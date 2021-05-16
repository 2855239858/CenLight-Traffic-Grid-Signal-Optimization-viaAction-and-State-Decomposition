import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import copy
from flow.core.agents.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs


latent_dim = 10

EP_MAX = 1000
EP_LEN = 200
GAMMA = 0.8
A_LR = 0.0005
C_LR = 0.0005
BATCH =50
A_UPDATE_STEPS = 10
C_UPDATE_STEPS = 10
METHOD = [
    dict(name='kl_pen', kl_target=0.01, lam=0.5),   # KL penalty
    dict(name='clip', epsilon=0.2),                 # Clipped surrogate objective, find this is better
][1]        # choose the method for optimization


###################Using Tensorflow's LTSM########################
#PPO1 builds the actor network by using tf's LTSM.
class PPO1(object):

    def __init__(self, s_dim=32, a_dim=1, name="meme"):
        runner1 = '/cpu:0'
        runner2 = '/gpu:0'
        with tf.device('/cpu:0'):
            self.sess = tf.Session()
            # self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
            self.tfs = tf.placeholder(tf.float32, [None, s_dim/a_dim], 'actor_state')
            self.a_dim = a_dim
            self.s_dim = s_dim
            self.name = name
            self.buffer_a = []
            self.buffer_s = []
            self.buffer_r = []
            self.global_steps = 0
            self.update_steps_a = 0
            self.update_steps_c = 0
            self.global_counter = 0
            self.pre_counter = 0
            self.batch_size = 0

            # self.rnn_input = tf.placeholder(tf.float32, [None, TIME_STEP, INPUT_SIZE])                   # (batch, height, width, channel)

            # critic
            with tf.variable_scope(self.name + '_critic'):
                l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu,bias_initializer = tf.constant_initializer(0.01), 
                kernel_initializer = tf.random_normal_initializer(0., .01))
                # l2 = tf.layers.dense(l1, 50, tf.nn.relu,bias_initializer = tf.constant_initializer(0.01), 
                # kernel_initializer = tf.random_normal_initializer(0., .01))
                # l2 = tf.layers.dense(l1, 32, tf.nn.relu,kernel_initializer = w_init,bias_initializer = tf.constant_initializer(0.01))
                self.v = tf.layers.dense(l1, 1)
                self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
                self.advantage = self.tfdc_r - self.v
                self.closs = tf.reduce_mean(tf.square(self.advantage))

                self.ctrain_op = tf.train.AdamOptimizer(C_LR).minimize(self.closs)

            # actor
            self.pi, pi_params = self._build_anet(self.name + '_pi', trainable=True)
            self.oldpi, oldpi_params = self._build_anet(self.name + '_oldpi', trainable=False)

            self.tfa = tf.placeholder(tf.int32, [None, self.a_dim], 'action')
            self.tfadv = tf.placeholder(tf.float32, [None, self.a_dim], 'advantage')

            self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

            ##调整概率分布的维度，方便获取概率
            # ratio = []
            # ratio_temp = []
            index = []
            pi_resize = tf.reshape(self.pi,[-1,2])
            oldpi_resize = tf.reshape(self.oldpi,[-1,2])
            # for i in range(self.batch_size):
            #     for j in range(self.a_dim):
            #         index.append([i,j,self.tfa[i][j]])
            self.a_indices = tf.stack([tf.range(tf.shape(tf.reshape(self.tfa,[-1]))[0], dtype=tf.int32), tf.reshape(self.tfa,[-1])], axis=1)
            pi_prob = tf.gather_nd(params=pi_resize, indices=self.a_indices)  
            oldpi_prob = tf.gather_nd(params=oldpi_resize, indices=self.a_indices) 
            self.ratio_temp1 = tf.reshape(pi_prob / (oldpi_prob + 1e-8),[-1,self.a_dim])
            # ratio_temp2 = tf.reshape(ratio_temp1,[-1,self.a_dim])
            # ratio = tf.reduce_mean(ratio_temp2,axis = 1)

            self.surr = self.ratio_temp1 * self.tfadv  # surrogate loss

            # for i in range(self.batch_size):
            #     ratio_temp = []
            #     for j in range(self.a_dim):
            #         ratio_temp.append(self.pi_resize[i][j][a_indices[i][j]]
            #             /(self.oldpi_resize[i][j][a_indices[i][j]] + 1e-8))
            #     ratio.append(tf.reduce_mean(ratio_temp))
            # surr = ratio * self.tfadv


            # a_indices = tf.stack([tf.range(tf.shape(self.tfa)[0], dtype=tf.int32), self.tfa], axis=1)
            # pi_prob = tf.gather_nd(params=self.pi, indices=a_indices)  # shape=(None, )
            # oldpi_prob = tf.gather_nd(params=self.oldpi, indices=a_indices)  # shape=(None, )
            # ratio = pi_prob / (oldpi_prob + 1e-8)
            # surr = ratio * self.tfadv  # surrogate loss


            self.aloss = -tf.reduce_mean(tf.minimum(  # clipped surrogate objective
                self.surr,
                tf.clip_by_value(self.ratio_temp1, 1. - 0.2, 1. + 0.2) * self.tfadv))

            self.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(self.aloss)

        ##以下为分开计算actor loss的部分

            self.aloss_seperated = -tf.reduce_mean(tf.reshape(tf.minimum(  # clipped surrogate objective
                self.surr,
                tf.clip_by_value(self.ratio_temp1, 1. - 0.2, 1. + 0.2) * self.tfadv),[-1,self.a_dim]),axis = 0)
            self.atrain_op_seperated = tf.train.AdamOptimizer(A_LR).minimize(self.aloss_seperated)



            self.sess.run(tf.global_variables_initializer())
            self.writer = tf.summary.FileWriter("baseline/rnn_discrete/" + self.name + "_log/", self.sess.graph)
            self.saver = tf.train.Saver(max_to_keep=20)
            tf.get_default_graph().finalize()

    def update_critic(self):
        s = np.vstack(self.buffer_s)
        r = np.vstack(self.buffer_r)
        critic_loss = self.sess.run(self.closs, {self.tfs: s, self.tfdc_r: r})
        self.summarize(critic_loss,self.pre_counter,'pre_Critic_loss')
        [self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r}) for _ in range(C_UPDATE_STEPS)]
        self.pre_counter += 1


    def update_actor(self):
        s = np.vstack(self.buffer_s)
        r = np.array(self.buffer_r)[:, np.newaxis]
        a = self.buffer_a
        self.batch_size = len(a)
        self.sess.run(self.update_oldpi_op)
        adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r})
        # print(np.array(adv).shape)
        # adv = (adv - adv.mean())/(adv.std()+1e-6)     # sometimes helpful
        actor_loss = self.sess.run(self.aloss, {self.tfs: s, self.tfa: a, self.tfadv: adv})
        self.summarize(actor_loss,self.global_counter,'Actor_loss')

        [self.sess.run(self.atrain_op, {self.tfs: s, self.tfa: a, self.tfadv: adv}) for _ in range(A_UPDATE_STEPS)]

        # update critic
        critic_loss = self.sess.run(self.closs, {self.tfs: s, self.tfdc_r: r})


        self.summarize(critic_loss,self.global_counter,'Critic_loss')
        # [self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r}) for _ in range(C_UPDATE_STEPS)]

        self.global_counter += 1


    def update_a_c(self):
        print("Update Actor and Critic")
        s = np.vstack(self.buffer_s)
        # r = np.array(self.buffer_r)[:, np.newaxis]
        # print(r)
        r = np.vstack(self.buffer_r)
        a = np.array(self.buffer_a).reshape([-1,self.a_dim])
        # self.batch_size = len(a)
        self.sess.run(self.update_oldpi_op)
        adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r})
        
        for i in range(len(adv)):
            adv[i] = abs(adv[i]) + 1

        # print(np.array(adv).shape)
        # adv = (adv - adv.mean())/(adv.std()+1e-6)     # sometimes helpful
        actor_loss = self.sess.run(self.aloss, {self.tfs: s, self.tfa: a, self.tfadv: adv})
        self.summarize(actor_loss,self.global_counter,'Actor_loss')

        [self.sess.run(self.atrain_op, {self.tfs: s, self.tfa: a, self.tfadv: adv}) for _ in range(A_UPDATE_STEPS)]

        # update critic
        critic_loss = self.sess.run(self.closs, {self.tfs: s, self.tfdc_r: r})
        self.summarize(critic_loss,self.global_counter,'Critic_loss')
        [self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r}) for _ in range(C_UPDATE_STEPS)]

        self.global_counter += 1



    def update(self):
        s = np.vstack(self.buffer_s)
        # r = np.array(self.buffer_r)[:, np.newaxis]
        # print(r)
        r = np.vstack(self.buffer_r)
        # s_addAc = np.array(s).reshape([-1,int(self.s_dim/self.a_dim)]).tolist()
        # for i in range(len(r)):
        #     s_addAc[i].append(self.buffer_a[i])

        print(self.sess.run(self.pi,{self.tfs:s}))
        a = np.array(self.buffer_a).reshape([-1,self.a_dim])
        # self.batch_size = len(a)
        self.sess.run(self.update_oldpi_op)
        adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r})
        adv_r = np.array(adv).reshape([-1,self.a_dim])
        # print(np.array(adv).shape)
        # adv = (adv - adv.mean())/(adv.std()+1e-6)     # sometimes helpful
        actor_loss = self.sess.run(self.aloss, {self.tfs: s, self.tfa: a, self.tfadv: adv_r})
        self.summarize(actor_loss,self.global_counter,'Actor_loss')


        # [self.sess.run(self.atrain_op, {self.tfs: s, self.tfa: a, self.tfadv: adv_r}) for _ in range(A_UPDATE_STEPS)]
        [self.sess.run(self.atrain_op_seperated, {self.tfs: s, self.tfa: a, self.tfadv: adv_r}) for _ in range(A_UPDATE_STEPS)]

        critic_loss = self.sess.run(self.closs, {self.tfs: s, self.tfdc_r: r})
        self.summarize(critic_loss,self.global_counter,'Critic_loss')
        [self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r}) for _ in range(C_UPDATE_STEPS)]

        self.global_counter += 1

    def _build_anet(self, name, trainable):
        # with tf.variable_scope(name):
        #     self.l1 = tf.layers.dense(self.tfs, 32, tf.nn.relu, trainable=trainable)
        #     self.l2 = tf.layers.dense(self.l1, 32, tf.nn.relu, trainable=trainable)
        #     out = tf.layers.dense(self.l2, self.a_dim, tf.nn.softmax, trainable=trainable)
        # params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIA_aRL5
        with tf.variable_scope(name):
            # RNN
            out_temp2 = []
            rnn_input = tf.reshape(self.tfs,[-1,self.a_dim,int(self.s_dim/self.a_dim)])
            rnn_cell = tf.nn.rnn_cell.LSTMCell(num_units=64,activation = tf.nn.tanh,
                initializer = tf.random_normal_initializer(0., .01))
            outputs, (h_c, h_n) = tf.nn.dynamic_rnn(
                rnn_cell,                   # cell you have chosen
                rnn_input,                      # input
                initial_state=None,         # the initial hidden state
                dtype=tf.float32,           # must given if set initial_state = None
                time_major=False,           # False: (batch, time step, input); True: (time step, batch, input)
            )
            for i in range(self.a_dim):
                out_temp1 = tf.layers.dense(outputs[:, i, :], 2,tf.nn.softmax,trainable = trainable, 
                kernel_initializer = tf.random_normal_initializer(0., .01),
                 bias_initializer = tf.constant_initializer(0.01))              # output based on the last output step      
                # out_temp2 = tf.layers.dense(out_temp1, 2,tf.nn.softmax,trainable = trainable,
                #     kernel_initializer = tf.random_normal_initializer(0., .01),
                #     bias_initializer = tf.constant_initializer(0.01))              # output based on the last output step      
                # out.append(out_temp2)
                out_temp2.append(out_temp1)
            out = tf.stack([out_temp2[k] for k in range(self.a_dim)], axis=1)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return out, params

    def choose_action(self, s):
        # _s = np.array(s).reshape([-1,TIME_STEP,INPUT_SIZE])

        # prob_weights = self.sess.run(self.pi, feed_dict={self.rnn_input: _s})
        # print(prob_weights)
        # action = np.random.choice(range(prob_weights.shape[1]),
        #                           p=prob_weights.ravel())  # select action w.r.t the actions prob
        # prob = tf.nn.softmax(prob_weights)
        _s = np.array(s).reshape([-1,int(self.s_dim/self.a_dim)])
        action = []
        prob = self.sess.run(self.pi,feed_dict={self.tfs: _s})
        prob_temp = np.array(prob).reshape([-1,2])
        # print(prob)
        for i in range(self.a_dim):
            action_temp = np.random.choice(range(prob_temp[i].shape[0]),
                                  p=prob_temp[i].ravel())  # select action w.r.t the actions prob
            action.append(action_temp)
            
        return action


    def choose_best_action(self, s):
        # _s = np.array(s).reshape([-1,TIME_STEP,INPUT_SIZE])

        # prob_weights = self.sess.run(self.pi, feed_dict={self.rnn_input: _s})
        # print(prob_weights)
        # action = np.random.choice(range(prob_weights.shape[1]),
        #                           p=prob_weights.ravel())  # select action w.r.t the actions prob
        # prob = tf.nn.softmax(prob_weights)
        action = []
        prob = self.sess.run(self.pi,feed_dict={self.tfs: s})
        for i in range(self.a_dim):
            action_temp = np.argmax(prob[i].ravel())  # select action w.r.t the actions prob
            action.append(action_temp)
            
        return action



    def get_state(self, s):
        s = s[np.newaxis, :]
        h = self.sess.run(self.l2, {self.tfs: s})[0]
        return h

    def get_v(self, s):
        _s = np.array(s)
        if _s.ndim < 2:
            s = _s[np.newaxis, :]
        # print(self.sess.run(self.v, {self.tfs: s}))
        return self.sess.run(self.v, {self.tfs: s})

    def experience_store(self, s, a, r):
        self.buffer_a.append(a)
        self.buffer_s.append(s)
        self.buffer_r.append(r)

    def empty_buffer(self):
        self.buffer_s, self.buffer_r, self.buffer_a = [], [], []

    ##每一步的reward进行一个discount，让越远的reward影响变小
    def trajction_process(self, s_):
        _s = np.array(s_).reshape([-1,int(self.s_dim/self.a_dim)]).tolist()
        # for i in range(len(a_)):
        #     _s[i].append(a_[i])

        v_s_ = self.get_v(_s)
        buffer_r = np.array(self.buffer_r).reshape([-1,self.a_dim])
        buffer = [[],[],[],[],[],[],[],[],[],[],[],[]]
        for r in buffer_r[::-1]:
            for i in range(self.a_dim):
                v_s_[i] = r[i] + GAMMA * v_s_[i]
                buffer[i].append(copy.deepcopy(v_s_[i]))
                
        for i in range(self.a_dim):
            buffer[i].reverse()
        out = np.stack([buffer[k] for k in range(self.a_dim)], axis=1)
        # print(self.buffer_r)

        self.buffer_r = np.array(out).reshape([-1])

    def summarize(self, reward, i, tag):
        summary = tf.Summary()
        summary.value.add(tag=tag, simple_value=reward)
        self.writer.add_summary(summary, i)
        self.writer.flush()

    def save_params(self,name,ep):
        save_path = self.saver.save(self.sess,'my_net/rnn_discrete/{}_ep{}.ckpt'.format(name,ep))
        print("Save to path:",save_path)
    def restore_params(self,name,ep):
        self.saver.restore(self.sess,'my_net/rnn_discrete/{}_ep{}.ckpt'.format(name,ep))
        print("Restore params from")

###################Self-built RNN network#########################
#One cycle layer of RNN.
class PPO2(object):
    #PPO2在PPO上自定义了actor的RNN网络结构，使能够让前一step的输出作为后一step的输入
    #In this class, the only verification is to rewrite the RNN neural network. (tip: Inputs is the same as class PPO)
    def __init__(self, s_dim=32, a_dim=1, name="meme"):
        runner1 = '/cpu:0'
        runner2 = '/gpu:0'
        with tf.device('/cpu:0'):
            self.sess = tf.Session()
            # self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
            self.a_dim = a_dim
            self.s_dim = s_dim
            self.name = name
            self.buffer_a = []
            self.buffer_s = []
            self.buffer_r = []
            self.global_steps = 0
            self.update_steps_a = 0
            self.update_steps_c = 0
            self.global_counter = 0
            self.pre_counter = 0

            self.hidden_net = 64
            self.output_net = 64

            # self.rnn_input = tf.placeholder(tf.float32, [None, TIME_STEP, INPUT_SIZE])                   # (batch, height, width, channel)


            # with tf.variable_scope('rnn_input_cell'):
            #     Uw = tf.get_variable('Uw', [self.s_dim, self.hidden_net],initializer = tf.constant_initializer(0.0))
            #     Ub = tf.get_variable('Ub', [self.hidden_net], initializer=tf.constant_initializer(0.0))
            # with tf.variable_scope('rnn_cycle_cell'):
            #     Ww = tf.get_variable('Ww', [self.hidden_net, self.hidden_net],initializer = tf.constant_initializer(0.0))
            #     Wb = tf.get_variable('Wb', [self.hidden_net], initializer=tf.constant_initializer(0.0))
            # with tf.variable_scope('rnn_output_cell'):
            #     Vw = tf.get_variable('Vw', [self.hidden_net, self.output_net],initializer = tf.constant_initializer(0.0))
            #     Vb = tf.get_variable('Vb', [self.output_net], initializer=tf.constant_initializer(0.0))

            self.tfs = tf.placeholder(tf.float32, [None, s_dim/a_dim], 'actor_state')
            self.tfa = tf.placeholder(tf.int32, [None], 'action')
            self.tfadv = tf.placeholder(tf.float32, [None, self.a_dim], 'advantage')
            
            # critic
            with tf.variable_scope(self.name + '_critic'):
                l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu,bias_initializer = tf.constant_initializer(0.01), 
                kernel_initializer = tf.random_normal_initializer(0., .01))
                self.v = tf.layers.dense(l1, 1)
                self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
                self.advantage = self.tfdc_r - self.v
                self.closs = tf.reduce_mean(tf.square(self.advantage))

                self.ctrain_op = tf.train.AdamOptimizer(C_LR).minimize(self.closs)

            # actor
            self.pi, pi_params = self._build_anet(self.name + '_pi', trainable=True)
            self.oldpi, oldpi_params = self._build_anet(self.name + '_oldpi', trainable=False)

            self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

            ##调整概率分布的维度，方便获取概率
            index = []
            self.pi_resize = tf.reshape(self.pi,[-1,2])
            self.oldpi_resize = tf.reshape(self.oldpi,[-1,2])

            self.a_indices = tf.stack([tf.range(tf.shape(tf.reshape(self.tfa,[-1]))[0], dtype=tf.int32), tf.reshape(self.tfa,[-1])], axis=1)
            pi_prob = tf.gather_nd(params=self.pi_resize, indices=self.a_indices)  
            oldpi_prob = tf.gather_nd(params=self.oldpi_resize, indices=self.a_indices) 
            self.ratio_temp1 = tf.reshape(pi_prob / (oldpi_prob + 1e-8),[-1,self.a_dim])

            self.surr = self.ratio_temp1 * self.tfadv  # surrogate loss

            self.aloss = -tf.reduce_mean(tf.minimum(  # clipped surrogate objective
                self.surr,
                tf.clip_by_value(self.ratio_temp1, 1. - 0.2, 1. + 0.2) * self.tfadv))

            self.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(self.aloss)

        ##以下为分开计算actor loss的部分

            # self.aloss_seperated = -tf.reduce_mean(tf.reshape(tf.minimum(  # clipped surrogate objective
            #     self.surr,
            #     tf.clip_by_value(self.ratio_temp1, 1. - 0.2, 1. + 0.2) * self.tfadv),[-1,self.a_dim]),axis = 0)
            # self.atrain_op_seperated = tf.train.AdamOptimizer(A_LR).minimize(self.aloss_seperated)

            self.sess.run(tf.global_variables_initializer())
            self.writer = tf.summary.FileWriter("baseline/rnn_discrete/" + self.name + "_log/", self.sess.graph)
            self.saver = tf.train.Saver(max_to_keep=20)
            tf.get_default_graph().finalize()

    def update(self):
        s = np.vstack(self.buffer_s)
        r = np.vstack(self.buffer_r)
        a = np.array(self.buffer_a)
        # print(s.shape)
        # print(a.shape)

        self.sess.run(self.update_oldpi_op)
        adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r})
        adv_r = np.array(adv).reshape([-1,self.a_dim])

        # tem = self.sess.run(self.pi,{self.tfs:s})
        # print(np.array(tem).shape)
        # tem2 = self.sess.run(self.pi_resize,{self.tfs:s})
        # print(np.array(tem2).shape)
        # old_pi = self.sess.run(self.oldpi_resize,{self.tfs:s})
        # print(np.array(old_pi).shape)
        # a_in = self.sess.run(self.a_indices,{self.tfa:a})
        # print(np.array(a_in).shape,a_in[6])


        actor_loss = self.sess.run(self.aloss, {self.tfs: s, self.tfa: a, self.tfadv: adv_r})
        self.summarize(actor_loss,self.global_counter,'Actor_loss')

        [self.sess.run(self.atrain_op, {self.tfs: s, self.tfa: a, self.tfadv: adv_r}) for _ in range(A_UPDATE_STEPS)]
        # [self.sess.run(self.atrain_op_seperated, {self.tfs: s, self.tfa: a, self.tfadv: adv_r}) for _ in range(A_UPDATE_STEPS)]

        critic_loss = self.sess.run(self.closs, {self.tfs: s, self.tfdc_r: r})
        self.summarize(critic_loss,self.global_counter,'Critic_loss')
        [self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r}) for _ in range(C_UPDATE_STEPS)]
        
        self.global_counter += 1

    def rnn_cell(self,rnn_input, state,name,trainable):
        #Yt = relu(St*Vw+Vb)
        #St = tanch(Xt*Uw + Ub + St-1*Ww+Wb)
        #Xt = [1,198] St-1 = [1,64]
        #Uw = [198,64] Ub = [64]
        #Ww = [64,64]   Wb = [64]
        #Vw = [64,64]      Vb = [64]
        with tf.variable_scope('rnn_input_cell_' + name, reuse=True):
            Uw = tf.get_variable('Uw', [int(self.s_dim/self.a_dim), self.hidden_net],trainable=trainable)
            Ub = tf.get_variable('Ub', [1,self.hidden_net], initializer=tf.constant_initializer(0.0),trainable=trainable)
        with tf.variable_scope('rnn_cycle_cell_' + name,  reuse=True):
            Ww = tf.get_variable('Ww', [self.hidden_net, self.hidden_net],trainable=trainable)
            Wb = tf.get_variable('Wb', [1,self.hidden_net], initializer=tf.constant_initializer(0.0),trainable=trainable)
        with tf.variable_scope('rnn_output_cell_' + name, reuse=True):
            Vw = tf.get_variable('Vw', [self.hidden_net, self.output_net],trainable=trainable)
            Vb = tf.get_variable('Vb', [1,self.output_net], initializer=tf.constant_initializer(0.0),trainable=trainable)

        St = tf.nn.tanh(tf.matmul(tf.cast(tf.reshape(rnn_input,[-1,int(self.s_dim/self.a_dim)]),tf.float32),tf.cast(Uw,tf.float32)) + tf.cast(Ub,tf.float32) + tf.matmul(tf.cast(state,tf.float32),tf.cast(Ww,tf.float32)) + tf.cast(Wb,tf.float32))
        Yt = tf.nn.relu(tf.matmul(tf.cast(St,tf.float32),tf.cast(Vw,tf.float32)) + tf.cast(Vb,tf.float32))
        # return
        return St,Yt

    def _build_anet(self, name, trainable):
        with tf.variable_scope(name):

            with tf.variable_scope('rnn_input_cell_' + name):
                Uw = tf.get_variable('Uw', [int(self.s_dim/self.a_dim), self.hidden_net],trainable=trainable)
                Ub = tf.get_variable('Ub', [1,self.hidden_net], initializer=tf.constant_initializer(0.0),trainable=trainable)
            with tf.variable_scope('rnn_cycle_cell_' + name):
                Ww = tf.get_variable('Ww', [self.hidden_net, self.hidden_net],trainable=trainable)
                Wb = tf.get_variable('Wb', [1,self.hidden_net], initializer=tf.constant_initializer(0.0),trainable=trainable)
            with tf.variable_scope('rnn_output_cell_' + name):
                Vw = tf.get_variable('Vw', [self.hidden_net, self.output_net],trainable=trainable)
                Vb = tf.get_variable('Vb', [1,self.output_net], initializer=tf.constant_initializer(0.0),trainable=trainable)

            # with tf.variable_scope('rnn_input_cell_' + self.name + '_oldpi'):
            #     Uw = tf.get_variable('Uw', [int(self.s_dim/self.a_dim), self.hidden_net],trainable=False)
            #     Ub = tf.get_variable('Ub', [self.hidden_net], initializer=tf.constant_initializer(0.0),trainable=False)
            # with tf.variable_scope('rnn_cycle_cell_' + self.name + '_oldpi'):
            #     Ww = tf.get_variable('Ww', [self.hidden_net, self.hidden_net],trainable=False)
            #     Wb = tf.get_variable('Wb', [self.hidden_net], initializer=tf.constant_initializer(0.0),trainable=False)
            # with tf.variable_scope('rnn_output_cell_' + self.name + '_oldpi'):
            #     Vw = tf.get_variable('Vw', [self.hidden_net, self.output_net],trainable=False)
            #     Vb = tf.get_variable('Vb', [self.output_net], initializer=tf.constant_initializer(0.0),trainable=False)

            # RNN
            out_temp1 = []
            out_temp2 = []
            out = []
            rnn_input = tf.reshape(self.tfs,[-1,self.a_dim,int(self.s_dim/self.a_dim)])
            state = np.zeros([1,self.hidden_net])
            for j in range(self.a_dim):
                state,y = self.rnn_cell(rnn_input[:,j,:],state,name,trainable)
                out_temp1.append(y)
            # print(tf.shape(out_temp1))
            for i in range(self.a_dim):
                out_temp2.append(tf.layers.dense(out_temp1[i], 2,tf.nn.softmax,trainable = trainable, 
                    kernel_initializer = tf.random_normal_initializer(0., .01),
                    bias_initializer = tf.constant_initializer(0.01)))
            out = tf.stack([out_temp2[k] for k in range(self.a_dim)], axis=1)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return out, params

    def choose_action(self, s):

        _s = np.array(s).reshape([-1,int(self.s_dim/self.a_dim)])
        action = []
        prob = self.sess.run(self.pi,feed_dict={self.tfs: _s})
        prob_temp = np.array(prob).reshape([-1,2])
        print(prob)
        for i in range(self.a_dim):
            action_temp = np.random.choice(range(prob_temp[i].shape[0]),
                                  p=prob_temp[i].ravel())  # select action w.r.t the actions prob
            action.append(action_temp)
            
        return action

    def get_state(self, s):
        s = s[np.newaxis, :]
        h = self.sess.run(self.l2, {self.tfs: s})[0]
        return h

    def get_v(self, s):
        _s = np.array(s)
        if _s.ndim < 2:
            s = _s[np.newaxis, :]
        # print(self.sess.run(self.v, {self.tfs: s}))
        return self.sess.run(self.v, {self.tfs: s})

    def experience_store(self, s, a, r):
        self.buffer_a.append(a)
        self.buffer_s.append(s)
        self.buffer_r.append(r)

    def empty_buffer(self):
        self.buffer_s, self.buffer_r, self.buffer_a = [], [], []

    ##每一步的reward进行一个discount，让越远的reward影响变小
    def trajction_process(self, s_):
        _s = np.array(s_).reshape([-1,int(self.s_dim/self.a_dim)]).tolist()
        # for i in range(len(a_)):
        #     _s[i].append(a_[i])

        v_s_ = self.get_v(_s)
        buffer_r = np.array(self.buffer_r).reshape([-1,self.a_dim])
        buffer = [[],[],[],[],[],[],[],[],[],[],[],[]]
        for r in buffer_r[::-1]:
            for i in range(self.a_dim):
                v_s_[i] = r[i] + GAMMA * v_s_[i]
                buffer[i].append(copy.deepcopy(v_s_[i]))
                
        for i in range(self.a_dim):
            buffer[i].reverse()
        out = np.stack([buffer[k] for k in range(self.a_dim)], axis=1)
        # print(self.buffer_r)

        self.buffer_r = np.array(out).reshape([-1])

    def summarize(self, reward, i, tag):
        summary = tf.Summary()
        summary.value.add(tag=tag, simple_value=reward)
        self.writer.add_summary(summary, i)
        self.writer.flush()

    def save_params(self,name,ep):
        save_path = self.saver.save(self.sess,'my_net/rnn_discrete/{}_ep{}.ckpt'.format(name,ep))
        print("Save to path:",save_path)
    def restore_params(self,name,ep):
        self.saver.restore(self.sess,'my_net/rnn_discrete/{}_ep{}.ckpt'.format(name,ep))
        print("Restore params from")

###################Two cycle layer RNN###########################
#PPO3 has two layers of RNN neural network.
#First layer doesn't output actions and we record the last step's hidden state
#as the second cycle layer's first step's hidden input state.(Ensuring that all actions
#  decided at each timestep are depanded on all signals' states.)
class PPO3(object):
    #PPO2在PPO上自定义了actor的RNN网络结构，使能够让前一step的输出作为后一step的输入
    #In this class, the only verification is to rewrite the RNN neural network. 
    #The input states of RNN are different too. (For each step of RNN, input states are states of signal and the signal's chosen action.)

    def __init__(self, s_dim=32, a_dim=1, name="meme", combine_action = 1):
        runner1 = '/cpu:0'
        runner2 = '/gpu:0'
        with tf.device('/cpu:0'):
            self.sess = tf.Session()
            # self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
            self.a_dim = a_dim
            self.s_dim = s_dim
            self.name = name
            self.buffer_a = []
            self.buffer_s = []
            self.buffer_r = []
            self.global_steps = 0
            self.update_steps_a = 0
            self.update_steps_c = 0
            self.global_counter = 0
            self.pre_counter = 0

            self.hidden_net = 64
            self.output_net = 64
            self.combine_action = combine_action

            # self.rnn_input = tf.placeholder(tf.float32, [None, TIME_STEP, INPUT_SIZE])                   # (batch, height, width, channel)


            # with tf.variable_scope('rnn_input_cell'):
            #     Uw = tf.get_variable('Uw', [self.s_dim, self.hidden_net],initializer = tf.constant_initializer(0.0))
            #     Ub = tf.get_variable('Ub', [self.hidden_net], initializer=tf.constant_initializer(0.0))
            # with tf.variable_scope('rnn_cycle_cell'):
            #     Ww = tf.get_variable('Ww', [self.hidden_net, self.hidden_net],initializer = tf.constant_initializer(0.0))
            #     Wb = tf.get_variable('Wb', [self.hidden_net], initializer=tf.constant_initializer(0.0))
            # with tf.variable_scope('rnn_output_cell'):
            #     Vw = tf.get_variable('Vw', [self.hidden_net, self.output_net],initializer = tf.constant_initializer(0.0))
            #     Vb = tf.get_variable('Vb', [self.output_net], initializer=tf.constant_initializer(0.0))

            self.tfa = tf.placeholder(tf.int32, [None], 'action')
            self.tfadv = tf.placeholder(tf.float32, [None, int(self.a_dim/self.combine_action)], 'advantage')
            self.tfs = tf.placeholder(tf.float32, [None, int(s_dim * self.combine_action/a_dim)], 'actor_state')
            # critic
            with tf.variable_scope(self.name + '_critic'):
                l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu,bias_initializer = tf.constant_initializer(0.01), 
                kernel_initializer = tf.random_normal_initializer(0., .01))
                self.v = tf.layers.dense(l1, 1)
                self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
                self.advantage = self.tfdc_r - self.v
                self.closs = tf.reduce_mean(tf.square(self.advantage))

                self.ctrain_op = tf.train.AdamOptimizer(C_LR).minimize(self.closs)

            # actor
            self.pi, pi_params = self._build_anet(self.name + '_pi', trainable=True)
            self.oldpi, oldpi_params = self._build_anet(self.name + '_oldpi', trainable=False)

            self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

            ##调整概率分布的维度，方便获取概率
            index = []
            self.pi_resize = tf.reshape(self.pi,[-1,2])
            self.oldpi_resize = tf.reshape(self.oldpi,[-1,2])

            self.a_indices = tf.stack([tf.range(tf.shape(tf.reshape(self.tfa,[-1]))[0], dtype=tf.int32), tf.reshape(self.tfa,[-1])], axis=1)
            pi_prob = tf.gather_nd(params=self.pi_resize, indices=self.a_indices)  
            oldpi_prob = tf.gather_nd(params=self.oldpi_resize, indices=self.a_indices) 
            self.ratio_temp1 = tf.reshape(tf.reduce_mean(tf.reshape(pi_prob / (oldpi_prob + 1e-8),[-1,self.combine_action]),axis= 1),
                                                        [-1,int(self.a_dim/self.combine_action)])

            self.surr = self.ratio_temp1 * self.tfadv  # surrogate loss

            self.aloss = -tf.reduce_mean(tf.minimum(  # clipped surrogate objective
                self.surr,
                tf.clip_by_value(self.ratio_temp1, 1. - 0.2, 1. + 0.2) * self.tfadv))

            self.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(self.aloss)

        ##以下为分开计算actor loss的部分

            self.aloss_seperated = -tf.reduce_mean(tf.reshape(tf.minimum(  # clipped surrogate objective
                self.surr,
                tf.clip_by_value(self.ratio_temp1, 1. - 0.2, 1. + 0.2) * self.tfadv),[-1,self.a_dim]),axis = 0)
            # self.atrain_op_seperated = [tf.train.AdamOptimizer(A_LR).minimize(self.aloss_seperated[k]) for k in range(self.a_dim)]
            self.atrain_op_seperated = [tf.train.AdamOptimizer(A_LR).minimize(self.aloss_seperated[k]) for k in range(1)]

            self.sess.run(tf.global_variables_initializer())
            self.writer = tf.summary.FileWriter("baseline/PPO3/" + self.name + "_log/", self.sess.graph)
            self.saver = tf.train.Saver(max_to_keep=5)
            # tf.get_default_graph().finalize()

    def update_critic(self):
        s = np.vstack(self.buffer_s)
        r = np.vstack(self.buffer_r)
        critic_loss = self.sess.run(self.closs, {self.tfs: s, self.tfdc_r: r})
        self.summarize(critic_loss,self.pre_counter,'pre_Critic_loss')
        [self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r}) for _ in range(C_UPDATE_STEPS)]
        self.pre_counter += 1

    def update(self):
        print("Update")
        s = np.vstack(self.buffer_s)
        c_s = s.reshape([-1,int(self.s_dim * self.combine_action/ self.a_dim)])
        r = np.vstack(self.buffer_r)
        a = np.array(self.buffer_a).reshape([-1])
        # print(s.shape)
        # print(a.shape)

        self.sess.run(self.update_oldpi_op)
        adv = self.sess.run(self.advantage, {self.tfs: c_s, self.tfdc_r: r})
        
        ##Calculating advantages one
        adv_r = np.array(adv).reshape([-1,int(self.a_dim/self.combine_action)])

        # ##Calculating advantages two
        # adv_mean, adv_std = mpi_statistics_scalar(adv)
        # adv_ori = (adv - adv_mean) / adv_std
        # adv_r = np.array(adv_ori).reshape([-1,int(self.a_dim/self.combine_action)])

        # tem = self.sess.run(self.pi,{self.tfs:s})
        # print(np.array(tem).shape)
        # tem2 = self.sess.run(self.pi_resize,{self.tfs:s})
        # print(np.array(tem2).shape)
        # old_pi = self.sess.run(self.oldpi_resize,{self.tfs:s})
        # print(np.array(old_pi).shape)
        # a_in = self.sess.run(self.a_indices,{self.tfa:a})
        # print(np.array(a_in).shape,a_in[6])


        actor_loss = self.sess.run(self.aloss, {self.tfs: c_s, self.tfa: a, self.tfadv: adv_r})
        self.summarize(actor_loss,self.global_counter,'Actor_loss')

        [self.sess.run(self.atrain_op, {self.tfs: c_s, self.tfa: a, self.tfadv: adv_r}) for _ in range(A_UPDATE_STEPS)]
        # [self.sess.run(self.atrain_op_seperated, {self.tfs: c_s, self.tfa: a, self.tfadv: adv_r}) for _ in range(A_UPDATE_STEPS)]

        critic_loss = self.sess.run(self.closs, {self.tfs: c_s, self.tfdc_r: r})
        self.summarize(critic_loss,self.global_counter,'Critic_loss')
        [self.sess.run(self.ctrain_op, {self.tfs:c_s, self.tfdc_r: r}) for _ in range(C_UPDATE_STEPS)]
        
        self.global_counter += 1

    def rnn_cell(self,rnn_input, state,name,trainable,last_prob):
        #Yt = relu(St*Vw+Vb)
        #St = tanch(Xt*Uw + Ub + St-1*Ww+Wb)
        #Xt = [none,198 + 2] St-1 = [none,64] Yt = [none,64]
        #Uw = [198 + 2,64] Ub = [64]
        #Ww = [64,64]   Wb = [64]
        #Vw = [64,64]      Vb = [64]
        with tf.variable_scope('rnn_input_cell_' + name, reuse=True):
            Uw = tf.get_variable('Uw', [int(self.s_dim/self.a_dim) + 2, self.hidden_net],trainable=trainable)
            Ub = tf.get_variable('Ub', [1,self.hidden_net], initializer=tf.constant_initializer(0.0),trainable=trainable)
        with tf.variable_scope('rnn_cycle_cell_' + name,  reuse=True):
            Ww = tf.get_variable('Ww', [self.hidden_net, self.hidden_net],trainable=trainable)
            Wb = tf.get_variable('Wb', [1,self.hidden_net], initializer=tf.constant_initializer(0.0),trainable=trainable)
        with tf.variable_scope('rnn_output_cell_' + name, reuse=True):
            Vw = tf.get_variable('Vw', [self.hidden_net, self.output_net],trainable=trainable)
            Vb = tf.get_variable('Vb', [1,self.output_net], initializer=tf.constant_initializer(0.0),trainable=trainable)
        if last_prob == None:
            St = tf.nn.tanh(tf.matmul(tf.cast(tf.reshape(tf.pad(rnn_input,[[0,0],[0,2]]),[-1,int(self.s_dim/self.a_dim) + 2]),tf.float32),tf.cast(Uw,tf.float32)) + tf.cast(Ub,tf.float32) + tf.matmul(tf.cast(state,tf.float32),tf.cast(Ww,tf.float32)) + tf.cast(Wb,tf.float32))
        else:
            St = tf.nn.tanh(tf.matmul(tf.cast(tf.concat([tf.reshape(rnn_input,[-1,int(self.s_dim/self.a_dim)]),last_prob],axis = 1),tf.float32),tf.cast(Uw,tf.float32)) + tf.cast(Ub,tf.float32) + tf.matmul(tf.cast(state,tf.float32),tf.cast(Ww,tf.float32)) + tf.cast(Wb,tf.float32))

        # St = tf.nn.tanh(tf.matmul(tf.cast(tf.reshape(tf.concat([rnn_input,last_prob],1),[-1,int(self.s_dim/self.a_dim)]),tf.float32),tf.cast(Uw,tf.float32)) + tf.cast(Ub,tf.float32) + tf.matmul(tf.cast(state,tf.float32),tf.cast(Ww,tf.float32)) + tf.cast(Wb,tf.float32))
        Yt = tf.nn.relu(tf.matmul(tf.cast(St,tf.float32),tf.cast(Vw,tf.float32)) + tf.cast(Vb,tf.float32))
        # return
        return St,Yt

    def _build_anet_FCN(self,name,trainable):
        with tf.variable_scope(name):
            input = tf.reshape(self.tfs,[-1,int(self.s_dim/self.a_dim)])
            l1 = tf.layers.dense(input, 64, tf.nn.relu,bias_initializer = tf.constant_initializer(0.01), 
                kernel_initializer = tf.random_normal_initializer(0., .01),trainable=trainable)
            # l2 = tf.layers.dense(l1, 100, tf.nn.relu,bias_initializer = tf.constant_initializer(0.01), 
            #     kernel_initializer = tf.random_normal_initializer(0., .01),trainable=trainable)
            out = tf.layers.dense(l1, 2,tf.nn.softmax,trainable=trainable)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return out,params

    def _build_anet(self, name, trainable):
        with tf.variable_scope(name):

            with tf.variable_scope('rnn_input_cell_' + name):
                Uw = tf.get_variable('Uw', [int(self.s_dim/self.a_dim) + 2, self.hidden_net],trainable=trainable)
                Ub = tf.get_variable('Ub', [1,self.hidden_net], initializer=tf.constant_initializer(0.0),trainable=trainable)
            with tf.variable_scope('rnn_cycle_cell_' + name):
                Ww = tf.get_variable('Ww', [self.hidden_net, self.hidden_net],trainable=trainable)
                Wb = tf.get_variable('Wb', [1,self.hidden_net], initializer=tf.constant_initializer(0.0),trainable=trainable)
            with tf.variable_scope('rnn_output_cell_' + name):
                Vw = tf.get_variable('Vw', [self.hidden_net, self.output_net],trainable=trainable)
                Vb = tf.get_variable('Vb', [1,self.output_net], initializer=tf.constant_initializer(0.0),trainable=trainable)

            # RNN
            out_temp1 = []
            out_temp2 = []
            out = []
            actions = []
            last_prob = None
            rnn_input = tf.reshape(self.tfs,[-1,self.a_dim,int(self.s_dim/self.a_dim)])
            state = np.zeros([1,self.hidden_net])
            #The first for cycle aims to get state include all signals' imformation
            #and   pass to the second RNN layer (through variate "state")
            for j in range(self.a_dim):
                state,y = self.rnn_cell(rnn_input[:,j,:],state,name,trainable,last_prob)
                out_temp1.append(tf.layers.dense(y, 2,tf.nn.softmax,trainable = trainable, 
                    kernel_initializer = tf.random_normal_initializer(0., .01),
                    bias_initializer = tf.constant_initializer(0.01)))
                last_prob = out_temp1[j]
            #The second cycle is aim to make actions depend on last cycle's final state.
            for j in range(self.a_dim):
                state,y = self.rnn_cell(rnn_input[:,j,:],state,name,trainable,last_prob)
                out_temp2.append(tf.layers.dense(y, 2,tf.nn.softmax,trainable = trainable, 
                    kernel_initializer = tf.random_normal_initializer(0., .01),
                    bias_initializer = tf.constant_initializer(0.01)))
                last_prob = out_temp2[j]
                # actions = np.random.choice(range(out_temp1[j]),p=out_temp1[j].ravel())
            out = tf.stack([out_temp2[k] for k in range(self.a_dim)], axis=1)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return out, params

    def choose_action(self, s):

        _s = np.array(s).reshape([-1,int(self.s_dim* self.combine_action/self.a_dim)])
        action = []
        prob = self.sess.run(self.pi,feed_dict={self.tfs: _s})
        prob_temp = np.array(prob).reshape([-1,2])
        # print(prob)

        for i in range(self.a_dim):
            action_temp = np.random.choice(range(prob_temp[i].shape[0]),
                                p=prob_temp[i].ravel())  # select action w.r.t the actions prob
            action.append(action_temp)

        #the next part we initial a seed of random number limited in (0,1]
        #when seed first less than 0.9(threshold) that choose action according to given probability.
        #but if seed less bigger than 0.9, then we choose action equally.
        # for i in range(self.a_dim):
        #     seed = np.random.rand()
        #     if seed < 0.9:
        #         action_temp = np.random.choice(range(prob_temp[i].shape[0]),
        #                             p=prob_temp[i].ravel())  # select action w.r.t the actions prob
        #         action.append(action_temp)
        #     else:
        #         seed = np.random.rand()
        #         if seed < 0.5:
        #             action.append(0)
        #         else:
        #             action.append(1)
        
        return action

    def get_state(self, s):
        s = s[np.newaxis, :]
        h = self.sess.run(self.l2, {self.tfs: s})[0]
        return h

    def get_v(self, s):
        _s = np.array(s)
        if _s.ndim < 2:
            s = _s[np.newaxis, :]
        # print(self.sess.run(self.v, {self.tfs: s}))
        return self.sess.run(self.v, {self.tfs: s})

    def experience_store(self, s, a, r):
        self.buffer_a.append(a)
        self.buffer_s.append(s)
        self.buffer_r.append(r)

    def empty_buffer(self):
        self.buffer_s, self.buffer_r, self.buffer_a = [], [], []


    def trajction_process_proximate(self):
        #This function aims to calculate proximate F(s) of each state s.
        v_s_ = np.mean(np.array(self.buffer_r).reshape([-1,self.a_dim]),axis = 0)
        #we assume that all of the following Rs are the mean value of simulated steps (200)
        #so, the following Rs are geometric progression.
        #Sn = a1 * (1 - GAMMA^^n) / (1 - GAMMA) proximate equals to a1/(1-GAMMA)
        # print(v_s_)
        v_s_ = v_s_ / (1 - GAMMA)
        # print(v_s_)
        buffer_r = np.array(self.buffer_r).reshape([-1,self.a_dim])
        buffer = [[],[],[],[],[],[],[],[],[],[],[],[]]
        for r in buffer_r[::-1]:
            for i in range(self.a_dim):
                v_s_[i] = r[i] + GAMMA * v_s_[i]
                buffer[i].append(copy.deepcopy(v_s_[i]))
                
        for i in range(self.a_dim):
            buffer[i].reverse()
        # print(np.array(buffer[0]))
        out = np.stack([buffer[k] for k in range(self.a_dim)], axis=1)

        self.buffer_r = np.array(out).reshape([-1])


    ##每一步的reward进行一个discount，让越远的reward影响变小
    def trajction_process(self, s_):
        _s = np.array(s_).reshape([-1,int(self.s_dim * self.combine_action/self.a_dim)]).tolist()
        # for i in range(len(a_)):
        #     _s[i].append(a_[i])
        # v_s_ = [0,0,0,0,0,0]
        v_s_ = self.get_v(_s)
        # print(v_s_)
        buffer_r = np.mean(np.array(self.buffer_r).reshape([-1,self.combine_action]), axis= 1).reshape([-1,int(self.a_dim / self.combine_action)])

        buffer = [[],[],[],[],[],[],[],[],[],[],[],[]]
        for r in buffer_r[::-1]:
            for i in range(int(self.a_dim/self.combine_action)):
                # print('v1:{}'.format(v_s_[i]))
                # print('r:{}'.format(r[i]))
                # print('v2:{}'.format(r[i] + GAMMA * v_s_[i]))
                v_s_[i] = (r[i] + GAMMA * v_s_[i])
                # print('v3:{}'.format(v_s_[i]))
                buffer[i].append(copy.deepcopy(v_s_[i]))
        for i in range(int(self.a_dim/self.combine_action)):
            buffer[i].reverse()

        # print(np.array(buffer[0]))
        out = np.stack([buffer[k] for k in range(int(self.a_dim/self.combine_action))], axis=1)
        # print(out)
        self.buffer_r = np.array(out).reshape([-1])

    def summarize(self, reward, i, tag):
        summary = tf.Summary()
        summary.value.add(tag=tag, simple_value=reward)
        self.writer.add_summary(summary, i)
        self.writer.flush()

    def save_params(self,name,ep):
        save_path = self.saver.save(self.sess,'my_net/rnn_discrete/{}_ep{}.ckpt'.format(name,ep))
        print("Save to path:",save_path)
    def restore_params(self,name,ep):
        self.saver.restore(self.sess,'my_net/rnn_discrete/{}_ep{}.ckpt'.format(name,ep))
        print("Restore params from")

###################RNN accepts all states at first timestep############
#This function works bad.
class PPO4(object):
    #PPO2在PPO上自定义了actor的RNN网络结构，使能够让前一step的输出作为后一step的输入
    #In this class, the only verification is to rewrite the RNN neural network. 
    #The input states of RNN are different too. 
    #In this class, we input all signals' states into RNN at once, and the following inputs of RNN is last 
    #chosen action.
    
    def __init__(self, s_dim=32, a_dim=1, name="meme"):
        runner1 = '/cpu:0'
        runner2 = '/gpu:0'
        with tf.device('/cpu:0'):
            self.sess = tf.Session()
            # self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
            self.a_dim = a_dim
            self.s_dim = s_dim
            self.name = name
            self.buffer_a = []
            self.buffer_s = []
            self.buffer_r = []
            self.global_steps = 0
            self.update_steps_a = 0
            self.update_steps_c = 0
            self.global_counter = 0
            self.pre_counter = 0

            self.hidden_net = 64
            self.output_net = 64

            # self.rnn_input = tf.placeholder(tf.float32, [None, TIME_STEP, INPUT_SIZE])                   # (batch, height, width, channel)


            # with tf.variable_scope('rnn_input_cell'):
            #     Uw = tf.get_variable('Uw', [self.s_dim, self.hidden_net],initializer = tf.constant_initializer(0.0))
            #     Ub = tf.get_variable('Ub', [self.hidden_net], initializer=tf.constant_initializer(0.0))
            # with tf.variable_scope('rnn_cycle_cell'):
            #     Ww = tf.get_variable('Ww', [self.hidden_net, self.hidden_net],initializer = tf.constant_initializer(0.0))
            #     Wb = tf.get_variable('Wb', [self.hidden_net], initializer=tf.constant_initializer(0.0))
            # with tf.variable_scope('rnn_output_cell'):
            #     Vw = tf.get_variable('Vw', [self.hidden_net, self.output_net],initializer = tf.constant_initializer(0.0))
            #     Vb = tf.get_variable('Vb', [self.output_net], initializer=tf.constant_initializer(0.0))

            self.tfs = tf.placeholder(tf.float32, [None, s_dim/a_dim], 'actor_state')
            self.tfa = tf.placeholder(tf.int32, [None], 'action')
            self.tfadv = tf.placeholder(tf.float32, [None, self.a_dim], 'advantage')
            
            # critic
            with tf.variable_scope(self.name + '_critic'):
                l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu,bias_initializer = tf.constant_initializer(0.01), 
                kernel_initializer = tf.random_normal_initializer(0., .01))
                self.v = tf.layers.dense(l1, 1)
                self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
                self.advantage = self.tfdc_r - self.v
                self.closs = tf.reduce_mean(tf.square(self.advantage))

                self.ctrain_op = tf.train.AdamOptimizer(C_LR).minimize(self.closs)

            # actor
            self.pi, pi_params = self._build_anet(self.name + '_pi', trainable=True)
            self.oldpi, oldpi_params = self._build_anet(self.name + '_oldpi', trainable=False)

            self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

            ##调整概率分布的维度，方便获取概率
            index = []
            self.pi_resize = tf.reshape(self.pi,[-1,2])
            self.oldpi_resize = tf.reshape(self.oldpi,[-1,2])

            self.a_indices = tf.stack([tf.range(tf.shape(tf.reshape(self.tfa,[-1]))[0], dtype=tf.int32), tf.reshape(self.tfa,[-1])], axis=1)
            pi_prob = tf.gather_nd(params=self.pi_resize, indices=self.a_indices)  
            oldpi_prob = tf.gather_nd(params=self.oldpi_resize, indices=self.a_indices) 
            self.ratio_temp1 = tf.reshape(pi_prob / (oldpi_prob + 1e-8),[-1,self.a_dim])

            self.surr = self.ratio_temp1 * self.tfadv  # surrogate loss

            self.aloss = -tf.reduce_mean(tf.minimum(  # clipped surrogate objective
                self.surr,
                tf.clip_by_value(self.ratio_temp1, 1. - 0.2, 1. + 0.2) * self.tfadv))

            self.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(self.aloss)

        ##以下为分开计算actor loss的部分

            # self.aloss_seperated = -tf.reduce_mean(tf.reshape(tf.minimum(  # clipped surrogate objective
            #     self.surr,
            #     tf.clip_by_value(self.ratio_temp1, 1. - 0.2, 1. + 0.2) * self.tfadv),[-1,self.a_dim]),axis = 0)
            # self.atrain_op_seperated = tf.train.AdamOptimizer(A_LR).minimize(self.aloss_seperated)

            self.sess.run(tf.global_variables_initializer())
            self.writer = tf.summary.FileWriter("baseline/rnn_discrete/" + self.name + "_log/", self.sess.graph)
            self.saver = tf.train.Saver(max_to_keep=20)
            tf.get_default_graph().finalize()

    def update(self):
        s = np.vstack(self.buffer_s)
        r = np.vstack(self.buffer_r)
        a = np.array(self.buffer_a)
        # print(s.shape)
        # print(a.shape)

        self.sess.run(self.update_oldpi_op)
        adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r})
        adv_r = np.array(adv).reshape([-1,self.a_dim])

        # tem = self.sess.run(self.pi,{self.tfs:s})
        # print(np.array(tem).shape)
        # tem2 = self.sess.run(self.pi_resize,{self.tfs:s})
        # print(np.array(tem2).shape)
        # old_pi = self.sess.run(self.oldpi_resize,{self.tfs:s})
        # print(np.array(old_pi).shape)
        # a_in = self.sess.run(self.a_indices,{self.tfa:a})
        # print(np.array(a_in).shape,a_in[6])


        actor_loss = self.sess.run(self.aloss, {self.tfs: s, self.tfa: a, self.tfadv: adv_r})
        self.summarize(actor_loss,self.global_counter,'Actor_loss')

        [self.sess.run(self.atrain_op, {self.tfs: s, self.tfa: a, self.tfadv: adv_r}) for _ in range(A_UPDATE_STEPS)]
        # [self.sess.run(self.atrain_op_seperated, {self.tfs: s, self.tfa: a, self.tfadv: adv_r}) for _ in range(A_UPDATE_STEPS)]

        critic_loss = self.sess.run(self.closs, {self.tfs: s, self.tfdc_r: r})
        self.summarize(critic_loss,self.global_counter,'Critic_loss')
        [self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r}) for _ in range(C_UPDATE_STEPS)]
        
        self.global_counter += 1

    def rnn_cell(self,rnn_input, state,name,trainable,last_prob):
        #Yt = relu(St*Vw+Vb)
        #St = tanch(Xt*Uw + Ub + St-1*Ww+Wb)
        #Xt = [none,198*a_dim + 2] St-1 = [none,64] Yt = [none,64]
        #Uw = [198*a_dim + 2,64] Ub = [64]
        #Ww = [64,64]   Wb = [64]
        #Vw = [64,64]      Vb = [64]
        with tf.variable_scope('rnn_input_cell_' + name, reuse=True):
            Uw = tf.get_variable('Uw', [self.s_dim + 2, self.hidden_net],trainable=trainable)
            Ub = tf.get_variable('Ub', [1,self.hidden_net], initializer=tf.constant_initializer(0.0),trainable=trainable)
        with tf.variable_scope('rnn_cycle_cell_' + name,  reuse=True):
            Ww = tf.get_variable('Ww', [self.hidden_net, self.hidden_net],trainable=trainable)
            Wb = tf.get_variable('Wb', [1,self.hidden_net], initializer=tf.constant_initializer(0.0),trainable=trainable)
        with tf.variable_scope('rnn_output_cell_' + name, reuse=True):
            Vw = tf.get_variable('Vw', [self.hidden_net, self.output_net],trainable=trainable)
            Vb = tf.get_variable('Vb', [1,self.output_net], initializer=tf.constant_initializer(0.0),trainable=trainable)
        if last_prob == None:
            St = tf.nn.tanh(tf.matmul(tf.cast(tf.reshape(tf.pad(rnn_input,[[0,0],[0,2]]),
            [-1,int(self.s_dim + 2)]),tf.float32),tf.cast(Uw,tf.float32)) + tf.cast(Ub,tf.float32)
             + tf.matmul(tf.cast(state,tf.float32),tf.cast(Ww,tf.float32)) + tf.cast(Wb,tf.float32))
        else:
            St = tf.nn.tanh(tf.matmul(tf.cast(tf.concat([tf.reshape(rnn_input,[-1,self.s_dim]),
            last_prob],axis = 1),tf.float32),tf.cast(Uw,tf.float32)) + tf.cast(Ub,tf.float32)
             + tf.matmul(tf.cast(state,tf.float32),tf.cast(Ww,tf.float32)) + tf.cast(Wb,tf.float32))

        # St = tf.nn.tanh(tf.matmul(tf.cast(tf.reshape(tf.concat([rnn_input,last_prob],1),[-1,int(self.s_dim/self.a_dim)]),tf.float32),tf.cast(Uw,tf.float32)) + tf.cast(Ub,tf.float32) + tf.matmul(tf.cast(state,tf.float32),tf.cast(Ww,tf.float32)) + tf.cast(Wb,tf.float32))
        Yt = tf.nn.relu(tf.matmul(tf.cast(St,tf.float32),tf.cast(Vw,tf.float32)) + tf.cast(Vb,tf.float32))
        # return
        return St,Yt

    def _build_anet(self, name, trainable):
        with tf.variable_scope(name):

            with tf.variable_scope('rnn_input_cell_' + name):
                Uw = tf.get_variable('Uw', [self.s_dim + 2, self.hidden_net],trainable=trainable)
                Ub = tf.get_variable('Ub', [1,self.hidden_net], initializer=tf.constant_initializer(0.0),trainable=trainable)
            with tf.variable_scope('rnn_cycle_cell_' + name):
                Ww = tf.get_variable('Ww', [self.hidden_net, self.hidden_net],trainable=trainable)
                Wb = tf.get_variable('Wb', [1,self.hidden_net], initializer=tf.constant_initializer(0.0),trainable=trainable)
            with tf.variable_scope('rnn_output_cell_' + name):
                Vw = tf.get_variable('Vw', [self.hidden_net, self.output_net],trainable=trainable)
                Vb = tf.get_variable('Vb', [1,self.output_net], initializer=tf.constant_initializer(0.0),trainable=trainable)

            # RNN
            out_temp1 = []
            out = []
            actions = []
            last_prob = None
            rnn_input = tf.reshape(self.tfs,[-1,1,self.s_dim])
            rnn_input = tf.pad(rnn_input,[[0,0],[0,int(self.a_dim-1)],[0,0]])
            state = np.zeros([1,self.hidden_net])
            for j in range(self.a_dim):
                state,y = self.rnn_cell(rnn_input[:,j,:],state,name,trainable,last_prob)
                out_temp1.append(tf.layers.dense(y, 2,tf.nn.softmax,trainable = trainable, 
                    kernel_initializer = tf.random_normal_initializer(0., .01),
                    bias_initializer = tf.constant_initializer(0.01)))
                last_prob = out_temp1[j]
                # actions = np.random.choice(range(out_temp1[j]),p=out_temp1[j].ravel())
            out = tf.stack([out_temp1[k] for k in range(self.a_dim)], axis=1)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return out, params

    def choose_action(self, s):

        _s = np.array(s).reshape([-1,int(self.s_dim/self.a_dim)])
        action = []
        prob = self.sess.run(self.pi,feed_dict={self.tfs: _s})
        prob_temp = np.array(prob).reshape([-1,2])
        # print(prob)
        for i in range(self.a_dim):
            action_temp = np.random.choice(range(prob_temp[i].shape[0]),
                                  p=prob_temp[i].ravel())  # select action w.r.t the actions prob
            action.append(action_temp)
            
        return action

    def get_state(self, s):
        s = s[np.newaxis, :]
        h = self.sess.run(self.l2, {self.tfs: s})[0]
        return h

    def get_v(self, s):
        _s = np.array(s)
        if _s.ndim < 2:
            s = _s[np.newaxis, :]
        # print(self.sess.run(self.v, {self.tfs: s}))
        return self.sess.run(self.v, {self.tfs: s})

    def experience_store(self, s, a, r):
        self.buffer_a.append(a)
        self.buffer_s.append(s)
        self.buffer_r.append(r)

    def empty_buffer(self):
        self.buffer_s, self.buffer_r, self.buffer_a = [], [], []

    ##每一步的reward进行一个discount，让越远的reward影响变小
    def trajction_process(self, s_):
        _s = np.array(s_).reshape([-1,int(self.s_dim/self.a_dim)]).tolist()
        # for i in range(len(a_)):
        #     _s[i].append(a_[i])

        v_s_ = self.get_v(_s)
        buffer_r = np.array(self.buffer_r).reshape([-1,self.a_dim])
        buffer = [[],[],[],[],[],[],[],[],[],[],[],[]]
        for r in buffer_r[::-1]:
            for i in range(self.a_dim):
                v_s_[i] = r[i] + GAMMA * v_s_[i]
                buffer[i].append(copy.deepcopy(v_s_[i]))
                
        for i in range(self.a_dim):
            buffer[i].reverse()
        out = np.stack([buffer[k] for k in range(self.a_dim)], axis=1)
        # print(self.buffer_r)

        self.buffer_r = np.array(out).reshape([-1])

    def summarize(self, reward, i, tag):
        summary = tf.Summary()
        summary.value.add(tag=tag, simple_value=reward)
        self.writer.add_summary(summary, i)
        self.writer.flush()

    def save_params(self,name,ep):
        save_path = self.saver.save(self.sess,'my_net/rnn_discrete/{}_ep{}.ckpt'.format(name,ep))
        print("Save to path:",save_path)
    def restore_params(self,name,ep):
        self.saver.restore(self.sess,'my_net/rnn_discrete/{}_ep{}.ckpt'.format(name,ep))
        print("Restore params from")


class vae(keras.Model):
    
    def __init__(self,latent_dim):
        super(vae,self).__init__()
        
        self.latent_dim = latent_dim
        
        self.encoder = keras.Sequential([
                keras.layers.InputLayer(input_shape = (28,28,1)),
                keras.layers.Conv2D(filters = 32,kernel_size = 3,strides = (2,2),activation = 'relu'),
                keras.layers.Conv2D(filters = 32,kernel_size = 3,strides = (2,2),activation = 'relu'),
                keras.layers.Flatten(),
                keras.layers.Dense(256,activation = 'relu'),
                keras.layers.Dense(self.latent_dim + self.latent_dim)
                ])
        self.decoder = keras.Sequential([
                keras.layers.InputLayer(input_shape = (latent_dim,)),
                keras.layers.Dense(units = 7 * 7 * 32,activation = 'relu'),
                keras.layers.Reshape(target_shape = (7,7,32)),
                keras.layers.Conv2DTranspose(
                        filters = 64,
                        kernel_size = 3,
                        strides = (2,2),
                        padding = "SAME",
                        activation = 'relu'),
                keras.layers.Conv2DTranspose(
                        filters = 32,
                        kernel_size = 3,
                        strides = (2,2),
                        padding = "SAME",
                        activation = 'relu'),
                keras.layers.Conv2DTranspose(
                        filters = 1,
                        kernel_size = 3,
                        strides = (1,1),
                        padding = "SAME"),
                keras.layers.Conv2DTranspose(
                        filters = 1,
                        kernel_size = 3,
                        strides = (1,1),
                        padding = "SAME",
                        activation = 'sigmoid'),
                ])
                

    def encode(self,x):
        mean_logvar = self.encoder(x)
        N = mean_logvar.shape[0] 
        mean = tf.slice(mean_logvar, [0, 0], [N, self.latent_dim])
        logvar = tf.slice(mean_logvar, [0, self.latent_dim], [N, self.latent_dim])
        return mean,logvar
        

    def decode(self,z,apply_sigmoid = False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits
      
    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

class train:

    @staticmethod
    def compute_loss(model,x):
        mean, logvar = model.encode(x)
        z = model.reparameterize(mean,logvar)
        x_logits = model.decode(z)
        
        # loss
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logits, labels=x)
        marginal_likelihood = - tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        marginal_likelihood = tf.reduce_mean(marginal_likelihood)

        KL_divergence = tf.reduce_sum(mean ** 2 + tf.exp(logvar) - logvar - 1, axis=1)
        KL_divergence = tf.reduce_mean(KL_divergence)

        ELBO = marginal_likelihood - KL_divergence
        loss = -ELBO
        return loss
    
    @staticmethod
    def compute_gradient(model,x):
        with tf.GradientTape() as tape:
            loss = train.compute_loss(model,x)
        gradient = tape.gradient(loss,model.trainable_variables)
        return gradient,loss
    
    @staticmethod
    def update(optimizer,gradients,variables):
        optimizer.apply_gradients(zip(gradients,variables))

###################Rebuild critic network##########################
#PPO5 is used as new approach to handle big grid (bigger than 3x3) by rebuilding
#critic network.
class PPO5(object):
    #This is designed according xuezhang
    #PPO2在PPO上自定义了actor的RNN网络结构，使能够让前一step的输出作为后一step的输入
    #In this class, the only verification is to rewrite the RNN neural network. 
    #The input states of RNN are different too. (For each step of RNN, input states are states of signal and the signal's chosen action.)
    
    def __init__(self, s_dim=32, a_dim=1, name="meme"):
        runner1 = '/cpu:0'
        runner2 = '/gpu:0'
        with tf.device('/cpu:0'):
            self.sess = tf.Session()
            # self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
            self.a_dim = a_dim
            self.s_dim = s_dim
            self.name = name
            self.buffer_a = []
            self.buffer_s = []
            self.buffer_r = []
            self.global_steps = 0
            self.update_steps_a = 0
            self.update_steps_c = 0
            self.global_counter = 0
            self.pre_counter = 0

            self.hidden_net = 64
            self.output_net = 64

            self.tfa = tf.placeholder(tf.int32, [None], 'action')
            self.tfadv = tf.placeholder(tf.float32, [None, self.a_dim], 'advantage')
            self.tfs = tf.placeholder(tf.float32, [None, s_dim/a_dim], 'actor_state')
            # critic
            with tf.variable_scope(self.name + '_critic'): 
                input_state = tf.reshape(self.tfs,[-1,self.a_dim,int(self.s_dim/self.a_dim)])

                # l3 = []
                # for i in range(self.a_dim):
                #     with tf.variable_scope(self.name + '_critic_' + str(i)):
                #         l1 = tf.layers.dense(input_state[:,i,:], 100, tf.nn.relu,bias_initializer = tf.constant_initializer(0.01), 
                #                     kernel_initializer = tf.random_normal_initializer(0., .01))
                #         l2_temp = tf.layers.dense(l1, 1)
                #         l2 = tf.reshape(l2_temp,[-1])
                #         l3.append(l2)
                # self.v = tf.stack([l3[k] for k in range(self.a_dim)], axis=1)

                l4 = []
                for i in range(self.a_dim):
                    with tf.variable_scope(self.name + '_critic_' + str(i)):
                        l1 = tf.layers.dense(input_state[:,i,:], 100, tf.nn.relu,bias_initializer = tf.constant_initializer(0.01), 
                                    kernel_initializer = tf.random_normal_initializer(0., .01))
                        # l2 = tf.layers.dense(l1, 50, tf.nn.relu,bias_initializer = tf.constant_initializer(0.01), 
                        #             kernel_initializer = tf.random_normal_initializer(0., .01))
                        l3_temp = tf.layers.dense(l1, 1)
                        l3 = tf.reshape(l3_temp,[-1])
                        l4.append(l3)
                self.v = tf.stack([l4[k] for k in range(self.a_dim)], axis=1)


                # l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu,bias_initializer = tf.constant_initializer(0.01), 
                # kernel_initializer = tf.random_normal_initializer(0., .01))
                # # l2 = tf.layers.dense(l1, 64, tf.nn.relu,bias_initializer = tf.constant_initializer(0.01), 
                # # kernel_initializer = tf.random_normal_initializer(0., .01))
                # self.v = tf.layers.dense(l1, 1)

                self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
                self.advantage = self.tfdc_r - tf.reshape(self.v,[-1,1])
                # self.closs = tf.reduce_mean(tf.square(self.advantage))
                self.closs = tf.reduce_mean(tf.reshape(tf.square(self.advantage),[-1,self.a_dim]),axis=1)

                # self.ctrain_op = tf.train.AdamOptimizer(C_LR).minimize(self.closs)
                self.ctrain_op = [tf.train.AdamOptimizer(C_LR).minimize(self.closs[i]) for i in range(self.a_dim)]

            # actor
            self.pi, pi_params = self._build_anet_FCN(self.name + '_pi', trainable=True)
            self.oldpi, oldpi_params = self._build_anet_FCN(self.name + '_oldpi', trainable=False)

            self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

            ##调整概率分布的维度，方便获取概率
            index = []
            self.pi_resize = tf.reshape(self.pi,[-1,2])
            self.oldpi_resize = tf.reshape(self.oldpi,[-1,2])

            self.a_indices = tf.stack([tf.range(tf.shape(tf.reshape(self.tfa,[-1]))[0], dtype=tf.int32), tf.reshape(self.tfa,[-1])], axis=1)
            pi_prob = tf.gather_nd(params=self.pi_resize, indices=self.a_indices)  
            oldpi_prob = tf.gather_nd(params=self.oldpi_resize, indices=self.a_indices) 
            self.ratio_temp1 = tf.reshape(pi_prob / (oldpi_prob + 1e-8),[-1,self.a_dim])

            self.surr = self.ratio_temp1 * self.tfadv  # surrogate loss

            self.aloss = -tf.reduce_mean(tf.minimum(  # clipped surrogate objective
                self.surr,
                tf.clip_by_value(self.ratio_temp1, 1. - 0.2, 1. + 0.2) * self.tfadv))

            self.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(self.aloss)

        ##以下为分开计算actor loss的部分

            self.aloss_seperated = -tf.reduce_mean(tf.reshape(tf.minimum(  # clipped surrogate objective
                self.surr,
                tf.clip_by_value(self.ratio_temp1, 1. - 0.2, 1. + 0.2) * self.tfadv),[-1,self.a_dim]),axis = 0)
            # self.atrain_op_seperated = [tf.train.AdamOptimizer(A_LR).minimize(self.aloss_seperated[k]) for k in range(self.a_dim)]
            self.atrain_op_seperated = tf.train.AdamOptimizer(A_LR).minimize(self.aloss_seperated[4])

            self.sess.run(tf.global_variables_initializer())
            self.writer = tf.summary.FileWriter("baseline/rnn_discrete/" + self.name + "_log/", self.sess.graph)
            self.saver = tf.train.Saver(max_to_keep=20)
            # tf.get_default_graph().finalize()

    def update_critic(self):
        s = np.vstack(self.buffer_s)
        r = np.vstack(self.buffer_r)
        critic_loss = self.sess.run(self.closs, {self.tfs: s, self.tfdc_r: r})
        self.summarize(critic_loss,self.pre_counter,'pre_Critic_loss')
        [self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r}) for _ in range(C_UPDATE_STEPS)]
        self.pre_counter += 1

    def update(self):
        print("Update")
        s = np.vstack(self.buffer_s).reshape([-1,int(self.s_dim/self.a_dim)])
        r = np.vstack(self.buffer_r)
        a = np.array(self.buffer_a).reshape([-1])

        # print(s.shape)
        # print(a.shape)
        # print(r)
        self.sess.run(self.update_oldpi_op)
        adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r})
        adv_r = np.array(adv).reshape([-1,self.a_dim])

        # print(r)
        v= self.sess.run(self.v,{self.tfs:s})
        # print(v)
        # print(adv_r)
        # tem = self.sess.run(self.pi,{self.tfs:s})
        # print(np.array(tem).shape)
        # tem2 = self.sess.run(self.pi_resize,{self.tfs:s})
        # print(np.array(tem2).shape)
        # old_pi = self.sess.run(self.oldpi_resize,{self.tfs:s})
        # print(np.array(old_pi).shape)
        # a_in = self.sess.run(self.a_indices,{self.tfa:a})
        # print(np.array(a_in).shape,a_in[6])


        actor_loss = self.sess.run(self.aloss, {self.tfs: s, self.tfa: a, self.tfadv: adv_r})
        self.summarize(actor_loss,self.global_counter,'Actor_loss')

        # [self.sess.run(self.atrain_op, {self.tfs: s, self.tfa: a, self.tfadv: adv_r}) for _ in range(A_UPDATE_STEPS)]
        [self.sess.run(self.atrain_op_seperated, {self.tfs: s, self.tfa: a, self.tfadv: adv_r}) for _ in range(A_UPDATE_STEPS)]

        critic_loss = np.mean(self.sess.run(self.closs, {self.tfs: s, self.tfdc_r: r}))
        self.summarize(critic_loss,self.global_counter,'Critic_loss')
        [self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r}) for _ in range(C_UPDATE_STEPS)]
        
        self.global_counter += 1

    def rnn_cell(self,rnn_input, state,name,trainable,last_prob):
        #Yt = relu(St*Vw+Vb)
        #St = tanch(Xt*Uw + Ub + St-1*Ww+Wb)
        #Xt = [none,198 + 2] St-1 = [none,64] Yt = [none,64]
        #Uw = [198 + 2,64] Ub = [64]
        #Ww = [64,64]   Wb = [64]
        #Vw = [64,64]      Vb = [64]
        with tf.variable_scope('rnn_input_cell_' + name, reuse=True):
            Uw = tf.get_variable('Uw', [int(self.s_dim/self.a_dim) + 2, self.hidden_net],trainable=trainable)
            Ub = tf.get_variable('Ub', [1,self.hidden_net], initializer=tf.constant_initializer(0.0),trainable=trainable)
        with tf.variable_scope('rnn_cycle_cell_' + name,  reuse=True):
            Ww = tf.get_variable('Ww', [self.hidden_net, self.hidden_net],trainable=trainable)
            Wb = tf.get_variable('Wb', [1,self.hidden_net], initializer=tf.constant_initializer(0.0),trainable=trainable)
        with tf.variable_scope('rnn_output_cell_' + name, reuse=True):
            Vw = tf.get_variable('Vw', [self.hidden_net, self.output_net],trainable=trainable)
            Vb = tf.get_variable('Vb', [1,self.output_net], initializer=tf.constant_initializer(0.0),trainable=trainable)
        if last_prob == None:
            St = tf.nn.tanh(tf.matmul(tf.cast(tf.reshape(tf.pad(rnn_input,[[0,0],[0,2]]),[-1,int(self.s_dim/self.a_dim) + 2]),tf.float32),tf.cast(Uw,tf.float32)) + tf.cast(Ub,tf.float32) + tf.matmul(tf.cast(state,tf.float32),tf.cast(Ww,tf.float32)) + tf.cast(Wb,tf.float32))
        else:
            St = tf.nn.tanh(tf.matmul(tf.cast(tf.concat([tf.reshape(rnn_input,[-1,int(self.s_dim/self.a_dim)]),last_prob],axis = 1),tf.float32),tf.cast(Uw,tf.float32)) + tf.cast(Ub,tf.float32) + tf.matmul(tf.cast(state,tf.float32),tf.cast(Ww,tf.float32)) + tf.cast(Wb,tf.float32))

        # St = tf.nn.tanh(tf.matmul(tf.cast(tf.reshape(tf.concat([rnn_input,last_prob],1),[-1,int(self.s_dim/self.a_dim)]),tf.float32),tf.cast(Uw,tf.float32)) + tf.cast(Ub,tf.float32) + tf.matmul(tf.cast(state,tf.float32),tf.cast(Ww,tf.float32)) + tf.cast(Wb,tf.float32))
        Yt = tf.nn.relu(tf.matmul(tf.cast(St,tf.float32),tf.cast(Vw,tf.float32)) + tf.cast(Vb,tf.float32))
        # return
        return St,Yt

    def _build_anet_FCN(self,name,trainable):
        with tf.variable_scope(name):
            input = tf.reshape(self.tfs,[-1,int(self.s_dim/self.a_dim)])
            l1 = tf.layers.dense(input, 64, tf.nn.relu,bias_initializer = tf.constant_initializer(0.01), 
                kernel_initializer = tf.random_normal_initializer(0., .01),trainable=trainable)
            # l2 = tf.layers.dense(l1, 64, tf.nn.relu,bias_initializer = tf.constant_initializer(0.01), 
            #     kernel_initializer = tf.random_normal_initializer(0., .01),trainable=trainable)
            out = tf.layers.dense(l1,2,tf.nn.softmax,trainable=trainable)

        #     input_1 = tf.reshape(self.tfs,[-1,self.a_dim,int(self.s_dim/self.a_dim)])
        #     l4 = []
        #     for i in range(self.a_dim):
        #         with tf.variable_scope(name + '_actor_' + str(i)):
        #             l1 = tf.layers.dense(input_1[:,i,:], 64, tf.nn.relu,bias_initializer = tf.constant_initializer(0.01), 
        #                         kernel_initializer = tf.random_normal_initializer(0., .01),trainable=trainable)
        #             # l2 = tf.layers.dense(l1, 64, tf.nn.relu,bias_initializer = tf.constant_initializer(0.01), 
        #             #             kernel_initializer = tf.random_normal_initializer(0., .01),trainable=trainable)
        #             l3_temp = tf.layers.dense(l1, 2,tf.nn.softmax,trainable=trainable)
        #             l3 = tf.reshape(l3_temp,[-1,2])
        #             l4.append(l3)
        # out = tf.stack([l4[k] for k in range(self.a_dim)], axis=1)

        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return out,params

    def _build_anet(self, name, trainable):
        with tf.variable_scope(name):

            with tf.variable_scope('rnn_input_cell_' + name):
                Uw = tf.get_variable('Uw', [int(self.s_dim/self.a_dim) + 2, self.hidden_net],trainable=trainable)
                Ub = tf.get_variable('Ub', [1,self.hidden_net], initializer=tf.constant_initializer(0.0),trainable=trainable)
            with tf.variable_scope('rnn_cycle_cell_' + name):
                Ww = tf.get_variable('Ww', [self.hidden_net, self.hidden_net],trainable=trainable)
                Wb = tf.get_variable('Wb', [1,self.hidden_net], initializer=tf.constant_initializer(0.0),trainable=trainable)
            with tf.variable_scope('rnn_output_cell_' + name):
                Vw = tf.get_variable('Vw', [self.hidden_net, self.output_net],trainable=trainable)
                Vb = tf.get_variable('Vb', [1,self.output_net], initializer=tf.constant_initializer(0.0),trainable=trainable)

            # RNN
            out_temp1 = []
            out_temp2 = []
            out = []
            actions = []
            last_prob = None
            rnn_input = tf.reshape(self.tfs,[-1,self.a_dim,int(self.s_dim/self.a_dim)])
            state = np.zeros([1,self.hidden_net])
            # #The first for cycle is to get state include all signals' imformation
            # for j in range(self.a_dim):
            #     state,y = self.rnn_cell(rnn_input[:,j,:],state,name,trainable,last_prob)
            #     out_temp1.append(tf.layers.dense(y, 2,tf.nn.softmax,trainable = trainable, 
            #         kernel_initializer = tf.random_normal_initializer(0., .01),
            #         bias_initializer = tf.constant_initializer(0.01)))
            #     last_prob = out_temp1[j]
            #The second cycle is aim to make actions depend on last cycle's final state.
            last_prob = None
            for j in range(self.a_dim):
                state,y = self.rnn_cell(rnn_input[:,j,:],state,name,trainable,last_prob)
                out_temp2.append(tf.layers.dense(y, 2,tf.nn.softmax,trainable = trainable, 
                    kernel_initializer = tf.random_normal_initializer(0., .01),
                    bias_initializer = tf.constant_initializer(0.01)))
                last_prob = out_temp2[j]
                # actions = np.random.choice(range(out_temp1[j]),p=out_temp1[j].ravel())
            out = tf.stack([out_temp2[k] for k in range(self.a_dim)], axis=1)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return out, params


    def choose_action(self, s):

        _s = np.array(s).reshape([-1,int(self.s_dim/self.a_dim)])
        action = []
        prob = self.sess.run(self.pi,feed_dict={self.tfs: _s})
        prob_temp = np.array(prob).reshape([-1,2])
        print(prob)

        # for i in range(self.a_dim):
        #     action_temp = np.random.choice(range(prob_temp[i].shape[0]),
        #                         p=prob_temp[i].ravel())  # select action w.r.t the actions prob
        #     action.append(action_temp)

        # the next part we initial a seed of random number limited in (0,1]
        # when seed first less than 0.9(threshold) that choose action according to given probability.
        # but if seed less bigger than 0.9, then we choose action equally.
        for i in range(self.a_dim):
            seed = np.random.rand()
            if seed < 0.8:
                action_temp = np.random.choice(range(prob_temp[i].shape[0]),
                                    p=prob_temp[i].ravel())  # select action w.r.t the actions prob
                action.append(action_temp)
            else:
                seed = np.random.rand()
                if seed < 0.5:
                    action.append(0)
                else:
                    action.append(1)
        
        return action

    def get_state(self, s):
        s = s[np.newaxis, :]
        h = self.sess.run(self.l2, {self.tfs: s})[0]
        return h

    def get_v(self, s):
        _s = np.array(s)
        if _s.ndim < 2:
            s = _s[np.newaxis, :]
        # print(self.sess.run(self.v, {self.tfs: s}))
        return self.sess.run(self.v, {self.tfs: s})

    def experience_store(self, s, a, r):
        self.buffer_a.append(a)
        self.buffer_s.append(s)
        self.buffer_r.append(r)
    def experience_store_r_s(self, s, r):
        # self.buffer_a.append(a)
        self.buffer_s.append(s)
        self.buffer_r.append(r)
    def experience_store_a(self, a):
        self.buffer_a.append(a)


    def empty_buffer(self):
        self.buffer_s, self.buffer_r, self.buffer_a = [], [], []


    def trajction_process_proximate(self):
        #This function aims to calculate proximate F(s) of each state s.
        v_s_ = np.mean(np.array(self.buffer_r).reshape([-1,self.a_dim]),axis = 0)
        #we assume that all of the following Rs are the mean value of simulated steps (200)
        #so, the following Rs are geometric progression.
        #Sn = a1 * (1 - GAMMA^^n) / (1 - GAMMA) proximate equals to a1/(1-GAMMA)

        v_s_ = v_s_ / (1 - GAMMA)
        buffer_r = np.array(self.buffer_r).reshape([-1,self.a_dim])
        buffer = [[],[],[],[],[],[],[],[],[],[],[],[]]
        for r in buffer_r[::-1]:
            for i in range(self.a_dim):
                v_s_[i] = r[i] + GAMMA * v_s_[i]
                buffer[i].append(copy.deepcopy(v_s_[i]))
                
        for i in range(self.a_dim):
            buffer[i].reverse()
        out = np.stack([buffer[k] for k in range(self.a_dim)], axis=1)
        # print(self.buffer_r)
        self.buffer_r = np.array(out).reshape([-1])


    ##每一步的reward进行一个discount，让越远的reward影响变小
    def trajction_process(self, s_):
        _s = np.array(s_).reshape([-1,int(self.s_dim/self.a_dim)]).tolist()
        # for i in range(len(a_)):
        #     _s[i].append(a_[i])

        v_s_ = np.array(self.get_v(_s)).reshape([-1])
        buffer_r = np.array(self.buffer_r).reshape([-1,self.a_dim])
        buffer = [[],[],[],[],[],[],[],[],[],[],[],[]]
        for r in buffer_r[::-1]:
            for i in range(self.a_dim):
                v_s_[i] = r[i] + GAMMA * v_s_[i]
                buffer[i].append(copy.deepcopy(v_s_[i]))
                
        for i in range(self.a_dim):
            buffer[i].reverse()
        out = np.stack([buffer[k] for k in range(self.a_dim)], axis=1)
        # print(self.buffer_r)

        self.buffer_r = np.array(out).reshape([-1])

    def summarize(self, reward, i, tag):
        summary = tf.Summary()
        summary.value.add(tag=tag, simple_value=reward)
        self.writer.add_summary(summary, i)
        self.writer.flush()

    def save_params(self,name,ep):
        save_path = self.saver.save(self.sess,'my_net/PPO3/{}_ep{}.ckpt'.format(name,ep))
        print("Save to path:",save_path)
    def restore_params(self,name,ep):
        self.saver.restore(self.sess,'my_net/PPO3/{}_ep{}.ckpt'.format(name,ep))
        print("Restore params from")