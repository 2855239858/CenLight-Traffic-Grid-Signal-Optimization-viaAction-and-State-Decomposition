import tensorflow as tf
import numpy as np
import copy

EP_MAX = 1000
EP_LEN = 200
GAMMA = 0.9
A_LR = 0.002
C_LR = 0.002
BATCH = 40
A_UPDATE_STEPS = 5
C_UPDATE_STEPS = 10
METHOD = [
    dict(name='kl_pen', kl_target=0.01, lam=0.5),   # KL penalty
    dict(name='clip', epsilon=0.2),                 # Clipped surrogate objective, find this is better
][1]        # choose the method for optimization


class PPO(object):

    def __init__(self, s_dim=32, a_dim=1, name="meme"):
        runner1 = '/cpu:0'
        runner2 = '/gpu:0'
        with tf.device('/cpu:0'):
            self.sess = tf.Session()
            self.tfs = tf.placeholder(tf.float32, [None, s_dim/a_dim], 'state')
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
            self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')

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
            ratio_temp1 = pi_prob / (oldpi_prob + 1e-8)
            # ratio_temp2 = tf.reshape(ratio_temp1,[-1,self.a_dim])
            # ratio = tf.reduce_mean(ratio_temp2,axis = 1)

            surr = ratio_temp1 * self.tfadv  # surrogate loss

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
                surr,
                tf.clip_by_value(ratio_temp1, 1. - 0.2, 1. + 0.2) * self.tfadv))

            self.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(self.aloss)
            self.sess.run(tf.global_variables_initializer())
            self.writer = tf.summary.FileWriter("baseline/rnn_discrete/" + self.name + "_log/", self.sess.graph)
            self.saver = tf.train.Saver(max_to_keep=20)
            tf.get_default_graph().finalize()

    def update_critic(self):
        s = np.vstack(self.buffer_s)
        r = np.array(self.buffer_r)[:, np.newaxis]
        critic_loss = self.sess.run(self.closs, {self.tfs: s, self.tfdc_r: r})
        self.summarize(critic_loss,self.pre_counter,'Pre_critic_loss')
        self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r})
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


    def update(self):
        s = np.vstack(self.buffer_s)
        # r = np.array(self.buffer_r)[:, np.newaxis]
        # print(r)
        r = np.vstack(self.buffer_r)
        a = np.array(self.buffer_a).reshape([-1,self.a_dim])
        # self.batch_size = len(a)
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
            rnn_cell = tf.nn.rnn_cell.LSTMCell(num_units=64,activation = tf.nn.tanh,trainable = trainable,
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
        _s = np.array(s_).reshape([-1,int(self.s_dim/self.a_dim)])
        v_s_ = self.get_v(_s)
        buffer_r = np.array(self.buffer_r).reshape([-1,self.a_dim])
        buffer = [[],[],[],[],[],[],[],[],[]]
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
#
# if __name__ == "__main__":
#     import gym
#     import matplotlib.pyplot as plt
#     env = gym.make('CartPole-v0').unwrapped
#     print(env.observation_space.shape)
#     print(env.action_space.shape)
#
#     ppo = PPO(s_dim=4, a_dim=2)
#     all_ep_r = []
#
#     for ep in range(EP_MAX):
#         s = env.reset()
#         buffer_s, buffer_a, buffer_r = [], [], []
#         ep_r = 0
#         for t in range(EP_LEN):  # in one episode
#
#             a = ppo.choose_action(s)
#             s_, r, done, _ = env.step(a)
#             if done:
#                 r = -10
#             ppo.experience_store(s,a,r)
#             s = s_
#             ep_r += r
#
#             # update ppo
#             if (t + 1) % BATCH == 0 or t == EP_LEN - 1:
#                 ppo.trajction_process(s)
#                 ppo.update()
#                 ppo.empty_buffer()
#
#             if done:
#                 break
#         if ep == 0:
#             all_ep_r.append(ep_r)
#         else:
#             all_ep_r.append(all_ep_r[-1] * 0.9 + ep_r * 0.1)
#         print(
#             'Ep: %i' % ep,
#             "|Ep_r: %i" % ep_r,
#         )
#
#     plt.plot(np.arange(len(all_ep_r)), all_ep_r)
#     plt.xlabel('Episode')
#     plt.ylabel('Moving averaged episode reward')
#     plt.show()
