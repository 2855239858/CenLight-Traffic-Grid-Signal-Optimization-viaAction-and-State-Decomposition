"""Multi-agent traffic light example (single shared policy)."""

from ray.rllib.agents.ppo.ppo_policy import PPOTFPolicy
from flow.envs.multiagent import MyMultiTrafficLightGridPOEnv
from flow.networks import TrafficLightGridNetwork
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.params import InFlows, SumoCarFollowingParams, VehicleParams
from flow.controllers import SimCarFollowingController, GridRouter
from ray.tune.registry import register_env
from flow.utils.registry import make_create_env
import numpy as np

# Experiment parameters
N_ROLLOUTS = 20  # number of rollouts per training iteration
N_CPUS = 3  # number of parallel workers

# Environment parameters
HORIZON = 200  # time horizon of a single rollout
V_ENTER = 30  # enter speed for departing vehicles
INNER_LENGTH = 300  # length of inner edges in the traffic light grid network
LONG_LENGTH = 100  # length of final edge in route
SHORT_LENGTH = 300  # length of edges that vehicles start on
# number of vehicles originating in the left, right, top, and bottom edges
N_LEFT, N_RIGHT, N_TOP, N_BOTTOM = 0, 0, 0, 0

EDGE_INFLOW = 300  # inflow rate of vehicles at every edge
N_ROWS = 2  # number of row of bidirectional lanes
N_COLUMNS = 2  # number of columns of bidirectional lanes


# we place a sufficient number of vehicles to ensure they confirm with the
# total number specified above. We also use a "right_of_way" speed mode to
# support traffic light compliance
vehicles = VehicleParams()
num_vehicles = (N_LEFT + N_RIGHT) * N_COLUMNS + (N_BOTTOM + N_TOP) * N_ROWS
vehicles.add(
    veh_id="human",
    acceleration_controller=(SimCarFollowingController, {}),
    car_following_params=SumoCarFollowingParams(
        min_gap=2.5,
        max_speed=V_ENTER,
        decel=7.5,  # avoid collisions at emergency stops
        speed_mode="right_of_way",
    ),
    routing_controller=(GridRouter, {}),
    num_vehicles=num_vehicles)

# inflows of vehicles are place on all outer edges (listed here)
outer_edges = []
outer_edges += ["left{}_{}".format(N_ROWS, i) for i in range(N_COLUMNS)]
outer_edges += ["right0_{}".format(i) for i in range(N_COLUMNS)]
outer_edges += ["bot{}_0".format(i) for i in range(N_ROWS)]
outer_edges += ["top{}_{}".format(i, N_COLUMNS) for i in range(N_ROWS)]

# equal inflows for each edge (as dictate by the EDGE_INFLOW constant)
inflow = InFlows()
for edge in outer_edges:
    inflow.add(
        veh_type="human",
        edge=edge,
#        vehs_per_hour=EDGE_INFLOW,
        # probability=0.10,
        vehs_per_hour = 900,
        departLane="free",
        departSpeed=V_ENTER)

myNetParams = NetParams(
        inflows=inflow,
        additional_params={
            "speed_limit": V_ENTER + 5,  # inherited from grid0 benchmark
            "grid_array": {
                "short_length": SHORT_LENGTH,
                "inner_length": INNER_LENGTH,
                "long_length": LONG_LENGTH,
                "row_num": N_ROWS,
                "col_num": N_COLUMNS,
                "cars_left": N_LEFT,
                "cars_right": N_RIGHT,
                "cars_top": N_TOP,
                "cars_bot": N_BOTTOM,
            },
            "horizontal_lanes": 1,
            "vertical_lanes": 1,
        },
    )

flow_params = dict(
    # name of the experiment
    exp_tag="grid_0_{}x{}_i{}_multiagent".format(N_ROWS, N_COLUMNS, EDGE_INFLOW),

    # name of the flow environment the experiment is running on
    env_name=MyMultiTrafficLightGridPOEnv,

    # name of the network class the experiment is running on
    network=TrafficLightGridNetwork,

    # simulator that is used by the experiment
    simulator='traci',

    # sumo-related parameters (see flow.core.params.SumoParams)
    sim=SumoParams(
        restart_instance=True,
        sim_step=1,
        render=False,
    ),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=HORIZON,
        warmup_steps=3,
        sims_per_step=3,
        additional_params={
            "target_velocity": 50,
            "switch_time": 3,
            "num_observed": 15,
            "discrete": False,
            "tl_type": "controlled",
            "num_local_edges": 4,
            "num_local_lights": 4,
        },
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # network's documentation or ADDITIONAL_NET_PARAMS component)
    net=myNetParams,

    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.params.VehicleParams)
    veh=vehicles,

    # parameters specifying the positioning of vehicles upon initialization
    # or reset (see flow.core.params.InitialConfig)
    initial=InitialConfig(
        spacing='custom',
        shuffle=True,
    ),
)

#############################以下为训练部分#################################
def cover_actions(c_a, s_a,num):
    # for i in range(len(c_a)):
    #     if c_a[i] == 1:
    #         s_a[i] = abs(s_a[i] - 1)
    for i in range(num):
        if i == c_a:
            s_a[i] = 1
    return s_a


def data_collection(env, vels, queues):
    vehicles = env.k.vehicle
    veh_speeds = vehicles.get_speed(vehicles.get_ids())
    v_temp = 0
    if np.isnan(np.mean(veh_speeds)):
        v_temp = 0
    else:
        v_temp = np.mean(veh_speeds)
    vels.append(v_temp)
    queued_vels = len([v for v in veh_speeds if v < 1])
    queues.append(queued_vels)
    return vels, queues

def normalize_formation(state,Agent_NUM):
    _state = [[] for i in range(Agent_NUM)]
    for i in range(Agent_NUM):
        _state[i] = state["center"+str(i)]
    return _state


def record_line(log_path, line):
    with open(log_path, 'a') as fp:
        fp.writelines(line)
        fp.writelines("\n")
    return True


if __name__ == "__main__":
    myTrafficNet = TrafficLightGridNetwork(
        name = 'grid',
        vehicles =  vehicles,
        net_params = myNetParams,
    )
    env = MyMultiTrafficLightGridPOEnv(
        env_params=flow_params['env'], sim_params=flow_params['sim'], network=myTrafficNet)

    #   print(env.scenario.get_edge_list())
    # Perpare agent.
    from flow.core.ppo_rnn_all import *
############################################################################
############################################################################
    Agent_NUM = N_ROWS * N_COLUMNS
    Reward_num = 1    #0代表多个rewards，1代表1个
    NAME = '2x2_900_RNN_ALL_SOFT_test'
    Epoch = 4000
    steps = 200
    delay = 0
    delay_step = 2
    pre_trainCritic = 0
    pre_trainActor = 0
    rnn_train_epi = 25
    pre_train_critic = 1000
    pre_train_actor = 2000
    num_state = 198
    rnn_agent = PPO(s_dim=num_state*Agent_NUM,a_dim=Agent_NUM,name=NAME)
############################################################################
############################################################################
    # rnn_agent.restore_params(NAME,10)
    global_counter = 0
    each_line_path = "collected_data/rnn_ppo_all/{}_plot_log.txt".format(NAME)
    test_epoch_path = "collected_data/rnn_ppo_all/{}_epoch_log.txt".format(NAME)
    
    ##预训练critic
    if pre_trainCritic == 1:
        for pre in range(pre_train_critic):
            print("当前pre_critic训练次数:")
            print(pre)

            global_counter += 1

            state = env.reset()
            state = normalize_formation(state,Agent_NUM)
            _state = np.array(state).reshape([-1,num_state * Agent_NUM])

            pre_ep_r = 0
            for pre_step in range(steps):
                actions = [0,0,0,0]
                if pre_step % 30 == 0:
                    actions = [1,1,1,1]

                next_state, rewards, done, _ = env.step(actions)
                rnn_agent.experience_store(_state, actions, rewards)

                # rnn_agent.update_critic(_state,rewards)
                
                # rnn_agent.experience_store(_state, actions, rewards)
                pre_ep_r += rewards

                state = next_state
                state = normalize_formation(state,Agent_NUM)
                _state = np.array(state).reshape([-1,num_state * Agent_NUM])  

                if (pre_step + 1) % BATCH == 0 or  pre_step == EP_LEN - 1:
                    print(3)
                    print(pre_step)
                    rnn_agent.trajction_process(_state)
                    rnn_agent.update_critic()
                    rnn_agent.empty_buffer()
                # if (pre_step + 1) % BATCH == 0 or pre_step == EP_LEN - 1:
                #     rnn_agent.trajction_process(_state)
                #     rnn_agent.update()
                #     rnn_agent.empty_buffer()

            print('{}_pre_steps rewards:'.format(NAME))
            print(pre_ep_r)
            rnn_agent.summarize(pre_ep_r, global_counter, 'reward')

    ##预训练actor
    if pre_trainActor == 1:
        for i in range(pre_train_actor):
                    print("当前pre_actor训练次数:")
                    print(i)

                    global_counter += 1
                    state = env.reset()
                    state = normalize_formation(state,Agent_NUM)
                    _state = np.array(state).reshape([-1,num_state * Agent_NUM])
                    
                    ep_r = 0.0

                    action_store = []
                    reward_store = 0
                    state_store = []
                    count_store = 0

                    for step in range(steps):
                        step_r = 0.0

                        _actions = rnn_agent.choose_action(_state)
                        # print(_actions)
                        # actions = np.zeros((Agent_NUM,), dtype=int)
                        # rl_actions = cover_actions(_actions, actions,Agent_NUM)
                        # print(rl_actions)
                        if delay == 1:
                            if step % delay_step != 0:
                                # print(1)
                                _actions = [0,0,0,0]
                            else:
                                # print(2)
                                if count_store != 0:
                                    rnn_agent.experience_store(state_store, action_store, reward_store)
                                state_store = _state
                                action_store = _actions
                                count_store += 1
                            # print(_actions)

                        next_state, rewards, done, _ = env.step(_actions)

                        reward_store = rewards

                        if Reward_num == 0:
                            for k in range(Agent_NUM):
                                step_r += rewards[k]/Agent_NUM
                                ep_r += rewards[k]/Agent_NUM 
                                rnn_agent.experience_store(_state, _actions, step_r)
                        else:
                                ep_r += rewards
                                if delay != 1:
                                    rnn_agent.experience_store(_state, _actions, rewards)

                        state = next_state
                        state = normalize_formation(state,Agent_NUM)
                        _state = np.array(state).reshape([-1,num_state * Agent_NUM])

                        # if (step + 1) % BATCH == 0 or step == EP_LEN - 1:

                        if delay == 1:
                            if (count_store + 1) % BATCH == 0 or  step == EP_LEN - 1:
                                print(3)
                                count_store = 0
                                rnn_agent.trajction_process(_state)
                                rnn_agent.update_actor()
                                rnn_agent.empty_buffer()
                        else:
                            if (step + 1) % BATCH == 0 or  step == EP_LEN - 1:
                                print(3)
                                count_store = 0
                                rnn_agent.trajction_process(_state)
                                rnn_agent.update_actor()
                                rnn_agent.empty_buffer()

                        _done = True
                        for i in range(Agent_NUM):
                            _done *= done["center"+str(i)]
                        # print('dome?')
                        # print(_done)
                        if _done:
                            break
                    print('pre_train actor steps rewards:')
                    print(ep_r)
                    rnn_agent.summarize(ep_r, global_counter, 'reward')



    ##开始训练
    for ep in range(Epoch):

        #RNN_PPO训练步骤
        for i in range(rnn_train_epi):
            print("当前训练次数:")
            print(i)

            global_counter += 1
            state = env.reset([0,0,0,0])
            state = normalize_formation(state,Agent_NUM)
            _state = np.array(state).reshape([-1,num_state * Agent_NUM])
            
            ep_r = 0.0

            action_store = []
            reward_store = 0
            state_store = []
            count_store = 0

            for step in range(steps):
                step_r = 0.0

                _actions = rnn_agent.choose_action(_state)
                # print(_actions)
                # actions = np.zeros((Agent_NUM,), dtype=int)
                # rl_actions = cover_actions(_actions, actions,Agent_NUM)
                # print(rl_actions)
                if delay == 1:
                    if step % delay_step != 0:
                        # print(1)
                        _actions = [0,0,0,0]
                    else:
                        # print(2)
                        if count_store != 0:
                            rnn_agent.experience_store(state_store, action_store, reward_store)
                        state_store = _state
                        action_store = _actions
                        count_store += 1
                    # print(_actions)

                next_state, rewards, done, _ = env.step(_actions)

                reward_store = rewards

                if Reward_num == 0:
                    for k in range(Agent_NUM):
                        step_r += rewards[k]/Agent_NUM
                        ep_r += rewards[k]/Agent_NUM 
                        rnn_agent.experience_store(_state, _actions, step_r)
                else:
                        ep_r += rewards
                        if delay != 1:
                            rnn_agent.experience_store(_state, _actions, rewards)

                state = next_state
                state = normalize_formation(state,Agent_NUM)
                _state = np.array(state).reshape([-1,num_state * Agent_NUM])

                # if (step + 1) % BATCH == 0 or step == EP_LEN - 1:

                if delay == 1:
                    if (count_store + 1) % BATCH == 0 or  step == EP_LEN - 1:
                        print(3)
                        count_store = 0
                        rnn_agent.trajction_process(_state)
                        rnn_agent.update()
                        rnn_agent.empty_buffer()
                else:
                    if (step + 1) % BATCH == 0 or  step == EP_LEN - 1:
                        print(3)
                        count_store = 0
                        rnn_agent.trajction_process(_state)
                        rnn_agent.update()
                        rnn_agent.empty_buffer()

                _done = True
                for i in range(Agent_NUM):
                    _done *= done["center"+str(i)]
                # print('dome?')
                # print(_done)
                if _done:
                    break
            print('steps rewards:')
            print(ep_r)
            rnn_agent.summarize(ep_r, global_counter, 'reward')
        if ep % 10 == 0:
            rnn_agent.save_params(NAME,ep)
        # test phase
        if ep >= 0:
            print('测试阶段：')
            print(ep)
            record_line(each_line_path, "*** Epoch: {} ***\n".format(ep))
            queue, speed, ret = [], [], []
            for i in range(3):
                ep_r, ep_q, ep_v = [], [], []
                state = env.reset()
                state = normalize_formation(state,Agent_NUM)
                _state = [n for a in state for n in a ]
                _state = np.array(_state).reshape([-1,num_state * Agent_NUM])
                
                for step in range(steps):
                    step_r  = 0

                    data_collection(env, ep_v, ep_q)

                    _actions = rnn_agent.choose_action(_state)
                    # actions = np.zeros((Agent_NUM,), dtype=int)
                    # rl_actions = cover_actions(_actions, actions,Agent_NUM)
                    next_state, rewards, done, _ = env.step(_actions)

                    if Reward_num == 0:
                        for k in range(Agent_NUM):
                            step_r += rewards[k]/Agent_NUM
                        ep_r.append(step_r)
                    else:
                        ep_r.append(rewards)

                    state = next_state
                    state = normalize_formation(state,Agent_NUM)
                    _state = [n for a in state for n in a ]
                    _state = np.array(_state).reshape([-1,num_state * Agent_NUM])

                    _done = True
                    for i in range(Agent_NUM):
                        _done *= done["center"+str(i)]
                    if _done:
                        break

                queue.append(np.array(ep_q).mean())
                speed.append(np.array(ep_v).mean())
                ret.append(np.array(ep_r).mean())

                record_line(each_line_path, "Queue: " + str(ep_q) + "\n")
                record_line(each_line_path, "Speed: " + str(ep_v) + "\n")
                record_line(each_line_path, "Return: " + str(ep_r) + "\n")
            # record...

            print("*** Epoch: {} ***\n".format(ep))
            print("| Queue: {}, std: {} |".format(np.array(queue).mean(), np.array(queue).std()))
            print("| Speed: {}, std: {} |".format(np.array(speed).mean(), np.array(speed).std()))
            print("| Return: {}, std: {} |".format(np.array(ret).mean(), np.array(ret).std()))
            print("*****************\n")
            record_line(test_epoch_path, "*** Epoch: {} ***\n".format(ep))
            record_line(test_epoch_path, "| Queue: {}, std: {} |".format(np.array(queue).mean(), np.array(queue).std()))
            record_line(test_epoch_path, "| Speed: {}, std: {} |".format(np.array(speed).mean(), np.array(speed).std()))
            record_line(test_epoch_path, "| Return: {}, std: {} |".format(np.array(ret).mean(), np.array(ret).std()))
            record_line(test_epoch_path, "*****************\n")
