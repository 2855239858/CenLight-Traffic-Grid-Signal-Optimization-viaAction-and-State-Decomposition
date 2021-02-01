"""Multi-agent traffic light example (single shared policy)."""

from ray.rllib.agents.ppo.ppo_policy import PPOTFPolicy
from flow.envs.multiagent import MyMultiTrafficLightGridPOEnv
from flow.networks import TrafficLightGridNetwork
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.params import InFlows, SumoCarFollowingParams, VehicleParams
from flow.controllers import SimCarFollowingController, GridRouter
from ray.tune.registry import register_env
from flow.utils.registry import make_create_env

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
N_ROWS = 2 # number of row of bidirectional lanes
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
        vehs_per_hour=600,
        # probability=0.25,
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
        warmup_steps=0,
        sims_per_step=3,
        additional_params={
            "target_velocity": 50,
            "switch_time": 3,
            "num_observed": 10,
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
def cover_actions(c_a, s_a):
    # for i in range(len(c_a)):
    #     if c_a[i] == 1:
    #         s_a[i] = abs(s_a[i] - 1)
    for i in range(9):
        if i != c_a:
            s_a[i] = 0
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
#    from flow.scenarios.grid.grid_scenario import SimpleGridScenario
#    from flow.scenarios.grid.gen import SimpleGridGenerator
#    from flow.core.traffic_lights import TrafficLights

    Actuated = False
    if Actuated:
        print("这个功能还没写")
        # tl_logic = TrafficLights(baseline=False)
        # phases = [{
        #     "duration": "20",
        #     "minDur": "10",
        #     "maxDur": "30",
        #     "state": "GGGrrrGGGrrr"第几次
        # }, {
        #     "duration": "20",
        #     "minDur": "3",agent第几次：
        #     "maxDur": "6",
        #     "state": "yyyrrryyyrrr"
        # }, {
        #     "duration": "20",
        #     "minDur": "10",
        #     "maxDur": "30",
        #     "state": "rrrGGGrrrGGG"
        # }, {
        #     "duration": "20",
        #     "minDur": "3",
        #     "maxDur": "6",
        #     "state": "rrryyyrrryyy"
        # }]
        # for i in range(m * n):
        #     tl_logic.add("center{}".format(i), tls_type="actuated", phases=phases, programID=1)
        # # scenario = SimpleGridScenario(
        # #     name="PO_TrafficLightGridEnv",
        # #     generator_class=SimpleGridGenerator,
        # #     vehicles=vehs,
        # #     net_params=net_params,reset
        # #     initial_config=initial_config,
        # #     traffic_lights=tl_logic)
        # env = TrafficLightGridPOEnv(
        #     env_params=flow_params['env'], sim_params=flow_params['sim'], network=TrafficLightGridNetwork)
        # ret = 0
        # env.reset()
        # for steps in range(400):
        #     state, reward, done, _ = env.step(env.action_space.sample())
        #     ret += sum(reward)
        #     if done:
        #         break
        # print(ret)
    else:
        # scenario = SimpleGridScenario(
        #     name="PO_TrafficLightGridEnv",
        #     generator_class=SimpleGridGenerator,
        #     vehicles=vehs,agent第几次：
        #     net_params=net_params,
        #     initial_config=initial_config,
        # )
        myTrafficNet = TrafficLightGridNetwork(
            name = 'grid',
            vehicles =  vehicles,
            net_params = myNetParams,
        )
        env = MyMultiTrafficLightGridPOEnv(
            env_params=flow_params['env'], sim_params=flow_params['sim'], network=myTrafficNet)

     #   print(env.scenario.get_edge_list())
        # Perpare agent.
        from flow.core.ppo_agent import *
############################################################################
############################################################################
        Agent_NUM = N_ROWS * N_COLUMNS
        Reward_num = 1    #0代表多个rewards，1代表1个
        NAME = '2x2_600_PPO_Hierarchy_SOFT_try5'
        Epoch = 4000
        steps = 200
        sub_train_epi = 25
        delay = 0
        delay_step = 3
        sub_agents = [PPO(s_dim=138, a_dim=2, name=NAME + str(i)) for i in range(Agent_NUM)]
############################################################################
############################################################################
        # centre_agent = PPO(s_dim=Agent_NUM, a_dim=Agent_NUM+1, name="centre")
        # sub_agent = PPO(s_dim=211, a_dim=1, name="subagent")


        # rnn_agent = PPO(s_dim=42*9,a_dim=10,name="PPO_RNN")

        # sub_agents[1].restore_params()
        # train
        # centre_train_epi = 15
        global_counter = 0
        each_line_path = "collected_data/hierarchy/{}_plot_log.txt".format(NAME)
        test_epoch_path = "collected_data/hierarchy/{}_epoch_log.txt".format(NAME)

        for ep in range(Epoch):

            # phase of sub-policy.
            for i in range(sub_train_epi):
                # global_counter += 1
                state = env.reset()
###############################################
                state = normalize_formation(state,Agent_NUM)

                print('sub-agent第几次：')
                print(i)
                ep_r = [0] * Agent_NUM

                for step in range(steps):
                    # print('step')
                    # print(step)
                    # print('hellllppppppppppppppppppp')
                    # print(sub_agents[0])
                    # sub_agents[1].choose_action(state[1])
                    # print(state)
                    actions = [sub_agents[k].choose_action(state[k]) for k in range(Agent_NUM)]
                    # hiddens = [sub_agents[k].get_state(state[k]) for k in range(Agent_NUM)]
                    # actions = [sub_agents[k].choose_action(state) for k in range(Agent_NUM)]
                    # hiddens = [sub_agents[k].get_state(state) for k in range(Agent_NUM)]
                    # # reshape
                    # concat_hiddens = np.array(hiddens).reshape(1, -1)
                    # #
                    # centre_action = centre_agent.choose_action(concat_hiddens[0])
                    # rl_actions = cover_actions(centre_action, actions)
                    # steps
                    next_state, rewards, done, _ = env.step(actions)
                    # print('rewards:')
                    # print(rewards)
                    # sub-agent train
                    for k in range(Agent_NUM):
                        ep_r[k] += rewards[k]
                    for agent in range(Agent_NUM):
                        sub_agents[agent].experience_store(state[agent], actions[agent], rewards[agent])

                    state = next_state
###############################################
                    state = normalize_formation(state,Agent_NUM)

                    # print('state"')
                    # print(state)
                    if (step + 1) % BATCH == 0 or  step == EP_LEN - 1:
                        print(3)
                        count_store = 0
                        for k in range(Agent_NUM):
                            sub_agents[k].trajction_process(state[k])
                            sub_agents[k].update()
                            sub_agents[k].empty_buffer()

                    # if (step + 1) % BATCH == 0 or step == EP_LEN - 1:
                    #     for k in range(Agent_NUM):
                    #         sub_agents[k].trajction_process(state[k])
                    #         sub_agents[k].update()
                    #         sub_agents[k].empty_buffer()
                    _done = True
                    for w in range(Agent_NUM):
                        _done *= done["center"+str(w)]
                    # print('dome?')
                    # print(done)
                    # print(_done)
                    if _done:
                        break
                # sub_agents[1].save_params()
                print('{} subagent steps mean rewards:'.format(NAME))
                print(sum(ep_r))
                [sub_agents[k].summarize(ep_r[k], i + (ep * sub_train_epi), 'reward') for k in range(Agent_NUM)]
                sub_agents[0].summarize(sum(ep_r),i + (ep * sub_train_epi), 'total reward')
                # centre_agent.summarize(sum(ep_r)/Agent_NUM, i + (ep * sub_train_epi), 'mean reward')

            # phase of centre policy.
#             for j in range(centre_train_epi):
#                 print('centre-agent第几次：')
#                 print(j)
#                 global_counter += 1
#                 state = env.reset()
# ##############################################
#                 state = normalize_formation(state,Agent_NUM)

#                 ep_r = 0
#                 for step in range(steps):
#                     step_r = 0.0
#                     actions = [sub_agents[k].choose_action(state[k]) for k in range(Agent_NUM)]
#                     actions = np.array(actions)
#                     # print(actions)
#                     centre_action = centre_agent.choose_action(actions)
#                     # print('centre-action')
#                     # print(centre_action)
#                     rl_actions = cover_actions(centre_action, actions)
#                     # steps
#                     next_state, rewards, done, _ = env.step(rl_actions)
#                     # centre-agent train
#                     for k in range(Agent_NUM):
#                         step_r += rewards[k]/Agent_NUM
#                         ep_r += rewards[k]/Agent_NUM

#                     centre_agent.experience_store(actions, centre_action, step_r)
#                     #
#                     state = next_state
# ################################################
#                     state = normalize_formation(state,Agent_NUM)

#                     if (step + 1) % BATCH == 0 or step == EP_LEN - 1:
#                         centre_agent.trajction_process(actions)
#                         centre_agent.update()
#                         centre_agent.empty_buffer()
#                     _done = True
#                     for w in range(Agent_NUM):
#                         _done *= done["center"+str(w)]
#                     # print('dome?')
#                     # print(_done)
#                     if _done:
#                         break
#                 print('centre steps mean rewards:')
#                 print(ep_r)
#                 centre_agent.summarize(ep_r,  j + (ep * centre_train_epi), 'reward')
#                 centre_agent.summarize(ep_r, global_counter, 'Global reward')

            if ep % 10 == 0:
                # centre_agent.save_params('centre_agent',ep)
                [sub_agents[k].save_params('sub_agent_{}'.format(str(k)),ep) for k in range(Agent_NUM)]

            # test phase
            if ep >= 0:
                print('test-phase  ep：')
                print(ep)
                record_line(each_line_path, "*** Epoch: {} ***\n".format(ep))
                queue, speed, ret = [], [], []
                for i in range(3):
                    ep_r, ep_q, ep_v = [], [], []
                    state = env.reset()
################################################
                    state = normalize_formation(state,Agent_NUM)
                    
                    for step in range(steps):

                        data_collection(env, ep_v, ep_q)

                        # actions = [sub_agents[k].choose_action(state[k])[0] for k in range(Agent_NUM)]
                        actions = [sub_agents[k].choose_action(state[k]) for k in range(Agent_NUM)]
                        # hiddens = [sub_agents[k].get_state(state[k]) for k in range(Agent_NUM)]
                        # # reshape
                        # concat_hiddens = np.array(hiddens).reshape(1, -1)
                        #
                        # centre_action = centre_agent.choose_action(concat_hiddens[0])
                        # actions = np.array(actions)
                        # centre_action = centre_agent.choose_action(actions)
                        # rl_actions = cover_actions(centre_action, actions)
                        # steps
                        next_state, rewards, done, _ = env.step(actions)
                        ep_r.append(sum(rewards))
                        state = next_state
#################################################
                        state = normalize_formation(state,Agent_NUM)

                        _done = True
                        for w in range(Agent_NUM):
                            _done *= done["center"+str(w)]
                        # print('dome?')
                        # print(_done)
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
