"""Example of modified minicity network with human-driven vehicles."""
from flow.controllers import IDMController
from flow.controllers import RLController
from flow.core.params import SumoParams, EnvParams, NetParams, InitialConfig
from flow.core.params import SumoCarFollowingParams, SumoLaneChangeParams
from flow.core.params import VehicleParams
from flow.envs.ring.accel import AccelEnv, ADDITIONAL_ENV_PARAMS
from flow.controllers.routing_controllers import MinicityRouter
from flow.networks import MiniCityNetwork
from flow.core.params import TrafficLightParams


vehicles = VehicleParams()
vehicles.add(
    veh_id="idm",
    acceleration_controller=(IDMController, {}),
    routing_controller=(MinicityRouter, {}),
    car_following_params=SumoCarFollowingParams(
        speed_mode=1,
    ),
    lane_change_params=SumoLaneChangeParams(
        lane_change_mode="no_lat_collide",
    ),
    initial_speed=0,
    num_vehicles=90)
vehicles.add(
    veh_id="rl",
    acceleration_controller=(RLController, {}),
    routing_controller=(MinicityRouter, {}),
    car_following_params=SumoCarFollowingParams(
        speed_mode="obey_safe_speed",
    ),
    initial_speed=0,
    num_vehicles=10)
################################################
tl_logic = TrafficLightParams(baseline=False)
phases = [{
    "duration": "31",
    "minDur": "8",
    "maxDur": "45",
    "state": "GrGrGrGrGrGr"
}, {
    "duration": "6",
    "minDur": "3",
    "maxDur": "6",
    "state": "yryryryryryr"
}, {
    "duration": "31",
    "minDur": "8",
    "maxDur": "45",
    "state": "rGrGrGrGrGrG"
}, {
    "duration": "6",
    "minDur": "3",
    "maxDur": "6",
    "state": "ryryryryryry"
}]
phases1 = [{
    "duration": "31",
    "minDur": "8",
    "maxDur": "45",
    "state": "GGggrrrrGGggrrrr"
}, {
    "duration": "5",
    "minDur": "3",
    "maxDur": "6",
    "state": "yyggrrrryyggrrrr"
}, {
    "duration": "6",
    "minDur": "3",
    "maxDur": "6",
    "state": "rrGGrrrrrrGGrrrr"
},{
    "duration": "5",
    "minDur": "3",
    "maxDur": "6",
    "state": "rryyrrrrrryyrrrr"
},{
    "duration": "31",
    "minDur": "8",
    "maxDur": "45",
    "state": "rrrrGGggrrrrGGgg"
},{
    "duration": "5",
    "minDur": "3",
    "maxDur": "6",
    "state": "rrrryyggrrrryygg"
},{
    "duration": "6",
    "minDur": "3",
    "maxDur": "6",
    "state": "rrrrrrGGrrrrrrGG"
},{
    "duration": "5",
    "minDur": "3",
    "maxDur": "6",
    "state": "rrrrrryyrrrrrryy"
}]
tl_logic.add("n_i3", phases=phases1, programID=1)
tl_logic.add("n_i4", phases=phases, programID=1)

additional_net_params = {"traffic_lights": True}
net_params = NetParams(additional_params=additional_net_params)
##################################################
flow_params = dict(
    # name of the experiment
    exp_tag='minicity',

    # name of the flow environment the experiment is running on
    env_name=AccelEnv,

    # name of the network class the experiment is running on
    network=MiniCityNetwork(name="city",
                            vehicles=VehicleParams(),
                            net_params=net_params,
                            initial_config=InitialConfig(),
                            traffic_lights=tl_logic),

    # simulator that is used by the experiment
    simulator='traci',

    # sumo-related parameters (see flow.core.params.SumoParams)
    sim=SumoParams(
        sim_step=0.25,
        render='drgb',
        save_render=False,
        sight_radius=30,
        pxpm=3,
        show_radius=True,
    ),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=750,
        additional_params=ADDITIONAL_ENV_PARAMS
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # network's documentation or ADDITIONAL_NET_PARAMS component)
    net=net_params,

    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.params.VehicleParams)
    veh=vehicles,

    # parameters specifying the positioning of vehicles upon initialization/
    # reset (see flow.core.params.InitialConfig)
    initial=InitialConfig(
        spacing="random",
        min_gap=5,
    ),
    tls=tl_logic,
)
