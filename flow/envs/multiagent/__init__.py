"""Empty init file to ensure documentation for multi-agent envs is created."""

from flow.envs.multiagent.base import MultiEnv
from flow.envs.multiagent.ring.wave_attenuation import \
    MultiWaveAttenuationPOEnv

from flow.envs.multiagent.ring.Mywave_attenuation import \
    MyMultiWaveAttenuationPOEnv

from flow.envs.multiagent.ring.accel import MultiAgentAccelEnv
from flow.envs.multiagent.traffic_light_grid import MultiTrafficLightGridPOEnv
from flow.envs.multiagent.mytraffic_light_grid import MyMultiTrafficLightGridPOEnv
from flow.envs.multiagent.highway import MultiAgentHighwayPOEnv

__all__ = ['MultiEnv', 'MultiAgentAccelEnv', 'MultiWaveAttenuationPOEnv',
           'MultiTrafficLightGridPOEnv','MyMultiTrafficLightGridPOEnv', 'MultiAgentHighwayPOEnv']
