from gym.envs.registration import register

register(
    id='gym_sumo-v0',
    entry_point='gym_sumo.envs:SumoEnv',
)

from gym.envs.registration import register

register(
    id='gym_sumo-v1',
    entry_point='gym_sumo.envs:SumoEnv_qew',
)


register(
    id='gym_sumo-v2',
    entry_point='gym_sumo.envs:SumoEnv_exit',
)
