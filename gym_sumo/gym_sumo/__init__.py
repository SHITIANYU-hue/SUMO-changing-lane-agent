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


register(
    id='gym_sumo-v3',
    entry_point='gym_sumo.envs:SumoEnv_multi',
)

register(
    id='gym_sumo-v4',
    entry_point='gym_sumo.envs:SumoEnv_merge',
)

register(
    id='gym_sumo-v5',
    entry_point='gym_sumo.envs:SumoEnv_intersection',
)
