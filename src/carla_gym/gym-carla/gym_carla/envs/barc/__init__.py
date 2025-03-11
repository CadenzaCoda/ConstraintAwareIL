from gym.envs.registration import register

register(
    id='barc-v0',
    entry_point='gym_carla.envs.barc.barc_env:BarcEnv',
    # max_episode_steps=100000,
)
from gym_carla.envs.barc.barc_env import BarcEnv
