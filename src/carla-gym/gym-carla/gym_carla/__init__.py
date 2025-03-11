from gym.envs.registration import register

register(
    id='carla-v0',
    entry_point='gym_carla.envs.carla_env:CarlaEnv',
)
import gym_carla.envs.barc
import gym_carla.envs.pointmass
