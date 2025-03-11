import gym
import numpy as np

import gym_carla

from controllers.barc_mpcc_conv import MPCCConvWrapper
from controllers.barc_lmpc import LMPCWrapper
from controllers.barc_pid import PIDWrapper

from loguru import logger


def main(controller: str, seed=0):
    """
    This is a minimalistic test script to test LMPC with the BARC gym environment.
    """
    controller_cls_mp = {
        'lmpc': LMPCWrapper,
        'mpcc-conv': MPCCConvWrapper,
        'pid': PIDWrapper,
    }
    dt = 0.1
    dt_sim = 0.01
    t0 = 0
    env = gym.make('barc-v0', track_name='L_track_barc',
                   t0=t0, dt=dt, dt_sim=dt_sim,
                   do_render=True,
                   max_n_laps=20,
                   enable_camera=False)
    # expert = LMPCWrapper(dt=dt, t0=t0,
    #                      track_obj=env.get_track())
    expert = controller_cls_mp[controller](dt=dt, t0=t0,
                                           track_obj=env.get_track())
    env.bind_controller(expert)

    ob, info = env.reset(seed=seed, options={'spawning': 'fixed'})
    expert.reset(seed=seed, options=info)
    rew, terminated, truncated = None, False, False

    lap_time = []

    while True:
        # ac, _ = expert.step(vehicle_state=info['vehicle_state'], terminated=terminated, lap_no=info['lap_no'])
        ac, _ = expert.step(**ob, **info)
        # ac = np.array([2.0, -0.01])
        # ac += np.random.randn(*ac.shape) * np.array([2.0, 0.1]) * 0.2
        ob, rew, terminated, truncated, info = env.step(ac)
        if terminated:
            lap_time.append(info['lap_time'])
        if truncated:
            print(f"Average lap time: {np.mean(lap_time[1:])}, Std: {np.std(lap_time[1:])}")
            logger.info("Rollout truncated.")
            ob, info = env.reset(seed=seed, options={'spawning': 'fixed'})
            expert.reset(seed=seed, options=info)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--controller', '-c', type=str,
                        choices=['lmpc', 'mpcc-conv', 'pid'])
    params = vars(parser.parse_args())

    main(**params)
