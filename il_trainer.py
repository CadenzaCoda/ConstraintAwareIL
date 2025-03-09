import os
from abc import ABC, abstractmethod
import copy
from collections import deque
from pathlib import Path
from typing import Optional

import gym
from torchvision.transforms import transforms

import gym_carla
import numpy as np
import torch
import yaml
from matplotlib import pyplot as plt
from tqdm import tqdm

from carla_gym.controllers.barc_lmpc import LMPCWrapper
from carla_gym.controllers.barc_mpcc_conv import MPCCConvWrapper
from carla_gym.controllers.barc_pid import PIDWrapper
import models.feedforward
from models import safeAC, visionSafeAC
from models.base_model import BaseModel

from utils import data_util
from torch.utils.data import DataLoader
import utils.pytorch_util as ptu
from utils.logging.writer import MultiPurposeWriter
# from util.expert_wrapper_carla import CARLA_PIDLaneFollowerWrapper
# from utils.transformations import CustomTransformSet, RandomCutout, Grayscale, RandomBrightness, \
#     RandomContrast

# import multiprocessing as mp
from loguru import logger
from labml import experiment


class IL_Trainer_CARLA(ABC):
    def __init__(self, carla_params,
                 initial_traj_len=1024,
                 eps_len=1024,
                 replay_buffer_maxsize=65536,
                 do_relabel_with_expert=True,
                 n_training_per_epoch=1,
                 comment: str = '',
                 no_saving: bool = False,
                 starting_step: int = 0,
                 eval_freq: int = 1,
                 batch_size: int = 1,
                 n_initial_training_epochs: int = 5,
                 beta: float = 0.25,
                 use_labml_tracker: bool = True,
                 pretrain_critic: bool = False,
                 ntfy_freq: int = -1,
                 beta_decay_freq: int = 5,
                 **agent_params
                 ):
        """

        @param expert: A wrapper designed for CARLA. Must have a step function that accepts a single frame of
        observation from gym_carla, and returns a 1d numpy array of expert action.
        The expert must be deterministic.
        @param carla_params:
        @param initial_traj_len:
        @param replay_buffer_maxsize:
        @param do_relabel_with_expert:
        @param n_training_per_epoch:
        @param comment:
        @param no_saving:
        @param starting_step:
        @param eval_freq:
        @param batch_size:
        @param n_initial_training_epochs:
        @param agent_params:
        """
        self.best_avg_lap_time = np.inf
        self.beta_decay_freq = beta_decay_freq
        self.eval_freq = eval_freq
        self.comment = comment
        self.initial_traj_len = initial_traj_len
        self.do_relabel_with_expert = do_relabel_with_expert
        self.n_training_per_epoch = n_training_per_epoch
        self.batch_size = batch_size
        self.no_saving = no_saving
        self.n_initial_training_epochs = n_initial_training_epochs
        self.beta = beta
        self.use_labml_tracker = use_labml_tracker
        self.agent_params = agent_params

        self.n_eval_success, self.n_eval_total = 0, 0
        self.eval_rewards_last10 = deque(maxlen=10)

        self.update_carla_params(carla_params)
        self.env = gym.make('barc-v0', **carla_params)
        self.eps_len = min(replay_buffer_maxsize, eps_len)

        # self.expert = LMPCWrapper(dt=carla_params['dt'], t0=carla_params['t0'], track_obj=self.env.get_track())
        self.expert = PIDWrapper(dt=carla_params['dt'], t0=carla_params['t0'], track_obj=self.env.get_track())
        self.agent: 'BaseModel' = None
        self.initialize_agent(comment=comment, **agent_params)

        self.replay_buffer: 'data_util.EfficientReplayBuffer' = None
        self.initialize_replay_buffer(replay_buffer_maxsize)

        self.writer: 'MultiPurposeWriter' = MultiPurposeWriter(model_name=self.agent.model_name,
                                                               log_dir=f"logs/{self.agent.model_name}_{comment or ''}",
                                                               comment=comment or '',
                                                               print_method=logger.info,
                                                               use_labml_tracker=use_labml_tracker,
                                                               ntfy_freq=ntfy_freq,
                                                               )

        # Load previous model weights and replay buffer.
        self.starting_step = starting_step
        self.agent.to(ptu.device)
        if starting_step > 0:
            self.agent.load()
            self.replay_buffer.load()
        if pretrain_critic:
            self.pretrain_critic()

    def update_carla_params(self, carla_params):
        pass

    @abstractmethod
    def initialize_agent(self, comment, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def initialize_replay_buffer(self, replay_buffer_maxsize):
        raise NotImplementedError

    def pretrain_critic(self):
        return

    def sample_trajectory(self, beta: float, pbar: Optional['tqdm'] = None,
                          max_traj_len=np.inf,
                          PATIENCE=2, TRUNCATE=5):
        """

        @param beta:
        @param pbar:
        @param max_traj_len:
        @param PATIENCE: Maximum allowed consecutive expert fails before truncating the trajectory.
        @param TRUNCATE: Number of examples to remove from the replay buffer if the trajectory is truncated.
        @return:
        """
        ob, info = self.env.reset(options={'controller': self.expert})
        self.agent.reset()
        self.agent.eval()
        self.expert.reset(options=info)
        self.replay_buffer.clear()
        terminated, truncated = False, False
        traj_len = 0
        fail_counter = 0

        while not truncated:
            ac = self.agent.get_action(*self.agent.parse_carla_obs(ob, info))
            # expert_ac, expert_info = self.expert.step(state=info['vehicle_state'],
            #                                           terminated=terminated,
            #                                           lap_no=info['lap_no'])
            expert_ac, expert_info = self.expert.step(**ob, **info)
            expert_ac = np.clip(expert_ac, self.env.action_space.low, self.env.action_space.high)

            # try:
            if expert_info['success']:
                # action = expert_ac if np.random.rand() <= beta else ac
                closed_loop_action = beta * expert_ac + (1 - beta) * ac
                next_ob, rew, terminated, truncated, info = self.env.step(closed_loop_action)
                # logger.debug(f"Action: {ac}, Expert action: {expert_ac}, v_long: {ob['state'][0]}")
                # self.add_frame(ob=ob, ac_agent=ac, ac_expert=expert_ac, rew=rew, terminated=terminated,
                #                truncated=truncated, info=info, next_ob=next_ob)
                self.replay_buffer.add_frame(ob, rew, terminated, truncated, info,
                                             action=expert_ac.astype(np.float32),
                                             closed_loop_action=closed_loop_action.astype(np.float32),
                                             next_state=next_ob['state'])
                fail_counter = 0

            else:
                logger.warning(f"Expert solved inaccurate with code {expert_info.get('status', 'unknown')}.")
                next_ob, rew, terminated, truncated, info = self.env.step(ac)
                fail_counter += 1
                if fail_counter >= PATIENCE:
                    truncated = True  # NEW: Truncate the trajectory if expert fails for <PATIENCE> consecutive steps.
                    # self.replay_buffer.popback(min(traj_len, TRUNCATE))

            # except ValueError as e:  # Capture out-of-track errors from simulator step.
            #     logger.warning(e)
            #     truncated = True
            #     break

            ob = next_ob
            traj_len += 1
            if pbar is not None:
                pbar.update(1)
            if traj_len >= max_traj_len:
                return traj_len
        return traj_len

    def sample_trajectories(self, beta: float, total_length=None, global_step=None):
        logger.info('Sampling trajectories for training...')
        total_length = total_length or self.eps_len
        batch_traj_len = 0
        n_resets = 0
        with tqdm(total=total_length, desc='Sampling', unit='steps') as pbar:
            while batch_traj_len < total_length:
                traj_len = self.sample_trajectory(beta, pbar=pbar, max_traj_len=total_length - batch_traj_len)
                batch_traj_len += traj_len
                n_resets += 1
        self.writer.do_logging({f'failure_rate': (n_resets - 1) / total_length}, global_step=global_step, mode='train')
        return batch_traj_len

    # def train_agent(self, global_step, val_loader: Optional[DataLoader] = None):
    #     logger.info("Training agent...")
    #     dataloader = self.replay_buffer.dataloader(shuffle=self.replay_buffer.random_eviction,
    #                                                batch_size=self.batch_size)
    #     info = self.agent.fit(train_loader=dataloader,
    #                           val_loader=val_loader,
    #                           n_epochs=self.n_training_per_epoch if global_step > 0 else self.n_initial_training_epochs,
    #                           )
    #     for mode, values in info.items():
    #         self.writer.do_logging(values, global_step=global_step, mode=mode)

    def train_module(self, module, global_step):
        logger.info(f"Training {module.__class__.__name__}...")
        # dataloader = self.replay_buffer.dataloader(shuffle=True, batch_size=self.batch_size,
        #                                            manifest=[module.feature_fields, module.label_fields])
        info = module.fit(train_dataset=self.replay_buffer,
                          n_epochs=self.n_training_per_epoch if global_step > 0 else self.n_initial_training_epochs,
                          global_step=global_step)
        for mode, values in info.items():
            self.writer.do_logging(values, global_step=global_step, mode=mode)

    def evaluate_agent(self, global_step):
        logger.info("Evaluating agent...")
        self.agent.reset()
        self.agent.eval()
        # ob, info = self.env.reset(options={'render': True})
        ob, info = self.env.reset(options={'controller': self.expert, 'spawning': 'fixed'})
        # self.writer.add_text(tag='val', text_string=self.env.current_weather, global_step=global_step)

        if self.do_relabel_with_expert:
            self.expert.reset(options=info)

        truncated, terminated = False, False
        # fastest_lap = np.inf
        lap_times = []
        rews = 0.
        traj_len = 0
        completed_laps = 0

        sts, acs, expert_acs, vs = [], [], [], []

        while not truncated:
            ac = self.agent.get_action(*self.agent.parse_carla_obs(ob, info))
            if self.do_relabel_with_expert:
                # try:
                expert_ac, expert_info = self.expert.step(**ob, **info)
                # except SolverException as e:
                #     expert_ac = self.expert._action

            sts.append(np.array([info['vehicle_state'].x.x, info['vehicle_state'].x.y]))
            acs.append(ac)
            vs.append(np.linalg.norm(ob['state'][:2]))
            if self.do_relabel_with_expert:
                expert_acs.append(expert_ac)

            ob, rew, terminated, truncated, info = self.env.step(ac)
            rews += rew
            traj_len += 1
            completed_laps = info['lap_no']
            if terminated:
                lap_times.append(info['lap_time'])
                # fastest_lap = min(fastest_lap, info['lap_time'])

        # Below are logging-related actions.
        self.writer.do_logging({
            'Traj Len': traj_len,
            'sum_reward': rews,
            'fastest_lap': np.max(lap_times) if len(lap_times) > 0 else np.inf,
            'avg_lap_time': np.mean(lap_times) if len(lap_times) > 0 else np.inf,
            'std_lap_time': np.std(lap_times[1:]) if len(lap_times) > 1 else np.nan,
            'completed_laps': completed_laps,
        }, global_step=global_step, mode='val')

        if completed_laps >= 20 or (len(lap_times) > 0 and np.mean(lap_times) < self.best_avg_lap_time):
            self.best_avg_lap_time = completed_laps
            self.agent.export(path=os.path.join(Path(__file__).parent / 'model_data' / 'significant_checkpoints'),
                              name=f"{self.comment}_{global_step}")

        self.n_eval_total += 1
        if traj_len == self.eps_len:
            self.n_eval_success += 1
        self.eval_rewards_last10.append(rews)

        sts, acs, expert_acs = np.asarray(sts), np.asarray(acs), np.asarray(expert_acs)

        fig, ((ax_u_a, ax_u_steer), (ax_v, ax_gps)) = plt.subplots(2, 2, figsize=(10, 10))
        self.env.get_track().plot_map(ax=ax_gps)

        if self.do_relabel_with_expert:
            ax_u_a.plot(expert_acs[:, 0], label='expert')
            ax_u_steer.plot(expert_acs[:, 1], label='expert')

        ax_u_a.plot(acs[:, 0], label='model')
        ax_u_a.set_title("$u_{a}$")
        ax_u_a.legend()

        ax_u_steer.plot(acs[:, 1], label='model')
        ax_u_steer.set_title("$u_{steer}$")
        ax_u_steer.legend()

        ax_v.plot(vs, label='velocity')
        ax_v.set_title("$v$")
        ax_v.legend()

        ax_gps.plot(sts[:, 0], sts[:, 1], )
        ax_gps.set_title("GPS trajectory")

        fig.suptitle(f"{self.agent.model_name}_{self.comment} (Global Step: {global_step})")

        self.writer.add_figure(tag='val', figure=fig, global_step=global_step)

    def training_loop(self, n_epochs: int):
        try:
            for global_step in range(self.starting_step, n_epochs):
                logger.info(f"Epoch {global_step} / {n_epochs}")
                self.agent.step_schedule()
                # self.sample_trajectories(beta=self.beta ** global_step)
                self.sample_trajectory(beta=self.beta ** np.ceil(global_step / self.beta_decay_freq),
                                       max_traj_len=self.initial_traj_len if global_step == 0 else self.eps_len)
                # self.sample_trajectories(beta=self.beta ** np.ceil(global_step / self.beta_decay_freq),
                #                          total_length=self.initial_traj_len if global_step == 0 else self.eps_len)
                self.replay_buffer.preprocess()
                self.train_module(self.agent, global_step)
                if self.no_saving:
                    continue
                if global_step % self.eval_freq == 0:
                    self.evaluate_agent(global_step=global_step)
                self.agent.export(path=os.path.join(Path(__file__).parent / 'model_data'), name=self.comment)
        finally:
            hparam_dict = {
                'initial_traj_len': self.initial_traj_len,
                'beta': self.beta,
                'eps_len': self.eps_len,
                **self.agent_params
            }
            metric_dict = {
                'success_rate': self.n_eval_success / self.n_eval_total if self.n_eval_total != 0 else None,
                'mean_rew_last10': np.mean(np.asarray(self.eval_rewards_last10)) if len(
                    self.eval_rewards_last10) else None,
            }
            # self.writer.add_hparams(hparam_dict, metric_dict)
            self.writer.flush()
            self.writer.ntfy(message="Training program terminated.")
            # self.replay_buffer.export()

    def main(self, n_epochs: int):
        if self.use_labml_tracker:
            with experiment.record(name=f"{trainer.agent.model_name}_{params['comment']}",
                                   exp_conf={**params, **agent_params}):
                self.training_loop(n_epochs=n_epochs)
        else:
            self.training_loop(n_epochs=n_epochs)


class IL_Trainer_CARLA_StateFF(IL_Trainer_CARLA):
    def update_carla_params(self, carla_params):
        carla_params.update({
            'enable_camera': False,
        })

    def initialize_agent(self, comment, **kwargs):
        self.agent = models.feedforward.StateFF(ac_dim=2, st_dim=6, size=64, n_layers=3, lr=1e-3, weight_decay=1e-4,
                                                feature_fields=['gps', 'velocity'], label_fields=['actions'])

    def initialize_replay_buffer(self, replay_buffer_maxsize):
        self.replay_buffer = data_util.EfficientReplayBuffer(replay_buffer_maxsize,
                                                             lazy_init=not params['experimental'])


class IL_Trainer_CARLA_ConditionedEncDec(IL_Trainer_CARLA):
    def sample_trajectory(self, beta: float, pbar: Optional['tqdm'] = None,
                          max_traj_len=np.inf,
                          PATIENCE=2, TRUNCATE=5):
        """
        Note that we have to overwrite this function because the conditioning signal comes from a field in info.

        @param beta:
        @param pbar:
        @param max_traj_len:
        @param PATIENCE: Maximum allowed consecutive expert fails before truncating the trajectory.
        @param TRUNCATE: Number of examples to remove from the replay buffer if the trajectory is truncated.
        @return:
        """
        ob, info = self.env.reset(options={'controller': self.expert})
        self.agent.reset()
        self.agent.eval()
        if self.do_relabel_with_expert:
            self.expert.reset(options=info)
        terminated, truncated = False, False
        traj_len = 0
        fail_counter = 0

        while not truncated:
            ac = self.agent.get_action(*self.agent.parse_carla_obs(ob))
            # expert_ac, expert_info = self.expert.step(state=info['vehicle_state'],
            #                                           terminated=terminated,
            #                                           lap_no=info['lap_no'])
            expert_ac, expert_info = self.expert.step(**ob, **info)
            self.env.clip_action(expert_ac)

            try:
                if expert_info['success']:
                    next_ob, rew, terminated, truncated, info = self.env.step(beta * expert_ac + (1 - beta) * ac)
                    # logger.debug(f"Action: {ac}, Expert action: {expert_ac}, v_long: {ob['state'][0]}")
                    self.add_frame(ob=ob, ac_agent=ac, ac_expert=expert_ac, rew=rew, terminated=terminated,
                                   truncated=truncated, info=info, next_ob=next_ob)
                    # self.replay_buffer.add_frame(ob, expert_ac.astype(np.float32), rew, truncated, info)  # The only line that's different
                    fail_counter = 0

                else:
                    logger.warning(f"Expert solved inaccurate with code {expert_info.get('status', 'unknown')}.")
                    next_ob, rew, terminated, truncated, info = self.env.step(ac)
                    fail_counter += 1
                    if fail_counter >= PATIENCE:
                        truncated = True  # NEW: Truncate the trajectory if expert fails for <PATIENCE> consecutive steps.
                        self.replay_buffer.popback(min(traj_len, TRUNCATE))

            except ValueError as e:  # Capture out-of-track errors from simulator step.
                logger.warning(e)
                truncated = True
                pass

            ob = next_ob
            traj_len += 1
            if pbar is not None:
                pbar.update(1)
            if traj_len >= max_traj_len:
                return traj_len
        return traj_len

    def initialize_agent(self, comment, **kwargs):
        raise NotImplementedError
        self.agent = ILAgent.ConditionedEncDec(ac_dim=2, st_dim=6, size=64, n_layers=3, lr=1e-3, weight_decay=1e-4,
                                               cond_dim=1)

    def initialize_replay_buffer(self, replay_buffer_maxsize):
        self.replay_buffer = data_util.EfficientReplayBufferSCA(replay_buffer_maxsize)


class IL_Trainer_CARLA_SafeAC(IL_Trainer_CARLA):
    def initialize_agent(self, comment, **kwargs):
        self.agent = safeAC.SafeAC(**kwargs)

    def initialize_replay_buffer(self, replay_buffer_maxsize):
        self.replay_buffer = data_util.EfficientReplayBufferPN(maxsize=replay_buffer_maxsize,
                                                               lazy_init=not params['experimental'],
                                                               # lazy_init=not params['experimental'],
                                                               )
        # self.replay_buffer = data_util.EfficientReplayBufferPN_LMPC(replay_buffer_maxsize, expert=self.expert)

    def pretrain_critic(self):
        pretraining_expert = LMPCWrapper(track_obj=self.env.get_track())
        raise NotImplementedError

    def sample_trajectory(self, beta: float, pbar: Optional['tqdm'] = None,
                          max_traj_len=np.inf,
                          PATIENCE=2, TRUNCATE=np.inf):
        """

        @param beta:
        @param pbar:
        @param max_traj_len:
        @param PATIENCE: Maximum allowed consecutive expert fails before truncating the trajectory.
        @param TRUNCATE: Number of examples to remove from the replay buffer if the trajectory is truncated.
        @return:
        """
        ob, info = self.env.reset(options={'controller': self.expert})
        self.agent.reset()
        self.agent.eval()
        self.expert.reset(options=info)
        self.replay_buffer.clear_buffer()
        terminated, truncated = False, False
        traj_len = 0
        fail_counter = 0

        while not truncated:
            ac = self.agent.get_action(*self.agent.parse_carla_obs(ob, info))
            # expert_ac, expert_info = self.expert.step(state=info['vehicle_state'],
            #                                           terminated=terminated,
            #                                           lap_no=info['lap_no'])
            expert_ac, expert_info = self.expert.step(**ob, **info)
            expert_ac = np.clip(expert_ac, self.env.action_space.low, self.env.action_space.high)
            closed_loop_action = beta * expert_ac + (1 - beta) * ac

            # try:
            if expert_info['success']:
                # action = expert_ac if np.random.rand() <= beta else ac
                # closed_loop_action = beta * expert_ac + (1 - beta) * ac
                next_ob, rew, terminated, truncated, info = self.env.step(closed_loop_action)
                # logger.debug(f"Action: {ac}, Expert action: {expert_ac}, v_long: {ob['state'][0]}")
                # self.add_frame(ob=ob, ac_agent=ac, ac_expert=expert_ac, rew=rew, terminated=terminated,
                #                truncated=truncated, info=info, next_ob=next_ob)
                self.replay_buffer.add_frame(ob, rew, terminated, truncated, info,
                                             action=expert_ac.astype(np.float32),
                                             closed_loop_action=closed_loop_action.astype(np.float32),
                                             next_state=next_ob['state'])
                fail_counter = 0

            else:
                logger.warning(f"Expert solved inaccurate with code {expert_info.get('status', 'unknown')}.")
                next_ob, rew, terminated, truncated, info = self.env.step(closed_loop_action)
                fail_counter += 1
                self.replay_buffer.D_neg.add_frame(ob, rew, terminated, truncated, info,
                                                   action=expert_ac.astype(np.float32),
                                                   closed_loop_action=closed_loop_action.astype(np.float32),
                                                   next_state=next_ob['state'])
                if fail_counter >= PATIENCE:
                    truncated = True  # NEW: Truncate the trajectory if expert fails for <PATIENCE> consecutive steps.
                    # self.replay_buffer.popback(min(traj_len, TRUNCATE))

            # except ValueError as e:  # Capture out-of-track errors from simulator step.
            #     logger.warning(e)
            #     truncated = True
            #     break

            traj_len += 1
            ob = next_ob
            if pbar is not None:
                pbar.update(1)
            if traj_len >= max_traj_len:
                return traj_len
        return traj_len

    def training_loop(self, n_epochs: int):
        try:
            for global_step in range(self.starting_step, n_epochs):
                logger.info(f"Epoch {global_step} / {n_epochs}")
                # self.agent.step_schedule()
                # self.sample_trajectories(beta=self.beta ** global_step)
                # self.sample_trajectory(beta=self.beta ** np.ceil(global_step / self.beta_decay_freq),
                #                        max_traj_len=self.initial_traj_len if global_step == 0 else self.eps_len)
                self.sample_trajectories(beta=self.beta ** np.ceil(global_step / self.beta_decay_freq),
                                         total_length=self.initial_traj_len if global_step == 0 else self.eps_len,
                                         global_step=global_step)
                if global_step % 10 == 0:
                    self.replay_buffer.preprocess()
                self.train_module(self.agent, global_step)
                if self.no_saving:
                    continue
                if global_step % self.eval_freq == 0:
                    self.evaluate_agent(global_step=global_step)
                self.agent.export(path=os.path.join(Path(__file__).parent / 'model_data'), name=self.comment)
        finally:
            self.writer.flush()
            self.writer.ntfy(message="Training program terminated.")
            # if not params['experimental']:
                # self.replay_buffer.export(path=os.path.join(Path(__file__).parent / 'model_data'), name=self.comment)


class IL_Trainer_CARLA_VisionSafeAC(IL_Trainer_CARLA_SafeAC):
    def initialize_agent(self, comment, **kwargs):
        logger.debug(f"{kwargs}")
        self.agent = visionSafeAC.VisionSafeAC(**kwargs)

    def pretrain_critic(self):
        checkpoint_path = Path(__file__).resolve().parent / 'model_data' / 'SafeAC_L_track_barc_safeac-v0.9.14-lam10-learned-local.pt'
        checkpoint = torch.load(checkpoint_path)
        model_state_dict = self.agent.state_dict()

        # print(checkpoint.keys())
        filtered_state_dict = {k: v for k, v in checkpoint.items() if ('critic' in k or 'dynamics' in k) and v.shape == model_state_dict[k].shape}
        print(filtered_state_dict.keys())

        # Load only the matching weights
        model_state_dict.update(filtered_state_dict)  # Update with the filtered keys
        self.agent.load_state_dict(model_state_dict, strict=False)  # Load updated state_dict
        self.agent.critic.initialized = True


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_epochs', type=int, default=500)
    parser.add_argument('--initial_traj_len', type=int, default=1024)
    parser.add_argument('--n_training_per_epoch', type=int, default=1)
    parser.add_argument('--n_initial_training_epochs', type=int, default=5)
    parser.add_argument('--replay_buffer_maxsize', type=int, default=102_400)
    # parser.add_argument('--no_relabeling', action='store_true')
    parser.add_argument('--comment', '-m', type=str, default='')
    parser.add_argument('--eps_len', type=int, default=1024)

    parser.add_argument('--town', type=str, default='L_track_barc')
    parser.add_argument('--host', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--dt', type=float, default=0.1)
    parser.add_argument('--no_saving', action='store_true')
    # parser.add_argument('--continue', action='store_true')
    parser.add_argument('--eval_freq', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--beta', type=float, default=0.95)
    parser.add_argument('--beta_decay_freq', type=int, default=1)

    # parser.add_argument('--evaluation', action='store_true')
    parser.add_argument('--evaluation', type=str, default=None)
    parser.add_argument('--validation', action='store_true')
    parser.add_argument('--continue', type=int, default=0)
    parser.add_argument('--freeze_weather', action='store_true')
    parser.add_argument('--fix_spawning', action='store_true')
    parser.add_argument('--pretrain_critic', action='store_true')

    parser.add_argument('--experimental', action='store_true')
    parser.add_argument('--ntfy_freq', type=int, default=100)

    params = vars(parser.parse_args())

    if params['experimental']:
        params.update({
            # 'initial_traj_len': 128,
            'comment': '_'.join((params['comment'], 'experimental')),
            'ntfy_freq': -1,
        })

    print(params)

    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])

    comment = '_'.join((params['town'], params['comment']))

    params['comment'] = '_'.join((params['town'], params['comment'])) + ('_evaluation' if params['evaluation'] else '')

    t0, dt, dt_sim = 0., 0.1, 0.01

    carla_params = dict(
        track_name=params['town'],
        t0=t0, dt=dt, dt_sim=dt_sim,
        do_render=False,
        max_n_laps=50,
        enable_camera=True,
        host=params['host'],
        port=params['port'],
    )  # Name kept as CARLA for backward compatibility.

    # TODO: RESUME FROM HERE!!!!

    with open("./config/visionSafeAC.yaml", 'r') as f:
        config = yaml.safe_load(f)

    agent_params = config['model_hparams']

    # trainer_cls = IL_Trainer_CARLA_StateFF
    # trainer_cls = IL_Trainer_CARLA_ConditionedEncDec
    trainer_cls = IL_Trainer_CARLA_VisionSafeAC

    trainer = trainer_cls(carla_params,
                          replay_buffer_maxsize=params['replay_buffer_maxsize'],
                          eps_len=params['eps_len'],
                          initial_traj_len=params['initial_traj_len'],
                          do_relabel_with_expert=True,
                          n_training_per_epoch=params['n_training_per_epoch'],
                          comment=params['comment'],
                          no_saving=params['no_saving'],
                          starting_step=params['continue'],
                          eval_freq=params['eval_freq'],
                          batch_size=params['batch_size'],
                          n_initial_training_epochs=params['n_initial_training_epochs'],
                          beta=params['beta'],
                          use_labml_tracker=not params['experimental'],
                          ntfy_freq=params['ntfy_freq'],
                          beta_decay_freq=params['beta_decay_freq'],
                          pretrain_critic=params['pretrain_critic'],
                          **agent_params
                          )

    if params['evaluation']:
        trainer.agent.load(path=Path(__file__).resolve().parent / 'model_data' / 'significant_checkpoints',
                           name=comment)
        # for epoch in range(1):
        trainer.evaluate_agent(global_step=0)

    else:
        val_loader = None
        if params['validation']:
            raise NotImplementedError
            val_dataset = copy.deepcopy(trainer.replay_buffer)
            val_dataset.load()
            val_loader = val_dataset.dataloader(batch_size=params['batch_size'], shuffle=False)

        trainer.main(n_epochs=params['n_epochs'])
