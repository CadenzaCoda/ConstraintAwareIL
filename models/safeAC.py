from collections import defaultdict
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from utils import pytorch_util as ptu
from .base_model import BaseModel

from loguru import logger

from utils.data_util import EfficientReplayBuffer, EfficientReplayBufferPN


class SafeActor(BaseModel):
    feature_fields = ['state']
    label_fields = ['action', 'state']

    def __init__(self, st_dim, ac_dim, size, n_layers, lr=1e-3, weight_decay=1e-5, critic=None, dynamics=None, lam=1.):
        """
        Model Input: states
        Model Output: actions

        Loss: MSE + NLL
        """
        super().__init__()
        self.actor = ptu.build_mlp(input_size=st_dim, output_size=ac_dim, size=size, n_layers=n_layers,
                                   activation='relu', )
        self.critic = critic  # The critic outputs logits (without sigmoid).
        self.dynamics = dynamics

        self.optimizer = Adam(self.actor.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = ExponentialLR(optimizer=self.optimizer, gamma=0.1 ** (1 / 100))
        self.loss_func = nn.MSELoss()
        self.log_sigmoid = nn.LogSigmoid()
        self.lam = lam

    def step_schedule(self):
        self.lam = min(10., self.lam * 1.007)  # Approx 2 ** (1 / 100)

    def loss(self, pred, label, k=1):
        u_pred, = pred
        u, states = label
        mse_loss = self.loss_func(u_pred, u)
        if not self.critic.initialized:
            return mse_loss, {'mse_loss': mse_loss.item()}
        x_next_pred, = self.dynamics(states, u_pred)
        logits, = self.critic(x_next_pred)
        nll_loss = -self.lam * self.log_sigmoid(logits).sum() / logits.size(0)

        return mse_loss + nll_loss, {'mse_loss': mse_loss.item(), 'nll_loss': nll_loss.item()}  # MSE + NLL.

    def forward(self, x):
        return (self.actor(x),)


class Dynamics(BaseModel):
    feature_fields = ['state', 'closed_loop_action']
    label_fields = ['next_state']

    def __init__(self, st_dim, ac_dim, size, n_layers, lr=1e-3, weight_decay=1e-5):
        super().__init__()
        self.dynamics = ptu.build_mlp(input_size=st_dim + ac_dim, output_size=st_dim, size=size, n_layers=n_layers,
                                      activation='relu', )
        self.optimizer = Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = ExponentialLR(optimizer=self.optimizer, gamma=0.1 ** (1 / 100))
        self.loss_func = nn.MSELoss()

    def forward(self, x, u):
        return (self.dynamics(torch.cat([x, u], dim=-1)) + x,)

    def loss(self, pred, labels):
        x_next_pred, = pred
        x_next, = labels
        return self.loss_func(x_next_pred, x_next), {}


class SafeCritic(BaseModel):
    feature_fields = ['state']
    label_fields = ['safe']

    def __init__(self, st_dim, ac_dim, size, n_layers, lr=1e-3, weight_decay=1e-5):
        super().__init__()
        self.critic = ptu.build_mlp(input_size=st_dim, output_size=1, size=size, n_layers=n_layers,
                                    activation='relu', )
        # self.dynamics = dynamics
        self.optimizer = Adam(self.critic.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = ExponentialLR(optimizer=self.optimizer, gamma=0.1 ** (1 / 100))
        self.loss_func = nn.BCEWithLogitsLoss()
        self.initialized = False

    def forward(self, x):
        # x_next, = self.dynamics(x, u)
        return (self.critic(x),)

    def loss(self, pred, label):
        logit, = pred
        label, = label
        return self.loss_func(logit, label), {
            'acc': (((logit < 0) & (label < 0.5)) | ((logit > 0) & (label > 0.5))).sum() / label.size(0)
        }


class SafeAC(BaseModel):

    def __init__(self, st_dim, ac_dim, n_layers, size, lr=1e-3, weight_decay=1e-5,
                 dyn_size=3, dyn_layers=64, critic_size=64, critic_layers=4,
                 lam=1., ):
        super().__init__()
        self.dynamics = Dynamics(st_dim=st_dim, ac_dim=ac_dim, size=dyn_size, n_layers=dyn_layers,
                                 lr=lr, weight_decay=weight_decay)
        self.critic = SafeCritic(st_dim=st_dim, ac_dim=ac_dim, size=critic_size, n_layers=critic_layers,
                                 lr=lr, weight_decay=weight_decay)
        self.actor = SafeActor(st_dim=st_dim, ac_dim=ac_dim, size=size, n_layers=n_layers,
                               lr=lr, weight_decay=weight_decay, lam=lam, critic=self.critic,
                               dynamics=self.dynamics)

    # def step_schedule(self):
        # self.actor.step_schedule()

    def forward(self, x):
        return self.actor(x)

    def fit(self, train_dataset: EfficientReplayBufferPN, n_epochs, val_dataset=None,
            global_step=None):
        info = defaultdict(lambda: {})
        dynamics_info = self.dynamics.fit(train_dataset=train_dataset, n_epochs=n_epochs)
        if train_dataset.D_neg.initialized and len(train_dataset.D_neg) > 0:
            self.critic.initialized = True
            critic_info = self.critic.fit(train_dataset=train_dataset, n_epochs=n_epochs)
        else:
            self.critic.initialized = False
            critic_info = {}
        actor_info = self.actor.fit(train_dataset=train_dataset.D_pos, n_epochs=n_epochs)

        for d in (dynamics_info, critic_info, actor_info):
            for k1, v1 in d.items():
                for k2, v2 in v1.items():
                    info[k1][k2] = v2
        return info

    def parse_carla_obs(self, obs, info):
        return self.actor.parse_carla_obs(obs, info)

    def get_action(self, *args):
        return self.actor.get_action(*args)

    def predict(self, obs, info, batched=False):
        return self.actor.predict(obs, info, batched=batched)
