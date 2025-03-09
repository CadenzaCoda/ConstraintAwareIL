import itertools
from collections import defaultdict
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torchvision.models import resnet18, ResNet18_Weights

from utils import pytorch_util as ptu
from .base_model import BaseModel
from .safeAC import Dynamics, SafeCritic, SafeAC

from loguru import logger

from utils.data_util import EfficientReplayBuffer, EfficientReplayBufferPN


class VisionSafeActor(BaseModel):
    feature_fields = ['camera', 'velocity']
    label_fields = ['action', 'state']

    def __init__(self, ob_dim, ac_dim, size, n_layers, lr=1e-3, weight_decay=1e-5, critic=None, dynamics=None, lam=1.):
        """
        Model Input: states
        Model Output: actions

        Loss: MSE + NLL
        """
        super().__init__()
        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.resnet.fc = nn.Sequential()
        self.mlp = ptu.build_mlp(input_size=512 + ob_dim, output_size=ac_dim, size=size, n_layers=n_layers,
                                 activation='relu', )
        self.critic = critic  # The critic outputs logits (without sigmoid).
        self.dynamics = dynamics

        self.optimizer = Adam(itertools.chain(self.resnet.parameters(), self.mlp.parameters()), lr=lr,
                              weight_decay=weight_decay)
        self.scheduler = ExponentialLR(optimizer=self.optimizer, gamma=1)  # 0.1 ** (1 / 100))
        self.loss_func = nn.MSELoss()
        self.log_sigmoid = nn.LogSigmoid()
        # self.img_mean = ptu.from_numpy(np.array([0.485, 0.456, 0.406])).view(1, 3, 1, 1).float()
        # self.img_std = ptu.from_numpy(np.array([0.229, 0.224, 0.225])).view(1, 3, 1, 1).float()
        self.lam = lam

    def step_schedule(self):
        self.lam = min(10., self.lam * 1.007)  # Approx 2 ** (1 / 100)

    def loss(self, pred, label, k=1, temperature=1):
        u_pred, = pred
        u, states = label
        mse_loss = self.loss_func(u_pred, u)
        if not self.critic.initialized:
            return mse_loss, {'mse_loss': mse_loss.item()}
        x_next_pred, = self.dynamics(states, u_pred)
        logits, = self.critic(x_next_pred)
        nll_loss = -self.lam * self.log_sigmoid(logits / temperature).sum() / logits.size(0)

        return mse_loss + nll_loss, {'mse_loss': mse_loss.item(), 'nll_loss': nll_loss.item()}  # MSE + NLL.

    def forward(self, img, vel):
        # Normalize the image first. 
        img = img.permute(0, 3, 1, 2) / 255.
        # img = (img / 255. - self.img_mean) / self.img_std
        # logger.debug(f"{img.size()}")
        l = self.resnet(img)
        out = self.mlp(torch.cat([l, vel], dim=1))
        return (out,)


class VisionSafeAC(SafeAC, nn.Module):
    def __init__(self, st_dim, ob_dim, ac_dim, n_layers, size, lr=1e-3, weight_decay=1e-5,
                 dyn_size=3, dyn_layers=64, critic_size=64, critic_layers=4,
                 lam=1., ):
        super().__init__(st_dim=st_dim, ac_dim=ac_dim, n_layers=n_layers, size=size)
        self.dynamics = Dynamics(st_dim=st_dim, ac_dim=ac_dim, size=dyn_size, n_layers=dyn_layers,
                                 lr=lr, weight_decay=weight_decay)
        self.critic = SafeCritic(st_dim=st_dim, ac_dim=ac_dim, size=critic_size, n_layers=critic_layers,
                                 lr=lr, weight_decay=weight_decay)
        self.actor = VisionSafeActor(ob_dim=ob_dim, ac_dim=ac_dim, size=size, n_layers=n_layers,
                                     lr=lr, weight_decay=weight_decay,
                                     lam=lam, critic=self.critic, dynamics=self.dynamics)

    def fit(self, train_dataset: EfficientReplayBufferPN, n_epochs, val_dataset=None, global_step=None):
        info = defaultdict(lambda: {})
        if global_step is None or global_step % 5 == 0:
            dynamics_info = self.dynamics.fit(train_dataset=train_dataset, n_epochs=n_epochs * 10)
        else:
            dynamics_info = {}

        if train_dataset.D_neg.initialized and len(train_dataset.D_neg) > 16 and (global_step is None or global_step % 10 == 0):
            self.critic.initialized = True
            critic_info = self.critic.fit(train_dataset=train_dataset, n_epochs=n_epochs * 10)
        else:
            critic_info = {}
        actor_info = self.actor.fit(train_dataset=train_dataset.D_pos, n_epochs=n_epochs)
        for d in (dynamics_info, actor_info, critic_info):
            for k1, v1 in d.items():
                for k2, v2 in v1.items():
                    info[k1][k2] = v2
        return info
