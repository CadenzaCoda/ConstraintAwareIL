import os
from abc import abstractmethod

import torch
import torch.nn as nn
from loguru import logger
from torch.optim import Adam

from mpclab_controllers.ILAgent import BaseILModel

from utils import pytorch_util as ptu


class SpeedConditionedFull(BaseILModel):
    """
    Speed conditioned model. Assuming full-state feedback.

    Insights:
    An encoder-decoder structure like the unet. Use shortcut from the encoder to the decoder! (concatenate)
    Speed conditioning signal is injected into the decoder.

    During training, dropout the conditioning signal at some probability.
    During inference, try using CFG.
    """
    input_fields = ['states', 'lap_times']
    output_fields = ['actions']

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.state_encoder = ptu.build_mlp(input_size=input_size, n_layers=0, size=hidden_size, output_size=hidden_size,
                                           activation='relu', output_activation='relu')
        self.action_decoder = ptu.build_mlp(input_size=hidden_size, n_layers=0, size=hidden_size,
                                            output_size=output_size,
                                            activation='relu')
        self.perf_encoder = ptu.build_mlp(input_size=1, n_layers=1, size=hidden_size, output_size=hidden_size,
                                          activation='relu')

    def forward(self, x, v):
        l_x = self.state_encoder(x)
        l_v = self.perf_encoder(v)
        u = self.action_decoder(l_x + l_v)
        return u

    def get_action(self, x, v):
        assert len(x.shape) == 1 and len(v.shape) == 1
        u = self(ptu.from_numpy(x[None]), ptu.from_numpy(v[None]))
        return ptu.to_numpy(u).squeeze(0)

    def fit(self, train_loader, n_epochs=1, lr=1e-3, weight_decay=1e-4):
        optimizer = Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        criteria = nn.MSELoss()

        train_loss = 0.
        train_examples = 0
        for epoch in range(n_epochs):
            self.train()
            for ob_no, perf_np, ac_na in train_loader:
                optimizer.zero_grad()
                ac_pred_na = self(ob_no, perf_np)
                loss = criteria(ac_pred_na, ac_na)
                train_loss += loss.item()
                train_examples += 1
                loss.backward()
                optimizer.step()
        return {
            'train': {'loss': train_loss / train_examples},
        }

    def parse_carla_obs(self, ob):
        raise NotImplementedError




