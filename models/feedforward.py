from typing import List, Tuple, Dict

import torch
import torch.nn as nn
from torch.optim import Adam
from .base_model import BaseModel
from utils import pytorch_util as ptu


class StateFF(BaseModel):
    feature_fields = ['gps', 'velocity']
    label_fields = ['actions']

    def __init__(self, ac_dim: int, st_dim: int,
                 size, n_layers, lr, weight_decay, **kwargs):
        super().__init__()
        self.mlp = ptu.build_mlp(input_size=st_dim, output_size=ac_dim, n_layers=n_layers, size=size,
                                 activation='relu')
        self.optimizer = Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.loss_func = nn.MSELoss()

    def forward(self, gps: torch.Tensor, velocity: torch.Tensor) -> List[torch.Tensor]:
        return [self.mlp(torch.cat([velocity, gps], dim=1))]

    def loss(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        return self.loss_func(pred[0], target[0]), {}
