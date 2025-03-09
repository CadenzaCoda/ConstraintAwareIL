import numpy as np
import os
from abc import abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Tuple, Dict
from utils import pytorch_util as ptu
from utils.data_util import EfficientReplayBuffer

import torch
import torch.nn as nn

from loguru import logger


class BaseModel(nn.Module):
    feature_fields = ()
    additional_input_fields = ()
    label_fields = ()

    def __init__(self):
        super().__init__()

    def step_schedule(self):
        pass

    @property
    def model_name(self):
        return self.__class__.__name__

    def reset(self):
        pass

    def parse_carla_obs(self, obs, info):
        try:
            out = [obs[field] for field in self.feature_fields] + [info[field] for field in
                                                                    self.additional_input_fields]
            return out
        except KeyError as e:
            logger.error(f"e")
            logger.error(f"Available fields are: {list(obs.keys()) + list(info.keys())}")

    def get_action(self, *args):
        self.eval()
        ret = [ptu.to_numpy(x)[0] for x in self(*[ptu.from_numpy(arg.copy()[None]) for arg in args])]
        return ret if len(ret) > 1 else ret[0]

    def predict(self, obs, info, batched=False):
        self.eval()
        inputs = self.parse_obs(obs, info)

        with torch.no_grad():
            for i, field in enumerate(inputs):
                inputs[i] = ptu.from_numpy(np.asarray(field))
                if not batched:
                    inputs[i] = inputs[i][None]
            output = self(*inputs)
            for i, field in enumerate(output):
                output[i] = ptu.to_numpy(field)
                if not batched:
                    output[i] = output[i][0]
            return output

    @abstractmethod
    def loss(self, pred, label) -> Tuple[torch.Tensor, Dict[str, float]]:
        raise NotImplementedError

    def fit(self, train_dataset: 'EfficientReplayBuffer', n_epochs, val_dataset=None):
        train_examples, train_loss = 0, 0.
        val_examples, val_loss = 0, 0.

        train_loader = train_dataset.dataloader(batch_size=64, shuffle=True, num_workers=0,
                                                manifest=[self.feature_fields, self.label_fields])
        val_loader = val_dataset.dataloader(batch_size=64, shuffle=False, num_workers=0,
                                            manifest=[self.feature_fields,
                                                      self.label_fields]) if val_dataset is not None else None
        train_scores = defaultdict(lambda: 0.)
        val_scores = defaultdict(lambda: 0.) if val_dataset is not None else None

        for epoch in range(n_epochs):
            self.train()
            for features, labels in train_loader:
                pred = self(*features)
                loss, train_info = self.loss(pred, labels)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                self.optimizer.step()
                train_loss += loss.item()
                train_examples += 1
                for k, v in train_info.items():
                    train_scores[k] += v
            if val_loader is None:
                continue
            self.eval()
            with torch.no_grad():
                for features, labels in val_loader:
                    pred = self(*features)
                    loss, val_info = self.loss(pred, labels)
                    val_loss += loss.item()
                    val_examples += 1
                    for k, v in val_info.items():
                        val_scores[k] += v
        self.scheduler.step()
        info = {
            'train': {f'{self.model_name}_loss': train_loss / train_examples,
                      **{f'{self.model_name}_{k}': v / train_examples for k, v in train_scores.items()}}
        }
        if val_loader is not None:
            info['val'] = {f'{self.model_name}_loss': val_loss / val_examples,
                           **{f'{self.model_name}_{k}': v / val_examples for k, v in val_scores.items()}}
        return info

    def export(self, path=None, name=None):
        if not Path(path).exists():
            os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), Path(path) / f"{self.model_name}{f'_{name}' if name is not None else ''}.pt")

    def load(self, path=None, name=None):
        file_path = Path(path) / f"{self.model_name}{f'_{name}' if name is not None else ''}.pt"
        if not file_path.exists():
            logger.warning('Weight files not found.')
            return
        self.load_state_dict(torch.load(file_path, weights_only=True, map_location=ptu.device))
        logger.info(f"Weights loaded successfully from {os.path.join(path, name)}!")
