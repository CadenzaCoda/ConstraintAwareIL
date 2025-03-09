import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18, ResNet18_Weights

from utils import pytorch_util as ptu


class SafetyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = ptu.build_mlp(
            
        )
