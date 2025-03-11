import numpy as np
import torch
import torch.nn as nn

from typing import Union
Activation = Union[str, nn.Module]

# device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
if torch.cuda.is_available():
    device = 'cuda:0'
    print("Using GPU")
elif torch.backends.mps.is_available():
    device = 'mps'
    print("Using MPS")
else:
    device = 'cpu'
    print("Using CPU")
device = torch.device(device)


def from_numpy(x: np.ndarray):
    return torch.from_numpy(x).type(torch.FloatTensor).to(device)


def to_numpy(x: torch.Tensor):
    return x.detach().cpu().numpy()


_str_to_activation = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'leaky_relu': nn.LeakyReLU(),
    'sigmoid': nn.Sigmoid(),
    'selu': nn.SELU(),
    'softplus': nn.Softplus(),
    'identity': nn.Identity(),
}


def build_mlp(
        input_size: int,
        output_size: int,
        n_layers: int,
        size: int,
        activation: Activation = 'tanh',
        output_activation: Activation = 'identity',
):
    """
        Builds a feedforward neural network

        arguments:
            input_placeholder: placeholder variable for the state (batch_size, input_size)
            scope: variable scope of the network

            n_layers: number of hidden layers
            size: dimension of each hidden layer
            activation: activation of each hidden layer

            input_size: size of the input layer
            output_size: size of the output layer
            output_activation: activation of the output layer

        returns:
            output_placeholder: the result of a forward pass through the hidden layers + the output layer
    """
    if isinstance(activation, str):
        activation = _str_to_activation[activation]
    if isinstance(output_activation, str):
        output_activation = _str_to_activation[output_activation]
    layers = []
    in_size = input_size
    for _ in range(n_layers):
        layers.append(nn.Linear(in_size, size))
        layers.append(activation)
        in_size = size
    layers.append(nn.Linear(in_size, output_size))
    layers.append(output_activation)
    return nn.Sequential(*layers).to(device)


if __name__ == '__main__':
    x = np.random.rand(2, 2, 2)
    print(x)
    x = from_numpy(x)
    print(x)
    x = to_numpy(x)
    print(x)
