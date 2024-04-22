import random
import torch
from torch import nn

class FNN(nn.Module):
    def __init__(self, layer_sizes, activation='tanh', initialize = 'zero', device='cpu'):
        super(FNN, self).__init__()
        layer_depth = len(layer_sizes) - 1
        if isinstance(activation, list):
            if not layer_depth == len(activation):
                raise ValueError(
                    "Total number of activation functions do not match with sum of hidden layers and output layer!"
                )
        else:
            activation = (layer_depth-1)*[activation] + ['none']

        if isinstance(initialize, list):
            if not layer_depth == len(initialize):
                raise ValueError(
                    "Total number of initialize strategy do not match with sum of hidden layers and output layer!"
                )
        else:
            initialize = (layer_depth - 1) * ['none'] + [initialize]

        self.linears = nn.Sequential()
        for i in range(1, len(layer_sizes)):
            self.linears.append(
                Layer(
                    layer_sizes[i - 1], layer_sizes[i],
                    activation=activation[i - 1],
                    initialize=initialize[i - 1],
                    dtype=torch.float32
                )
            )

    def forward(self, X):
        return self.linears(X)

class Layer(nn.Module):
    def __init__(self, in_channels, out_channels, activation='tanh',initialize='none',dtype=torch.float32):
        super().__init__()

        self.layer = nn.Sequential(nn.Linear(
            in_channels, out_channels
        ))

        if activation == 'relu':
            self.layer.append(nn.ReLU())
        elif activation == 'tanh':
            self.layer.append(nn.Tanh())
        elif activation == 'leakyrelu':
            self.layer.append(nn.LeakyReLU())
        elif activation == 'sigmoid':
            self.layer.append(nn.Sigmoid())
        elif activation == 'identity':
            self.layer.append(nn.Identity())

        self.Initial_param(initialize=initialize)

    def Initial_param(self, initialize='none'):
        if initialize=='none':
            for name, param in self.layer.named_parameters():
                if name.endswith('weight'):
                    nn.init.xavier_normal_(param)
                elif name.endswith('bias'):
                    nn.init.zeros_(param)
        elif initialize=='zero':
            for name, param in self.layer.named_parameters():
                if name.endswith('weight'):
                    nn.init.zeros_(param)
                elif name.endswith('bias'):
                    nn.init.zeros_(param)

    def forward(self, x):
        return self.layer(x)