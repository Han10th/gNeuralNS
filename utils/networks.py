import random
import torch
from torch import nn

def multiple_FNN(N,layer_sizes, activation='tanh', initialize = 'zero',device='cpu'):
    listofFNN = []
    for i in range(N):
        listofFNN += [FNN(layer_sizes, activation, initialize).to(device)]
    return listofFNN
class FNN(nn.Module):
    def __init__(self, layer_sizes, activation='tanh', initialize = 'zero'):
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

        self.layers = nn.Sequential()
        for i in range(1, len(layer_sizes)):
            self.layers.append(
                Layer(
                    layer_sizes[i - 1], layer_sizes[i],
                    activation=activation[i - 1],
                    initialize=initialize[i - 1],
                    dtype=torch.float32
                )
            )

    def forward(self, X):
        return self.layers(X)
class SFNN(nn.Module):
    def __init__(self, layer_sizes, n_output, activation='tanh', initialize = 'zero', device='cpu'):
        super(SFNN, self).__init__()
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

        self.layers = torch.nn.ModuleList([])
        for k in range(n_output):
            layer = nn.Sequential()
            for i in range(1, len(layer_sizes)):
                layer.append(
                    Layer(
                        layer_sizes[i - 1], layer_sizes[i],
                        activation=activation[i - 1],
                        initialize=initialize[i - 1],
                        dtype=torch.float32
                    )
                )
            self.layers += [layer]
    def forward(self, X):
        return torch.cat([layer(X) for layer in self.layers], dim=1)
class PFNN(nn.Module):
    def __init__(self, activation='tanh', initialize = 'zero', device='cpu'):
        super(PFNN, self).__init__()
        ##   ERROR PROCESSING   ##
        n_output = len(layer_sizes[-1])

        self.layers = nn.Sequential()
        for i in range(1, len(layer_sizes)):
            prev_layer_size = layer_sizes[i - 1]
            curr_layer_size = layer_sizes[i]
            if isinstance(curr_layer_size, (list, tuple)):
                if isinstance(prev_layer_size, (list, tuple)):
                    self.layers.append(
                        torch.nn.ModuleList([
                            Layer(
                                prev_layer_size[j], curr_layer_size[j],
                                activation=activation[i - 1],
                                initialize=initialize[i - 1],
                                dtype=torch.float32
                            )
                            for j in range(n_output)
                        ])
                    )
                else:  # e.g. 64 -> [8, 8, 8]
                    self.layers.append(
                        torch.nn.ModuleList([
                            Layer(
                                prev_layer_size, curr_layer_size[j],
                                activation=activation[i - 1],
                                initialize=initialize[i - 1],
                                dtype=torch.float32
                            )
                            for j in range(n_output)
                        ])
                    )
            else:
                self.layers.append(
                    Layer(
                        prev_layer_size, curr_layer_size,
                        activation=activation[i - 1],
                        initialize=initialize[i - 1],
                        dtype=torch.float32
                    )
                )
    def forward(self, X):
        x = X
        for layer in self.layers[:-1]:
            if isinstance(layer, torch.nn.ModuleList):
                if isinstance(x, list):
                    x = [f(x_) for f, x_ in zip(layer, x)]
                else:
                    x = [f(x) for f in layer]
            else:
                x = layer(x)

        # output layers
        if isinstance(x, list):
            x = torch.cat([f(x_) for f, x_ in zip(self.layers[-1], x)], dim=1)
        else:
            x = self.layers[-1](x)
        return x
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