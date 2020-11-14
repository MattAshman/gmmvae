import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['LinearNN']


class LinearNN(nn.Module):
    """A fully connected neural network.
    :param in_dim: An int, dimension of the input variable.
    :param out_dim: An int, dimension of the output variable.
    :param hidden_dims: A list, dimensions of hidden layers.
    :param nonlinearity: A function, the non-linearity to apply in between
    layers.
    """
    def __init__(self, in_dim, out_dim, hidden_dims=(64, 64),
                 nonlinearity=F.relu):
        super().__init__()

        self.nonlinearity = nonlinearity
        self.layers = nn.ModuleList()
        for i in range(len(hidden_dims) + 1):
            if i == 0:
                self.layers.append(nn.Linear(in_dim, hidden_dims[i]))
            elif i == len(hidden_dims):
                self.layers.append(nn.Linear(hidden_dims[i-1], out_dim))
            else:
                self.layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))

        # Weight initialisation.
        for layer in self.layers:
            nn.init.xavier_normal_(layer.weight)
            if layer.bias.data is not None:
                nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        """Returns output of the network.
        :param x: A Tensor, input of shape [M, in_dim].
        """
        for layer in self.layers[:-1]:
            x = self.nonlinearity(layer(x))

        x = self.layers[-1](x)
        return x
