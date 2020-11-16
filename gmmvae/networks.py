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


class LinearGaussian(nn.Module):
    """A fully connected neural network for parameterising a diagonal
    Gaussian distribution.

    :param in_dim (int): dimension of the input variable.
    :param out_dim (int): dimension of the output variable.
    :param hidden_dims (list, optional): dimensions of hidden layers.
    :param initial_sigma (float, optional): initial output sigma.
    :param initial_mu (float, optional): initial output mean.
    :param sigma (float, optional): if not None, sets the initial
    homoscedastic output sigma.
    :param train_sigma (bool, optional): whether to train the homoscedastic
    output sigma.
    :param min_sigma (float, optional): sets the minimum output sigma.
    :param nonlinearity (function, optional): non-linearity to apply in
    between layers.
    """
    def __init__(self, in_dim, out_dim, hidden_dims=(64, 64), min_sigma=0.,
                 nonlinearity=F.relu):
        super().__init__()

        self.out_dim = out_dim
        self.min_sigma = min_sigma
        self.network = LinearNN(in_dim, 2*out_dim, hidden_dims, nonlinearity)

    def forward(self, x):
        """Returns parameters of a diagonal Gaussian distribution."""
        x = self.network(x)
        mu = x[..., :self.out_dim]
        sigma = self.min_sigma + F.softplus(x[..., self.out_dim:])

        return mu, sigma


class LinearGaussianIndexed(nn.Module):

    def __init__(self, in_dim, out_dim, k, hidden_dims=(64, 64), min_sigma=0.,
                 nonlinearity=F.relu):
        super().__init__()

        self.out_dim = out_dim
        self.k = k
        self.min_sigma = min_sigma
        self.networks = nn.ModuleList()
        for _ in range(k):
            self.networks.append(LinearNN(in_dim, 2*out_dim, hidden_dims,
                                          nonlinearity))

    def forward(self, x, y):
        """Returns parameters of a diagonal Gaussian distribution."""
        output = torch.zeros((x.shape[0], 2*self.out_dim))
        for k in range(self.k):
            idx = torch.where(y == k)[0]
            output[idx] = self.networks[k](x[idx, :])

        mu = output[..., :self.out_dim]
        sigma = self.min_sigma + F.softplus(output[..., self.out_dim:])

        return mu, sigma
