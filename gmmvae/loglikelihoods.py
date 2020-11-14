import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.distributions.normal import Normal
from .networks import LinearNN

__all__ = ['NNHomoGaussian', 'NNHeteroGaussian', 'NNBernoulli']


class HomoGaussian(nn.Module):

    def __init__(self, dim, sigma=None, sigma_grad=True, min_sigma=1e-3):
        super().__init__()

        self.min_sigma = min_sigma

        if sigma is None:
            self.log_sigma = nn.Parameter(torch.zeros(dim),
                                          requires_grad=sigma_grad)
        else:
            self.log_sigma = nn.Parameter(torch.onees(dim) * np.log(sigma),
                                          requires_grad=sigma_grad)

    def forward(self, mu, x):
        sigma = self.log_sigma.exp().clamp(min=self.min_sigma)
        px = Normal(mu, sigma)

        return px.log_prob(x)

    def predict(self, mu):
        sigma = self.log_sigma.exp().clamp(min=self.min_sigma)
        sigma = torch.ones_like(mu) * sigma

        return mu, sigma


class Bernoulli(nn.Module):

    def __init__(self):
        super().__init__()

        self.loglikelihood = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, f, y):
        return -self.loglikelihood(f, y)

    def predict(self, f):
        return torch.sigmoid(f)


class NNHomoGaussian(nn.Module):
    """A fully connected neural network for parameterising a diagonal
    Gaussian distribution with homoscedastic noise.
    :param in_dim (int): dimension of the input variable.
    :param out_dim (int): dimension of the output variable.
    :param hidden_dims (list, optional): dimensions of hidden layers.
    :param sigma (float, optional): if not None, sets the initial
    homoscedastic output sigma.
    :param sigma_grad (bool, optional): whether to train the homoscedastic
    output sigma.
    :param min_sigma (float, optional): sets the minimum output sigma.
    :param nonlinearity (function, optional): non-linearity to apply in
    between layers.
    """
    def __init__(self, in_dim, out_dim, hidden_dims=(64, 64), sigma=None,
                 sigma_grad=True, min_sigma=1e-3, nonlinearity=F.relu):
        super().__init__()

        self.network = LinearNN(in_dim, out_dim, hidden_dims, nonlinearity)
        self.loglikelhood = HomoGaussian(out_dim, sigma, sigma_grad, min_sigma)

    def forward(self, z, x):
        mu = self.network(z)

        return self.loglikelhood(mu, x)

    def predict(self, z):
        mu = self.network(z)

        return self.loglikelhood.predict(mu)


class NNHeteroGaussian(nn.Module):
    """A fully connected neural network for parameterising a diagonal
    Gaussian distribution with heteroscedastic noise.
    :param in_dim (int): dimension of the input variable.
    :param out_dim (int): dimension of the output variable.
    :param hidden_dims (list, optional): dimensions of hidden layers.
    :param sigma (float, optional): if not None, sets the initial
    homoscedastic output sigma.
    :param sigma_grad (bool, optional): whether to train the homoscedastic
    output sigma.
    :param min_sigma (float, optional): sets the minimum output sigma.
    :param nonlinearity (function, optional): non-linearity to apply in
    between layers.
    """
    def __init__(self, in_dim, out_dim, hidden_dims=(64, 64), min_sigma=1e-3,
                 nonlinearity=F.relu):
        super().__init__()

        self.out_dim = out_dim
        self.min_sigma = min_sigma
        self.network = LinearNN(in_dim, 2 * out_dim, hidden_dims, nonlinearity)

    def forward(self, z, x):
        output = self.network(z)
        mu = output[..., :self.out_dim]
        raw_sigma = output[..., self.out_dim:]
        sigma = F.softplus(raw_sigma) + self.min_sigma
        px = Normal(mu, sigma)

        return px.log_prob(x)

    def predict(self, z):
        output = self.network(z)
        mu = output[..., :self.out_dim]
        raw_sigma = output[..., self.out_dim:]
        sigma = F.softplus(raw_sigma) + self.min_sigma

        return mu, sigma


class NNBernoulli(nn.Module):
    """A fully connected neural network for parameterising a Bernoulli
    distribution.
    :param in_dim (int): dimension of the input variable.
    :param out_dim (int): dimension of the output variable.
    :param hidden_dims (list, optional): dimensions of hidden layers.
    :param nonlinearity (function, optional): non-linearity to apply in
    between layers.
    """
    def __init__(self, in_dim, out_dim, hidden_dims=(64, 64),
                 nonlinearity=F.relu):
        super().__init__()

        self.network = LinearNN(in_dim, out_dim, hidden_dims, nonlinearity)
        self.loglikelihood = Bernoulli()

    def forward(self, z, x):
        f = self.network(z)

        return self.loglikelihood(f, x)

    def predict(self, z):
        f = self.network(z)

        return self.loglikelihood.predict(f)
