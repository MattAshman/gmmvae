import torch.nn as nn

from torch.nn import functional as F
from torch.distributions import Categorical
from .base import Likelihood
from .networks import LinearNN

__all__ = ['Categorical', 'NNCategorical']


class Categorical(Likelihood):

    def __init__(self):
        super().__init__()

    def forward(self, z):
        return Categorical(F.softmax(z, dim=1))

    def log_prob(self, z, y):
        log_prob = F.log_softmax(z, dim=1)
        return (y * log_prob).sum(dim=1)


class NNCategorical(nn.Module):
    """A fully connected neural network for parameterising a discrete
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
        self.likelihood = Categorical()

    def forward(self, z):
        f = self.network(z)

        return self.likelihood(f)

    def log_prob(self, z, y):
        f = self.network(z)

        return self.likelihood.log_prob(f, y)
