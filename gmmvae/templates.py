import torch
import torch.nn as nn

from torch.nn import functional as F
from torch.distributions import Normal
from .likelihoods import Likelihood, Bernoulli, HomoGaussian


class MNISTClassificationNet(nn.Module):
    def __init__(self, nonlinearity=F.relu):
        super().__init__()

        self.nonlinearity = nonlinearity
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

        self.nloglikelihood = F.nll_loss

    def forward(self, x):
        # Convolutional layers.
        x = self.conv1(x)
        x = self.nonlinearity(x)
        x = self.conv2(x)
        x = self.nonlinearity(x)
        x = F.max_pool2d(x, 2)

        # Linear layers.
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.nonlinearity(x)
        x = self.dropout2(x)
        x = self.fc2(x)

        output = F.log_softmax(x, dim=1)

        return output

    def nll(self, x, y):
        output = self.forward(x)

        return self.nloglikelihood(output, y), output


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class CELEBAVariationalDist(Likelihood):
    def __init__(self, out_dim, hidden_dim=256, min_sigma=1e-3):
        super().__init__()

        # Setup the three linear transformations used.
        self.out_dim = out_dim
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(128, hidden_dim, 4, 1),
            nn.ReLU(True),
            View((-1, hidden_dim * 1 * 1)),
            nn.Linear(hidden_dim, out_dim * 2)
        )

        self.min_sigma = min_sigma

    def forward(self, x):
        output = self.network(x)
        mu = output[..., :self.out_dim]
        raw_sigma = output[..., self.out_dim:]
        sigma = F.softplus(raw_sigma).clamp(min=self.min_sigma)
        px = Normal(mu, sigma)

        return px


class CELEBALikelihood(Likelihood):
    def __init__(self, in_dim, hidden_dim=256):
        super().__init__()

        # Setup the two linear transformations used.
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            View((-1, hidden_dim, 1, 1)),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_dim, 128, 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1)
        )

        # TODO: what's the appropriate size here?
        self.likelihood = HomoGaussian()

    def forward(self, z):
        mu = self.network(z)

        return self.likelihood(mu)


class CIFAR10VariationalDist(Likelihood):
    def __init__(self, out_dim, hidden_dim=256, min_sigma=1e-3):
        super().__init__()

        self.out_dim = out_dim
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(128, hidden_dim, 4, 1),
            nn.ReLU(True),
            View((-1, hidden_dim * 1 * 1)),
            nn.Linear(hidden_dim, out_dim * 2)
        )

        self.min_sigma = min_sigma

    def forward(self, x):
        output = self.network(x)
        mu = output[..., :self.out_dim]
        raw_sigma = output[..., self.out_dim:]
        sigma = F.softplus(raw_sigma).clamp(min=self.min_sigma)
        px = Normal(mu, sigma)

        return px


class CIFAR10Likelihood(Likelihood):
    def __init__(self, in_dim, hidden_dim=256, min_sigma=1e-3):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            View((-1, hidden_dim, 1, 1)),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_dim, 128, 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1)
        )

        # TODO: what's the appropriate size here?
        self.likelihood = HomoGaussian(min_sigma=min_sigma)

    def forward(self, z):
        mu = self.network(z)

        return self.likelihood(mu)
