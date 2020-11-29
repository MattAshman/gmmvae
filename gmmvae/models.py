import torch
import torch.nn as nn
import numpy as np

from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from torch.distributions.kl import kl_divergence

__all__ = ['VAE', 'EntroVAE', 'GMMVAE']


class VAE(nn.Module):
    def __init__(self, likelihood, variational_dist, z_dim):
        super().__init__()

        self.likelihood = likelihood
        self.variational_dist = variational_dist
        self.z_dim = z_dim

    def pz(self):
        pz_mu = torch.zeros(self.z_dim)
        pz_sigma = torch.ones(self.z_dim)
        pz = Normal(pz_mu, pz_sigma)

        return pz

    def qz(self, x):
        qz = self.variational_dist(x)

        return qz

    def elbo(self, x, num_samples=1):
        """Monte Carlo estimate of the evidence lower bound."""
        pz = self.pz()
        qz = self.qz(x)

        kl = kl_divergence(qz, pz).sum()

        # z_samples is shape (num_samples, batch, z_dim).
        z_samples = qz.rsample((num_samples,))

        log_px_z = 0
        for z in z_samples:
            log_px_z += self.likelihood.log_prob(z, x).sum()

        log_px_z /= num_samples
        elbo = (log_px_z - kl) / x.shape[0]

        return elbo

    def sample(self, num_samples=1, z=None, x=None):
        if z is None:
            if x is None:
                qz = self.pz()
            else:
                qz = self.qz(x)

            z = qz.sample((num_samples,))

        px_z = self.likelihood(z)
        x_samples = px_z.sample()

        return x_samples, px_z


class EntroVAE(VAE):
    def __init__(self, likelihood, variational_dist, z_dim, init_scale=1.):
        super().__init__(likelihood, variational_dist, z_dim)

        self.logscale = nn.Parameter(torch.ones(z_dim) * np.log(init_scale))

    def qz(self, x, h):
        qz = self.variational_dist(x)

        # Replace the scale with entropy scaled sigma.
        qz_sigma = h.unsqueeze(1).matmul(self.logscale.exp().unsqueeze(0))
        qz = Normal(qz.mean, qz_sigma)

        return qz

    def elbo(self, x, h, num_samples=1):
        """Monte Carlo estimate of the evidence lower bound."""
        pz = self.pz()
        qz = self.qz(x, h)

        kl = kl_divergence(qz, pz).sum()

        # z_samples is shape (num_samples, batch, z_dim).
        z_samples = qz.rsample((num_samples,))

        log_px_z = 0
        for z in z_samples:
            log_px_z += self.likelihood.log_prob(z, x).sum()

        log_px_z /= num_samples
        elbo = (log_px_z - kl) / x.shape[0]

        return elbo


class GMMVAE(VAE):

    def __init__(self, likelihood, variational_dist, z_dim, k,
                 init_sigma=1., diag=False):
        super().__init__(likelihood, variational_dist, z_dim)

        self.z_dim = z_dim
        self.k = k

        # Initialise GMM parameters.
        self.pz_y_mu = nn.Parameter(torch.randn((k, z_dim)),
                                    requires_grad=True)
        if diag:
            self.pz_y_logsigma = nn.Parameter(
                (torch.ones(k, 1) * init_sigma).log(), requires_grad=True)
        else:
            self.pz_y_logsigma = nn.Parameter(
                (torch.ones((k, z_dim)) * init_sigma).log(),
                requires_grad=True)

    def py_z(self, z, pi):
        # Compute the marginal likelihood, p(z) = \sum_k p(z|y)p(y).
        pzy = torch.zeros_like(pi)
        for k in range(self.k):
            pz_y = Normal(self.pz_y_mu[k, :], self.pz_y_logsigma[k, :].exp())
            pzy[:, k] = pz_y.log_prob(z).sum(1)
            pzy[:, k] += pi[:, k].log()

        pz = torch.logsumexp(pzy, dim=1)

        # Compute the posterior p(y|z) = p(z, y) / p(z)
        py_z = pzy - pz.unsqueeze(1)
        py_z = Categorical(py_z.exp())

        return py_z

    def elbo(self, x, pi, num_samples=1):
        """Monte Carlo estimate of the evidence lower bound."""
        qz = self.qz(x)

        # z_samples is shape (num_samples, batch, z_dim).
        z_samples = qz.rsample((num_samples,))

        log_px_z = 0
        kl_y = 0
        kl_z = 0
        for z in z_samples:
            log_px_z += self.likelihood.log_prob(z, x).sum()

            py_z = self.py_z(z, pi)
            kl_y += kl_divergence(py_z, Categorical(pi)).sum()

            for k in range(self.k):
                pz_y = Normal(self.pz_y_mu[k, :],
                              self.pz_y_logsigma[k, :].exp())

                kl_z_k = py_z.probs[:, k] * kl_divergence(qz, pz_y).sum(1)
                kl_z += kl_z_k.sum()

        log_px_z /= num_samples
        kl_y /= num_samples
        kl_z /= num_samples
        elbo = (log_px_z - kl_y - kl_z) / x.shape[0]

        return elbo

    def sample(self, num_samples=1, z=None, x=None, pi=None):
        if z is None:
            if x is None:
                # Sample p(y).
                if pi is None:
                    pi = torch.ones(self.k) / self.k

                py = Categorical(pi)
                y = py.sample((num_samples,))

                # Sample p(z|y).
                qz = Normal(self.pz_y_mu[y, :], self.pz_y_logsigma[y, :].exp())
            else:
                # Sample q(z).
                qz = self.qz(x)

            z = qz.sample((num_samples,))

        # Sample p(x|z).
        px_z = self.likelihood(z)
        x_samples = px_z.sample()

        return x_samples, px_z


class WeightedVAE(VAE):
    def __init__(self, likelihood, variational_dist, z_dim, k):
        super().__init__(likelihood, variational_dist, z_dim)

        self.k = k

        # Initialise parameters of K Gaussians.
        self.mu = nn.Parameter(torch.randn((k, z_dim)), requires_grad=True)
        self.logsigma = nn.Parameter(torch.ones(k, z_dim).log(),
                                     requires_grad=True)

    def pz_y(self, y):
        # Compute the prior probability p(z|y).
        sigma = y.matmul(self.logsigma.exp().pow(-2)).pow(-0.5)
        mu = sigma.pow(2) * y.matmul(self.logsigma.exp().pow(-2) * self.mu)
        pz_y = Normal(mu, sigma)

        return pz_y

    def elbo(self, x, y, num_samples=1):
        """Monte Carlo estimate of the evidence lower bound."""
        pz_y = self.pz_y(y)
        qz = self.qz(x)

        kl = kl_divergence(qz, pz_y).sum()

        # z_samples is shape (num_samples, batch, z_dim).
        z_samples = qz.rsample((num_samples,))

        log_px_z = 0
        for z in z_samples:
            log_px_z += self.likelihood.log_prob(z, x).sum()

        log_px_z /= num_samples
        elbo = (log_px_z - kl) / x.shape[0]

        return elbo

    def sample(self, num_samples=1, y=None, z=None, x=None):
        if z is None:
            if x is None:
                # Sample p(z|y).
                if y is None:
                    y = torch.ones(self.k) / self.k

                qz = self.pz_y(y)
            else:
                # Sample q(z).
                qz = self.qz(x)

            z = qz.sample((num_samples,))

        px_z = self.likelihood(z)
        x_samples = px_z.sample()

        return x_samples


class GMMVAEFixedCls(VAE):

    def __init__(self, likelihood, cls, variational_dist, z_dim, k,
                 init_sigma=1.):
        super().__init__(likelihood, variational_dist, z_dim)

        self.cls = cls
        self.z_dim = z_dim
        self.k = k

        # Initialise GMM parameters.
        self.pz_y_mu = nn.Parameter(torch.randn((k, z_dim)) * 0.1,
                                    requires_grad=True)
        self.pz_y_logsigma = nn.Parameter(
            (torch.ones((k, z_dim)) * init_sigma).log(), requires_grad=True)

    def qy(self, x, cls_output=None):
        if cls_output is None:
            cls_output = self.cls(x).detach()

        qy = Categorical(cls_output.exp())

        return qy

    def qz(self, x, y):
        qz = self.variational_dist(x, y)

        return qz

    def elbo(self, x, cls_output=None, pi=None, num_samples=1):
        """Monte Carlo estimate of the evidence lower bound."""
        qy = self.qy(x, cls_output)
        x = x.flatten(start_dim=1)

        log_px_z = 0
        for _ in range(num_samples):
            y = qy.sample()
            qz = self.qz(x, y)
            z = qz.rsample()
            log_px_z += self.likelihood.log_prob(z, x).sum()

        log_px_z /= num_samples

        if pi is None:
            pi = torch.ones(self.k) / self.k

        py = Categorical(pi)
        kl_y = kl_divergence(qy, py).sum()

        kl_z = 0
        for k in range(self.k):
            pz_y = Normal(self.pz_y_mu[k, :],
                          self.pz_y_logsigma[k, :].exp())
            qz = self.qz(x, torch.ones(x.shape[0]).fill_(k))

            kl_z_k = qy.probs[:, k] * kl_divergence(qz, pz_y).sum(1)
            kl_z += kl_z_k.sum()

        elbo = (log_px_z - kl_y - kl_z) / x.shape[0]

        return elbo


class GMMVAECls(GMMVAEFixedCls):

    def __init__(self, likelihood, cls, variational_dist, z_dim, k,
                 init_sigma=1.):
        super().__init__(likelihood, cls, variational_dist, z_dim, k,
                         init_sigma)

        self.cls_nloglikelihood = nn.functional.nll_loss

    def cls_nll(self, x, y):
        output = self.cls(x)

        return self.cls_nloglikelihood(output, y), output


class InfiniteGMMVAE(nn.Module):

    def __init__(self, likelihood_x, likelihood_z, variational_dist_z,
                 variational_dist_w, z_dim, w_dim, k):
        super().__init__()

        self.likelihood_x = likelihood_x
        self.likelihood_z = likelihood_z
        self.variational_dist_z = variational_dist_z
        self.variational_dist_w = variational_dist_w
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.k = k

    def pw(self):
        pw_mu = torch.zeros(self.w_dim)
        pw_sigma = torch.ones(self.w_dim)
        pw = Normal(pw_mu, pw_sigma)

        return pw

    def qz(self, x):
        qz = self.variational_dist_z(x)

        return qz

    def qw(self, x):
        qw = self.variational_dist_w(x)

        return qw

    def py_wz(self, z, w, pi):
        # Compute the marginal likelihood, p(z|w) = \sum_k p(z|w,y)p(y).
        pzy_w = torch.zeros_like(pi)
        for k in range(self.k):
            pz_wy_mu, pz_wy_sigma = self.decoder_z(
                w, torch.ones_like(w).fill_(k))
            pz_wy = Normal(pz_wy_mu, pz_wy_sigma)
            pzy_w[:, k] = pz_wy.log_prob(z).sum(1)
            pzy_w[:, k] += pi[:, k].log()

        pz_w = torch.logsumexp(pzy_w, dim=1)

        # Compute the posterior p(y|z) = p(z, y) / p(z)
        py_wz = pzy_w - pz_w.unsqueeze(1)
        py_wz = Categorical(py_wz.exp())

        return py_wz

    def elbo(self, x, pi, num_samples=1):
        """Monte Carlo estimate of the evidence lower bound."""
        qz = self.qz(x)
        qw = self.qw(x)
        pw = self.pw()

        # z_samples is shape (num_samples, batch, z_dim).
        z_samples = qz.rsample((num_samples,))
        w_samples = qw.rsample((num_samples,))

        log_px_z = 0
        kl_y = 0
        kl_z = 0
        for z, w in zip(z_samples, w_samples):
            log_px_z += self.likelihood_x.log_prob(z, x).sum()

            py_wz = self.py_wz(z, w, pi)
            kl_y += kl_divergence(py_wz, Categorical(pi)).sum()

            for k in range(self.k):
                pz_wy = self.likelihood_z(w, torch.ones(w.shape[0]).fill_(k))
                kl_z_k = py_wz.probs[:, k] * kl_divergence(qz, pz_wy).sum(1)
                kl_z += kl_z_k.sum()

        log_px_z /= num_samples
        kl_y /= num_samples
        kl_z /= num_samples
        kl_w = kl_divergence(qw, pw).sum()
        elbo = (log_px_z - kl_y - kl_z - kl_w) / x.shape[0]

        return elbo

    def sample(self, num_samples=1, w=None, z=None, x=None, pi=None):
        if z is None:
            if x is None:
                # Sample p(y).
                if pi is None:
                    pi = torch.ones(self.k) / self.k

                py = Categorical(pi)
                y = py.sample((num_samples,))

                if w is None:
                    # Sample p(w).
                    pw = self.pw()
                    w = pw.sample((num_samples,))

                # Sample p(z|w,y).
                qz = self.likelihood_z(w, y)
            else:
                # Sample q(z).
                qz = self.qz(x)

            z = qz.sample((num_samples,))

        # Sample p(x|z).
        px_z = self.loglikelihood_x(z)
        x_samples = px_z.sample()

        return x_samples


class HierarchicalVAE(VAE):
    def __init__(self, likelihood_x, likelihood_z, variational_dist, z_dim):
        super().__init__(likelihood_x, variational_dist, z_dim)

        self.likelihood_z = likelihood_z

    def pz_y(self, y):
        pz_y = self.likelihood_z(y)

        return pz_y

    def qz_x(self, x):
        qz_x = self.variational_dist(x)

        return qz_x

    def qz(self, x, y):
        pz_y = self.pz_y(y)
        qz_x = self.qz_x(x)
        pz_y_mu, pz_y_sigma = pz_y.mean, pz_y.stddev
        qz_x_mu, qz_x_sigma = qz_x.mean, qz_x.stddev

        # Convert to natural parameters.
        pz_y_np2 = -0.5 * pz_y_sigma.pow(-2)
        pz_y_np1 = pz_y_mu * pz_y_sigma.pow(-2)
        qz_x_np2 = -0.5 * qz_x_sigma.pow(-2)
        qz_x_np1 = qz_x_mu * qz_x_sigma.pow(-2)

        # Combine natural parameters and convert back.
        qz_np1 = pz_y_np1 + qz_x_np1
        qz_np2 = pz_y_np2 + qz_x_np2
        qz_sigma = (-0.5 * qz_np2.pow(-1)).pow(0.5)
        qz_mu = qz_sigma.pow(2) * qz_np1

        qz = Normal(qz_mu, qz_sigma)

        return qz


    def elbo(self, x, y, num_samples=1):
        """Monte Carlo estimate of the evidence lower bound."""
        pz_y = self.pz_y(y)
        qz = self.qz(x, y)

        kl = kl_divergence(qz, pz_y).sum()

        # z_samples is shape (num_samples, batch, z_dim).
        z_samples = qz.rsample((num_samples,))

        log_px_z = 0
        for z in z_samples:
            log_px_z += self.likelihood.log_prob(z, x).sum()

        log_px_z /= num_samples
        elbo = (log_px_z - kl) / x.shape[0]

        return elbo

    def sample(self, y, num_samples=1):
        pz_y = self.pz_y(y)
        z = pz_y.sample((num_samples,))
        px_z = self.likelihood(z)
        x_samples = px_z.sample()

        return x_samples, px_z


class MultiModalVAE(nn.Module):
    def __init__(self, likelihood_x, likelihood_y, variational_dist_x,
                 variational_dist_y, z_dim):
        super().__init__()

        self.likelihood_x = likelihood_x
        self.likelihood_y = likelihood_y
        self.variational_dist_x = variational_dist_x
        self.variational_dist_y = variational_dist_y
        self.z_dim = z_dim

    def pz(self):
        pz_mu = torch.zeros(self.z_dim)
        pz_sigma = torch.ones(self.z_dim)
        pz = Normal(pz_mu, pz_sigma)

        return pz

    def qz_x(self, x):
        qz_x = self.variational_dist_x(x)

        return qz_x

    def qz_y(self, y):
        qz_y = self.variational_dist_y(y)

        return qz_y

    def qz(self, x, y):
        qz_x = self.qz_x(x)
        qz_y = self.qz_y(y)
        qz_x_mu, qz_x_sigma = qz_x.mean, qz_x.stddev
        qz_y_mu, qz_y_sigma = qz_y.mean, qz_y.stddev

        # Convert to natural parameters.
        qz_x_np2 = -0.5 * qz_x_sigma.pow(-2)
        qz_x_np1 = qz_x_mu * qz_x_sigma.pow(-2)
        qz_y_np2 = -0.5 * qz_y_sigma.pow(-2)
        qz_y_np1 = qz_y_mu * qz_y_sigma.pow(-2)

        # Combine natural parameters and convert back.
        qz_np1 = qz_x_np1 + qz_y_np1
        qz_np2 = qz_x_np2 + qz_y_np2
        qz_sigma = (-0.5 * qz_np2.pow(-1)).pow(0.5)
        qz_mu = qz_sigma.pow(2) * qz_np1

        qz = Normal(qz_mu, qz_sigma)

        return qz

    def elbo(self, x, y, num_samples=1):
        """Monte Carlo estimate of the evidence lower bound."""
        pz = self.pz()
        qz = self.qz(x, y)
        qz_y = self.qz_y(y)
        qz_x = self.qz_x(x)

        kl = kl_divergence(qz, pz).sum()
        kl_y = kl_divergence(qz_y, pz).sum()
        kl_x = kl_divergence(qz_x, pz).sum()

        # z_samples is shape (num_samples, batch, z_dim).
        z_samples = qz.rsample((num_samples,))
        z_y_samples = qz_y.rsample((num_samples,))
        z_x_samples = qz_x.rsample((num_samples,))

        log_pxy_z = 0
        log_px_z = 0
        log_py_z = 0
        for z, z_y, z_x in zip(z_samples, z_y_samples, z_x_samples):
            log_pxy_z += self.likelihood_x.log_prob(z, x).sum()
            log_pxy_z += self.likelihood_y.log_prob(z, y).sum()
            log_py_z += self.likelihood_x.log_prob(z_y, x).sum()
            log_px_z += self.likelihood_y.log_prob(z_x, y).sum()

        log_pxy_z /= num_samples
        elbo = (log_pxy_z - kl) / x.shape[0]

        return elbo

    def sample(self, num_samples=1, x=None, y=None):
        if x is None and y is None:
            qz = self.pz()
        elif x is None and y is not None:
            qz = self.qz_y(y)
        elif x is not None and y is None:
            qz = self.qz_x(x)
        else:
            qz = self.qz(x, y)

        z = qz.sample((num_samples,))
        px_z = self.likelihood_x(z)
        py_z = self.likelihood_y(z)
        x_samples, y_samples = px_z.sample(), py_z.sample()

        return x_samples, y_samples, px_z, py_z
