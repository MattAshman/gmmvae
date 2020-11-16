import torch
import torch.nn as nn
import numpy as np

from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from torch.distributions.kl import kl_divergence

__all__ = ['VAE', 'EntroVAE', 'GMMVAE']


class VAE(nn.Module):
    def __init__(self, encoder, loglikelihood, z_dim):
        super().__init__()

        self.encoder = encoder
        self.loglikelihood = loglikelihood
        self.z_dim = z_dim

    def pz(self, x):
        pz_mu = torch.zeros(x.shape[0], self.z_dim)
        pz_sigma = torch.ones(x.shape[0], self.z_dim)
        pz = Normal(pz_mu, pz_sigma)

        return pz

    def qz(self, x):
        qz_mu, qz_sigma = self.encoder(x)
        qz = Normal(qz_mu, qz_sigma)

        return qz

    def elbo(self, x, num_samples=1):
        """Monte Carlo estimate of the evidence lower bound."""
        pz = self.pz(x)
        qz = self.qz(x)

        kl = kl_divergence(qz, pz).sum()

        # z_samples is shape (num_samples, batch, z_dim).
        z_samples = qz.rsample((num_samples,))

        log_px_z = 0
        for z in z_samples:
            log_px_z += self.loglikelihood(z, x).sum()

        log_px_z /= num_samples
        elbo = (log_px_z - kl) / x.shape[0]

        return elbo

    def sample(self, num_samples=1):
        z = torch.randn((num_samples, self.z_dim))
        samples = self.loglikelihood.predict(z)

        return samples

    def predict_x(self, z):
        x = self.loglikelihood.predict(z)

        return x

    def reconstruct_x(self, x):
        z, _ = self.encoder(x)
        x_recon = self.loglikelihood.predict(z)

        return x_recon


class EntroVAE(VAE):
    def __init__(self, encoder, loglikelihood, z_dim, init_scale=1.):
        super().__init__(encoder, loglikelihood, z_dim)

        self.logscale = nn.Parameter(torch.ones(z_dim) * np.log(init_scale))

    def qz(self, x, h):
        qz_mu = self.encoder(x)[0]
        qz_sigma = h.unsqueeze(1).matmul(self.logscale.exp().unsqueeze(0))
        qz = Normal(qz_mu, qz_sigma)

        return qz

    def elbo(self, x, h, num_samples=1):
        """Monte Carlo estimate of the evidence lower bound."""
        pz = self.pz(x)
        qz = self.qz(x, h)

        kl = kl_divergence(qz, pz).sum()

        # z_samples is shape (num_samples, batch, z_dim).
        z_samples = qz.rsample((num_samples,))

        log_px_z = 0
        for z in z_samples:
            log_px_z += self.loglikelihood(z, x).sum()

        log_px_z /= num_samples
        elbo = (log_px_z - kl) / x.shape[0]

        return elbo


class GMMVAE(nn.Module):

    def __init__(self, loglikelihood, encoder, z_dim, k, init_sigma=1.):
        super().__init__()

        self.loglikelihood = loglikelihood
        self.encoder = encoder
        self.z_dim = z_dim
        self.k = k

        # Initialise GMM parameters.
        self.pz_y_mu = nn.Parameter(torch.randn((k, z_dim)) * 0.1,
                                    requires_grad=True)
        self.pz_y_logsigma = nn.Parameter(
            (torch.ones((k, z_dim)) * init_sigma).log(), requires_grad=True)

    def qz(self, x):
        qz_mu, qz_sigma = self.encoder(x)
        qz = Normal(qz_mu, qz_sigma)

        return qz

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
            log_px_z += self.loglikelihood(z, x).sum()

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

    def sample(self, pi=None, num_samples=1):
        if pi is None:
            pi = torch.ones(self.k) / self.k

        # Sample p(y).
        py = Categorical(pi)
        y = py.sample((num_samples,))

        # Sample p(z|y).
        pz_y = Normal(self.pz_y_mu[y, :], self.pz_y_logsigma[y, :].exp())
        z = pz_y.sample()

        # Sample p(x|z).
        samples = self.loglikelihood.predict(z)

        return samples

    def predict_x(self, z):
        x = self.loglikelihood.predict(z)

        return x

    def reconstruct_x(self, x):
        z, _ = self.encoder(x)
        x_recon = self.loglikelihood.predict(z)

        return x_recon


class GMMVAEFixedCls(nn.Module):

    def __init__(self, loglikelihood, cls, encoder, z_dim, k, init_sigma=1.):
        super().__init__()

        self.loglikelihood = loglikelihood
        self.cls = cls
        self.encoder = encoder
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
        qz_mu, qz_sigma = self.encoder(x, y)
        qz = Normal(qz_mu, qz_sigma)

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
            log_px_z += self.loglikelihood(z, x).sum()

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


class GMMVAECls(nn.Module):

    def __init__(self, loglikelihood, cls, encoder, z_dim, k, init_sigma=1.):
        super().__init__()

        self.loglikelihood = loglikelihood
        self.cls = cls
        self.encoder = encoder
        self.z_dim = z_dim
        self.k = k
        self.cls_nloglikelihood = nn.functional.nll_loss

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
        qz_mu, qz_sigma = self.encoder(x, y)
        qz = Normal(qz_mu, qz_sigma)

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
            log_px_z += self.loglikelihood(z, x).sum()

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

    def cls_nll(self, x, y):
        output = self.cls(x)

        return self.cls_nloglikelihood(output, y), output


class InfiniteGMMVAE(nn.Module):

    def __init__(self, loglikelihood, decoder_z, encoder_z, encoder_w, z_dim,
                 w_dim, k):
        super().__init__()

        self.loglikelhood = loglikelihood
        self.decoder_z = decoder_z
        self.encoder_z = encoder_z
        self.encoder_w = encoder_w
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.k = k

    def pw(self):
        pw_mu = torch.zeros(self.w_dim)
        pw_sigma = torch.ones(self.w_dim)
        pw = Normal(pw_mu, pw_sigma)

        return pw

    def qz(self, x):
        qz_mu, qz_sigma = self.encoder_z(x)
        qz = Normal(qz_mu, qz_sigma)

        return qz

    def qw(self, x):
        qw_mu, qw_sigma = self.encoder_w(x)
        qw = Normal(qw_mu, qw_sigma)

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
            log_px_z += self.loglikelihood(z, x).sum()

            py_wz = self.py_wz(z, w, pi)
            kl_y += kl_divergence(py_wz, Categorical(pi)).sum()

            for k in range(self.k):
                pz_wy_mu, pz_wy_sigma = self.decoder_z(
                    w, torch.ones_like(w).fill_(k))
                pz_wy = Normal(pz_wy_mu, pz_wy_sigma)

                kl_z_k = py_wz.probs[:, k] * kl_divergence(qz, pz_wy).sum(1)
                kl_z += kl_z_k.sum()

        log_px_z /= num_samples
        kl_y /= num_samples
        kl_z /= num_samples
        kl_w = kl_divergence(qw, pw).sum()
        elbo = (log_px_z - kl_y - kl_z - kl_w) / x.shape[0]

        return elbo

    def sample(self, pi=None, num_samples=1):
        if pi is None:
            pi = torch.ones(self.k) / self.k

        # Sample p(y).
        py = Categorical(pi)
        y = py.sample((num_samples,))

        # Sample p(w).
        pw = self.pw()
        w = pw.sample((num_samples,))

        # Sample p(z|w,y).
        pz_wy_mu, pz_wy_sigma = self.decoder_z(w, y)
        pz_wy = Normal(pz_wy_mu, pz_wy_sigma)
        z = pz_wy.sample()

        # Sample p(x|z).
        samples = self.loglikelihood.predict(z)

        return samples
