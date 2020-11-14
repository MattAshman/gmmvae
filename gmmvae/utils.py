import numpy as np


def gaussian_diagonal_ll(x, m, v):
    sd = (x - m) ** 2
    ll = -0.5 * (2 * np.pi * v).log() - 0.5 * v.pow(-1) * sd

    # Sum over dimensions.
    ll = ll.sum(1)

    return ll


def gaussian_diagonal_kl(m1, v1, m2, v2):
    kl = 0.5 * ((v2 / v1).log() + (v1 + (m1 - m2) ** 2) / v2 - 1)

    # Sum over dimensions.
    kl = kl.sum(1)

    return kl
