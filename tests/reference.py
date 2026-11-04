"""Independent float64 NumPy reference, written per sample from the paper's equations."""

import numpy as np

PENALTIES = {
    "logistic": lambda z: np.logaddexp(0.0, -z),
    "hinge": lambda z: max(0.0, 1.0 - z),
    "smooth_hinge": lambda z: 0.0 if z >= 1 else (1 - z) ** 2 / 2 if z > 0 else 0.5 - z,
    "modified_least_squares": lambda z: 0.0 if z >= 1 else (1 - z) ** 2,
}


def ordinal_loss(z, y, t, construction, penalty):
    f = PENALTIES[penalty]
    total = 0.0
    for zi, yi in zip(np.ravel(z).astype(float), y):
        if construction == "all":  # eq 13
            total += sum(f(zi - tl) if l < yi else f(tl - zi) for l, tl in enumerate(t))
        else:  # eq 12: only the thresholds bounding class yi
            total += (f(zi - t[yi - 1]) if yi > 0 else 0.0) + (f(t[yi] - zi) if yi < len(t) else 0.0)
    return total / len(y)


def ordistic_loss(z, y, means, log_priors=None):
    log_priors = np.zeros(len(means)) if log_priors is None else np.asarray(log_priors, float)
    means = np.asarray(means, float)
    total = 0.0
    for zi, yi in zip(np.ravel(z).astype(float), y):
        energy = means * zi + log_priors - means**2 / 2
        total += np.logaddexp.reduce(energy) - energy[yi]
    return total / len(y)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def log_probs(z, t):
    """log P(y = k | z), shape (batch, K). Subtracts on the side away from 1 to stay accurate."""
    z = np.asarray(z, float).reshape(-1, 1)
    inf = np.full_like(z, np.inf)
    a = np.concatenate([-inf, np.asarray(t, float) - z, inf], axis=1)
    lo, hi = a[:, :-1], a[:, 1:]
    with np.errstate(over="ignore"):
        return np.log(np.where(lo > 0, _sigmoid(-lo) - _sigmoid(-hi), _sigmoid(hi) - _sigmoid(lo)))


def cumulative_link_loss(z, y, t):
    return -log_probs(z, t)[np.arange(len(y)), y].mean()
