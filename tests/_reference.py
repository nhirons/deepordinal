"""Backend-free float64 reference for the cumulative logit model."""

import numpy as np


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def log_probs(logits, thresholds):
    """log P(y = k | z), shape (batch, K), accurate in float64 for |z| up to ~700."""
    z = np.asarray(logits, np.float64).reshape(-1, 1)
    t = np.asarray(thresholds, np.float64)
    a = np.concatenate([np.full((z.shape[0], 1), -np.inf), t[None] - z, np.full((z.shape[0], 1), np.inf)], 1)
    lo, hi = a[:, :-1], a[:, 1:]
    # sigmoid(hi) - sigmoid(lo) == sigmoid(-lo) - sigmoid(-hi); use the side away from 1.
    with np.errstate(over="ignore"):
        p = np.where(lo > 0, _sigmoid(-lo) - _sigmoid(-hi), _sigmoid(hi) - _sigmoid(lo))
    return np.log(p)
