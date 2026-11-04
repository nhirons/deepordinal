"""Backend-free helpers shared by every backend."""

import numpy as np


def check_output_dim(output_dim):
    if output_dim < 2:
        raise ValueError(f"output_dim must be at least 2, got {output_dim}")


def initial_raw_thresholds(output_dim):
    """Raw parameters whose thresholds are ``logit(k/K)``: uniform probabilities at a zero logit.

    The layout is the first threshold, then the inverse softplus of each gap.
    """
    k = np.arange(1, output_dim)
    thresholds = np.log(k / (output_dim - k))
    gaps = np.diff(thresholds)
    return np.concatenate([thresholds[:1], gaps + np.log(-np.expm1(-gaps))]).astype(np.float32)
