import numpy as np


def check_output_dim(output_dim):
    if output_dim < 2:
        raise ValueError(f"output_dim must be at least 2, got {output_dim}")


def initial_raw_thresholds(output_dim):
    """Raw threshold parameters whose ordered thresholds are logit(k/K).

    With these thresholds a zero logit assigns every class probability 1/K.
    The raw vector stores the first threshold followed by the inverse softplus
    of each gap, matching ``t = cumsum([raw[0], softplus(raw[1:])])``.
    """
    k = np.arange(1, output_dim, dtype=np.float64)
    thresholds = np.log(k / (output_dim - k))
    gaps = np.diff(thresholds)
    inv_softplus = gaps + np.log(-np.expm1(-gaps))
    return np.concatenate([thresholds[:1], inv_softplus]).astype(np.float32)
