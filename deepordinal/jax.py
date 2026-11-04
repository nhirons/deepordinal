import jax
import jax.numpy as jnp
from flax import nnx

__all__ = ["OrdinalOutput", "ordinal_loss", "ordistic_loss"]


def _penalty(z, name):
    if name == "hinge":
        return jnp.maximum(0.0, 1.0 - z)
    elif name == "smooth_hinge":
        return jnp.where(
            z >= 1,
            jnp.zeros_like(z),
            jnp.where(z > 0, (1 - z) ** 2 / 2, 0.5 - z),
        )
    elif name == "modified_least_squares":
        return jnp.where(z >= 1, jnp.zeros_like(z), (1 - z) ** 2)
    elif name == "logistic":
        return jax.nn.softplus(-z)
    else:
        raise ValueError(f"Unknown penalty: {name}")


def ordinal_loss(logits, targets, thresholds, construction="all", penalty="logistic"):
    """Rennie & Srebro ordinal loss (IJCAI 2005).

    Args:
        logits: (batch,) or (batch, 1): raw predictor output z(x).
        targets: (batch,): integer labels in [0, K).
        thresholds: (K-1,): sorted interior thresholds.
        construction: ``'all'`` or ``'immediate'``.
        penalty: ``'hinge'``, ``'smooth_hinge'``, ``'modified_least_squares'``,
            or ``'logistic'``.

    Returns:
        Scalar mean loss over the batch.
    """
    if construction not in ("all", "immediate"):
        raise ValueError(f"Unknown construction: {construction}")
    logits = jnp.reshape(jnp.asarray(logits), (-1,))
    targets = jnp.asarray(targets).astype(jnp.int32)[:, None]  # (batch, 1)
    thresholds = jnp.asarray(thresholds, dtype=logits.dtype)
    K = thresholds.shape[0] + 1
    # Threshold l (0-indexed) separates classes <= l from classes > l. The logit
    # should sit above threshold l when l < y and below it otherwise, so the
    # margin is s(l; y) * (theta_l - z) with s = -1 if l < y else +1 (eq 13).
    l_idx = jnp.arange(K - 1)  # (K-1,)
    signs = 1 - 2 * (l_idx < targets).astype(logits.dtype)  # (batch, K-1)
    margins = signs * (thresholds[None, :] - logits[:, None])  # (batch, K-1)
    penalties = _penalty(margins, penalty)
    if construction == "immediate":
        # eq 12 keeps only the thresholds bounding the target class: y-1 and y.
        # The first and last classes have one bounding threshold each.
        penalties = penalties * ((l_idx == targets - 1) | (l_idx == targets))
    loss = penalties.sum(axis=1)  # (batch,)

    return loss.mean()


def ordistic_loss(logits, targets, means, log_priors=None):
    """Ordistic loss (Rennie & Srebro, Section 4).

    Args:
        logits: (batch,) or (batch, 1): raw predictor output z(x).
        targets: (batch,): integer labels in [0, K).
        means: (K,): class means (mu_1=-1, mu_K=1 by convention; interior learned).
        log_priors: (K,) or None: log-prior terms pi_i. Defaults to zeros.

    Returns:
        Scalar mean negative log-likelihood over the batch.
    """
    logits = jnp.reshape(jnp.asarray(logits), (-1,))
    targets = jnp.asarray(targets).astype(jnp.int32)
    means = jnp.asarray(means, dtype=logits.dtype)
    K = means.shape[0]
    if log_priors is None:
        log_priors = jnp.zeros((K,), dtype=logits.dtype)
    else:
        log_priors = jnp.asarray(log_priors, dtype=logits.dtype)
    # energy_ik = mu_k * z_i + pi_k - mu_k^2 / 2
    energy = means[None, :] * logits[:, None] + log_priors[None, :] - means[None, :] ** 2 / 2
    # loss = -log P(y|z) = -energy[y] + log(sum_k exp(energy[k]))
    target_energy = jnp.take_along_axis(energy, targets[:, None], axis=1)[:, 0]
    log_partition = jax.nn.logsumexp(energy, axis=1)
    return (log_partition - target_energy).mean()


class OrdinalOutput(nnx.Module):
    """Ordinal regression output layer (Flax NNX).

    Projects an arbitrary input down to a single logit and converts it
    into *output_dim* class probabilities via learned, sorted thresholds.

    The layer learns ``output_dim - 1`` interior thresholds ``t(1), ..., t(K-1)``
    (with ``t(0) = -inf`` and ``t(K) = +inf`` fixed) and computes::

        P(y = k | x) = sigmoid(t(k+1) - logit) - sigmoid(t(k) - logit)

    Args:
        input_dim: Size of the input feature dimension.
        output_dim: Number of ordinal classes.
        rngs: ``nnx.Rngs`` used to initialize the parameters.
    """

    def __init__(self, input_dim, output_dim, *, rngs: nnx.Rngs):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear = nnx.Linear(input_dim, 1, rngs=rngs)
        init = jax.nn.initializers.glorot_uniform()
        t = init(rngs.params(), (1, output_dim - 1)).reshape(-1)
        self.interior_thresholds = nnx.Param(jnp.sort(t))

    def __call__(self, x):
        logit = self.linear(x)  # (batch, 1)
        t = self.interior_thresholds[...].astype(logit.dtype)
        cdf = jax.nn.sigmoid(t - logit)  # P(y <= k), (batch, K-1)
        ones = jnp.ones_like(logit)
        return jnp.concatenate([cdf, ones], axis=-1) - jnp.concatenate([1 - ones, cdf], axis=-1)
