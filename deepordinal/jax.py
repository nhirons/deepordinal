import jax
import jax.numpy as jnp
from flax import nnx

from deepordinal._common import check_output_dim, initial_raw_thresholds

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
    into *output_dim* class probabilities via learned, ordered thresholds
    ``t(1) < ... < t(K-1)``::

        P(y = k | x) = sigmoid(t(k+1) - logit) - sigmoid(t(k) - logit)

    where ``sigmoid(t(0) - logit)`` is 0 and ``sigmoid(t(K) - logit)`` is 1.

    The thresholds are ordered by construction. The layer stores
    ``raw_thresholds``: the first threshold, then the inverse softplus of each
    gap. ``interior_thresholds`` recovers them as
    ``cumsum([raw[0], softplus(raw[1:])])``, so no optimizer step can reorder
    them and the probabilities are never negative.

    The projection has no bias. Adding a constant to the logit and to every
    threshold gives the same model, so the thresholds absorb the bias.

    Thresholds initialize at ``logit(k/K)``, so a zero logit gives every class
    probability ``1/K``.

    Args:
        input_dim: Size of the input feature dimension.
        output_dim: Number of ordinal classes, at least 2.
        rngs: ``nnx.Rngs`` used to initialize the projection.
    """

    def __init__(self, input_dim, output_dim, *, rngs: nnx.Rngs):
        check_output_dim(output_dim)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear = nnx.Linear(input_dim, 1, use_bias=False, rngs=rngs)
        self.raw_thresholds = nnx.Param(jnp.asarray(initial_raw_thresholds(output_dim)))

    @property
    def interior_thresholds(self):
        """Ordered thresholds, shape ``(K-1,)``. Differentiable."""
        raw = self.raw_thresholds[...]
        return jnp.cumsum(jnp.concatenate([raw[:1], jax.nn.softplus(raw[1:])]))

    def logit(self, x):
        """Raw logit ``z(x)``, shape ``(..., 1)``. Pass this to the loss functions."""
        return self.linear(x)

    def __call__(self, x):
        logit = self.logit(x)  # (..., 1)
        t = self.interior_thresholds.astype(logit.dtype)
        cdf = jax.nn.sigmoid(t - logit)  # P(y <= k), (..., K-1)
        ones = jnp.ones_like(logit)
        return jnp.concatenate([cdf, ones], axis=-1) - jnp.concatenate([1 - ones, cdf], axis=-1)
