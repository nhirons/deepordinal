import math

import tensorflow as tf
from tensorflow.keras import initializers
from tensorflow.keras.layers import Layer

from deepordinal._common import check_output_dim, initial_raw_thresholds

__all__ = ["OrdinalOutput", "SortedInitializer", "cumulative_link_loss", "ordinal_loss", "ordistic_loss"]

def _penalty(z, name):
    if name == "hinge":
        return tf.maximum(0.0, 1.0 - z)
    elif name == "smooth_hinge":
        return tf.where(
            z >= 1.0,
            tf.zeros_like(z),
            tf.where(z > 0.0, (1.0 - z) ** 2 / 2.0, 0.5 - z),
        )
    elif name == "modified_least_squares":
        return tf.where(z >= 1.0, tf.zeros_like(z), (1.0 - z) ** 2)
    elif name == "logistic":
        return tf.math.softplus(-z)
    else:
        raise ValueError(f"Unknown penalty: {name}")


def ordinal_loss(logits, targets, thresholds, construction="all", penalty="logistic"):
    """Rennie & Srebro ordinal loss (IJCAI 2005).

    Args:
        logits: (batch,) or (batch, 1) — raw predictor output z(x).
        targets: (batch,) — integer labels in [0, K).
        thresholds: (K-1,) — sorted interior thresholds.
        construction: ``'all'`` or ``'immediate'``.
        penalty: ``'hinge'``, ``'smooth_hinge'``, ``'modified_least_squares'``,
            or ``'logistic'``.

    Returns:
        Scalar mean loss over the batch.
    """
    if construction not in ("all", "immediate"):
        raise ValueError(f"Unknown construction: {construction}")
    logits = tf.cast(tf.reshape(logits, [-1]), tf.float32)
    targets = tf.expand_dims(tf.cast(targets, tf.int32), 1)  # (batch, 1)
    thresholds = tf.cast(thresholds, tf.float32)
    K = thresholds.shape[0] + 1
    # Threshold l (0-indexed) separates classes <= l from classes > l. The logit
    # should sit above threshold l when l < y and below it otherwise, so the
    # margin is s(l; y) * (theta_l - z) with s = -1 if l < y else +1 (eq 13).
    l_idx = tf.expand_dims(tf.range(K - 1), 0)  # (1, K-1)
    signs = 1.0 - 2.0 * tf.cast(l_idx < targets, tf.float32)  # (batch, K-1)
    margins = signs * (tf.expand_dims(thresholds, 0) - tf.expand_dims(logits, 1))
    penalties = _penalty(margins, penalty)
    if construction == "immediate":
        # eq 12 keeps only the thresholds bounding the target class: y-1 and y.
        # The first and last classes have one bounding threshold each.
        mask = tf.logical_or(tf.equal(l_idx, targets - 1), tf.equal(l_idx, targets))
        penalties = penalties * tf.cast(mask, tf.float32)
    loss = tf.reduce_sum(penalties, axis=1)

    return tf.reduce_mean(loss)


def ordistic_loss(logits, targets, means, log_priors=None):
    """Ordistic loss (Rennie & Srebro, Section 4).

    Args:
        logits: (batch,) or (batch, 1) — raw predictor output z(x).
        targets: (batch,) — integer labels in [0, K).
        means: (K,) — class means.
        log_priors: (K,) or None — log-prior terms. Defaults to zeros.

    Returns:
        Scalar mean negative log-likelihood over the batch.
    """
    logits = tf.cast(tf.reshape(logits, [-1]), tf.float32)
    targets = tf.cast(targets, tf.int32)
    means = tf.cast(means, tf.float32)
    K = means.shape[0]
    if log_priors is None:
        log_priors = tf.zeros([K], dtype=tf.float32)
    else:
        log_priors = tf.cast(log_priors, tf.float32)
    energy = (
        tf.expand_dims(means, 0) * tf.expand_dims(logits, 1)
        + tf.expand_dims(log_priors, 0)
        - tf.expand_dims(means, 0) ** 2 / 2.0
    )
    batch_idx = tf.range(tf.shape(targets)[0])
    indices = tf.stack([batch_idx, targets], axis=1)
    target_energy = tf.gather_nd(energy, indices)
    log_partition = tf.reduce_logsumexp(energy, axis=1)
    return tf.reduce_mean(log_partition - target_energy)


_LOG2 = math.log(2.0)


def _log1mexp(a):
    # log(1 - exp(-a)) for a > 0 (Maechler 2012)
    return tf.where(a < _LOG2, tf.math.log(-tf.math.expm1(-a)), tf.math.log1p(-tf.exp(-a)))


def _log_probs(logit, thresholds):
    """log P(y = k | z) for the cumulative logit model, shape ``(..., K)``.

    ``logit`` has shape ``(..., 1)``. With ``a(k) = t(k) - z``::

        log P(0)     = log sigmoid(a(1))
        log P(K-1)   = log sigmoid(-a(K-1))
        log P(k)     = log sigmoid(a(k+1)) + log sigmoid(-a(k)) + log(1 - exp(a(k) - a(k+1)))

    The middle identity is exact, and every term is a softplus or a
    ``log1mexp``, so nothing cancels.
    """
    a = thresholds - logit  # (..., K-1)
    first = -tf.math.softplus(-a[..., :1])
    last = -tf.math.softplus(a[..., -1:])
    hi, lo = a[..., 1:], a[..., :-1]
    mid = -tf.math.softplus(-hi) - tf.math.softplus(lo) + _log1mexp(hi - lo)
    return tf.concat([first, mid, last], axis=-1)


def cumulative_link_loss(logits, targets, thresholds):
    """Negative log-likelihood of the cumulative logit model.

    This is the likelihood of the model ``OrdinalOutput`` represents, also
    called the proportional-odds or ordered logit model::

        P(y = k | z) = sigmoid(t(k+1) - z) - sigmoid(t(k) - z)

    The log-probabilities are computed without subtracting probabilities,
    so they stay accurate when every ``sigmoid(t(k) - z)`` rounds to 0 or 1.

    Args:
        logits: (batch,) or (batch, 1): raw predictor output z(x).
        targets: (batch,): integer labels in [0, K).
        thresholds: (K-1,): strictly increasing interior thresholds.

    Returns:
        Scalar mean negative log-likelihood over the batch.
    """
    logits = tf.cast(tf.reshape(logits, [-1, 1]), tf.float32)
    targets = tf.cast(tf.reshape(targets, [-1]), tf.int32)
    thresholds = tf.cast(thresholds, tf.float32)
    log_p = _log_probs(logits, thresholds)  # (batch, K)
    return -tf.reduce_mean(tf.gather(log_p, targets, axis=1, batch_dims=1))


class SortedInitializer(initializers.Initializer):
    """Wraps a Keras initializer and returns its output sorted along the last axis."""

    def __init__(self, base="glorot_uniform"):
        self.base = initializers.get(base)

    def __call__(self, shape, dtype=None):
        values = self.base(shape, dtype=dtype)
        return tf.sort(values, axis=-1)

    def get_config(self):
        return {"base": initializers.serialize(self.base)}


class OrdinalOutput(Layer):
    """Ordinal regression output layer (Keras).

    Projects an arbitrary input down to a single logit and converts it
    into *output_dim* class probabilities via learned, ordered thresholds
    ``t(1) <= ... <= t(K-1)``::

        P(y = k | x) = sigmoid(t(k+1) - logit) - sigmoid(t(k) - logit)

    where ``sigmoid(t(0) - logit)`` is 0 and ``sigmoid(t(K) - logit)`` is 1.

    The thresholds are ordered by construction. The layer stores
    ``raw_thresholds``: the first threshold, then the inverse softplus of each
    gap. ``interior_thresholds`` recovers them as
    ``cumsum([raw[0], softplus(raw[1:])])``, so no optimizer step can reorder
    them and the probabilities are never negative. Gaps are positive in exact
    arithmetic; a raw gap below about -100 underflows to 0 in float32, which
    ties two thresholds and gives the class between them probability 0.

    The projection has no bias. Adding a constant to the logit and to every
    threshold gives the same model, so the thresholds absorb the bias.

    Thresholds initialize at ``logit(k/K)``, so a zero logit gives every class
    probability ``1/K``.

    Args:
        output_dim: Number of ordinal classes, at least 2.
    """

    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        check_output_dim(output_dim)
        self.output_dim = output_dim

    def build(self, input_shape):
        self.kernel = self.add_weight(
            name="kernel",
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform",
            trainable=True,
        )
        raw = initial_raw_thresholds(self.output_dim)
        self.raw_thresholds = self.add_weight(
            name="raw_thresholds",
            shape=(self.output_dim - 1,),
            initializer=lambda shape, dtype=None: tf.constant(raw, dtype=dtype),
            trainable=True,
        )
        super().build(input_shape)

    @property
    def interior_thresholds(self):
        """Ordered thresholds, shape ``(K-1,)``. Differentiable."""
        raw = self.raw_thresholds
        return tf.cumsum(tf.concat([raw[:1], tf.math.softplus(raw[1:])], axis=0))

    def logit(self, inputs):
        """Raw logit ``z(x)``, shape ``(..., 1)``. Pass this to the loss functions.

        Unlike ``layer(inputs)``, this method bypasses ``Layer.__call__``, so it
        builds the layer and casts inputs to the compute dtype itself.
        """
        inputs = tf.convert_to_tensor(inputs)
        if not self.built:
            self.build(inputs.shape)
        return tf.matmul(tf.cast(inputs, self.compute_dtype), self.kernel)

    def log_probs(self, inputs):
        """Log class probabilities, shape ``(..., K)``. Numerically stable."""
        return _log_probs(self.logit(inputs), self.interior_thresholds)

    def call(self, inputs):
        return tf.exp(self.log_probs(inputs))

    def compute_output_shape(self, input_shape):
        return (*input_shape[:-1], self.output_dim)

    def get_config(self):
        config = super().get_config()
        config["output_dim"] = self.output_dim
        return config
