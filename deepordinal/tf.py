"""TensorFlow/Keras backend."""

import math

import tensorflow as tf

from deepordinal._common import check_output_dim, initial_raw_thresholds

__all__ = ["OrdinalOutput", "cumulative_link_loss", "ordinal_loss", "ordistic_loss"]


def _penalty(z, name):
    zero = tf.zeros_like(z)
    if name == "logistic":
        return tf.math.softplus(-z)
    if name == "hinge":
        return tf.nn.relu(1 - z)
    if name == "smooth_hinge":
        return tf.where(z >= 1, zero, tf.where(z > 0, (1 - z) ** 2 / 2, 0.5 - z))
    if name == "modified_least_squares":
        return tf.where(z >= 1, zero, (1 - z) ** 2)
    raise ValueError(f"Unknown penalty: {name}")


def _nll(log_p, targets):
    y = tf.cast(tf.reshape(targets, [-1]), tf.int32)
    return -tf.reduce_mean(tf.gather(log_p, y, axis=1, batch_dims=1))


def _as_float(x, like=None):
    x = tf.convert_to_tensor(x)
    return x if like is None else tf.cast(x, like.dtype)


def ordinal_loss(logits, targets, thresholds, construction="all", penalty="logistic"):
    """Threshold ordinal loss (Rennie & Srebro 2005, eqs 12 and 13).

    Args:
        logits: (batch,) or (batch, 1): raw predictor output z, from ``layer.logit``.
        targets: (batch,): integer labels in [0, K).
        thresholds: (K-1,): ordered thresholds, from ``layer.interior_thresholds``.
        construction: ``"all"`` penalizes every threshold (eq 13).
            ``"immediate"`` penalizes only the two bounding the target class (eq 12).
        penalty: ``"logistic"``, ``"hinge"``, ``"smooth_hinge"``, or
            ``"modified_least_squares"`` (eqs 5 to 9).

    Returns:
        Scalar mean loss over the batch.
    """
    if construction not in ("all", "immediate"):
        raise ValueError(f"Unknown construction: {construction}")
    z = tf.reshape(_as_float(logits), [-1, 1])
    y = tf.reshape(tf.cast(targets, tf.int32), [-1, 1])
    thresholds = _as_float(thresholds, like=z)
    # Threshold l separates classes <= l from classes > l, so the margin is
    # positive when z is on the correct side of it.
    l = tf.range(tf.shape(thresholds)[0])
    penalties = _penalty(tf.where(l < y, z - thresholds, thresholds - z), penalty)
    if construction == "immediate":
        penalties = penalties * tf.cast((l == y - 1) | (l == y), z.dtype)
    return tf.reduce_mean(tf.reduce_sum(penalties, axis=1))


def ordistic_loss(logits, targets, means, log_priors=None):
    """Ordistic loss (Rennie & Srebro 2005, section 4).

    Negative log-likelihood of ``P(y = k | z)`` proportional to
    ``exp(means[k] * z + log_priors[k] - means[k] ** 2 / 2)``.

    Args:
        logits: (batch,) or (batch, 1): raw predictor output z.
        targets: (batch,): integer labels in [0, K).
        means: (K,): class means. The paper fixes the first at -1 and the last at 1.
        log_priors: (K,) or None: log-prior terms. Defaults to zeros.

    Returns:
        Scalar mean negative log-likelihood over the batch.
    """
    z = tf.reshape(_as_float(logits), [-1, 1])
    means = _as_float(means, like=z)
    energy = means * z - means**2 / 2
    if log_priors is not None:
        energy = energy + _as_float(log_priors, like=z)
    return _nll(tf.nn.log_softmax(energy, axis=1), targets)


def cumulative_link_loss(logits, targets, thresholds):
    """Negative log-likelihood of the cumulative logit model ``OrdinalOutput`` represents.

    Args:
        logits: (batch,) or (batch, 1): raw predictor output z, from ``layer.logit``.
        targets: (batch,): integer labels in [0, K).
        thresholds: (K-1,): increasing thresholds, from ``layer.interior_thresholds``.

    Returns:
        Scalar mean negative log-likelihood over the batch.
    """
    z = tf.reshape(_as_float(logits), [-1, 1])
    return _nll(_log_probs(z, _as_float(thresholds, like=z)), targets)


def _log1mexp(x):
    """log(1 - exp(-x)) for x > 0 (Maechler 2012)."""
    return tf.where(x < math.log(2), tf.math.log(-tf.math.expm1(-x)), tf.math.log1p(-tf.exp(-x)))


def _log_probs(logit, thresholds):
    """``log P(y = k | z)``, shape ``(..., K)``, for logit shape ``(..., 1)``.

    With ``a = thresholds - z``, the exact identity
    ``sigmoid(a[k]) - sigmoid(a[k-1]) = sigmoid(a[k]) * sigmoid(-a[k-1]) * (1 - exp(a[k-1] - a[k]))``
    turns every term into a softplus or a ``log1mexp``, so nothing cancels.
    A plain difference of sigmoids rounds to 0 in float32 once ``|z|`` passes about 17.
    """
    a = thresholds - logit
    first = -tf.math.softplus(-a[..., :1])
    last = -tf.math.softplus(a[..., -1:])
    hi, lo = a[..., 1:], a[..., :-1]
    middle = -tf.math.softplus(-hi) - tf.math.softplus(lo) + _log1mexp(hi - lo)
    return tf.concat([first, middle, last], axis=-1)


class OrdinalOutput(tf.keras.layers.Layer):
    """Ordinal output layer: features, then one logit z, then K class probabilities.

    ``P(y = k | x) = sigmoid(t[k] - z) - sigmoid(t[k-1] - z)``, where the first
    term is 1 for the last class and the second is 0 for the first class.

    The K-1 thresholds ``t = interior_thresholds`` are ordered by construction.
    The trainable ``raw_thresholds`` hold the first threshold and the inverse
    softplus of each gap. They start at ``logit(k/K)``, so a zero logit gives
    every class probability 1/K. The projection has no bias, because shifting
    z and every threshold together leaves the model unchanged.

    Args:
        output_dim: number of classes K, at least 2.
    """

    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        check_output_dim(output_dim)
        self.output_dim = output_dim

    def build(self, input_shape):
        raw = initial_raw_thresholds(self.output_dim)
        self.kernel = self.add_weight(name="kernel", shape=(input_shape[-1], 1), initializer="glorot_uniform")
        self.raw_thresholds = self.add_weight(
            name="raw_thresholds",
            shape=raw.shape,
            initializer=lambda shape, dtype=None: tf.constant(raw, dtype=dtype),
        )
        super().build(input_shape)

    @property
    def interior_thresholds(self):
        raw = self.raw_thresholds
        return tf.cumsum(tf.concat([raw[:1], tf.math.softplus(raw[1:])], axis=0))

    def logit(self, inputs):
        # Bypasses Layer.__call__, so build and cast to the compute dtype here.
        inputs = tf.convert_to_tensor(inputs)
        if not self.built:
            self.build(inputs.shape)
        return tf.matmul(tf.cast(inputs, self.compute_dtype), self.kernel)

    def log_probs(self, inputs):
        return _log_probs(self.logit(inputs), self.interior_thresholds)

    def call(self, inputs):
        return tf.exp(self.log_probs(inputs))

    def compute_output_shape(self, input_shape):
        return (*input_shape[:-1], self.output_dim)

    def get_config(self):
        return {**super().get_config(), "output_dim": self.output_dim}
