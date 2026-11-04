import tensorflow as tf
from tensorflow.keras import initializers
from tensorflow.keras.layers import Layer

__all__ = ["OrdinalOutput", "SortedInitializer", "ordinal_loss", "ordistic_loss"]

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
    """Ordinal regression output layer.

    Projects an arbitrary input down to a single logit and converts it
    into *output_dim* class probabilities via learned, sorted thresholds.

    The layer learns ``output_dim - 1`` interior thresholds ``t(1)…t(K-1)``
    (with ``t(0) = -∞`` and ``t(K) = +∞`` fixed) and computes::

        P(y = k | x) = σ(t(k+1) - logit) - σ(t(k) - logit)
    """

    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim

    def build(self, input_shape):
        self.kernel = self.add_weight(
            name="kernel",
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.bias = self.add_weight(
            name="bias",
            shape=(1,),
            initializer="zeros",
            trainable=True,
        )
        self.interior_thresholds = self.add_weight(
            name="thresholds",
            shape=(1, self.output_dim - 1),
            initializer=SortedInitializer("glorot_uniform"),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        logit = tf.matmul(inputs, self.kernel) + self.bias  # (batch, 1)
        cdf = tf.sigmoid(self.interior_thresholds - logit)  # P(y <= k), (batch, K-1)
        ones = tf.ones_like(logit)
        return tf.concat([cdf, ones], axis=-1) - tf.concat([1.0 - ones, cdf], axis=-1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_config(self):
        config = super().get_config()
        config["output_dim"] = self.output_dim
        return config
