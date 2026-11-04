"""Keras-specific behavior. Shared behavior is in test_deepordinal.py."""

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from deepordinal.tf import OrdinalOutput


def test_config_round_trip():
    layer = OrdinalOutput.from_config(OrdinalOutput(4).get_config())
    assert layer.output_dim == 4


def test_works_in_sequential_model():
    model = tf.keras.Sequential([tf.keras.Input((6,)), tf.keras.layers.Dense(8), OrdinalOutput(4)])
    assert model.predict(np.zeros((3, 6), np.float32), verbose=0).shape == (3, 4)


def test_logit_builds_and_casts_like_call():
    # logit() bypasses Layer.__call__, which normally builds the layer and casts inputs.
    layer = OrdinalOutput(4)
    h = tf.random.normal((2, 3), dtype=tf.float64)
    logit = layer.logit(h)
    assert layer.built and logit.dtype == tf.float32
    np.testing.assert_allclose(logit.numpy(), tf.matmul(tf.cast(h, tf.float32), layer.kernel).numpy())


def test_losses_compute_in_logits_dtype():
    from deepordinal.tf import cumulative_link_loss, ordinal_loss, ordistic_loss

    z, y = tf.constant([0.5, -1.0], tf.float64), tf.constant([0, 2])
    t = tf.constant([-1.0, 1.0])  # float32, cast to the logits' float64
    for loss in (ordinal_loss(z, y, t), cumulative_link_loss(z, y, t), ordistic_loss(z, y, tf.constant([-1.0, 0.0, 1.0]))):
        assert loss.dtype == tf.float64
