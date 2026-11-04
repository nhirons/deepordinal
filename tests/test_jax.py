"""JAX-specific behavior. Shared behavior is in test_deepordinal.py."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
nnx = pytest.importorskip("flax.nnx")

from deepordinal.jax import OrdinalOutput, cumulative_link_loss, ordinal_loss, ordistic_loss


def test_layer_and_losses_jit():
    layer = OrdinalOutput(6, 4, rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.key(0), (8, 6))
    y = jax.numpy.arange(8) % 4
    np.testing.assert_allclose(nnx.jit(lambda m, x: m(x))(layer, x), layer(x), rtol=1e-6)
    z, t = layer.logit(x), layer.interior_thresholds
    for loss, third in [(ordinal_loss, t), (cumulative_link_loss, t), (ordistic_loss, jax.numpy.linspace(-1, 1, 4))]:
        np.testing.assert_allclose(jax.jit(loss)(z, y, third), loss(z, y, third), rtol=1e-6)
