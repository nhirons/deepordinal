"""Cross-backend parity: every installed backend computes the same numbers."""

import itertools

import numpy as np
import pytest

BACKENDS = {}
try:
    import torch
    import deepordinal.torch as dt

    BACKENDS["torch"] = (
        lambda fn, *a, **kw: fn(*[torch.as_tensor(x) for x in a], **kw).detach().numpy(),
        dt,
    )
except ImportError:
    pass
try:
    import tensorflow as tf
    import deepordinal.tf as dtf

    BACKENDS["tf"] = (lambda fn, *a, **kw: fn(*[tf.constant(x) for x in a], **kw).numpy(), dtf)
except ImportError:
    pass
try:
    import jax.numpy as jnp
    import deepordinal.jax as dj

    BACKENDS["jax"] = (lambda fn, *a, **kw: np.asarray(fn(*[jnp.asarray(x) for x in a], **kw)), dj)
except ImportError:
    pass

if len(BACKENDS) < 2:
    pytest.skip("parity tests need at least two backends", allow_module_level=True)

rng = np.random.default_rng(0)
K = 5
LOGITS = (rng.normal(size=64) * 3).astype(np.float32)
TARGETS = rng.integers(0, K, size=64).astype(np.int64)
THRESHOLDS = np.sort(rng.normal(size=K - 1)).astype(np.float32)
MEANS = np.sort(rng.normal(size=K)).astype(np.float32)


def _all_equal(results):
    ref_name, ref = next(iter(results.items()))
    for name, val in results.items():
        np.testing.assert_allclose(val, ref, rtol=1e-5, atol=1e-6, err_msg=f"{name} vs {ref_name}")


@pytest.mark.parametrize(
    "construction,penalty",
    list(itertools.product(["all", "immediate"], ["hinge", "smooth_hinge", "modified_least_squares", "logistic"])),
)
def test_ordinal_loss_parity(construction, penalty):
    _all_equal({
        name: run(mod.ordinal_loss, LOGITS, TARGETS, THRESHOLDS, construction=construction, penalty=penalty)
        for name, (run, mod) in BACKENDS.items()
    })


def test_ordistic_loss_parity():
    _all_equal({
        name: run(mod.ordistic_loss, LOGITS, TARGETS, MEANS) for name, (run, mod) in BACKENDS.items()
    })


def test_cumulative_link_loss_parity():
    _all_equal({
        name: run(mod.cumulative_link_loss, LOGITS, TARGETS, THRESHOLDS) for name, (run, mod) in BACKENDS.items()
    })
