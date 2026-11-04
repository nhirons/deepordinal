import pytest

jax = pytest.importorskip("jax")
nnx = pytest.importorskip("flax.nnx")

import jax.numpy as jnp
import numpy as np

from deepordinal.jax import OrdinalOutput, _penalty, ordinal_loss, ordistic_loss


def _x(shape, seed=0):
    return jax.random.normal(jax.random.key(seed), shape)


def test_output_shape():
    layer = OrdinalOutput(input_dim=8, output_dim=5, rngs=nnx.Rngs(0))
    out = layer(_x((4, 8)))
    assert out.shape == (4, 5)


def test_probabilities_sum_to_one():
    layer = OrdinalOutput(input_dim=4, output_dim=3, rngs=nnx.Rngs(0))
    out = layer(_x((16, 4)))
    np.testing.assert_allclose(out.sum(axis=-1), np.ones(16), atol=1e-5)


def test_probabilities_non_negative():
    layer = OrdinalOutput(input_dim=4, output_dim=6, rngs=nnx.Rngs(0))
    out = layer(_x((32, 4)))
    assert bool((out >= 0).all())


def test_thresholds_initialized_sorted():
    layer = OrdinalOutput(input_dim=4, output_dim=5, rngs=nnx.Rngs(0))
    t = layer.interior_thresholds
    assert t.shape == (4,)
    np.testing.assert_array_equal(t, jnp.sort(t))


def test_gradients_flow():
    layer = OrdinalOutput(input_dim=4, output_dim=3, rngs=nnx.Rngs(0))
    x = _x((8, 4))
    grads = nnx.grad(lambda m: m(x)[:, 0].sum())(layer)
    assert jnp.any(grads.linear.kernel[...] != 0)
    assert jnp.any(grads.raw_thresholds[...] != 0)


def test_seed_reproducibility():
    a = OrdinalOutput(input_dim=4, output_dim=3, rngs=nnx.Rngs(42))
    b = OrdinalOutput(input_dim=4, output_dim=3, rngs=nnx.Rngs(42))
    x = _x((4, 4))
    np.testing.assert_allclose(a(x), b(x))


def test_thresholds_stay_ordered_for_any_raw_values():
    layer = OrdinalOutput(input_dim=4, output_dim=8, rngs=nnx.Rngs(0))
    layer.raw_thresholds[...] = _x((7,), seed=3) * 10
    t = layer.interior_thresholds
    assert bool((t[1:] >= t[:-1]).all())
    assert bool((layer(_x((64, 4)) * 10) >= 0).all())


def test_initial_thresholds_give_uniform_probabilities_at_zero_logit():
    layer = OrdinalOutput(input_dim=4, output_dim=5, rngs=nnx.Rngs(0))
    np.testing.assert_allclose(layer(jnp.zeros((3, 4))), np.full((3, 5), 0.2), atol=1e-6)


def test_no_bias():
    layer = OrdinalOutput(input_dim=4, output_dim=3, rngs=nnx.Rngs(0))
    assert layer.linear.bias is None
    leaves = jax.tree_util.tree_leaves(nnx.state(layer, nnx.Param))
    assert sorted(leaf.shape for leaf in leaves) == [(2,), (4, 1)]  # raw_thresholds, kernel


def test_logit_method():
    layer = OrdinalOutput(input_dim=4, output_dim=3, rngs=nnx.Rngs(0))
    x = _x((5, 4))
    np.testing.assert_allclose(layer.logit(x), layer.linear(x))
    assert layer.logit(x).shape == (5, 1)


def test_output_dim_too_small_raises():
    with pytest.raises(ValueError, match="at least 2"):
        OrdinalOutput(input_dim=4, output_dim=1, rngs=nnx.Rngs(0))


def test_extra_leading_dims():
    layer = OrdinalOutput(input_dim=4, output_dim=5, rngs=nnx.Rngs(0))
    out = layer(_x((2, 3, 4)))
    assert out.shape == (2, 3, 5)
    np.testing.assert_allclose(out.sum(axis=-1), np.ones((2, 3)), atol=1e-5)


def test_single_sample():
    layer = OrdinalOutput(input_dim=2, output_dim=4, rngs=nnx.Rngs(0))
    out = layer(_x((1, 2)))
    assert out.shape == (1, 4)
    np.testing.assert_allclose(out.sum(), 1.0, atol=1e-5)


def test_jit():
    layer = OrdinalOutput(input_dim=4, output_dim=3, rngs=nnx.Rngs(0))
    x = _x((8, 4))
    np.testing.assert_allclose(nnx.jit(lambda m, x: m(x))(layer, x), layer(x), atol=1e-6)


# --- Loss function tests ---


class TestPenaltyFunctions:
    z = jnp.array([-1.0, 0.0, 0.5, 1.0, 2.0])

    def test_hinge_values(self):
        np.testing.assert_allclose(_penalty(self.z, "hinge"), [2.0, 1.0, 0.5, 0.0, 0.0])

    def test_smooth_hinge_values(self):
        np.testing.assert_allclose(_penalty(self.z, "smooth_hinge"), [1.5, 0.5, 0.125, 0.0, 0.0])

    def test_modified_least_squares_values(self):
        np.testing.assert_allclose(
            _penalty(self.z, "modified_least_squares"), [4.0, 1.0, 0.25, 0.0, 0.0]
        )

    def test_logistic_values(self):
        np.testing.assert_allclose(_penalty(jnp.array([0.0]), "logistic"), [0.6931471805599453], atol=1e-5)

    def test_all_penalties_non_negative(self):
        z = jnp.linspace(-3, 3, 100)
        for name in ["hinge", "smooth_hinge", "modified_least_squares", "logistic"]:
            assert bool((_penalty(z, name) >= -1e-7).all()), f"{name} produced negative values"

    def test_unknown_penalty_raises(self):
        with pytest.raises(ValueError, match="Unknown penalty"):
            _penalty(jnp.array([0.0]), "bad")


class TestOrdinalLoss:
    def _make_inputs(self):
        thresholds = jnp.array([-1.0, 0.0, 1.0])  # K=4 classes
        logits = jnp.array([0.5, -0.5])
        targets = jnp.array([1, 2])  # 0-indexed
        return logits, targets, thresholds

    def test_all_threshold_runs(self):
        loss = ordinal_loss(*self._make_inputs(), construction="all", penalty="logistic")
        assert loss.shape == ()
        assert float(loss) >= 0

    def test_immediate_threshold_runs(self):
        loss = ordinal_loss(*self._make_inputs(), construction="immediate", penalty="logistic")
        assert loss.shape == ()
        assert float(loss) >= 0

    def test_all_penalties_work(self):
        for penalty in ["hinge", "smooth_hinge", "modified_least_squares", "logistic"]:
            loss = ordinal_loss(*self._make_inputs(), penalty=penalty)
            assert float(loss) >= 0, f"{penalty} loss is negative"

    def test_perfect_prediction_low_loss(self):
        thresholds = jnp.array([-2.0, 0.0, 2.0])
        logits = jnp.array([-3.0])  # class 0: should be below -2
        loss_correct = ordinal_loss(logits, jnp.array([0]), thresholds, penalty="hinge")
        loss_wrong = ordinal_loss(logits, jnp.array([3]), thresholds, penalty="hinge")
        assert loss_correct < loss_wrong

    def test_gradients_flow_through_loss(self):
        thresholds = jnp.array([-1.0, 0.0, 1.0])
        logits = jnp.array([0.5])
        targets = jnp.array([1])
        g_logits, g_thresh = jax.grad(
            lambda l, t: ordinal_loss(l, targets, t, penalty="logistic"), argnums=(0, 1)
        )(logits, thresholds)
        assert jnp.any(g_logits != 0)
        assert jnp.any(g_thresh != 0)

    @pytest.mark.parametrize("construction", ["all", "immediate"])
    @pytest.mark.parametrize("penalty", ["hinge", "smooth_hinge", "modified_least_squares", "logistic"])
    def test_gradients_finite_at_boundary_classes(self, construction, penalty):
        thresholds = jnp.array([-1.0, 0.0, 1.0])
        logits = jnp.array([0.5, -0.5])
        targets = jnp.array([0, 3])
        g_logits, g_thresh = jax.grad(
            lambda l, t: ordinal_loss(l, targets, t, construction=construction, penalty=penalty),
            argnums=(0, 1),
        )(logits, thresholds)
        assert bool(jnp.isfinite(g_logits).all())
        assert bool(jnp.isfinite(g_thresh).all())

    def test_batch_reduction(self):
        loss = ordinal_loss(
            jnp.array([0.5, -0.5, 0.0, 1.5]), jnp.array([0, 1, 2, 3]), jnp.array([-1.0, 0.0, 1.0])
        )
        assert loss.shape == ()

    def test_unknown_construction_raises(self):
        with pytest.raises(ValueError, match="Unknown construction"):
            ordinal_loss(*self._make_inputs(), construction="bad")

    def test_logits_2d(self):
        loss = ordinal_loss(jnp.array([[0.5], [-0.5]]), jnp.array([1, 2]), jnp.array([-1.0, 0.0, 1.0]))
        assert loss.shape == ()

    def test_hand_worked_all_threshold_hinge(self):
        # K=3, thresholds=[0, 2], logit=1, target=1 -> f(1) + f(1) = 0
        loss = ordinal_loss(
            jnp.array([1.0]), jnp.array([1]), jnp.array([0.0, 2.0]), construction="all", penalty="hinge"
        )
        np.testing.assert_allclose(loss, 0.0)

    def test_hand_worked_immediate_hinge(self):
        loss = ordinal_loss(
            jnp.array([1.0]), jnp.array([1]), jnp.array([0.0, 2.0]), construction="immediate", penalty="hinge"
        )
        np.testing.assert_allclose(loss, 0.0)

    def test_jit(self):
        args = self._make_inputs()
        np.testing.assert_allclose(jax.jit(ordinal_loss)(*args), ordinal_loss(*args), rtol=1e-6)

    def test_matches_layer_end_to_end(self):
        layer = OrdinalOutput(input_dim=4, output_dim=3, rngs=nnx.Rngs(0))
        x = _x((8, 4))
        y = jnp.array([0, 1, 2, 0, 1, 2, 0, 1])

        def loss_fn(m):
            return ordinal_loss(m.logit(x), y, m.interior_thresholds)

        grads = nnx.grad(loss_fn)(layer)
        assert bool(jnp.isfinite(grads.raw_thresholds[...]).all())

    def test_hand_worked_immediate_boundary_classes(self):
        # K=3, thresholds=[0, 2], logit=0.5. Class 0 has only an upper threshold:
        # f(0 - 0.5) = 1.5. Class 2 has only a lower threshold: f(0.5 - 2) = 2.5.
        loss = ordinal_loss(
            jnp.array([0.5, 0.5]), jnp.array([0, 2]), jnp.array([0.0, 2.0]),
            construction="immediate", penalty="hinge",
        )
        np.testing.assert_allclose(loss, 2.0)


class TestOrdisticLoss:
    def test_runs(self):
        loss = ordistic_loss(jnp.array([0.5, -0.5]), jnp.array([0, 2]), jnp.array([-1.0, 0.0, 1.0]))
        assert loss.shape == ()
        assert float(loss) >= 0

    def test_with_log_priors(self):
        loss = ordistic_loss(
            jnp.array([0.5]), jnp.array([1]), jnp.array([-1.0, 0.0, 1.0]),
            log_priors=jnp.array([0.0, 0.1, -0.1]),
        )
        assert loss.shape == ()

    def test_gradients_flow(self):
        targets = jnp.array([1])
        g_logits, g_means = jax.grad(
            lambda l, m: ordistic_loss(l, targets, m), argnums=(0, 1)
        )(jnp.array([0.5]), jnp.array([-1.0, 0.0, 1.0]))
        assert jnp.any(g_logits != 0)
        assert jnp.any(g_means != 0)

    def test_non_negative(self):
        logits = _x((20,))
        targets = jax.random.randint(jax.random.key(1), (20,), 0, 3)
        loss = ordistic_loss(logits, targets, jnp.array([-1.0, 0.0, 1.0]))
        assert float(loss) >= 0
