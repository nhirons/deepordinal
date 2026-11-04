import pytest

tf = pytest.importorskip("tensorflow")

from deepordinal.tf import OrdinalOutput, _log_probs, cumulative_link_loss, ordinal_loss, ordistic_loss

import _reference as ref


def _t(a):
    return tf.constant(a)


def _np(x):
    return x.numpy()


def test_output_shape():
    layer = OrdinalOutput(output_dim=5)
    x = tf.random.normal((4, 8))
    out = layer(x)
    assert out.shape == (4, 5)


def test_probabilities_sum_to_one():
    layer = OrdinalOutput(output_dim=3)
    x = tf.random.normal((16, 4))
    out = layer(x)
    sums = tf.reduce_sum(out, axis=-1)
    tf.debugging.assert_near(sums, tf.ones(16), atol=1e-5)


def test_probabilities_non_negative():
    layer = OrdinalOutput(output_dim=6)
    x = tf.random.normal((32, 4))
    out = layer(x)
    assert tf.reduce_all(out >= 0).numpy()


def test_thresholds_initialized_sorted():
    layer = OrdinalOutput(output_dim=5)
    layer.build((None, 4))
    t = layer.interior_thresholds.numpy().flatten()
    assert list(t) == sorted(t)


def test_gradients_flow():
    layer = OrdinalOutput(output_dim=3)
    x = tf.random.normal((8, 4))
    with tf.GradientTape() as tape:
        out = layer(x)
        loss = tf.reduce_sum(out)
    grads = tape.gradient(loss, layer.trainable_variables)
    assert all(g is not None for g in grads)


def test_get_config_roundtrip():
    layer = OrdinalOutput(output_dim=4)
    config = layer.get_config()
    assert config["output_dim"] == 4
    restored = OrdinalOutput.from_config(config)
    assert restored.output_dim == 4


def test_thresholds_stay_ordered_for_any_raw_values():
    layer = OrdinalOutput(output_dim=8)
    layer(tf.zeros((1, 4)))
    layer.raw_thresholds.assign(tf.random.normal((7,)) * 10)
    t = layer.interior_thresholds.numpy()
    assert (t[1:] >= t[:-1]).all()
    assert tf.reduce_all(layer(tf.random.normal((64, 4)) * 10) >= 0).numpy()


def test_initial_thresholds_give_uniform_probabilities_at_zero_logit():
    layer = OrdinalOutput(output_dim=5)
    out = layer(tf.zeros((3, 4)))
    np.testing.assert_allclose(out.numpy(), np.full((3, 5), 0.2), atol=1e-6)


def test_no_bias():
    layer = OrdinalOutput(output_dim=3)
    layer(tf.zeros((1, 4)))
    names = sorted(w.name.split("/")[-1].split(":")[0] for w in layer.trainable_weights)
    assert names == ["kernel", "raw_thresholds"]


def test_logit_method():
    layer = OrdinalOutput(output_dim=3)
    x = tf.random.normal((5, 4))
    layer(x)
    np.testing.assert_allclose(layer.logit(x).numpy(), tf.matmul(x, layer.kernel).numpy())


def test_thresholds_gradient():
    layer = OrdinalOutput(output_dim=4)
    x = tf.random.normal((8, 3))
    layer(x)
    with tf.GradientTape() as tape:
        loss = ordinal_loss(layer.logit(x), tf.constant([0, 1, 2, 3] * 2), layer.interior_thresholds)
    grads = tape.gradient(loss, layer.trainable_weights)
    assert all(g is not None and np.isfinite(g.numpy()).all() for g in grads)


def test_logit_builds_unbuilt_layer():
    layer = OrdinalOutput(output_dim=3)
    assert layer.logit(tf.zeros((2, 4))).shape == (2, 1)
    assert layer.built


def test_logit_casts_inputs_like_call():
    layer = OrdinalOutput(output_dim=4)
    h = tf.random.normal((2, 3), dtype=tf.float64)
    assert layer(h).dtype == tf.float32
    logit = layer.logit(h)
    assert logit.dtype == tf.float32
    np.testing.assert_allclose(logit.numpy(), tf.matmul(tf.cast(h, tf.float32), layer.kernel).numpy())


def test_output_dim_too_small_raises():
    with pytest.raises(ValueError, match="at least 2"):
        OrdinalOutput(output_dim=1)


def test_extra_leading_dims():
    layer = OrdinalOutput(output_dim=5)
    out = layer(tf.random.normal((2, 3, 4)))
    assert out.shape == (2, 3, 5)
    tf.debugging.assert_near(tf.reduce_sum(out, axis=-1), tf.ones((2, 3)), atol=1e-5)


def test_single_sample():
    layer = OrdinalOutput(output_dim=4)
    x = tf.random.normal((1, 2))
    out = layer(x)
    assert out.shape == (1, 4)
    tf.debugging.assert_near(tf.reduce_sum(out), 1.0, atol=1e-5)


# --- Loss function tests ---

import numpy as np


class TestPenaltyFunctions:
    def test_hinge_values(self):
        from deepordinal.tf import _penalty
        z = tf.constant([-1.0, 0.0, 0.5, 1.0, 2.0])
        expected = [2.0, 1.0, 0.5, 0.0, 0.0]
        np.testing.assert_allclose(_penalty(z, "hinge").numpy(), expected)

    def test_smooth_hinge_values(self):
        from deepordinal.tf import _penalty
        z = tf.constant([-1.0, 0.0, 0.5, 1.0, 2.0])
        expected = [1.5, 0.5, 0.125, 0.0, 0.0]
        np.testing.assert_allclose(_penalty(z, "smooth_hinge").numpy(), expected)

    def test_modified_least_squares_values(self):
        from deepordinal.tf import _penalty
        z = tf.constant([-1.0, 0.0, 0.5, 1.0, 2.0])
        expected = [4.0, 1.0, 0.25, 0.0, 0.0]
        np.testing.assert_allclose(_penalty(z, "modified_least_squares").numpy(), expected)

    def test_logistic_values(self):
        from deepordinal.tf import _penalty
        z = tf.constant([0.0])
        result = _penalty(z, "logistic").numpy()
        np.testing.assert_allclose(result, [0.6931471805599453], atol=1e-5)

    def test_all_penalties_non_negative(self):
        from deepordinal.tf import _penalty
        z = tf.linspace(-3.0, 3.0, 100)
        for name in ["hinge", "smooth_hinge", "modified_least_squares", "logistic"]:
            assert tf.reduce_all(_penalty(z, name) >= -1e-7).numpy(), f"{name} produced negative values"

    def test_unknown_penalty_raises(self):
        from deepordinal.tf import _penalty
        with pytest.raises(ValueError, match="Unknown penalty"):
            _penalty(tf.constant([0.0]), "bad")


class TestOrdinalLoss:
    def _make_inputs(self):
        thresholds = tf.constant([-1.0, 0.0, 1.0])
        logits = tf.constant([0.5, -0.5])
        targets = tf.constant([1, 2])
        return logits, targets, thresholds

    def test_all_threshold_runs(self):
        logits, targets, thresholds = self._make_inputs()
        loss = ordinal_loss(logits, targets, thresholds, construction="all", penalty="logistic")
        assert loss.shape == ()
        assert loss.numpy() >= 0

    def test_immediate_threshold_runs(self):
        logits, targets, thresholds = self._make_inputs()
        loss = ordinal_loss(logits, targets, thresholds, construction="immediate", penalty="logistic")
        assert loss.shape == ()
        assert loss.numpy() >= 0

    def test_all_penalties_work(self):
        logits, targets, thresholds = self._make_inputs()
        for penalty in ["hinge", "smooth_hinge", "modified_least_squares", "logistic"]:
            loss = ordinal_loss(logits, targets, thresholds, penalty=penalty)
            assert loss.numpy() >= 0, f"{penalty} loss is negative"

    def test_perfect_prediction_low_loss(self):
        thresholds = tf.constant([-2.0, 0.0, 2.0])
        logits = tf.constant([-3.0])
        targets = tf.constant([0])
        loss_correct = ordinal_loss(logits, targets, thresholds, penalty="hinge")
        targets_wrong = tf.constant([3])
        loss_wrong = ordinal_loss(logits, targets_wrong, thresholds, penalty="hinge")
        assert loss_correct.numpy() < loss_wrong.numpy()

    def test_gradients_flow_through_loss(self):
        thresholds = tf.Variable([-1.0, 0.0, 1.0])
        logits = tf.Variable([0.5])
        targets = tf.constant([1])
        with tf.GradientTape() as tape:
            loss = ordinal_loss(logits, targets, thresholds, penalty="logistic")
        grads = tape.gradient(loss, [logits, thresholds])
        assert all(g is not None for g in grads)

    def test_batch_reduction(self):
        thresholds = tf.constant([-1.0, 0.0, 1.0])
        logits = tf.constant([0.5, -0.5, 0.0, 1.5])
        targets = tf.constant([0, 1, 2, 3])
        loss = ordinal_loss(logits, targets, thresholds)
        assert loss.shape == ()

    def test_unknown_construction_raises(self):
        logits, targets, thresholds = self._make_inputs()
        with pytest.raises(ValueError, match="Unknown construction"):
            ordinal_loss(logits, targets, thresholds, construction="bad")

    def test_logits_2d(self):
        thresholds = tf.constant([-1.0, 0.0, 1.0])
        logits = tf.constant([[0.5], [-0.5]])
        targets = tf.constant([1, 2])
        loss = ordinal_loss(logits, targets, thresholds)
        assert loss.shape == ()

    def test_hand_worked_all_threshold_hinge(self):
        thresholds = tf.constant([0.0, 2.0])
        logits = tf.constant([1.0])
        targets = tf.constant([1])
        loss = ordinal_loss(logits, targets, thresholds, construction="all", penalty="hinge")
        np.testing.assert_allclose(loss.numpy(), 0.0, atol=1e-6)

    def test_hand_worked_immediate_hinge(self):
        thresholds = tf.constant([0.0, 2.0])
        logits = tf.constant([1.0])
        targets = tf.constant([1])
        loss = ordinal_loss(logits, targets, thresholds, construction="immediate", penalty="hinge")
        np.testing.assert_allclose(loss.numpy(), 0.0, atol=1e-6)

    def test_hand_worked_immediate_boundary_classes(self):
        # K=3, thresholds=[0, 2], logit=0.5. Class 0 has only an upper threshold:
        # f(0 - 0.5) = 1.5. Class 2 has only a lower threshold: f(0.5 - 2) = 2.5.
        thresholds = tf.constant([0.0, 2.0])
        logits = tf.constant([0.5, 0.5])
        targets = tf.constant([0, 2])
        loss = ordinal_loss(logits, targets, thresholds, construction="immediate", penalty="hinge")
        np.testing.assert_allclose(loss.numpy(), 2.0, atol=1e-6)

    @pytest.mark.parametrize("construction", ["all", "immediate"])
    @pytest.mark.parametrize("penalty", ["hinge", "smooth_hinge", "modified_least_squares", "logistic"])
    def test_gradients_finite_at_boundary_classes(self, construction, penalty):
        thresholds = tf.Variable([-1.0, 0.0, 1.0])
        logits = tf.Variable([0.5, -0.5])
        targets = tf.constant([0, 3])
        with tf.GradientTape() as tape:
            loss = ordinal_loss(logits, targets, thresholds, construction=construction, penalty=penalty)
        for g in tape.gradient(loss, [logits, thresholds]):
            assert np.isfinite(g.numpy()).all()


class TestOrdisticLoss:
    def test_runs(self):
        means = tf.constant([-1.0, 0.0, 1.0])
        logits = tf.constant([0.5, -0.5])
        targets = tf.constant([0, 2])
        loss = ordistic_loss(logits, targets, means)
        assert loss.shape == ()
        assert loss.numpy() >= 0

    def test_with_log_priors(self):
        means = tf.constant([-1.0, 0.0, 1.0])
        log_priors = tf.constant([0.0, 0.1, -0.1])
        logits = tf.constant([0.5])
        targets = tf.constant([1])
        loss = ordistic_loss(logits, targets, means, log_priors=log_priors)
        assert loss.shape == ()

    def test_gradients_flow(self):
        means = tf.Variable([-1.0, 0.0, 1.0])
        logits = tf.Variable([0.5])
        targets = tf.constant([1])
        with tf.GradientTape() as tape:
            loss = ordistic_loss(logits, targets, means)
        grads = tape.gradient(loss, [logits, means])
        assert all(g is not None for g in grads)

    def test_non_negative(self):
        means = tf.constant([-1.0, 0.0, 1.0])
        logits = tf.random.normal([20])
        targets = tf.random.uniform([20], 0, 3, dtype=tf.int32)
        loss = ordistic_loss(logits, targets, means)
        assert loss.numpy() >= 0


class TestCumulativeLinkLoss:
    T = np.array([-1.0, 0.0, 1.5], np.float32)

    def test_log_probs_match_reference(self):
        z = np.linspace(-8, 8, 41).astype(np.float32)
        np.testing.assert_allclose(_np(_log_probs(_t(z[:, None]), _t(self.T))), ref.log_probs(z, self.T), rtol=1e-4, atol=1e-5)

    @pytest.mark.parametrize("z", [-80.0, -30.0, 30.0, 80.0])
    def test_log_probs_accurate_at_extreme_logits(self, z):
        # In float32 every sigmoid(t - z) rounds to 0 or 1 here, so subtracting
        # probabilities gives log(0) for every class except one.
        got = _np(_log_probs(_t(np.array([[z]], np.float32)), _t(self.T)))
        assert np.isfinite(got).all()
        np.testing.assert_allclose(got, ref.log_probs([z], self.T), rtol=1e-5, atol=1e-6)

    def test_hand_worked_two_classes(self):
        # K=2, t=[0], z=0: P(0) = P(1) = 1/2.
        loss = cumulative_link_loss(_t(np.array([0.0, 0.0], np.float32)), _t(np.array([0, 1])), _t(np.array([0.0], np.float32)))
        np.testing.assert_allclose(_np(loss), np.log(2.0), rtol=1e-6)

    def test_loss_is_mean_target_nll(self):
        rng = np.random.default_rng(0)
        z = (rng.normal(size=32) * 4).astype(np.float32)
        y = rng.integers(0, 4, size=32)
        expected = -ref.log_probs(z, self.T)[np.arange(32), y].mean()
        np.testing.assert_allclose(_np(cumulative_link_loss(_t(z), _t(y), _t(self.T))), expected, rtol=1e-5)

    def test_differs_from_immediate_logistic_by_gap_term(self):
        # sigmoid(a) - sigmoid(b) = sigmoid(a) sigmoid(-b) (1 - exp(b - a)), so the NLL
        # exceeds the immediate-threshold logistic loss by -log(1 - exp(-gap)) for
        # interior classes and matches it for the first and last classes.
        rng = np.random.default_rng(1)
        z = rng.normal(size=16).astype(np.float32)
        for y in range(4):
            yy = np.full(16, y)
            nll = _np(cumulative_link_loss(_t(z), _t(yy), _t(self.T)))
            imm = _np(ordinal_loss(_t(z), _t(yy), _t(self.T), construction="immediate", penalty="logistic"))
            gap = -np.log(-np.expm1(-np.diff(self.T.astype(np.float64))[y - 1])) if 0 < y < 3 else 0.0
            np.testing.assert_allclose(nll - imm, gap, rtol=1e-4, atol=1e-5)

    def test_gradients_finite_at_extreme_logits(self):
        z = tf.Variable([-80.0, -30.0, 0.0, 30.0, 80.0])
        t = tf.Variable(self.T)
        with tf.GradientTape() as tape:
            loss = cumulative_link_loss(z, tf.constant([1, 2, 0, 1, 2]), t)
        for g in tape.gradient(loss, [z, t]):
            assert np.isfinite(g.numpy()).all()

    def test_layer_log_probs_and_forward(self):
        layer = OrdinalOutput(output_dim=5)
        x = tf.random.normal((2, 3, 4)) * 20
        lp = layer.log_probs(x)
        assert lp.shape == (2, 3, 5)
        np.testing.assert_allclose(layer(x).numpy(), np.exp(lp.numpy()), rtol=1e-6)
        np.testing.assert_allclose(np.exp(lp.numpy()).sum(-1), np.ones((2, 3)), atol=1e-5)
