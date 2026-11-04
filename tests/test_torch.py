import numpy as np
import pytest

torch = pytest.importorskip("torch")

from deepordinal.torch import OrdinalOutput, _log_probs, cumulative_link_loss, ordinal_loss, ordistic_loss

import _reference as ref


def _t(a):
    return torch.as_tensor(a)


def _np(x):
    return x.detach().numpy()


def test_output_shape():
    layer = OrdinalOutput(input_dim=8, output_dim=5)
    x = torch.randn(4, 8)
    out = layer(x)
    assert out.shape == (4, 5)


def test_probabilities_sum_to_one():
    layer = OrdinalOutput(input_dim=4, output_dim=3)
    x = torch.randn(16, 4)
    out = layer(x)
    torch.testing.assert_close(out.sum(dim=-1), torch.ones(16), atol=1e-5, rtol=0)


def test_probabilities_non_negative():
    layer = OrdinalOutput(input_dim=4, output_dim=6)
    x = torch.randn(32, 4)
    out = layer(x)
    assert (out >= 0).all()


def test_thresholds_initialized_sorted():
    layer = OrdinalOutput(input_dim=4, output_dim=5)
    t = layer.interior_thresholds.detach()
    sorted_t, _ = t.sort()
    torch.testing.assert_close(t, sorted_t)


def test_gradients_flow():
    layer = OrdinalOutput(input_dim=4, output_dim=3)
    x = torch.randn(8, 4)
    out = layer(x)
    loss = out.sum()
    loss.backward()
    assert layer.linear.weight.grad is not None
    assert layer.raw_thresholds.grad is not None


def test_seed_reproducibility():
    torch.manual_seed(42)
    a = OrdinalOutput(input_dim=4, output_dim=3)
    torch.manual_seed(42)
    b = OrdinalOutput(input_dim=4, output_dim=3)
    x = torch.randn(4, 4)
    torch.testing.assert_close(a(x), b(x))


def test_thresholds_stay_ordered_for_any_raw_values():
    layer = OrdinalOutput(input_dim=4, output_dim=8)
    with torch.no_grad():
        layer.raw_thresholds.copy_(torch.randn(7) * 10)
    t = layer.interior_thresholds
    assert (t[1:] >= t[:-1]).all()
    assert (layer(torch.randn(64, 4) * 10) >= 0).all()


def test_initial_thresholds_give_uniform_probabilities_at_zero_logit():
    layer = OrdinalOutput(input_dim=4, output_dim=5)
    out = layer(torch.zeros(3, 4))
    torch.testing.assert_close(out, torch.full((3, 5), 0.2))


def test_no_bias():
    layer = OrdinalOutput(input_dim=4, output_dim=3)
    assert layer.linear.bias is None
    names = {n for n, _ in layer.named_parameters()}
    assert names == {"linear.weight", "raw_thresholds"}


def test_logit_method():
    layer = OrdinalOutput(input_dim=4, output_dim=3)
    x = torch.randn(5, 4)
    torch.testing.assert_close(layer.logit(x), layer.linear(x))
    assert layer.logit(x).shape == (5, 1)


def test_output_dim_too_small_raises():
    with pytest.raises(ValueError, match="at least 2"):
        OrdinalOutput(input_dim=4, output_dim=1)


def test_extra_leading_dims():
    layer = OrdinalOutput(input_dim=4, output_dim=5)
    out = layer(torch.randn(2, 3, 4))
    assert out.shape == (2, 3, 5)
    torch.testing.assert_close(out.sum(dim=-1), torch.ones(2, 3), atol=1e-5, rtol=0)


def test_single_sample():
    layer = OrdinalOutput(input_dim=2, output_dim=4)
    x = torch.randn(1, 2)
    out = layer(x)
    assert out.shape == (1, 4)
    torch.testing.assert_close(out.sum(), torch.tensor(1.0), atol=1e-5, rtol=0)


# --- Loss function tests ---


class TestPenaltyFunctions:
    """Test each penalty function for known input/output values."""

    def test_hinge_values(self):
        z = torch.tensor([-1.0, 0.0, 0.5, 1.0, 2.0])
        expected = torch.tensor([2.0, 1.0, 0.5, 0.0, 0.0])
        from deepordinal.torch import _penalty
        torch.testing.assert_close(_penalty(z, "hinge"), expected)

    def test_smooth_hinge_values(self):
        z = torch.tensor([-1.0, 0.0, 0.5, 1.0, 2.0])
        expected = torch.tensor([1.5, 0.5, 0.125, 0.0, 0.0])
        from deepordinal.torch import _penalty
        torch.testing.assert_close(_penalty(z, "smooth_hinge"), expected)

    def test_modified_least_squares_values(self):
        z = torch.tensor([-1.0, 0.0, 0.5, 1.0, 2.0])
        expected = torch.tensor([4.0, 1.0, 0.25, 0.0, 0.0])
        from deepordinal.torch import _penalty
        torch.testing.assert_close(_penalty(z, "modified_least_squares"), expected)

    def test_logistic_values(self):
        z = torch.tensor([0.0])
        from deepordinal.torch import _penalty
        result = _penalty(z, "logistic")
        torch.testing.assert_close(result, torch.tensor([0.6931471805599453]), atol=1e-5, rtol=0)

    def test_all_penalties_non_negative(self):
        from deepordinal.torch import _penalty
        z = torch.linspace(-3, 3, 100)
        for name in ["hinge", "smooth_hinge", "modified_least_squares", "logistic"]:
            assert (_penalty(z, name) >= -1e-7).all(), f"{name} produced negative values"

    def test_unknown_penalty_raises(self):
        from deepordinal.torch import _penalty
        with pytest.raises(ValueError, match="Unknown penalty"):
            _penalty(torch.tensor([0.0]), "bad")


class TestOrdinalLoss:
    """Test ordinal_loss with both constructions."""

    def _make_inputs(self):
        thresholds = torch.tensor([-1.0, 0.0, 1.0])  # K=4 classes
        logits = torch.tensor([0.5, -0.5])
        targets = torch.tensor([1, 2])  # 0-indexed
        return logits, targets, thresholds

    def test_all_threshold_runs(self):
        logits, targets, thresholds = self._make_inputs()
        loss = ordinal_loss(logits, targets, thresholds, construction="all", penalty="logistic")
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_immediate_threshold_runs(self):
        logits, targets, thresholds = self._make_inputs()
        loss = ordinal_loss(logits, targets, thresholds, construction="immediate", penalty="logistic")
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_all_penalties_work(self):
        logits, targets, thresholds = self._make_inputs()
        for penalty in ["hinge", "smooth_hinge", "modified_least_squares", "logistic"]:
            loss = ordinal_loss(logits, targets, thresholds, penalty=penalty)
            assert loss.item() >= 0, f"{penalty} loss is negative"

    def test_perfect_prediction_low_loss(self):
        # Logit perfectly in the middle of its segment should have low loss
        thresholds = torch.tensor([-2.0, 0.0, 2.0])
        logits = torch.tensor([-3.0])  # class 0: should be below -2
        targets = torch.tensor([0])
        loss_correct = ordinal_loss(logits, targets, thresholds, penalty="hinge")
        # Now misclassify
        targets_wrong = torch.tensor([3])
        loss_wrong = ordinal_loss(logits, targets_wrong, thresholds, penalty="hinge")
        assert loss_correct < loss_wrong

    def test_gradients_flow_through_loss(self):
        thresholds = torch.tensor([-1.0, 0.0, 1.0], requires_grad=True)
        logits = torch.tensor([0.5], requires_grad=True)
        targets = torch.tensor([1])
        loss = ordinal_loss(logits, targets, thresholds, penalty="logistic")
        loss.backward()
        assert logits.grad is not None
        assert thresholds.grad is not None

    def test_batch_reduction(self):
        thresholds = torch.tensor([-1.0, 0.0, 1.0])
        logits = torch.tensor([0.5, -0.5, 0.0, 1.5])
        targets = torch.tensor([0, 1, 2, 3])
        loss = ordinal_loss(logits, targets, thresholds)
        assert loss.shape == ()

    def test_unknown_construction_raises(self):
        logits, targets, thresholds = self._make_inputs()
        with pytest.raises(ValueError, match="Unknown construction"):
            ordinal_loss(logits, targets, thresholds, construction="bad")

    def test_logits_2d(self):
        thresholds = torch.tensor([-1.0, 0.0, 1.0])
        logits = torch.tensor([[0.5], [-0.5]])
        targets = torch.tensor([1, 2])
        loss = ordinal_loss(logits, targets, thresholds)
        assert loss.shape == ()

    def test_hand_worked_all_threshold_hinge(self):
        # K=3 (classes 0,1,2), thresholds=[0, 2], logit=1, target=1 (paper y=2)
        # s(l=1;y=2)=-1, s(l=2;y=2)=+1
        # f(-1*(0-1)) + f(+1*(2-1)) = f(1) + f(1) = 0 + 0 = 0
        thresholds = torch.tensor([0.0, 2.0])
        logits = torch.tensor([1.0])
        targets = torch.tensor([1])
        loss = ordinal_loss(logits, targets, thresholds, construction="all", penalty="hinge")
        torch.testing.assert_close(loss, torch.tensor(0.0))

    def test_hand_worked_immediate_hinge(self):
        # K=3, thresholds=[0, 2], logit=1, target=1 (0-indexed)
        # theta_low = thresholds[1] = 0, theta_high = thresholds[1] = 2
        # f(1-0) + f(2-1) = f(1) + f(1) = 0 + 0 = 0
        thresholds = torch.tensor([0.0, 2.0])
        logits = torch.tensor([1.0])
        targets = torch.tensor([1])
        loss = ordinal_loss(logits, targets, thresholds, construction="immediate", penalty="hinge")
        torch.testing.assert_close(loss, torch.tensor(0.0))

    def test_hand_worked_immediate_boundary_classes(self):
        # K=3, thresholds=[0, 2], logit=0.5. Class 0 has only an upper threshold:
        # f(0 - 0.5) = 1.5. Class 2 has only a lower threshold: f(0.5 - 2) = 2.5.
        thresholds = torch.tensor([0.0, 2.0])
        logits = torch.tensor([0.5, 0.5])
        targets = torch.tensor([0, 2])
        loss = ordinal_loss(logits, targets, thresholds, construction="immediate", penalty="hinge")
        torch.testing.assert_close(loss, torch.tensor(2.0))

    @pytest.mark.parametrize("construction", ["all", "immediate"])
    @pytest.mark.parametrize("penalty", ["hinge", "smooth_hinge", "modified_least_squares", "logistic"])
    def test_gradients_finite_at_boundary_classes(self, construction, penalty):
        thresholds = torch.tensor([-1.0, 0.0, 1.0], requires_grad=True)
        logits = torch.tensor([0.5, -0.5], requires_grad=True)
        targets = torch.tensor([0, 3])
        ordinal_loss(logits, targets, thresholds, construction=construction, penalty=penalty).backward()
        assert torch.isfinite(logits.grad).all()
        assert torch.isfinite(thresholds.grad).all()


class TestOrdisticLoss:
    def test_runs(self):
        means = torch.tensor([-1.0, 0.0, 1.0])
        logits = torch.tensor([0.5, -0.5])
        targets = torch.tensor([0, 2])
        loss = ordistic_loss(logits, targets, means)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_with_log_priors(self):
        means = torch.tensor([-1.0, 0.0, 1.0])
        log_priors = torch.tensor([0.0, 0.1, -0.1])
        logits = torch.tensor([0.5])
        targets = torch.tensor([1])
        loss = ordistic_loss(logits, targets, means, log_priors=log_priors)
        assert loss.shape == ()

    def test_gradients_flow(self):
        means = torch.tensor([-1.0, 0.0, 1.0], requires_grad=True)
        logits = torch.tensor([0.5], requires_grad=True)
        targets = torch.tensor([1])
        loss = ordistic_loss(logits, targets, means)
        loss.backward()
        assert logits.grad is not None
        assert means.grad is not None

    def test_non_negative(self):
        means = torch.tensor([-1.0, 0.0, 1.0])
        logits = torch.randn(20)
        targets = torch.randint(0, 3, (20,))
        loss = ordistic_loss(logits, targets, means)
        assert loss.item() >= 0


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
        z = torch.tensor([-80.0, -30.0, 0.0, 30.0, 80.0], requires_grad=True)
        t = torch.tensor(self.T, requires_grad=True)
        cumulative_link_loss(z, torch.tensor([1, 2, 0, 1, 2]), t).backward()
        assert torch.isfinite(z.grad).all() and torch.isfinite(t.grad).all()

    def test_layer_log_probs_and_forward(self):
        layer = OrdinalOutput(input_dim=4, output_dim=5)
        x = torch.randn(2, 3, 4) * 20
        lp = layer.log_probs(x)
        assert lp.shape == (2, 3, 5)
        torch.testing.assert_close(layer(x), lp.exp())
        torch.testing.assert_close(lp.exp().sum(-1), torch.ones(2, 3), atol=1e-5, rtol=0)
