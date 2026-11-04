"""Backend-agnostic tests. Each runs once per installed backend, against tests/reference.py."""

import numpy as np
import pytest

import reference as ref

K = 5
rng = np.random.default_rng(0)
Z = (rng.normal(size=60) * 3).astype(np.float32)
Y = rng.permutation(np.arange(60) % K)  # every class, including both boundary classes
T = np.sort(rng.normal(size=K - 1)).astype(np.float32)
MEANS = np.sort(rng.normal(size=K)).astype(np.float32)
LOG_PRIORS = rng.normal(size=K).astype(np.float32)

CONSTRUCTIONS = ["all", "immediate"]
PENALTIES = list(ref.PENALTIES)


def run(backend, fn, *args, **kwargs):
    return backend.numpy(getattr(backend.mod, fn)(*map(backend.tensor, args), **kwargs))


def assert_finite(*arrays):
    for a in arrays:
        assert np.isfinite(a).all()


# --- ordinal_loss ---


@pytest.mark.parametrize("construction", CONSTRUCTIONS)
@pytest.mark.parametrize("penalty", PENALTIES)
def test_ordinal_loss_matches_reference(backend, construction, penalty):
    got = run(backend, "ordinal_loss", Z, Y, T, construction=construction, penalty=penalty)
    np.testing.assert_allclose(got, ref.ordinal_loss(Z, Y, T, construction, penalty), rtol=1e-5)


@pytest.mark.parametrize("construction", CONSTRUCTIONS)
@pytest.mark.parametrize("penalty", PENALTIES)
def test_ordinal_loss_gradients_finite(backend, construction, penalty):
    y = backend.tensor(Y)
    loss = lambda z, t: backend.mod.ordinal_loss(z, y, t, construction=construction, penalty=penalty)
    assert_finite(*backend.grad(loss, Z, T))


@pytest.mark.parametrize(
    "construction, logits, targets, expected",
    [
        # K=3, thresholds [0, 2]. z=1 sits inside class 1 with margin 1 on both sides.
        ("all", [1.0], [1], 0.0),
        ("immediate", [1.0], [1], 0.0),
        # Boundary classes have one bounding threshold: f(0 - 0.5) = 1.5, f(0.5 - 2) = 2.5.
        ("immediate", [0.5, 0.5], [0, 2], 2.0),
        # All-threshold adds the far threshold too: (1.5 + 0) and (0.5 + 2.5).
        ("all", [0.5, 0.5], [0, 2], 2.25),
    ],
)
def test_ordinal_loss_hand_worked_hinge(backend, construction, logits, targets, expected):
    logits, targets, thresholds = np.float32(logits), np.array(targets), np.float32([0.0, 2.0])
    got = run(backend, "ordinal_loss", logits, targets, thresholds, construction=construction, penalty="hinge")
    np.testing.assert_allclose(got, expected, atol=1e-6)


def test_ordinal_loss_rejects_unknown_options(backend):
    with pytest.raises(ValueError, match="Unknown construction"):
        run(backend, "ordinal_loss", Z, Y, T, construction="bad")
    with pytest.raises(ValueError, match="Unknown penalty"):
        run(backend, "ordinal_loss", Z, Y, T, penalty="bad")


# --- ordistic_loss ---


@pytest.mark.parametrize("log_priors", [None, LOG_PRIORS])
def test_ordistic_loss_matches_reference(backend, log_priors):
    extra = () if log_priors is None else (log_priors,)
    got = run(backend, "ordistic_loss", Z, Y, MEANS, *extra)
    np.testing.assert_allclose(got, ref.ordistic_loss(Z, Y, MEANS, log_priors), rtol=1e-5)


def test_ordistic_loss_gradients(backend):
    y = backend.tensor(Y)
    grads = backend.grad(lambda z, m: backend.mod.ordistic_loss(z, y, m), Z, MEANS)
    assert_finite(*grads)
    assert all(np.any(g != 0) for g in grads)


# --- cumulative_link_loss and log-probabilities ---


def test_cumulative_link_loss_matches_reference(backend):
    got = run(backend, "cumulative_link_loss", Z, Y, T)
    np.testing.assert_allclose(got, ref.cumulative_link_loss(Z, Y, T), rtol=1e-5)


def test_cumulative_link_loss_two_classes_by_hand(backend):
    # K=2, threshold 0, logit 0: both classes have probability 1/2.
    got = run(backend, "cumulative_link_loss", np.float32([0, 0]), np.array([0, 1]), np.float32([0]))
    np.testing.assert_allclose(got, np.log(2), rtol=1e-6)


@pytest.mark.parametrize("z", [-80.0, -30.0, -8.0, 0.0, 8.0, 30.0, 80.0])
def test_log_probs_accurate_at_any_logit(backend, z):
    # Past |z| ~ 17 every float32 sigmoid(t - z) rounds to 0 or 1, so a difference
    # of sigmoids would give log(0) for every class but one.
    logit = np.float32([[z]])
    got = backend.numpy(backend.mod._log_probs(backend.tensor(logit), backend.tensor(T)))
    assert_finite(got)
    np.testing.assert_allclose(got, ref.log_probs(logit, T), rtol=1e-5, atol=1e-6)


def test_cumulative_link_loss_gradients_finite_at_extreme_logits(backend):
    z, y = np.float32([-80, -30, 0, 30, 80]), backend.tensor(np.array([1, 2, 0, 3, 4]))
    assert_finite(*backend.grad(lambda z, t: backend.mod.cumulative_link_loss(z, y, t), z, T))


@pytest.mark.parametrize("target", range(K))
def test_cumulative_link_loss_exceeds_immediate_logistic_by_gap_term(backend, target):
    # sigmoid(a) - sigmoid(b) = sigmoid(a) sigmoid(-b) (1 - exp(b - a)), so the NLL adds
    # -log(1 - exp(-gap)) for interior classes and equals immediate-logistic at the ends.
    y = np.full(len(Z), target)
    nll = run(backend, "cumulative_link_loss", Z, y, T)
    immediate = run(backend, "ordinal_loss", Z, y, T, construction="immediate", penalty="logistic")
    gap = np.diff(T.astype(float))[target - 1] if 0 < target < K - 1 else np.inf
    np.testing.assert_allclose(nll - immediate, -np.log(-np.expm1(-gap)), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("loss", ["ordinal_loss", "ordistic_loss", "cumulative_link_loss"])
def test_losses_accept_column_logits(backend, loss):
    third = MEANS if loss == "ordistic_loss" else T
    np.testing.assert_allclose(run(backend, loss, Z[:, None], Y, third), run(backend, loss, Z, Y, third))


# --- OrdinalOutput ---


@pytest.mark.parametrize("shape", [(4, 6), (1, 6), (2, 3, 6)])
def test_layer_outputs_distributions(backend, shape):
    probs = backend.numpy(backend.layer(6, K)(backend.tensor(rng.normal(size=shape).astype(np.float32) * 10)))
    assert probs.shape == (*shape[:-1], K)
    assert (probs >= 0).all()
    np.testing.assert_allclose(probs.sum(-1), 1, atol=1e-5)


def test_layer_matches_reference(backend):
    layer = backend.layer(6, K)
    backend.set_raw_thresholds(layer, rng.normal(size=K - 1).astype(np.float32))
    x = backend.tensor(rng.normal(size=(32, 6)).astype(np.float32) * 10)
    expected = ref.log_probs(backend.numpy(layer.logit(x)), backend.numpy(layer.interior_thresholds))
    np.testing.assert_allclose(backend.numpy(layer.log_probs(x)), expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(backend.numpy(layer(x)), np.exp(expected), rtol=1e-5, atol=1e-7)


def test_layer_starts_uniform(backend):
    layer = backend.layer(6, K)
    k = np.arange(1, K)
    np.testing.assert_allclose(backend.numpy(layer.interior_thresholds), np.log(k / (K - k)), atol=1e-6)
    probs = backend.numpy(layer(backend.tensor(np.zeros((3, 6), np.float32))))
    np.testing.assert_allclose(probs, 1 / K, atol=1e-6)


def test_layer_thresholds_ordered_for_any_parameters(backend):
    layer = backend.layer(6, 8)
    backend.set_raw_thresholds(layer, rng.normal(size=7).astype(np.float32) * 10)
    t = backend.numpy(layer.interior_thresholds)
    assert (np.diff(t) >= 0).all()
    assert (backend.numpy(layer(backend.tensor(rng.normal(size=(64, 6)).astype(np.float32) * 10))) >= 0).all()


def test_layer_has_no_bias(backend):
    sizes = sorted(int(np.prod(s)) for s in backend.param_shapes(backend.layer(6, K)))
    assert sizes == [K - 1, 6]  # raw_thresholds and the projection weight


@pytest.mark.parametrize("loss", ["ordinal_loss", "cumulative_link_loss"])
def test_layer_trains_through_losses(backend, loss):
    layer = backend.layer(6, K)
    x = backend.tensor(rng.normal(size=(len(Y), 6)).astype(np.float32))
    y = backend.tensor(Y)
    grads = backend.layer_grads(layer, lambda m: getattr(backend.mod, loss)(m.logit(x), y, m.interior_thresholds))
    assert len(grads) == 2
    assert_finite(*grads)
    assert all(np.any(g != 0) for g in grads)


def test_layer_rejects_fewer_than_two_classes(backend):
    with pytest.raises(ValueError, match="at least 2"):
        backend.layer(6, 1)
