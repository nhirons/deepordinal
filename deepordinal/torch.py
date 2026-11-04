"""PyTorch backend."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepordinal._common import check_output_dim, initial_raw_thresholds

__all__ = ["OrdinalOutput", "cumulative_link_loss", "ordinal_loss", "ordistic_loss"]


def _penalty(z, name):
    if name == "logistic":
        return F.softplus(-z)
    if name == "hinge":
        return F.relu(1 - z)
    if name == "smooth_hinge":
        return torch.where(z >= 1, 0.0, torch.where(z > 0, (1 - z) ** 2 / 2, 0.5 - z))
    if name == "modified_least_squares":
        return torch.where(z >= 1, 0.0, (1 - z) ** 2)
    raise ValueError(f"Unknown penalty: {name}")


def _nll(log_p, targets):
    return -log_p.gather(1, targets.long().reshape(-1, 1)).mean()


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
    z = logits.reshape(-1, 1)
    y = targets.long().reshape(-1, 1)
    # Threshold l separates classes <= l from classes > l, so the margin is
    # positive when z is on the correct side of it.
    l = torch.arange(thresholds.shape[0], device=z.device)
    penalties = _penalty(torch.where(l < y, z - thresholds, thresholds - z), penalty)
    if construction == "immediate":
        penalties = penalties * ((l == y - 1) | (l == y))
    return penalties.sum(dim=1).mean()


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
    energy = means * logits.reshape(-1, 1) - means**2 / 2
    if log_priors is not None:
        energy = energy + log_priors
    return _nll(energy.log_softmax(dim=1), targets)


def cumulative_link_loss(logits, targets, thresholds):
    """Negative log-likelihood of the cumulative logit model ``OrdinalOutput`` represents.

    Args:
        logits: (batch,) or (batch, 1): raw predictor output z, from ``layer.logit``.
        targets: (batch,): integer labels in [0, K).
        thresholds: (K-1,): increasing thresholds, from ``layer.interior_thresholds``.

    Returns:
        Scalar mean negative log-likelihood over the batch.
    """
    return _nll(_log_probs(logits.reshape(-1, 1), thresholds), targets)


def _log1mexp(x):
    """log(1 - exp(-x)) for x > 0 (Maechler 2012)."""
    return torch.where(x < math.log(2), torch.log(-torch.expm1(-x)), torch.log1p(-torch.exp(-x)))


def _log_probs(logit, thresholds):
    """``log P(y = k | z)``, shape ``(..., K)``, for logit shape ``(..., 1)``.

    With ``a = thresholds - z``, the exact identity
    ``sigmoid(a[k]) - sigmoid(a[k-1]) = sigmoid(a[k]) * sigmoid(-a[k-1]) * (1 - exp(a[k-1] - a[k]))``
    turns every term into a softplus or a ``log1mexp``, so nothing cancels.
    A plain difference of sigmoids rounds to 0 in float32 once ``|z|`` passes about 17.
    """
    a = thresholds - logit
    first = -F.softplus(-a[..., :1])
    last = -F.softplus(a[..., -1:])
    hi, lo = a[..., 1:], a[..., :-1]
    middle = -F.softplus(-hi) - F.softplus(lo) + _log1mexp(hi - lo)
    return torch.cat([first, middle, last], dim=-1)


class OrdinalOutput(nn.Module):
    """Ordinal output layer: features, then one logit z, then K class probabilities.

    ``P(y = k | x) = sigmoid(t[k] - z) - sigmoid(t[k-1] - z)``, where the first
    term is 1 for the last class and the second is 0 for the first class.

    The K-1 thresholds ``t = interior_thresholds`` are ordered by construction.
    The trainable ``raw_thresholds`` hold the first threshold and the inverse
    softplus of each gap. They start at ``logit(k/K)``, so a zero logit gives
    every class probability 1/K. The projection has no bias, because shifting
    z and every threshold together leaves the model unchanged.

    Args:
        input_dim: input feature size.
        output_dim: number of classes K, at least 2.
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        check_output_dim(output_dim)
        self.output_dim = output_dim
        self.linear = nn.Linear(input_dim, 1, bias=False)
        self.raw_thresholds = nn.Parameter(torch.from_numpy(initial_raw_thresholds(output_dim)))

    @property
    def interior_thresholds(self):
        raw = self.raw_thresholds
        return torch.cat([raw[:1], F.softplus(raw[1:])]).cumsum(dim=0)

    def logit(self, x):
        return self.linear(x)

    def log_probs(self, x):
        return _log_probs(self.logit(x), self.interior_thresholds)

    def forward(self, x):
        return self.log_probs(x).exp()
