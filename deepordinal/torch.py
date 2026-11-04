import torch
import torch.nn as nn
import torch.nn.functional as F

from deepordinal._common import check_output_dim, initial_raw_thresholds

__all__ = ["OrdinalOutput", "ordinal_loss", "ordistic_loss"]


def _penalty(z, name):
    if name == "hinge":
        return torch.clamp(1 - z, min=0)
    elif name == "smooth_hinge":
        return torch.where(
            z >= 1,
            torch.zeros_like(z),
            torch.where(z > 0, (1 - z) ** 2 / 2, 0.5 - z),
        )
    elif name == "modified_least_squares":
        return torch.where(z >= 1, torch.zeros_like(z), (1 - z) ** 2)
    elif name == "logistic":
        return torch.nn.functional.softplus(-z)
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
    logits = logits.reshape(-1)
    targets = targets.long().unsqueeze(1)  # (batch, 1)
    K = thresholds.shape[0] + 1
    # Threshold l (0-indexed) separates classes <= l from classes > l. The logit
    # should sit above threshold l when l < y and below it otherwise, so the
    # margin is s(l; y) * (theta_l - z) with s = -1 if l < y else +1 (eq 13).
    l_idx = torch.arange(K - 1, device=logits.device)  # (K-1,)
    signs = 1 - 2 * (l_idx < targets).to(logits.dtype)  # (batch, K-1)
    margins = signs * (thresholds.unsqueeze(0) - logits.unsqueeze(1))  # (batch, K-1)
    penalties = _penalty(margins, penalty)
    if construction == "immediate":
        # eq 12 keeps only the thresholds bounding the target class: y-1 and y.
        # The first and last classes have one bounding threshold each.
        penalties = penalties * ((l_idx == targets - 1) | (l_idx == targets))
    loss = penalties.sum(dim=1)  # (batch,)

    return loss.mean()


def ordistic_loss(logits, targets, means, log_priors=None):
    """Ordistic loss (Rennie & Srebro, Section 4).

    Args:
        logits: (batch,) or (batch, 1) — raw predictor output z(x).
        targets: (batch,) — integer labels in [0, K).
        means: (K,) — class means (mu_1=-1, mu_K=1 by convention; interior learned).
        log_priors: (K,) or None — log-prior terms pi_i. Defaults to zeros.

    Returns:
        Scalar mean negative log-likelihood over the batch.
    """
    logits = logits.reshape(-1)
    targets = targets.long()
    K = means.shape[0]
    if log_priors is None:
        log_priors = torch.zeros(K, device=logits.device, dtype=logits.dtype)
    # energy_ik = mu_k * z_i + pi_k - mu_k^2 / 2
    energy = means.unsqueeze(0) * logits.unsqueeze(1) + log_priors.unsqueeze(0) - means.unsqueeze(0) ** 2 / 2
    # loss = -log P(y|z) = -energy[y] + log(sum_k exp(energy[k]))
    target_energy = energy[torch.arange(len(targets), device=targets.device), targets]
    log_partition = torch.logsumexp(energy, dim=1)
    return (log_partition - target_energy).mean()


class OrdinalOutput(nn.Module):
    """Ordinal regression output layer.

    Projects an arbitrary input down to a single logit and converts it
    into *output_dim* class probabilities via learned, ordered thresholds
    ``t(1) < ... < t(K-1)``::

        P(y = k | x) = sigmoid(t(k+1) - logit) - sigmoid(t(k) - logit)

    where ``sigmoid(t(0) - logit)`` is 0 and ``sigmoid(t(K) - logit)`` is 1.

    The thresholds are ordered by construction. The layer stores
    ``raw_thresholds``: the first threshold, then the inverse softplus of each
    gap. ``interior_thresholds`` recovers them as
    ``cumsum([raw[0], softplus(raw[1:])])``, so no optimizer step can reorder
    them and the probabilities are never negative.

    The projection has no bias. Adding a constant to the logit and to every
    threshold gives the same model, so the thresholds absorb the bias.

    Thresholds initialize at ``logit(k/K)``, so a zero logit gives every class
    probability ``1/K``.

    Args:
        input_dim: Size of the input feature dimension.
        output_dim: Number of ordinal classes, at least 2.
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        check_output_dim(output_dim)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear = nn.Linear(input_dim, 1, bias=False)
        self.raw_thresholds = nn.Parameter(torch.from_numpy(initial_raw_thresholds(output_dim)))

    @property
    def interior_thresholds(self):
        """Ordered thresholds, shape ``(K-1,)``. Differentiable."""
        raw = self.raw_thresholds
        return torch.cat([raw[:1], F.softplus(raw[1:])]).cumsum(dim=0)

    def logit(self, x):
        """Raw logit ``z(x)``, shape ``(..., 1)``. Pass this to the loss functions."""
        return self.linear(x)

    def forward(self, x):
        logit = self.logit(x)  # (..., 1)
        cdf = torch.sigmoid(self.interior_thresholds - logit)  # P(y <= k), (..., K-1)
        ones = torch.ones_like(logit)
        return torch.cat([cdf, ones], dim=-1) - torch.cat([1 - ones, cdf], dim=-1)
