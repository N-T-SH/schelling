"""Proper scoring rules on the typed outputs. No label smoothing: it biases calibration."""

from __future__ import annotations

import torch

EPS = 1e-12


def _reduce(x: torch.Tensor, reduction: str) -> torch.Tensor:
    return x.mean() if reduction == "mean" else x


def hard_cross_entropy(
    probs: torch.Tensor, y: torch.Tensor, reduction: str = "mean"
) -> torch.Tensor:
    p = probs.gather(1, y[:, None]).squeeze(1).clamp_min(EPS)
    return _reduce(-torch.log(p), reduction)


def soft_cross_entropy(
    probs: torch.Tensor, target: torch.Tensor, reduction: str = "mean"
) -> torch.Tensor:
    """Cross-entropy against a distribution (aggregate records)."""
    return _reduce(-(target * torch.log(probs.clamp_min(EPS))).sum(-1), reduction)


def kl_divergence(p: torch.Tensor, q: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """KL(p || q)."""
    kl = (p * (torch.log(p.clamp_min(EPS)) - torch.log(q.clamp_min(EPS)))).sum(-1)
    return _reduce(kl, reduction)


def corn_loss(cond: torch.Tensor, y: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """CORN loss on conditional probabilities cond[:, k] = P(y > k | y >= k).

    Each threshold k is a binary problem on the examples with y >= k and target 1[y > k]. Summed
    over thresholds this is exactly the negative log-likelihood of the implied distribution.
    """
    k = torch.arange(cond.shape[1], device=cond.device)[None, :]
    at_risk = (y[:, None] >= k).to(cond.dtype)
    target = (y[:, None] > k).to(cond.dtype)
    bce = -(
        target * torch.log(cond.clamp_min(EPS))
        + (1 - target) * torch.log((1 - cond).clamp_min(EPS))
    )
    return _reduce((at_risk * bce).sum(-1), reduction)
