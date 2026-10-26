"""Typed output heads. Pure tensor code so it can ship as Hub custom code.

Each head separates raw logits (what temperature calibration refits) from probabilities.
"""

from __future__ import annotations

import torch
from torch import nn


def mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over valid tokens: (B, T, H), (B, T) -> (B, H)."""
    m = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * m).sum(1) / m.sum(1).clamp_min(1.0)


def _mlp(d_in: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Linear(hidden, 1))


def noul_probs(logit: torch.Tensor, temperature: float) -> torch.Tensor:
    """(B,) -> (B, 2) [p_no, p_yes], p_yes strictly inside (0, 1)."""
    p = torch.sigmoid(logit / temperature).clamp(1e-6, 1 - 1e-6)
    return torch.stack([1 - p, p], dim=-1)


def choice_probs(
    logits: torch.Tensor, option_mask: torch.Tensor, temperature: float
) -> torch.Tensor:
    return torch.softmax((logits / temperature).masked_fill(~option_mask, float("-inf")), dim=-1)


def score_conditionals(
    threshold_logits: torch.Tensor, level_mask: torch.Tensor, temperature: float
) -> torch.Tensor:
    """P(y > k | y >= k) for k = 0..L-2; padded thresholds are exactly zero."""
    return torch.sigmoid(threshold_logits / temperature) * level_mask[:, 1:].to(
        threshold_logits.dtype
    )


def score_probs(cond: torch.Tensor) -> torch.Tensor:
    surv = torch.cumprod(cond, dim=1)
    return torch.cat([1 - surv[:, :1], surv[:, :-1] - surv[:, 1:], surv[:, -1:]], dim=-1)


class NoulHead(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.mlp = _mlp(dim, hidden)

    def logit(self, h: torch.Tensor) -> torch.Tensor:
        return self.mlp(h).squeeze(-1)

    def forward(self, h: torch.Tensor, temperature: float) -> torch.Tensor:
        return noul_probs(self.logit(h), temperature)


class ChoiceHead(nn.Module):
    """logit_i = <W h, e_i>. Shared weights across options make it permutation equivariant."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=False)
        self.scale = dim**-0.5

    def logits(self, h: torch.Tensor, options: torch.Tensor) -> torch.Tensor:
        """h (B, H), options (B, K, H) -> (B, K)."""
        return (self.proj(h).unsqueeze(1) * options).sum(-1) * self.scale

    def forward(
        self, h: torch.Tensor, options: torch.Tensor, option_mask: torch.Tensor, temperature: float
    ) -> torch.Tensor:
        return choice_probs(self.logits(h, options), option_mask, temperature)


class ScoreHead(nn.Module):
    """CORN-style ordinal head.

    t_k conditions on the context and on the two neighbouring level texts; P(y > k | y >= k) =
    sigmoid(t_k), so the survival function P(y > k) is a product of probabilities and is
    non-increasing by construction. Padded levels get zero conditional probability, so they
    receive zero mass.
    """

    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.mlp = _mlp(2 * dim, hidden)

    def threshold_logits(self, h: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
        """(B, H), (B, L, H) -> (B, L-1)."""
        between = 0.5 * (levels[:, :-1] + levels[:, 1:])
        ctx = h.unsqueeze(1).expand(-1, between.shape[1], -1)
        return self.mlp(torch.cat([ctx, between], dim=-1)).squeeze(-1)

    def conditionals(
        self, h: torch.Tensor, levels: torch.Tensor, level_mask: torch.Tensor, temperature: float
    ) -> torch.Tensor:
        return score_conditionals(self.threshold_logits(h, levels), level_mask, temperature)

    def survival(
        self, h: torch.Tensor, levels: torch.Tensor, level_mask: torch.Tensor, temperature: float
    ) -> torch.Tensor:
        return torch.cumprod(self.conditionals(h, levels, level_mask, temperature), dim=1)

    def forward(
        self, h: torch.Tensor, levels: torch.Tensor, level_mask: torch.Tensor, temperature: float
    ) -> torch.Tensor:
        return score_probs(self.conditionals(h, levels, level_mask, temperature))
