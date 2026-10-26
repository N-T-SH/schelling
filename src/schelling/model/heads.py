"""Typed output heads. Pure tensor code so it can ship as Hub custom code."""

from __future__ import annotations

import torch
from torch import nn


def mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over valid tokens: (B, T, H), (B, T) -> (B, H)."""
    m = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * m).sum(1) / m.sum(1).clamp_min(1.0)


def _mlp(d_in: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Linear(hidden, 1))


class NoulHead(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.mlp = _mlp(dim, hidden)

    def forward(self, h: torch.Tensor, temperature: float) -> torch.Tensor:
        """(B, H) -> (B, 2) probabilities [p_no, p_yes], p_yes strictly inside (0, 1)."""
        p = torch.sigmoid(self.mlp(h).squeeze(-1) / temperature).clamp(1e-6, 1 - 1e-6)
        return torch.stack([1 - p, p], dim=-1)


class ChoiceHead(nn.Module):
    """logit_i = <W h, e_i>. Shared weights across options make it permutation equivariant."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=False)
        self.scale = dim**-0.5

    def forward(
        self, h: torch.Tensor, options: torch.Tensor, option_mask: torch.Tensor, temperature: float
    ) -> torch.Tensor:
        """h (B, H), options (B, K, H), option_mask (B, K) bool -> (B, K) probabilities."""
        logits = (self.proj(h).unsqueeze(1) * options).sum(-1) * self.scale / temperature
        return torch.softmax(logits.masked_fill(~option_mask, float("-inf")), dim=-1)


class ScoreHead(nn.Module):
    """CORN-style ordinal head.

    t_k conditions on the context and on the two neighbouring level texts; P(y > k | y > k-1) =
    sigmoid(t_k), so the survival function P(y > k) is a product of probabilities and is
    non-increasing by construction. Padded levels get zero conditional probability, so they
    receive zero mass.
    """

    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.mlp = _mlp(2 * dim, hidden)

    def survival(
        self, h: torch.Tensor, levels: torch.Tensor, level_mask: torch.Tensor, temperature: float
    ) -> torch.Tensor:
        """(B, H), (B, L, H), (B, L) -> P(y > k) for k = 0..L-2, shape (B, L-1)."""
        between = 0.5 * (levels[:, :-1] + levels[:, 1:])
        ctx = h.unsqueeze(1).expand(-1, between.shape[1], -1)
        t = self.mlp(torch.cat([ctx, between], dim=-1)).squeeze(-1) / temperature
        cond = torch.sigmoid(t) * level_mask[:, 1:].to(t.dtype)
        return torch.cumprod(cond, dim=1)

    def forward(
        self, h: torch.Tensor, levels: torch.Tensor, level_mask: torch.Tensor, temperature: float
    ) -> torch.Tensor:
        s = self.survival(h, levels, level_mask, temperature)
        first = 1 - s[:, :1]
        middle = s[:, :-1] - s[:, 1:]
        last = s[:, -1:]
        return torch.cat([first, middle, last], dim=-1)
