"""Two-tower combiner: an item-only prior plus a person-dependent residual (decision 0004).

Works on backbone vectors, so it can be trained on cached vectors without the backbone.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class TTOut:
    prior: torch.Tensor  # item-only logits
    residual: torch.Tensor  # person-dependent logits, exactly zero at initialisation
    mask: torch.Tensor | None  # valid options or levels, None for Noul

    @property
    def logits(self) -> torch.Tensor:
        return self.prior + self.residual


def _last(net: nn.Sequential) -> nn.Linear:
    layer = net[-1]
    assert isinstance(layer, nn.Linear)
    return layer


def _mlp(d_in: int, hidden: int, d_out: int, zero_last: bool = False) -> nn.Sequential:
    net = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Linear(hidden, d_out))
    if zero_last:
        nn.init.zeros_(_last(net).weight)
        nn.init.zeros_(_last(net).bias)
    return net


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """(B, K, d), (B, K) -> (B, d); an all-masked row gives zeros."""
    m = mask.unsqueeze(-1).to(x.dtype)
    return (x * m).sum(1) / m.sum(1).clamp_min(1.0)


class PersonEncoder(nn.Module):
    """Set encoder over attribute vectors: no positional encoding, so order never matters."""

    def __init__(self, d: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(
            d, heads, dim_feedforward=2 * d, dropout=dropout, batch_first=True, norm_first=True
        )

    def forward(self, a: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """(B, A, d), (B, A) bool -> (B, d). A person with no attributes gets the zero vector."""
        none = ~mask.any(dim=1)
        safe = mask.clone()
        safe[none, 0] = True  # keep attention well defined; the row is zeroed below
        h = self.layer(a, src_key_padding_mask=~safe)
        p = masked_mean(h, safe)
        return p.masked_fill(none.unsqueeze(-1), 0.0)


class TwoTowerHeads(nn.Module):
    def __init__(
        self, hidden: int, d: int, person_heads: int, dropout: float, use_person: bool
    ) -> None:
        super().__init__()
        self.use_person = use_person
        self.u_proj, self.e_proj, self.a_proj = (nn.Linear(hidden, d) for _ in range(3))
        self.drop = nn.Dropout(dropout)
        # prior: item (and options) only
        self.noul_prior = _mlp(d, d, 1)
        self.choice_prior = _mlp(3 * d, d, 1)
        self.score_prior = _mlp(3 * d, d, 1)
        # residual: person-dependent, zero-initialised last layers
        self.person = PersonEncoder(d, person_heads, dropout)
        self.mix = _mlp(3 * d, d, d)
        self.noul_res = _mlp(d, d, 1, zero_last=True)
        self.choice_res = _mlp(2 * d, d, 1, zero_last=True)
        self.score_res = _mlp(2 * d, d, 1, zero_last=True)

    def reset(self) -> None:
        """PyTorch's default init for every layer, then a zero residual. (The framework's weight
        init sets every Linear to std 0.02, which leaves these small networks near-silent.)"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()
        self.zero_residual()

    def zero_residual(self) -> None:
        """Re-zero the residual outputs (the framework's weight init overwrites the zero init)."""
        for net in (self.noul_res, self.choice_res, self.score_res):
            nn.init.zeros_(_last(net).weight)
            nn.init.zeros_(_last(net).bias)

    def init_residual(self, std: float) -> None:
        """Give the residual branch non-zero output (tests; training starts from zero)."""
        for net in (self.noul_res, self.choice_res, self.score_res):
            nn.init.normal_(_last(net).weight, std=std)

    def forward(
        self,
        task: str,
        u: torch.Tensor,
        attr_vecs: torch.Tensor,
        attr_mask: torch.Tensor,
        e: torch.Tensor | None,
        option_mask: torch.Tensor | None,
    ) -> TTOut:
        """u (B, H) question vectors; attr_vecs (B, A, H); e (B, K, H) option or level vectors."""
        q = self.drop(self.u_proj(u))
        if task == "noul":
            prior = self.noul_prior(q).squeeze(-1)
            r = self._person_mix(q, attr_vecs, attr_mask)
            res = self.noul_res(r).squeeze(-1) if self.use_person else torch.zeros_like(prior)
            return TTOut(prior, res, None)
        assert e is not None and option_mask is not None
        ev = self.drop(self.e_proj(e))
        ebar = masked_mean(ev, option_mask)
        if task == "choice":
            ctx = torch.cat(
                [q.unsqueeze(1).expand_as(ev), ev, ebar.unsqueeze(1).expand_as(ev)], dim=-1
            )
            prior = self.choice_prior(ctx).squeeze(-1)
            unit = ev
        else:  # score: thresholds sit between neighbouring levels
            unit = 0.5 * (ev[:, :-1] + ev[:, 1:])
            ctx = torch.cat(
                [q.unsqueeze(1).expand_as(unit), unit, ebar.unsqueeze(1).expand_as(unit)], dim=-1
            )
            prior = self.score_prior(ctx).squeeze(-1)
        r = self._person_mix(q, attr_vecs, attr_mask)
        if self.use_person:
            res_net = self.choice_res if task == "choice" else self.score_res
            res = res_net(torch.cat([r.unsqueeze(1).expand_as(unit), unit], dim=-1)).squeeze(-1)
        else:
            res = torch.zeros_like(prior)
        return TTOut(prior, res, option_mask)

    def _person_mix(
        self, q: torch.Tensor, attr_vecs: torch.Tensor, attr_mask: torch.Tensor
    ) -> torch.Tensor:
        p = self.person(self.drop(self.a_proj(attr_vecs)), attr_mask)
        return self.mix(torch.cat([p, q, p * q], dim=-1))
