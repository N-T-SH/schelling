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
        self,
        hidden: int,
        d: int,
        person_heads: int,
        dropout: float,
        use_person: bool,
        use_history: bool = False,
    ) -> None:
        super().__init__()
        self.use_person = use_person
        self.use_history = use_history
        self.d = d
        self.u_proj, self.e_proj, self.a_proj = (nn.Linear(hidden, d) for _ in range(3))
        self.drop = nn.Dropout(dropout)
        # Per-dimension input statistics. Frozen mean-pooled backbone vectors of different items
        # are nearly parallel (median cosine about 0.95), so they are standardised first. Identity
        # until `set_input_stats` is called with the training vectors.
        self.u_mean: torch.Tensor
        self.u_std: torch.Tensor
        self.e_mean: torch.Tensor
        self.e_std: torch.Tensor
        self.a_mean: torch.Tensor
        self.a_std: torch.Tensor
        for name in ("u", "e", "a"):
            self.register_buffer(f"{name}_mean", torch.zeros(hidden))
            self.register_buffer(f"{name}_std", torch.ones(hidden))
        # prior: item (and options) only
        self.noul_prior = _mlp(d, d, 1)
        self.choice_prior = _mlp(3 * d, d, 1)
        self.score_prior = _mlp(3 * d, d, 1)
        # residual: person-dependent, zero-initialised last layers
        self.person = PersonEncoder(d, person_heads, dropout)
        # history tower (C01): each earlier answer is (question vector, chosen option vector); the
        # set is pooled by attention keyed to the target question. Absent in M4b models.
        self.mix = _mlp((5 if use_history else 3) * d, d, d)
        if use_history:
            self.hist_item = _mlp(2 * d, d, d)
            self.hist_query = nn.Linear(d, d)
            self.hist_key = nn.Linear(d, d)
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

    @torch.no_grad()
    def set_input_stats(self, u: torch.Tensor, e: torch.Tensor, a: torch.Tensor) -> None:
        """Fit the standardisation on the (unique) training vectors of each kind."""
        for name, x in (("u", u), ("e", e), ("a", a)):
            if len(x) > 1:
                getattr(self, f"{name}_mean").copy_(x.mean(0))
                getattr(self, f"{name}_std").copy_(x.std(0).clamp_min(1e-3))

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
        hist_u: torch.Tensor | None = None,
        hist_e: torch.Tensor | None = None,
        hist_mask: torch.Tensor | None = None,
        hist_tables: tuple[torch.Tensor, torch.Tensor] | None = None,
        hist_idx: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> TTOut:
        """u (B, H) question vectors; attr_vecs (B, A, H); e (B, K, H) option or level vectors;
        hist_u and hist_e (B, L, H) are the earlier questions and the options the person chose.
        The same history can instead be given as vector tables plus (B, L) row indices, which is
        mathematically identical and far cheaper when many rows share a few hundred vectors."""
        u = (u - self.u_mean) / self.u_std
        attr_vecs = (attr_vecs - self.a_mean) / self.a_std
        q = self.drop(self.u_proj(u))
        if task == "noul":
            prior = self.noul_prior(q).squeeze(-1)
            r = self._person_mix(
                q, attr_vecs, attr_mask, hist_u, hist_e, hist_mask, hist_tables, hist_idx
            )
            res = self.noul_res(r).squeeze(-1) if self.use_person else torch.zeros_like(prior)
            return TTOut(prior, res, None)
        assert e is not None and option_mask is not None
        ev = self.drop(self.e_proj((e - self.e_mean) / self.e_std))
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
        r = self._person_mix(
            q, attr_vecs, attr_mask, hist_u, hist_e, hist_mask, hist_tables, hist_idx
        )
        if self.use_person:
            res_net = self.choice_res if task == "choice" else self.score_res
            res = res_net(torch.cat([r.unsqueeze(1).expand_as(unit), unit], dim=-1)).squeeze(-1)
        else:
            res = torch.zeros_like(prior)
        return TTOut(prior, res, option_mask)

    def _history(
        self,
        q: torch.Tensor,
        hist_u: torch.Tensor | None,
        hist_e: torch.Tensor | None,
        hist_mask: torch.Tensor | None,
        tables: tuple[torch.Tensor, torch.Tensor] | None = None,
        idx: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Attention over the person's earlier answers; zeros when there are none."""
        if hist_mask is None:
            return torch.zeros_like(q)
        if tables is not None and idx is not None:
            # Project the few hundred distinct vectors once, then gather. The first layer of
            # `hist_item` is split so it too runs on the tables rather than on every history row.
            pu = self.drop(self.u_proj((tables[0] - self.u_mean) / self.u_std))
            pe = self.drop(self.e_proj((tables[1] - self.e_mean) / self.e_std))
            first, last = self.hist_item[0], self.hist_item[2]
            assert isinstance(first, nn.Linear)
            w = first.weight
            a = pu @ w[:, : self.d].T + first.bias
            b = pe @ w[:, self.d :].T
            x = last(torch.nn.functional.gelu(a[idx[0].clamp(min=0)] + b[idx[1].clamp(min=0)]))
        elif hist_u is not None and hist_e is not None:
            hu = self.drop(self.u_proj((hist_u - self.u_mean) / self.u_std))
            he = self.drop(self.e_proj((hist_e - self.e_mean) / self.e_std))
            x = self.hist_item(torch.cat([hu, he], dim=-1))
        else:
            return torch.zeros_like(q)
        score = (self.hist_query(q).unsqueeze(1) * self.hist_key(x)).sum(-1) / self.d**0.5
        alpha = torch.softmax(score.masked_fill(~hist_mask, -1e9), dim=-1) * hist_mask
        return (alpha.unsqueeze(-1) * x).sum(1)  # an all-masked row gives zeros

    def _person_mix(
        self,
        q: torch.Tensor,
        attr_vecs: torch.Tensor,
        attr_mask: torch.Tensor,
        hist_u: torch.Tensor | None = None,
        hist_e: torch.Tensor | None = None,
        hist_mask: torch.Tensor | None = None,
        hist_tables: tuple[torch.Tensor, torch.Tensor] | None = None,
        hist_idx: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        p = self.person(self.drop(self.a_proj(attr_vecs)), attr_mask)
        if not self.use_history:
            return self.mix(torch.cat([p, q, p * q], dim=-1))
        h = self._history(q, hist_u, hist_e, hist_mask, hist_tables, hist_idx)
        return self.mix(torch.cat([p, h, q, p * q, h * q], dim=-1))
