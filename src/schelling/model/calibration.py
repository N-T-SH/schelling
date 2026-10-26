"""Post-hoc temperature scaling, one scalar per head type, fitted on validation logits."""

from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import minimize_scalar

from schelling.model.heads import choice_probs, noul_probs, score_conditionals, score_probs

EPS = 1e-12


def probs_from_logits(
    task: str, logits: torch.Tensor, mask: torch.Tensor | None, temperature: float
) -> torch.Tensor:
    """Noul logits are (N,) or (N, 1); Choice (N, K) with mask; Score thresholds (N, L-1) with
    the level mask (N, L)."""
    if task == "noul":
        return noul_probs(logits.reshape(-1), temperature)
    assert mask is not None
    if task == "choice":
        return choice_probs(logits, mask, temperature)
    return score_probs(score_conditionals(logits, mask, temperature))


def nll_at_temperature(
    task: str, logits: torch.Tensor, mask: torch.Tensor | None, y: torch.Tensor, temperature: float
) -> float:
    probs = probs_from_logits(task, logits, mask, temperature)
    p = probs.gather(1, y[:, None]).squeeze(1).clamp_min(EPS)
    return float(-torch.log(p).mean())


def fit_temperature(
    task: str, logits: torch.Tensor, mask: torch.Tensor | None, y: torch.Tensor
) -> float:
    """Minimise validation NLL over log T, so T is positive by construction."""
    res = minimize_scalar(
        lambda log_t: nll_at_temperature(task, logits, mask, y, float(np.exp(log_t))),
        bounds=(-3.0, 3.0),
        method="bounded",
    )
    return float(np.exp(res.x))


def pad_stack(chunks: list[torch.Tensor], fill: float | bool) -> torch.Tensor:
    """Stack (n_i, w_i) tensors into (sum n_i, max w), padding with `fill`."""
    width = max(c.shape[1] for c in chunks)
    return torch.cat(
        [torch.nn.functional.pad(c, (0, width - c.shape[1]), value=fill) for c in chunks], dim=0
    )
