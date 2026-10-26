"""Population-level fidelity: one draw per held-out person per item, repeated over seeds."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from schelling.eval import metrics as m

ORDERED = ("score", "noul")  # qtypes where SD and Wasserstein over the index are meaningful
MODE_COLLAPSE_MARGIN = 0.2


def sample_draws(probs: Sequence[np.ndarray], n_seeds: int, seed: int) -> np.ndarray:
    """Inverse-CDF draws, shape (rows, n_seeds); deterministic given `seed`."""
    rng = np.random.default_rng(seed)
    n_rows = len(probs)
    u = rng.random((n_rows, n_seeds))
    out = np.empty((n_rows, n_seeds), dtype=np.int64)
    width = max(len(p) for p in probs)
    for lo in range(0, n_rows, 20_000):
        hi = min(lo + 20_000, n_rows)
        padded = np.zeros((hi - lo, width))
        sizes = np.array([len(p) for p in probs[lo:hi]])
        for k, p in enumerate(probs[lo:hi]):
            padded[k, : len(p)] = p
        cdf = np.cumsum(padded, axis=1)
        draws = (u[lo:hi, :, None] > cdf[:, None, :]).sum(axis=2)
        out[lo:hi] = np.minimum(draws, (sizes - 1)[:, None])
    return out


def _none(x: float) -> float | None:
    return None if x != x else float(x)  # NaN is not valid JSON


def _summary(per_seed: dict[str, list[float | None]]) -> dict[str, dict[str, float | None]]:
    out: dict[str, dict[str, float | None]] = {}
    for name, vals in per_seed.items():
        arr = np.array([np.nan if v is None else v for v in vals], dtype=float)
        if np.isnan(arr).all():
            out[name] = {"mean": None, "sd": None}
        else:
            out[name] = {
                "mean": _none(float(np.nanmean(arr))),
                "sd": _none(float(np.nanstd(arr, ddof=1))) if len(arr) > 1 else 0.0,
            }
    return out


def population_from_draws(
    frame: pd.DataFrame, probs: Sequence[np.ndarray], draws: np.ndarray, min_persons: int
) -> dict[str, Any]:
    """`frame`, `probs` and `draws` are row-aligned. Items with fewer persons are skipped."""
    n_seeds = draws.shape[1]
    positions = frame.reset_index(drop=True).groupby("item_id", sort=True).indices
    names = ("tvd", "w1", "sd_ratio", "entropy_ratio", "mode_collapse_rate")
    per_item: list[dict[str, np.ndarray]] = []
    y = frame["response"].to_numpy()
    qtypes = frame["qtype"].to_numpy()
    for idx in positions.values():
        if len(idx) < min_persons:
            continue
        n = len(probs[idx[0]])
        human = np.bincount(y[idx], minlength=n) / len(idx)
        ordered = qtypes[idx[0]] in ORDERED
        vals = {k: np.full(n_seeds, np.nan) for k in names}
        for s in range(n_seeds):
            synth = np.bincount(draws[idx, s], minlength=n) / len(idx)
            vals["tvd"][s] = m.tvd(synth, human)
            vals["entropy_ratio"][s] = m.entropy_ratio(synth, human)
            vals["mode_collapse_rate"][s] = float(synth.max() - human.max() > MODE_COLLAPSE_MARGIN)
            if ordered:
                vals["w1"][s] = m.wasserstein1(synth, human)
                vals["sd_ratio"][s] = m.sd_ratio(synth, human)
        per_item.append(vals)
    if not per_item:
        return {"n_items": 0, "per_seed": {}, "summary": None}
    per_seed: dict[str, list[float | None]] = {}
    for k in names:
        stack = np.vstack([v[k] for v in per_item])  # items x seeds
        if k == "mode_collapse_rate":
            agg = stack.mean(axis=0)
        elif np.isnan(stack).all():
            agg = np.full(n_seeds, np.nan)
        else:
            agg = np.nanmedian(stack, axis=0)
        per_seed[k] = [_none(float(v)) for v in agg]
    return {"n_items": len(per_item), "per_seed": per_seed, "summary": _summary(per_seed)}


def population_metrics(
    frame: pd.DataFrame,
    probs: Sequence[np.ndarray],
    n_seeds: int = 20,
    seed: int = 0,
    min_persons: int = 30,
) -> dict[str, Any]:
    return population_from_draws(frame, probs, sample_draws(probs, n_seeds, seed), min_persons)


def subgroup_fidelity(
    frame: pd.DataFrame,
    probs: Sequence[np.ndarray],
    persons: pd.DataFrame,
    columns: Sequence[str],
    n_seeds: int = 20,
    seed: int = 0,
    min_persons: int = 30,
    draws: np.ndarray | None = None,
) -> dict[str, Any]:
    """Median-over-items TVD per subgroup level, and the gap between worst and best level."""
    draws = sample_draws(probs, n_seeds, seed) if draws is None else draws
    merged = frame.reset_index(drop=True).merge(
        persons[["person_id", *columns]], on="person_id", how="left"
    )
    result: dict[str, Any] = {}
    for col in columns:
        levels: dict[str, float] = {}
        for level in sorted(merged[col].dropna().unique()):
            mask = (merged[col] == level).to_numpy()
            sub_probs = [p for p, keep in zip(probs, mask, strict=True) if keep]
            res = population_from_draws(merged[mask], sub_probs, draws[mask], min_persons)
            if res["summary"] is not None and res["summary"]["tvd"]["mean"] is not None:
                levels[str(level)] = float(res["summary"]["tvd"]["mean"])
        gap = max(levels.values()) - min(levels.values()) if len(levels) > 1 else None
        result[col] = {"levels": levels, "worst_group_gap": gap}
    return result


def correlation_distance(frame: pd.DataFrame, draws: np.ndarray, item_ids: Sequence[str]) -> float:
    """Mean over seeds of the Frobenius distance between item-by-item correlation matrices."""
    f = frame.reset_index(drop=True)
    keep = f["item_id"].isin(item_ids).to_numpy()
    f = f[keep]
    draws = draws[keep]

    def corr(values: np.ndarray) -> np.ndarray:
        wide = pd.DataFrame({"p": f["person_id"], "i": f["item_id"], "v": values}).pivot_table(
            index="p", columns="i", values="v", aggfunc="first"
        )
        return wide.reindex(columns=list(item_ids)).corr(min_periods=10).fillna(0.0).to_numpy()

    human = corr(f["response"].to_numpy())
    dists = [float(np.linalg.norm(corr(draws[:, s]) - human)) for s in range(draws.shape[1])]
    return float(np.mean(dists))
