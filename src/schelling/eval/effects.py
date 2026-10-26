"""Treatment-effect agreement between human and synthetic populations (docs/decisions/0002).

Contrasts come from `effects_pairs.yaml`. The human reference effect uses only the evaluated
partition's persons. Two false-positive rates are reported: over human-null pairs only
(`pairs`, used for iteration) and over human-null pairs plus placebo contrasts (`pairs_and_placebo`).
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy import stats

PAIRS_PATH = Path(__file__).with_name("effects_pairs.yaml")
ALPHA = 0.05
N_PLACEBO = 20
N_BOOT = 2000


def load_pairs(path: Path = PAIRS_PATH) -> list[dict[str, str]]:
    pairs: list[dict[str, str]] = yaml.safe_load(path.read_text())["pairs"]
    ids = [p["id"] for p in pairs]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate pair ids")
    for p in pairs:
        if p["arm_a"] == p["arm_b"]:
            raise ValueError(f"{p['id']}: identical arms")
    return pairs


def experiment_of_item() -> dict[str, str]:
    return {a: p["experiment"] for p in load_pairs() for a in (p["arm_a"], p["arm_b"])}


def effect_size_agreement(
    human: np.ndarray,
    model: np.ndarray,
    null_mask: np.ndarray,
    model_significant: np.ndarray,
) -> dict[str, Any]:
    real = ~null_mask
    corr = (
        float(np.corrcoef(human, model)[0, 1])
        if len(human) > 2 and human.std() > 0 and model.std() > 0
        else None
    )
    return {
        "sign_agreement": float(np.mean(np.sign(human[real]) == np.sign(model[real])))
        if real.any()
        else None,
        "correlation": corr,
        "false_positive_rate": float(np.mean(model_significant[null_mask]))
        if null_mask.any()
        else None,
    }


def holm(p: Sequence[float]) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values."""
    p_arr = np.asarray(p, dtype=float)
    order = np.argsort(p_arr)
    adj = np.empty_like(p_arr)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(p_arr) - rank) * p_arr[idx])
        adj[idx] = min(1.0, running)
    return adj


def _d(a: np.ndarray, b: np.ndarray, sd: float) -> float:
    return float((b.mean() - a.mean()) / sd) if sd > 0 else float("nan")


def _pooled_sd(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    return float(np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2)))


def _welch_p(a: np.ndarray, b: np.ndarray) -> float:
    if a.var() == 0 and b.var() == 0:
        return 1.0 if a.mean() == b.mean() else 0.0
    return float(stats.ttest_ind(a, b, equal_var=False).pvalue)


def _none(x: float) -> float | None:
    return None if x != x else float(x)


def _mean_sd(vals: Sequence[float | None]) -> dict[str, float | None]:
    arr = np.array([np.nan if v is None else v for v in vals], dtype=float)
    if np.isnan(arr).all():
        return {"mean": None, "sd": None}
    return {
        "mean": _none(float(np.nanmean(arr))),
        "sd": _none(float(np.nanstd(arr, ddof=1))) if len(arr) > 1 else 0.0,
    }


def _placebo_groups(persons: np.ndarray, k: int) -> np.ndarray:
    return np.array([int(hashlib.sha256(f"{k}|{p}".encode()).hexdigest(), 16) % 2 for p in persons])


def run_effects(
    frame: pd.DataFrame,
    draws: np.ndarray,
    min_persons: int = 30,
    seed: int = 0,
    n_boot: int = N_BOOT,
    pairs: list[dict[str, str]] | None = None,
) -> dict[str, Any] | None:
    """`frame` and `draws` are row-aligned (rows x seeds). Returns None if no pair is estimable."""
    pairs = load_pairs() if pairs is None else pairs
    f = frame.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    n_seeds = draws.shape[1]
    rows_of = {
        item: np.flatnonzero((f["item_id"] == item).to_numpy()) for item in f["item_id"].unique()
    }
    y = f["response"].to_numpy()

    used: list[dict[str, Any]] = []
    skipped: dict[str, str] = {}
    for p in pairs:
        ia, ib = (
            rows_of.get(p["arm_a"], np.array([], dtype=int)),
            rows_of.get(p["arm_b"], np.array([], dtype=int)),
        )
        if len(ia) < min_persons or len(ib) < min_persons:
            skipped[p["id"]] = f"fewer than {min_persons} persons in an arm ({len(ia)}, {len(ib)})"
            continue
        ya, yb = y[ia], y[ib]
        sd = _pooled_sd(ya, yb)
        human_d = _d(ya, yb, sd)
        boots = np.array(
            [
                _d(ya[rng.integers(0, len(ya), len(ya))], yb[rng.integers(0, len(yb), len(yb))], sd)
                for _ in range(n_boot)
            ]
        )
        lo, hi = (float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5)))
        synth_d, p_raw = [], []
        for s in range(n_seeds):
            sa, sb = draws[ia, s].astype(float), draws[ib, s].astype(float)
            synth_d.append(_d(sa, sb, sd))
            p_raw.append(_welch_p(sa, sb))
        used.append(
            {
                "pair": p,
                "ia": ia,
                "ib": ib,
                "human_d": human_d,
                "ci": (lo, hi),
                "human_null": bool(lo <= 0 <= hi),
                "synth_d": synth_d,
                "p_raw": p_raw,
            }
        )
    if not used:
        return None

    # Holm within experiment, separately per seed.
    for u in used:
        u["sig"] = np.zeros(n_seeds, dtype=bool)
    for exp in sorted({u["pair"]["experiment"] for u in used}):
        fam = [u for u in used if u["pair"]["experiment"] == exp]
        for s in range(n_seeds):
            adj = holm([u["p_raw"][s] for u in fam])
            for u, a in zip(fam, adj, strict=True):
                u["sig"][s] = a < ALPHA

    # Placebo contrasts: split persons within one arm by a hash of person id (null by construction).
    arm_items = sorted({u["pair"][k] for u in used for k in ("arm_a", "arm_b")})
    placebo_sig = []  # contrasts x seeds
    for item in arm_items:
        rows = rows_of[item]
        persons = f.loc[rows, "person_id"].to_numpy()
        for k in range(N_PLACEBO):
            g = _placebo_groups(persons, k)
            if min((g == 0).sum(), (g == 1).sum()) < 2:
                continue
            placebo_sig.append(
                [
                    _welch_p(
                        draws[rows[g == 0], s].astype(float), draws[rows[g == 1], s].astype(float)
                    )
                    < ALPHA
                    for s in range(n_seeds)
                ]
            )
    placebo = np.array(placebo_sig, dtype=bool).reshape(-1, n_seeds)

    human = np.array([u["human_d"] for u in used])
    null_mask = np.array([u["human_null"] for u in used])
    per_seed: dict[str, list[float | None]] = {
        k: []
        for k in (
            "correlation",
            "sign_agreement",
            "false_positive_rate_pairs",
            "false_positive_rate_pairs_and_placebo",
        )
    }
    for s in range(n_seeds):
        model = np.array([u["synth_d"][s] for u in used])
        sig = np.array([u["sig"][s] for u in used])
        ok = ~(np.isnan(human) | np.isnan(model))
        agg = effect_size_agreement(human[ok], model[ok], null_mask[ok], sig[ok])
        per_seed["correlation"].append(agg["correlation"])
        per_seed["sign_agreement"].append(agg["sign_agreement"])
        per_seed["false_positive_rate_pairs"].append(agg["false_positive_rate"])
        n_null, n_pl = int(null_mask[ok].sum()), placebo.shape[0]
        per_seed["false_positive_rate_pairs_and_placebo"].append(
            (int(sig[ok][null_mask[ok]].sum()) + int(placebo[:, s].sum())) / (n_null + n_pl)
            if n_null + n_pl
            else None
        )
    return {
        "n_pairs": len(used),
        "n_human_null": int(null_mask.sum()),
        "n_placebo_contrasts": int(placebo.shape[0]),
        "iteration_metric": "false_positive_rate_pairs",
        "pairs": {
            u["pair"]["id"]: {
                "experiment": u["pair"]["experiment"],
                "n_a": len(u["ia"]),
                "n_b": len(u["ib"]),
                "human_d": _none(u["human_d"]),
                "human_ci": [u["ci"][0], u["ci"][1]],
                "human_null": u["human_null"],
                "synthetic_d_per_seed": [_none(v) for v in u["synth_d"]],
            }
            for u in used
        },
        "per_seed": per_seed,
        "summary": {k: _mean_sd(v) for k, v in per_seed.items()},
        "placebo_false_positive_rate_per_seed": [float(v) for v in placebo.mean(axis=0)]
        if placebo.size
        else [],
        "skipped": skipped,
    }
