"""Paired comparison of two saved prediction files on the same rows.

Uses `schelling.eval.metrics.nll` for every number, so nothing here redefines a metric. The
bootstrap resamples whole units (items or persons): rows inside a unit are not independent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from schelling.eval import metrics


def load_predictions(run_dir: Path) -> pd.DataFrame:
    return pd.read_parquet(run_dir / "predictions.parquet")[["record_id", "probs"]]


def join_rows(records: pd.DataFrame, a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Rows present in both prediction files, with the observed response attached."""
    keep = records[["record_id", "source", "item_id", "person_id", "response"]]
    out = keep.merge(a.rename(columns={"probs": "probs_a"}), on="record_id", validate="1:1")
    return out.merge(b.rename(columns={"probs": "probs_b"}), on="record_id", validate="1:1")


def subsample_persons(rows: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """A fixed, seeded set of persons, the same for every model compared."""
    persons = np.sort(rows["person_id"].unique())
    if n >= len(persons):
        return rows
    chosen = np.random.default_rng(seed).choice(persons, size=n, replace=False)
    return rows[rows["person_id"].isin(chosen)]


def unit_nll(rows: pd.DataFrame, unit: str) -> pd.DataFrame:
    """Mean NLL of each model within each unit (one line per item or person)."""
    out = []
    for key, g in rows.groupby(unit, sort=True):
        y = g["response"].to_numpy(dtype=int)
        out.append(
            {
                unit: key,
                "n": len(g),
                "a": metrics.nll([np.asarray(p) for p in g["probs_a"]], y),
                "b": metrics.nll([np.asarray(p) for p in g["probs_b"]], y),
            }
        )
    return pd.DataFrame(out)


def paired_bootstrap(
    units: pd.DataFrame, n_boot: int = 10_000, seed: int = 0, level: float = 0.95
) -> dict[str, Any]:
    """Mean over units of (a - b), with a percentile interval from resampling units."""
    diff = (units["a"] - units["b"]).to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    means = diff[idx].mean(axis=1)
    lo, hi = np.quantile(means, [(1 - level) / 2, 1 - (1 - level) / 2])
    return {
        "n_units": len(diff),
        "mean_a": float(units["a"].mean()),
        "mean_b": float(units["b"].mean()),
        "mean_diff": float(diff.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "units_a_better": int((diff < 0).sum()),
        "n_boot": n_boot,
        "seed": seed,
        "level": level,
    }
