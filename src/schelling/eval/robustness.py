"""Option-permutation invariance (paraphrase invariance and memorisation probe: M5)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from schelling.baselines.base import Baseline


def permutation_invariance(
    predictor: Baseline, frame: pd.DataFrame, n_perms: int = 10, seed: int = 0, max_rows: int = 500
) -> float:
    """Max |change| in the probability of the same option across random option orders."""
    rows = frame[(frame["qtype"] == "choice") & frame["options"].notna()].reset_index(drop=True)
    if rows.empty:
        return 0.0
    rng = np.random.default_rng(seed)
    if len(rows) > max_rows:
        rows = rows.iloc[np.sort(rng.choice(len(rows), max_rows, replace=False))].reset_index(
            drop=True
        )
    base = predictor.predict_proba(rows)
    worst = 0.0
    for _ in range(n_perms):
        perms = [rng.permutation(len(o)) for o in rows["options"]]
        shuffled = rows.copy()
        shuffled["options"] = [
            [o[j] for j in perm] for o, perm in zip(rows["options"], perms, strict=True)
        ]
        shuffled["response"] = [
            int(np.where(perm == y)[0][0]) for perm, y in zip(perms, rows["response"], strict=True)
        ]
        out = predictor.predict_proba(shuffled)
        for p0, p1, perm in zip(base, out, perms, strict=True):
            back = np.empty_like(p1)
            back[perm] = p1  # new position k holds original option perm[k]
            worst = max(worst, float(np.abs(back - p0).max()))
    return worst
