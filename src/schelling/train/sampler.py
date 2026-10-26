"""Temperature sampling over studies so a huge source does not drown the small ones."""

from __future__ import annotations

import numpy as np
import pandas as pd


class StudySampler:
    """Draw record indices: a study with probability proportional to n**temperature, then a
    record uniformly within it. temperature 1 is the natural mix, 0 equalises studies."""

    def __init__(self, studies: pd.Series, temperature: float, seed: int) -> None:
        codes, uniques = pd.factorize(studies.to_numpy())
        self.members = [np.flatnonzero(codes == i) for i in range(len(uniques))]
        sizes = np.array([len(m) for m in self.members], dtype=float)
        weights = sizes**temperature
        self.p = weights / weights.sum()
        self.rng = np.random.default_rng(seed)

    def sample(self, n: int) -> np.ndarray:
        which = self.rng.choice(len(self.members), size=n, p=self.p)
        out = np.empty(n, dtype=np.int64)
        for s in np.unique(which):
            sel = np.flatnonzero(which == s)
            m = self.members[s]
            out[sel] = m[self.rng.integers(0, len(m), size=len(sel))]
        return out
