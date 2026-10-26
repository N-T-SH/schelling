"""B2: demographic-cell mean, backing off to B1 when a cell has fewer than `min_cell` people.

Cells are defined by the columns of `CELL_COLUMNS` that exist in the persons table. Records of
persons without attributes (Psych-101) use B1.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from schelling.baselines.base import Baseline, row_labels
from schelling.baselines.item_mean import ItemMean, _dist

CELL_COLUMNS = ("sex", "age_band", "education")


class DemoCellMean:
    name = "b2"

    def __init__(self, min_cell: int = 20, alpha: float = 0.5) -> None:
        self.min_cell, self.alpha = min_cell, alpha
        self.b1 = ItemMean(alpha)
        self.cell_of: dict[str, tuple[str, ...]] = {}
        self.counts: dict[tuple[str, tuple[str, ...]], dict[str, int]] = {}
        self.sizes: dict[tuple[str, tuple[str, ...]], int] = {}

    def _assign_cells(self, persons: pd.DataFrame) -> None:
        cols = [c for c in CELL_COLUMNS if c in persons.columns]
        self.cell_of = {}
        if cols:
            for pid, *vals in persons[["person_id", *cols]].itertuples(index=False):
                self.cell_of[str(pid)] = tuple(str(v) for v in vals)

    def fit(self, train: pd.DataFrame, persons: pd.DataFrame) -> Baseline:
        self.b1.fit(train, persons)
        self._assign_cells(persons)
        counts: dict[tuple[str, tuple[str, ...]], dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        sizes: dict[tuple[str, tuple[str, ...]], int] = defaultdict(int)
        for pid, item, labels, y in zip(
            train["person_id"], train["item_id"], row_labels(train), train["response"], strict=True
        ):
            cell = self.cell_of.get(str(pid))
            if cell is not None:
                counts[(item, cell)][labels[y]] += 1
                sizes[(item, cell)] += 1
        self.counts, self.sizes = dict(counts), dict(sizes)
        return self

    def predict_proba(self, frame: pd.DataFrame) -> list[np.ndarray]:
        fallback = self.b1.predict_proba(frame)
        out = []
        for pid, item, labels, base in zip(
            frame["person_id"], frame["item_id"], row_labels(frame), fallback, strict=True
        ):
            cell = self.cell_of.get(str(pid))
            key = (item, cell) if cell is not None else None
            if key is not None and self.sizes.get(key, 0) >= self.min_cell:
                out.append(_dist(labels, self.counts[key], self.alpha))
            else:
                out.append(base)
        return out
