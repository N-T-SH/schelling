"""B1: marginal human distribution per item from the train persons.

Counts are keyed by option text, so the baseline is invariant to option order by construction.
`alpha` is a Jeffreys-style pseudo-count per option; with alpha=0 it is the exact marginal.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from schelling.baselines.base import Baseline, row_labels


def _counts(frame: pd.DataFrame) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for item, labels, y in zip(frame["item_id"], row_labels(frame), frame["response"], strict=True):
        out[item][labels[y]] += 1
    return out


def _dist(labels: list[str], counts: dict[str, int] | None, alpha: float) -> np.ndarray:
    c = np.array([(counts or {}).get(lab, 0) + alpha for lab in labels], dtype=float)
    return c / c.sum() if c.sum() > 0 else np.full(len(labels), 1 / len(labels))


class ItemMean:
    name = "b1"

    def __init__(self, alpha: float = 0.5) -> None:
        self.alpha = alpha
        self.counts: dict[str, dict[str, int]] = {}
        self.n_unseen = 0

    def fit(self, train: pd.DataFrame, persons: pd.DataFrame) -> Baseline:
        self.counts = _counts(train)
        return self

    def predict_proba(self, frame: pd.DataFrame) -> list[np.ndarray]:
        self.n_unseen = 0
        out = []
        for item, labels in zip(frame["item_id"], row_labels(frame), strict=True):
            seen = item in self.counts
            self.n_unseen += not seen
            out.append(_dist(labels, self.counts.get(item) if seen else None, self.alpha))
        return out
