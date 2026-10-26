"""Common protocol for baselines (and, from M3, the model) so the harness scores them alike."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd


class Baseline(Protocol):
    name: str

    def fit(self, train: pd.DataFrame, persons: pd.DataFrame) -> Baseline: ...

    def predict_proba(self, frame: pd.DataFrame) -> list[np.ndarray]: ...


def labels_of(qtype: str, options: list[str] | None, levels: list[str] | None) -> list[str]:
    """Text label of each answer index; counts keyed by label are invariant to option order."""
    if qtype == "noul":
        return ["no", "yes"]
    out = options if options is not None else levels
    assert out is not None
    return list(out)


def row_labels(frame: pd.DataFrame) -> list[list[str]]:
    return [
        labels_of(q, None if o is None else list(o), None if lv is None else list(lv))
        for q, o, lv in zip(frame["qtype"], frame["options"], frame["levels"], strict=True)
    ]
