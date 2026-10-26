"""B0: uniform over the answer options."""

from __future__ import annotations

import numpy as np
import pandas as pd

from schelling.baselines.base import Baseline, row_labels


class Uniform:
    name = "b0"

    def fit(self, train: pd.DataFrame, persons: pd.DataFrame) -> Baseline:
        return self

    def predict_proba(self, frame: pd.DataFrame) -> list[np.ndarray]:
        return [np.full(len(lab), 1 / len(lab)) for lab in row_labels(frame)]
