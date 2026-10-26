"""B3: gradient-boosted classifier on person attributes plus item id (LightGBM).

One multiclass model per (qtype, number of options) group, because answer indices mean
different things per item and the item id is a categorical feature. Features are positional,
so unlike B1/B2 this baseline is not option-order invariant; the robustness check reports it.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from schelling.baselines.base import Baseline, row_labels


class GradientBoosted:
    name = "b3"

    def __init__(
        self, n_rounds: int = 100, seed: int = 0, learning_rate: float = 0.1, num_leaves: int = 31
    ) -> None:
        self.n_rounds, self.seed = n_rounds, seed
        self.params = {
            "objective": "multiclass",
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "min_data_in_leaf": 20,
            "verbosity": -1,
            "seed": seed,
            "deterministic": True,
            "force_row_wise": True,
            "num_threads": 4,
            "cat_smooth": 10,
        }
        self.models: dict[tuple[str, int], lgb.Booster | np.ndarray] = {}
        self.attr_cols: list[str] = []
        self.attr_codes: dict[str, dict[str, int]] = {}
        self.item_codes: dict[str, int] = {}
        self.persons = pd.DataFrame()

    def _features(self, frame: pd.DataFrame) -> np.ndarray:
        cols = [frame["item_id"].map(self.item_codes).to_numpy(dtype=float)]
        merged = frame[["person_id"]].merge(self.persons, on="person_id", how="left")
        for c in self.attr_cols:
            cols.append(merged[c].map(self.attr_codes[c]).to_numpy(dtype=float))
        return np.column_stack(cols)

    def fit(self, train: pd.DataFrame, persons: pd.DataFrame) -> Baseline:
        self.persons = persons.assign(person_id=persons["person_id"].astype(str)).drop_duplicates(
            "person_id"
        )
        self.attr_cols = [c for c in self.persons.columns if c != "person_id"]
        self.attr_codes = {
            c: {v: i for i, v in enumerate(sorted(self.persons[c].dropna().unique()))}
            for c in self.attr_cols
        }
        self.item_codes = {v: i for i, v in enumerate(sorted(train["item_id"].unique()))}
        x_all = self._features(train)
        sizes = np.array([len(lab) for lab in row_labels(train)])
        y_all = train["response"].to_numpy()
        keys = list(zip(train["qtype"], sizes, strict=True))
        for key in sorted(set(keys)):
            mask = np.array([k == key for k in keys])
            n = int(key[1])
            y = y_all[mask]
            if mask.sum() < 50 or len(np.unique(y)) < 2:
                counts = np.bincount(y, minlength=n) + 0.5
                self.models[(key[0], n)] = counts / counts.sum()
                continue
            ds = lgb.Dataset(x_all[mask], y, categorical_feature=[0], free_raw_data=False)
            self.models[(key[0], n)] = lgb.train({**self.params, "num_class": n}, ds, self.n_rounds)
        return self

    def predict_proba(self, frame: pd.DataFrame) -> list[np.ndarray]:
        sizes = np.array([len(lab) for lab in row_labels(frame)])
        x = self._features(frame)
        out: list[np.ndarray | None] = [None] * len(frame)
        qtypes = frame["qtype"].to_numpy()
        for (qtype, n), model in self.models.items():
            mask = (qtypes == qtype) & (sizes == n)
            if not mask.any():
                continue
            probs = (
                np.tile(model, (int(mask.sum()), 1))
                if isinstance(model, np.ndarray)
                else model.predict(x[mask])
            )
            probs = np.clip(probs, 1e-6, None)
            probs = probs / probs.sum(axis=1, keepdims=True)
            for pos, p in zip(np.flatnonzero(mask), probs, strict=True):
                out[pos] = p
        return [np.full(int(sizes[i]), 1 / sizes[i]) if p is None else p for i, p in enumerate(out)]
