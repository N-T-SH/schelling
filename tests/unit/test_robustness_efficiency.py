import numpy as np
import pandas as pd
import pytest

from schelling.baselines.item_mean import ItemMean
from schelling.eval.efficiency import latency
from schelling.eval.robustness import permutation_invariance

pytestmark = pytest.mark.unit


class PositionBiased:
    """Gives more mass to whichever option is listed first, regardless of its text."""

    def fit(self, train, persons):  # type: ignore[no-untyped-def]
        return self

    def predict_proba(self, frame: pd.DataFrame) -> list[np.ndarray]:
        out = []
        for _, r in frame.iterrows():
            n = 2 if r.qtype == "noul" else len(r.options if r.options is not None else r.levels)
            p = np.ones(n)
            p[0] = 3.0
            out.append(p / p.sum())
        return out


def test_item_mean_is_permutation_invariant(twin_frame) -> None:  # type: ignore[no-untyped-def]
    frame, persons = twin_frame
    model = ItemMean().fit(frame, persons)
    assert permutation_invariance(model, frame, n_perms=10, seed=0) < 1e-9


def test_position_bias_is_detected(twin_frame) -> None:  # type: ignore[no-untyped-def]
    frame, _ = twin_frame
    assert permutation_invariance(PositionBiased(), frame, n_perms=10, seed=0) > 0.1  # type: ignore[arg-type]


def test_latency_keys_and_ordering(twin_frame) -> None:  # type: ignore[no-untyped-def]
    frame, persons = twin_frame
    res = latency(ItemMean().fit(frame, persons), frame, n_requests=30)
    assert res["p95_ms"] >= res["p50_ms"] > 0 and res["decisions_per_s"] > 0
