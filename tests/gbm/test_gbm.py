"""B3 tests. They live apart because LightGBM and torch each bundle an OpenMP runtime and
crash when loaded into one process on macOS; tests/unit/test_gbm_isolated.py runs this
directory in a fresh interpreter that never imports torch."""

import pandas as pd
import pytest

from schelling.baselines.gbm import GradientBoosted

pytestmark = pytest.mark.unit


def _n_options(row: pd.Series) -> int:  # type: ignore[type-arg]
    return 2 if row.qtype == "noul" else len(row.options if row.options is not None else row.levels)


def test_gbm_outputs_valid_distributions(twin_frame) -> None:  # type: ignore[no-untyped-def]
    frame, persons = twin_frame
    model = GradientBoosted(n_rounds=5, seed=0).fit(frame, persons)
    probs = model.predict_proba(frame.head(50))
    for p, (_, row) in zip(probs, frame.head(50).iterrows(), strict=True):
        assert len(p) == _n_options(row)
        assert p.sum() == pytest.approx(1.0) and (p >= 0).all()
