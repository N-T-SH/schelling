import numpy as np
import pandas as pd
import pytest

from schelling.baselines.demo_cell import DemoCellMean
from schelling.baselines.gbm import GradientBoosted
from schelling.baselines.item_mean import ItemMean
from schelling.baselines.uniform import Uniform

pytestmark = pytest.mark.unit


def _n_options(row: pd.Series) -> int:  # type: ignore[type-arg]
    return 2 if row.qtype == "noul" else len(row.options if row.options is not None else row.levels)


def test_uniform(twin_frame) -> None:  # type: ignore[no-untyped-def]
    frame, persons = twin_frame
    probs = Uniform().fit(frame, persons).predict_proba(frame.head(20))
    for p, (_, row) in zip(probs, frame.head(20).iterrows(), strict=True):
        assert p == pytest.approx(np.full(_n_options(row), 1 / _n_options(row)))


def test_item_mean_reproduces_train_marginal_exactly(twin_frame) -> None:  # type: ignore[no-untyped-def]
    frame, persons = twin_frame
    model = ItemMean(alpha=0.0).fit(frame, persons)
    item = frame[frame.item_id == "QID102"]
    expected = np.bincount(item.response, minlength=4) / len(item)
    got = model.predict_proba(item.head(1))[0]
    assert got == pytest.approx(expected)


def test_item_mean_pseudo_count_keeps_probabilities_positive(twin_frame) -> None:  # type: ignore[no-untyped-def]
    frame, persons = twin_frame
    probs = ItemMean().fit(frame, persons).predict_proba(frame)
    assert all((p > 0).all() and p.sum() == pytest.approx(1.0) for p in probs)


def test_item_mean_falls_back_to_uniform_on_unseen_item(twin_frame) -> None:  # type: ignore[no-untyped-def]
    frame, persons = twin_frame
    train = frame[frame.item_id != "QID102"]
    model = ItemMean().fit(train, persons)
    p = model.predict_proba(frame[frame.item_id == "QID102"].head(1))[0]
    assert p == pytest.approx(np.full(4, 0.25))
    assert model.n_unseen == 1


def _cell_frame(n_per_cell: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, persons = [], []
    for sex in ["Male", "Female"]:
        for k in range(n_per_cell):
            pid = f"{sex}{k}"
            persons.append({"person_id": pid, "sex": sex, "age_band": "30-49", "education": "HS"})
            rows.append(
                {
                    "item_id": "q",
                    "person_id": pid,
                    "qtype": "choice",
                    "options": ["a", "b"],
                    "levels": None,
                    "response": 0 if sex == "Male" else 1,
                    "source": "twin2k",
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(persons)


def test_demo_cell_backs_off_to_item_mean_for_small_cells() -> None:
    frame, persons = _cell_frame(10)  # < 20 people per cell
    b2 = DemoCellMean(min_cell=20, alpha=0.5).fit(frame, persons).predict_proba(frame.head(1))[0]
    b1 = ItemMean(alpha=0.5).fit(frame, persons).predict_proba(frame.head(1))[0]
    assert b2 == pytest.approx(b1)


def test_demo_cell_uses_cell_when_large_enough() -> None:
    frame, persons = _cell_frame(30)
    p = DemoCellMean(min_cell=20, alpha=0.5).fit(frame, persons).predict_proba(frame.head(1))[0]
    assert p[0] > 0.9  # males all answered option 0


def test_demo_cell_equals_item_mean_when_source_has_no_attributes() -> None:
    frame, _ = _cell_frame(30)
    frame["source"] = "psych101"
    empty = pd.DataFrame({"person_id": []})
    a = DemoCellMean().fit(frame, empty).predict_proba(frame)
    b = ItemMean().fit(frame, empty).predict_proba(frame)
    assert all(np.allclose(x, y) for x, y in zip(a, b, strict=True))


def test_gbm_outputs_valid_distributions(twin_frame) -> None:  # type: ignore[no-untyped-def]
    frame, persons = twin_frame
    model = GradientBoosted(n_rounds=5, seed=0).fit(frame, persons)
    probs = model.predict_proba(frame.head(50))
    for p, (_, row) in zip(probs, frame.head(50).iterrows(), strict=True):
        assert len(p) == _n_options(row)
        assert p.sum() == pytest.approx(1.0) and (p >= 0).all()
