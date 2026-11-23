import numpy as np
import pandas as pd
import pytest

from schelling import compare
from schelling.eval import metrics

pytestmark = pytest.mark.unit


def _rows() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(0)
    recs, a, b = [], [], []
    for item in range(6):
        for person in range(20):
            rid = f"r{item}-{person}"
            y = int(rng.integers(0, 3))
            recs.append(
                {
                    "record_id": rid,
                    "source": "twin2k",
                    "item_id": f"i{item}",
                    "person_id": f"p{person}",
                    "response": y,
                }
            )
            good = np.full(3, 0.1)
            good[y] = 0.8
            a.append({"record_id": rid, "probs": good})
            b.append({"record_id": rid, "probs": np.full(3, 1 / 3)})
    return pd.DataFrame(recs), pd.DataFrame(a), pd.DataFrame(b)


def test_join_keeps_only_shared_rows() -> None:
    recs, a, b = _rows()
    rows = compare.join_rows(recs, a, b.iloc[:50])
    assert len(rows) == 50
    assert set(rows.columns) >= {"probs_a", "probs_b", "response", "item_id", "person_id"}


def test_unit_nll_matches_the_harness_metric() -> None:
    recs, a, b = _rows()
    rows = compare.join_rows(recs, a, b)
    units = compare.unit_nll(rows, "item_id")
    assert len(units) == 6
    g = rows[rows["item_id"] == "i0"]
    want = metrics.nll(list(g["probs_a"]), g["response"].to_numpy())
    assert units.loc[units["item_id"] == "i0", "a"].item() == pytest.approx(want)
    assert units["b"].to_numpy() == pytest.approx(np.log(3))


def test_bootstrap_sign_interval_and_determinism() -> None:
    recs, a, b = _rows()
    units = compare.unit_nll(compare.join_rows(recs, a, b), "item_id")
    out = compare.paired_bootstrap(units, n_boot=500, seed=1)
    assert out["mean_diff"] == pytest.approx(-np.log(0.8) - np.log(3))
    assert out["ci_low"] <= out["mean_diff"] <= out["ci_high"] < 0
    assert out["units_a_better"] == 6
    assert out == compare.paired_bootstrap(units, n_boot=500, seed=1)


def test_person_subsample_is_fixed_and_whole_persons() -> None:
    recs, a, b = _rows()
    rows = compare.join_rows(recs, a, b)
    s1 = compare.subsample_persons(rows, 5, seed=3)
    s2 = compare.subsample_persons(rows, 5, seed=3)
    assert s1["person_id"].nunique() == 5
    assert len(s1) == 5 * 6
    assert sorted(s1["record_id"]) == sorted(s2["record_id"])
    assert len(compare.subsample_persons(rows, 100, seed=3)) == len(rows)


def test_item_nll_of_one_run_is_the_mean_row_nll_within_each_item() -> None:
    recs, a, _ = _rows()
    got = compare.item_nll(recs, a)
    assert list(got.index) == [f"i{i}" for i in range(6)]
    assert got.to_numpy() == pytest.approx(np.full(6, -np.log(0.8)))


def test_item_nll_refuses_a_run_that_misses_rows() -> None:
    recs, a, _ = _rows()
    with pytest.raises(ValueError, match="missing"):
        compare.item_nll(recs, a.iloc[:-1])


def test_seed_mean_averages_runs_item_by_item() -> None:
    s1 = pd.Series([1.0, 3.0], index=["x", "y"])
    s2 = pd.Series([2.0, 5.0], index=["x", "y"])
    assert compare.seed_mean([s1, s2]).to_dict() == {"x": 1.5, "y": 4.0}
    with pytest.raises(ValueError, match="items"):
        compare.seed_mean([s1, s2.iloc[:1]])
