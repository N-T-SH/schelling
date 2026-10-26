"""Treatment-effect runner (docs/decisions/0002)."""

import numpy as np
import pandas as pd
import pytest

from schelling.eval import effects as fx

pytestmark = pytest.mark.unit


def test_effect_agreement_sign_correlation_and_false_positives() -> None:
    human = np.array([0.5, -0.4, 0.0, 0.0])
    model = np.array([0.4, -0.3, 0.3, 0.0])
    res = fx.effect_size_agreement(
        human,
        model,
        null_mask=np.array([False, False, True, True]),
        model_significant=np.array([True, True, True, False]),
    )
    assert res["sign_agreement"] == 1.0 and res["correlation"] > 0.8
    assert res["false_positive_rate"] == pytest.approx(0.5)


def test_pairs_file_is_well_formed() -> None:
    pairs = fx.load_pairs()
    assert len({p["id"] for p in pairs}) == len(pairs) == 16
    assert len({p["experiment"] for p in pairs}) == 8
    # An item belongs to exactly one experiment (needed for the heldout_experiment split).
    assert len(fx.experiment_of_item()) == len({a for p in pairs for a in (p["arm_a"], p["arm_b"])})


def test_holm_hand_computed() -> None:
    assert fx.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


PAIR = [{"id": "T1", "experiment": "x", "arm_a": "A", "arm_b": "B"}]


def _frame(n: int, shift: float, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ya = np.clip(np.round(rng.normal(2, 1, n)), 0, 5).astype(int)
    yb = np.clip(np.round(rng.normal(2 + shift, 1, n)), 0, 5).astype(int)
    return pd.DataFrame(
        {
            "item_id": ["A"] * n + ["B"] * n,
            "person_id": [f"a{i}" for i in range(n)] + [f"b{i}" for i in range(n)],
            "response": np.concatenate([ya, yb]),
        }
    )


def test_real_effect_is_found_and_reproduced_by_a_faithful_model() -> None:
    f = _frame(400, shift=1.0)
    draws = f["response"].to_numpy()[:, None].repeat(3, axis=1)  # model copies the humans
    res = fx.run_effects(f, draws, min_persons=30, pairs=PAIR, n_boot=300)
    assert res is not None
    p = res["pairs"]["T1"]
    assert p["human_ci"][0] > 0 and not p["human_null"]
    assert p["synthetic_d_per_seed"] == pytest.approx([p["human_d"]] * 3)
    assert res["summary"]["sign_agreement"]["mean"] == 1.0


def test_null_effect_has_interval_covering_zero_and_model_without_effect_is_not_significant() -> (
    None
):
    f = _frame(400, shift=0.0, seed=1)
    rng = np.random.default_rng(5)
    draws = rng.integers(0, 4, size=(len(f), 4))  # independent of arm
    res = fx.run_effects(f, draws, min_persons=30, pairs=PAIR, n_boot=300)
    assert res is not None and res["pairs"]["T1"]["human_null"]
    assert res["n_human_null"] == 1
    assert res["summary"]["false_positive_rate_pairs"]["mean"] is not None
    # placebo contrasts: 2 arm items x 20 hash splits
    assert res["n_placebo_contrasts"] == 40


def test_model_that_invents_an_effect_counts_as_false_positive() -> None:
    f = _frame(400, shift=0.0, seed=2)
    draws = np.where((f["item_id"] == "B").to_numpy()[:, None], 3, 0) + np.zeros(
        (len(f), 2), dtype=int
    )
    res = fx.run_effects(f, draws, min_persons=30, pairs=PAIR, n_boot=300)
    assert res is not None
    assert res["summary"]["false_positive_rate_pairs"]["mean"] == 1.0


def test_pairs_with_too_few_persons_are_skipped_and_reported() -> None:
    f = _frame(10, shift=1.0)
    res = fx.run_effects(f, np.zeros((len(f), 2), dtype=int), min_persons=30, pairs=PAIR)
    assert res is None
    f2 = pd.concat([_frame(400, 1.0), _frame(5, 1.0).assign(item_id=lambda d: d.item_id + "2")])
    pairs = [*PAIR, {"id": "T2", "experiment": "y", "arm_a": "A2", "arm_b": "B2"}]
    res2 = fx.run_effects(
        f2, f2["response"].to_numpy()[:, None], min_persons=30, pairs=pairs, n_boot=100
    )
    assert res2 is not None and "T2" in res2["skipped"]


def test_run_effects_is_deterministic() -> None:
    f = _frame(200, shift=0.5)
    draws = np.random.default_rng(0).integers(0, 4, size=(len(f), 3))
    a = fx.run_effects(f, draws, min_persons=30, pairs=PAIR, n_boot=200, seed=1)
    b = fx.run_effects(f, draws, min_persons=30, pairs=PAIR, n_boot=200, seed=1)
    assert a == b


def test_pooling_two_folds_equals_scoring_them_together() -> None:
    f1, f2 = (
        _frame(300, 1.0, seed=3),
        _frame(300, 0.0, seed=4).assign(item_id=lambda d: d.item_id.map({"A": "C", "B": "D"})),
    )
    pairs = [PAIR[0], {"id": "T2", "experiment": "y", "arm_a": "C", "arm_b": "D"}]
    both = pd.concat([f1, f2], ignore_index=True)
    draws = np.random.default_rng(0).integers(0, 4, size=(len(both), 3))
    joint = fx.run_effects(both, draws, min_persons=30, pairs=pairs, n_boot=100)
    r1 = fx.run_effects(f1, draws[: len(f1)], min_persons=30, pairs=pairs, n_boot=100)
    r2 = fx.run_effects(f2, draws[len(f1) :], min_persons=30, pairs=pairs, n_boot=100)
    pooled = fx.pool_runs([r1, r2])
    assert pooled is not None and joint is not None
    for k in pooled["per_seed"]:
        assert pooled["per_seed"][k] == joint["per_seed"][k], k
    assert pooled["n_pairs"] == joint["n_pairs"] == 2
    assert set(pooled["pairs"]) == {"T1", "T2"}


def test_pooling_rejects_a_pair_scored_twice() -> None:
    f = _frame(100, 1.0)
    r = fx.run_effects(f, f["response"].to_numpy()[:, None], min_persons=30, pairs=PAIR, n_boot=50)
    with pytest.raises(ValueError):
        fx.pool_runs([r, r])
