import numpy as np
import pandas as pd
import pytest

from schelling.eval import population as pop

pytestmark = pytest.mark.unit


def test_sampler_is_deterministic_given_seed() -> None:
    probs = [np.array([0.2, 0.8]), np.array([0.5, 0.3, 0.2])] * 10
    a = pop.sample_draws(probs, n_seeds=5, seed=3)
    b = pop.sample_draws(probs, n_seeds=5, seed=3)
    c = pop.sample_draws(probs, n_seeds=5, seed=4)
    assert (a == b).all() and not (a == c).all()


def test_sampler_frequencies_match_probabilities_within_binomial_interval() -> None:
    p = np.array([0.1, 0.6, 0.3])
    n = 10_000
    draws = pop.sample_draws([p] * n, n_seeds=1, seed=0)[:, 0]
    for k, pk in enumerate(p):
        sd = np.sqrt(n * pk * (1 - pk))
        assert abs((draws == k).sum() - n * pk) < 3.29 * sd  # 99.9% two-sided


def _synthetic(
    n_persons: int, sharpen: float, seed: int = 0
) -> tuple[pd.DataFrame, list[np.ndarray]]:
    """Humans answer from true per-person probabilities; the model reports a sharpened copy."""
    rng = np.random.default_rng(seed)
    rows, probs = [], []
    for item in range(20):
        base = rng.dirichlet(np.ones(5) * 3)  # shared item distribution; people differ around it
        for person in range(n_persons):
            true = rng.dirichlet(60 * base)
            y = int(rng.choice(5, p=true))
            model = true**sharpen
            rows.append(
                {
                    "item_id": f"i{item}",
                    "person_id": f"p{person}",
                    "source": "s",
                    "qtype": "score",
                    "response": y,
                }
            )
            probs.append(model / model.sum())
    return pd.DataFrame(rows), probs


def test_a08_correct_probabilities_give_sd_ratio_near_one() -> None:
    frame, probs = _synthetic(400, sharpen=1.0)
    res = pop.population_metrics(frame, probs, n_seeds=10, seed=0, min_persons=30)
    assert res["summary"]["sd_ratio"]["mean"] == pytest.approx(1.0, abs=0.05)


def test_a08_sharpened_probabilities_collapse_variance() -> None:
    frame, probs = _synthetic(400, sharpen=3.0)
    res = pop.population_metrics(frame, probs, n_seeds=10, seed=0, min_persons=30)
    assert res["summary"]["sd_ratio"]["mean"] < 0.9


def test_population_metrics_report_per_seed_and_skip_small_items() -> None:
    frame, probs = _synthetic(40, sharpen=1.0)
    res = pop.population_metrics(frame, probs, n_seeds=4, seed=0, min_persons=30)
    assert len(res["per_seed"]["tvd"]) == 4 and res["n_items"] == 20
    skipped = pop.population_metrics(frame, probs, n_seeds=4, seed=0, min_persons=1000)
    assert skipped["n_items"] == 0 and skipped["summary"] is None


def test_mode_collapse_rate_counts_items_where_modal_share_exceeds_human_by_point_two() -> None:
    n = 200
    frame = pd.DataFrame(
        {
            "item_id": "i",
            "person_id": [f"p{k}" for k in range(n)],
            "source": "s",
            "qtype": "score",
            "response": [k % 2 for k in range(n)],
        }
    )
    collapsed = [np.array([1.0, 0.0])] * n  # always option 0; human modal share is 0.5
    res = pop.population_metrics(frame, collapsed, n_seeds=3, seed=0, min_persons=30)
    assert res["summary"]["mode_collapse_rate"]["mean"] == 1.0


def test_subgroup_fidelity_reports_worst_group_gap() -> None:
    rng = np.random.default_rng(0)
    n = 600
    groups = np.where(np.arange(n) % 2 == 0, "a", "b")
    y = np.where(groups == "a", rng.integers(0, 2, n), 1)  # group b always answers 1
    frame = pd.DataFrame(
        {
            "item_id": "i",
            "person_id": [f"p{k}" for k in range(n)],
            "source": "s",
            "qtype": "noul",
            "response": y,
        }
    )
    probs = [np.array([0.5, 0.5])] * n  # fits group a, not group b
    persons = pd.DataFrame({"person_id": frame.person_id, "g": groups})
    res = pop.subgroup_fidelity(frame, probs, persons, ["g"], n_seeds=5, seed=0, min_persons=30)
    assert res["g"]["worst_group_gap"] > 0.3
    assert set(res["g"]["levels"]) == {"a", "b"}


def test_correlation_distance_zero_when_draws_equal_humans() -> None:
    rng = np.random.default_rng(0)
    people = [f"p{k}" for k in range(200)]
    base = rng.normal(size=200)
    rows = [
        {"item_id": f"i{j}", "person_id": p, "response": int(base[k] + rng.normal() > 0)}
        for j in range(4)
        for k, p in enumerate(people)
    ]
    frame = pd.DataFrame(rows)
    same = frame["response"].to_numpy()[:, None]
    assert pop.correlation_distance(frame, same, ["i0", "i1", "i2", "i3"]) == pytest.approx(0.0)
