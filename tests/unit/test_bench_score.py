"""D01-lite metrics on synthetic data with known answers."""

import numpy as np
import pandas as pd
import pytest

from schelling.bench import score

pytestmark = pytest.mark.unit

TRUE = {
    "i1": np.array([0.7, 0.3]),
    "i2": np.array([0.2, 0.5, 0.3]),
    "A": np.array([0.8, 0.2]),
    "B": np.array([0.3, 0.7]),
}


def _human(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for item, p in TRUE.items():
        for j in range(n):
            rows.append(
                {"item_id": item, "person_id": f"p{j}", "response": int(rng.choice(len(p), p=p))}
            )
    return pd.DataFrame(rows)


def _prompts(persons: list[str]) -> pd.DataFrame:
    rows = []
    for item, p in TRUE.items():
        labels = [f"o{k}" for k in range(len(p))]
        rows.append(
            {
                "prompt_id": f"{item}-pop",
                "item_id": item,
                "arm": "population",
                "person_id": None,
                "labels": labels,
            }
        )
        for who in persons:
            rows.append(
                {
                    "prompt_id": f"{item}-{who}",
                    "item_id": item,
                    "arm": "persona",
                    "person_id": who,
                    "labels": labels,
                }
            )
    return pd.DataFrame(rows)


PAIRS = [{"id": "E1", "experiment": "x", "arm_a": "A", "arm_b": "B"}]


def test_truth_scores_near_the_ceiling_and_uniform_scores_lower() -> None:
    persons = [f"p{j}" for j in range(20)]
    prompts = _prompts(persons)
    truth = {r.prompt_id: TRUE[r.item_id] for r in prompts.itertuples()}
    uniform = {
        r.prompt_id: np.full(len(TRUE[r.item_id]), 1 / len(TRUE[r.item_id]))
        for r in prompts.itertuples()
    }
    h = _human()
    good = score.score_model(prompts, truth, h, h, PAIRS, n_splits=50, n_boot=200)
    bad = score.score_model(prompts, uniform, h, h, PAIRS, n_splits=50, n_boot=200)
    # The true distribution is compared with one noisy half; a human half with another,
    # so truth beats the split-half ceiling slightly and uniform falls well short.
    assert good["population"]["accuracy_over_ceiling"] >= 0.99
    assert (
        bad["population"]["accuracy_over_ceiling"]
        < good["population"]["accuracy_over_ceiling"] - 0.1
    )
    assert good["population"]["sd_ratio_median"] == pytest.approx(1.0, abs=0.1)
    assert good["coverage"] == {"population": 1.0, "persona": 1.0}
    assert good["effects"]["persona"]["sign_agreement"] == 1.0
    assert good["effects"]["pairs"][0]["human"]["d"] > 0


def test_persona_that_knows_the_answer_lowers_person_nll() -> None:
    persons = [f"p{j}" for j in range(20)]
    prompts = _prompts(persons)
    h = _human()
    answer = {(r.person_id, r.item_id): r.response for r in h.itertuples()}
    probs = {}
    for r in prompts.itertuples():
        n = len(TRUE[r.item_id])
        if r.arm == "population":
            probs[r.prompt_id] = TRUE[r.item_id]
        else:
            q = np.full(n, 0.05 / (n - 1))
            q[answer[(r.person_id, r.item_id)]] = 0.95
            probs[r.prompt_id] = q
    out = score.score_model(prompts, probs, h, h, PAIRS, n_splits=10, n_boot=200)
    ind = out["individual"]
    assert ind["n_persons"] == 20 and ind["n_rows"] == 20 * len(TRUE)
    assert ind["nll_persona"] < ind["nll_population"]
    assert ind["persona_minus_population"]["ci_high"] < 0


def test_missing_and_unparsed_answers_fall_back_to_uniform_and_are_counted() -> None:
    prompts = _prompts(["p0", "p1"])
    probs = {r.prompt_id: TRUE[r.item_id] for r in prompts.itertuples() if r.arm == "persona"}
    probs["i1-p0"] = None
    out = score.score_model(prompts, probs, _human(), _human(), PAIRS, n_splits=5, n_boot=50)
    assert out["coverage"]["population"] == 0.0
    assert out["coverage"]["persona"] == pytest.approx(7 / 8)


def test_split_half_ceiling_is_below_one_and_compare_items_is_paired() -> None:
    rng = np.random.default_rng(0)
    resp = rng.choice(2, size=200, p=[0.6, 0.4])
    acc, ceil = score.split_half(np.array([0.6, 0.4]), resp, 100, rng)
    assert 0.8 < ceil < 1.0 and acc > 0.8
    a = [{"item_id": "x", "m": 0.9}, {"item_id": "y", "m": 0.8}]
    b = [{"item_id": "x", "m": 0.7}, {"item_id": "y", "m": 0.6}, {"item_id": "z", "m": 0.1}]
    cmp = score.compare_items(a, b, "m", 100, 0)
    assert cmp["n_items"] == 2 and cmp["mean"] == pytest.approx(0.2)


def test_item_without_population_prompt_is_left_out_of_person_comparison() -> None:
    persons = [f"p{j}" for j in range(10)]
    prompts = _prompts(persons)
    prompts = prompts[prompts["prompt_id"] != "i1-pop"]  # e.g. the reply did not parse
    probs = {
        pid: TRUE[item] for pid, item in zip(prompts["prompt_id"], prompts["item_id"], strict=True)
    }
    out = score.score_model(prompts, probs, _human(), _human(), PAIRS, n_splits=5, n_boot=50)
    assert out["individual"]["n_rows"] == 10 * (len(TRUE) - 1)
