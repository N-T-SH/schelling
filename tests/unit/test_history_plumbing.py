"""C01 leakage tests (assumption A04): history built from a scored frame must never contain the
target, its repeats or its question text, and never another person's answers."""

import numpy as np
import pandas as pd
import pytest

from schelling.infer.history import history_for_frame

pytestmark = pytest.mark.unit


def _frame() -> pd.DataFrame:
    rows = []
    for pid in ("p1", "p2"):
        for i in range(1, 7):
            rows.append(
                {
                    "record_id": f"{pid}-{i}-w1",
                    "person_id": pid,
                    "item_id": f"Q{i}",
                    "qtype": "choice",
                    "question": f"question {i}?",
                    "options": [f"{pid}-a{i}", f"{pid}-b{i}", f"SENTINEL-{pid}-{i}"],
                    "levels": None,
                    "response": 2,
                    "wave": 1,
                }
            )
    rows.append({**rows[0], "record_id": "p1-1-w4", "wave": 4})  # retest repeat of Q1
    rows.append({**rows[1], "record_id": "dup", "item_id": "Q99"})  # same text as Q2, new id
    return pd.DataFrame(rows).reset_index(drop=True)


def _blob(hist: list) -> str:  # type: ignore[type-arg]
    return " ".join(f"{h.question} {h.options} {h.answer}" for h in hist)


def test_the_target_item_is_absent_from_its_own_history() -> None:
    f = _frame()
    for row, hist in zip(f.itertuples(), history_for_frame(f, k=50), strict=True):
        blob = _blob(hist)
        assert f"{row.question}" not in blob
        assert f"SENTINEL-{row.person_id}-{row.item_id[1:]}" not in blob or row.item_id == "Q99"


def test_wave_four_repeats_never_enter_history_and_mask_their_item() -> None:
    f = _frame()
    out = history_for_frame(f, k=50)
    repeat = f.index[f["record_id"] == "p1-1-w4"][0]
    assert "question 1?" not in _blob(out[repeat])
    for i, hist in enumerate(out):
        if f.loc[i, "person_id"] == "p1" and f.loc[i, "item_id"] != "Q1":
            assert "question 1?" in _blob(hist)  # only wave-1 copy, once
            assert sum(h.question == "question 1?" for h in hist) == 1


def test_duplicate_question_text_is_masked_even_under_a_new_item_id() -> None:
    f = _frame()
    out = history_for_frame(f, k=50)
    dup = f.index[f["record_id"] == "dup"][0]
    assert "question 2?" not in _blob(out[dup])
    q2 = f.index[(f["item_id"] == "Q2") & (f["person_id"] == "p1") & (f["wave"] == 1)][0]
    assert "question 2?" not in _blob(out[q2])


def test_history_never_crosses_persons() -> None:
    f = _frame()
    for row, hist in zip(f.itertuples(), history_for_frame(f, k=50), strict=True):
        other = "p2" if row.person_id == "p1" else "p1"
        assert other not in _blob(hist)


def test_changing_only_the_targets_answer_leaves_its_history_unchanged() -> None:
    f = _frame()
    base = history_for_frame(f, k=50)
    g = f.copy()
    i = 3
    g.loc[i, "response"] = 0
    g.at[i, "options"] = ["changed", "changed", "changed"]
    assert history_for_frame(g, k=50)[i] == base[i]


def test_k_limits_history_and_selection_is_deterministic() -> None:
    f = _frame()
    a, b = history_for_frame(f, k=3), history_for_frame(f, k=3)
    assert a == b and all(len(h) <= 3 for h in a)
    assert history_for_frame(f, k=0) == [[] for _ in range(len(f))]


def test_row_order_of_the_frame_does_not_change_a_rows_history() -> None:
    f = _frame()
    base = dict(zip(f["record_id"], history_for_frame(f, k=4), strict=True))
    perm = f.sample(frac=1.0, random_state=np.random.RandomState(0)).reset_index(drop=True)
    shuffled = dict(zip(perm["record_id"], history_for_frame(perm, k=4), strict=True))
    assert shuffled == base


def test_noul_and_score_rows_become_valid_history_items() -> None:
    f = pd.DataFrame(
        [
            {
                "record_id": "a",
                "person_id": "p",
                "item_id": "N1",
                "qtype": "noul",
                "question": "do you?",
                "options": None,
                "levels": None,
                "response": 1,
                "wave": 1,
            },
            {
                "record_id": "b",
                "person_id": "p",
                "item_id": "S1",
                "qtype": "score",
                "question": "how much?",
                "options": None,
                "levels": np.array(["lo", "hi"]),
                "response": 0,
                "wave": 1,
            },
            {
                "record_id": "c",
                "person_id": "p",
                "item_id": "C1",
                "qtype": "choice",
                "question": "pick",
                "options": np.array(["x", "y"]),
                "levels": None,
                "response": 1,
                "wave": 1,
            },
        ]
    )
    out = history_for_frame(f, k=5)
    assert [(h.question, h.options[h.answer]) for h in out[2]] == [
        ("do you?", "yes"),
        ("how much?", "lo"),
    ]


def test_scored_items_carry_history_without_the_target_and_default_to_none() -> None:
    from schelling.infer.baseline_adapter import frame_items

    f = _frame()
    f = f.assign(state="")
    plain = frame_items(f, {})
    assert all(req.person.history == [] for req, _ in plain)
    with_hist = frame_items(f, {}, history_k=50)
    for row, (req, q) in zip(f.itertuples(), with_hist, strict=True):
        assert row.question not in _blob(req.person.history)
        assert getattr(q, "prompt", getattr(q, "proposition", "")) == row.question


def test_model_baseline_history_is_off_for_models_scored_before_c01() -> None:
    from tests.helpers import tiny_tokenizer, tiny_tt_model

    from schelling.infer.baseline_adapter import ModelBaseline
    from schelling.infer.predictor import Predictor

    m = tiny_tt_model(seed=0)
    assert ModelBaseline(Predictor(m, tiny_tokenizer())).history_k == 0
    m.config.history_items = 7
    assert ModelBaseline(Predictor(m, tiny_tokenizer())).history_k == 7
