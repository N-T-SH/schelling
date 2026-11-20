"""D01-lite: decision-item selection and prompt building. No network, no models."""

import numpy as np
import pandas as pd
import pytest

from schelling.bench import prompts
from schelling.bench.items import DECISION_STUDIES, select_items, select_persons

pytestmark = pytest.mark.unit


def _row(item: str, study: str, cond: str | None = None, qtype: str = "choice") -> dict:
    return {
        "item_id": item,
        "study_id": study,
        "condition": cond,
        "qtype": qtype,
        "question": f"question {item}?",
        "options": ["left", "right"] if qtype == "choice" else None,
        "levels": ["low", "mid", "high"] if qtype == "score" else None,
        "person_id": "p0",
        "response": 0,
        "source": "twin2k",
    }


def _items() -> pd.DataFrame:
    rows = [_row(f"QID84_{i}", "Economic preferences") for i in range(1, 11)]  # price list
    rows += [_row("QID117", "Economic preferences")]  # single
    rows += [_row(f"QID300_{i}", "Product Preferences - Pricing") for i in range(1, 31)]
    rows += [_row("QID157", "Disease - gain", cond="Disease - gain")]
    rows += [_row("QID999", "False consensus", cond="False consensus", qtype="score")]
    rows += [_row("QID25_1", "Personality", qtype="score")]
    rows += [_row("QID500", "Cognitive tests")]
    rows += [_row("QID158", "Disease-loss", cond="Disease-loss")]
    return pd.DataFrame(rows)


def test_only_decision_items_and_only_usable_ones() -> None:
    frame = _items()
    usable = set(frame["item_id"]) - {"QID158"}  # QID158 stands in for a sealed test item
    got = select_items(frame, usable, seed=0)
    studies = set(got["study_id"])
    assert "Personality" not in studies and "Cognitive tests" not in studies
    assert "False consensus" not in studies  # an attitude, not a decision
    assert "QID158" not in set(got["item_id"])
    assert "QID157" in set(got["item_id"]) and "QID117" in set(got["item_id"])
    assert {"Economic preferences", "Product Preferences - Pricing"} <= DECISION_STUDIES


def test_price_lists_keep_three_rows_and_pricing_is_subsampled() -> None:
    got = select_items(_items(), set(_items()["item_id"]), seed=0)
    rows = sorted(got.loc[got["task"] == "QID84", "item_id"], key=lambda s: int(s.split("_")[1]))
    # First, middle, last. With 10 rows the middle is position 4.5, which rounds half to even.
    assert rows == ["QID84_1", "QID84_5", "QID84_10"]
    assert (got["study_id"] == "Product Preferences - Pricing").sum() == 12


def test_selection_is_deterministic() -> None:
    a = select_items(_items(), set(_items()["item_id"]), seed=0)
    b = select_items(_items(), set(_items()["item_id"]), seed=0)
    assert list(a["item_id"]) == list(b["item_id"])


def test_person_sample_is_seeded_and_sized() -> None:
    ids = [f"p{i}" for i in range(50)]
    assert select_persons(ids, 10, seed=1) == select_persons(list(reversed(ids)), 10, seed=1)
    assert len(select_persons(ids, 10, seed=1)) == 10
    assert select_persons(ids, 100, seed=1) == sorted(ids)


def test_labels_keep_presented_order() -> None:
    assert prompts.labels("choice", ["b", "a"], None) == ["b", "a"]
    assert prompts.labels("score", None, ["low", "high"]) == ["low", "high"]
    assert prompts.labels("noul", None, None) == ["no", "yes"]


def test_completion_prompt_ends_at_the_choice_marker() -> None:
    text = prompts.completion_prompt("Pick one?", ["left", "right"], None)
    assert text.endswith("You press <<")
    assert "Press A for: left" in text and "Press B for: right" in text
    assert "Participant" not in text


def test_persona_lists_attributes_and_nothing_else() -> None:
    attrs = {"sex": "Female", "age_band": "30-49"}
    text = prompts.completion_prompt("Pick one?", ["left", "right"], attrs)
    chat = prompts.chat_messages("Pick one?", ["left", "right"], attrs)[-1]["content"]
    for body in (text, chat):
        assert "Female" in body and "30-49" in body
    assert "persona" not in text.lower()


def test_build_records_have_stable_ids_and_both_arms() -> None:
    items = select_items(_items(), set(_items()["item_id"]), seed=0)
    persons = pd.DataFrame([{"person_id": "p1", "sex": "Male"}, {"person_id": "p2", "sex": "F"}])
    recs = prompts.build_records(items, persons, ["p1", "p2"])
    arms = pd.Series([r["arm"] for r in recs]).value_counts().to_dict()
    assert arms == {"persona": 2 * len(items), "population": len(items)}
    ids = [r["prompt_id"] for r in recs]
    assert len(set(ids)) == len(ids)
    assert ids == [r["prompt_id"] for r in prompts.build_records(items, persons, ["p1", "p2"])]
    rec = recs[0]
    assert rec["keys"] == list("AB")[: len(rec["labels"])]
    assert rec["completion"].endswith("<<")


def test_keys_cover_twenty_options() -> None:
    labels = [f"opt{i}" for i in range(20)]
    text = prompts.completion_prompt("Q?", labels, None)
    assert "Press T for: opt19" in text
    assert np.all([k.isalpha() for k in prompts.KEYS])
