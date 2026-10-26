import pandas as pd
import pytest

from schelling.data.contamination import contamination_report, jaccard_shingles

pytestmark = pytest.mark.unit


def test_jaccard_basics() -> None:
    assert jaccard_shingles("a b c d", "a b c d") == 1.0
    assert jaccard_shingles("a b c d", "w x y z") == 0.0


def test_exact_and_near_duplicates_found() -> None:
    train = pd.DataFrame(
        {"item_id": ["t1", "t2"], "question": ["How much do you like cats a lot", "Pick a colour"]}
    )
    held = pd.DataFrame(
        {
            "item_id": ["h1", "h2", "h3"],
            "question": [
                "How much do you like cats a lot",  # exact
                "how much do you like cats a lot?!",  # exact after normalisation
                "Completely unrelated sentence about tax law and rates",
            ],
        }
    )
    rep = contamination_report(train, held, threshold=0.8)
    assert rep["exact"] == ["h1", "h2"]
    assert rep["near"] == []
    assert rep["remove_item_ids"] == ["h1", "h2"]


def test_near_duplicate_above_threshold_reported() -> None:
    base = (
        "please indicate how strongly you agree with the following statement about your daily life"
    )
    train = pd.DataFrame({"item_id": ["t"], "question": [base]})
    held = pd.DataFrame({"item_id": ["h"], "question": [base + " today"]})
    rep = contamination_report(train, held, threshold=0.8)
    assert rep["near"] == ["h"] and rep["exact"] == []
