import numpy as np
import pandas as pd
import pytest

from schelling.eval.retest import retest_fraction

pytestmark = pytest.mark.unit


def test_retest_fraction_hand_computed() -> None:
    # Three persons, one item. Wave 1 vs wave 4 answers agree for 2 of 3 -> ceiling 2/3.
    w1 = pd.DataFrame({"person_id": list("abc"), "item_id": "q", "wave": 1, "response": [0, 1, 1]})
    w4 = pd.DataFrame({"person_id": list("abc"), "item_id": "q", "wave": 4, "response": [0, 1, 0]})
    probs = [np.array([0.5, 0.5])] * 3  # model puts 0.5 on every answer
    res = retest_fraction(w4, probs, w1)
    assert res["ceiling"] == pytest.approx(2 / 3)
    assert res["model_agreement"] == pytest.approx(0.5)
    assert res["fraction_of_ceiling"] == pytest.approx(0.75)
    assert res["n"] == 3


def test_retest_ignores_wave4_rows_without_a_wave1_answer() -> None:
    w1 = pd.DataFrame({"person_id": ["a"], "item_id": "q", "wave": 1, "response": [0]})
    w4 = pd.DataFrame({"person_id": ["a", "z"], "item_id": "q", "wave": 4, "response": [0, 1]})
    res = retest_fraction(w4, [np.array([0.5, 0.5])] * 2, w1)
    assert res["n"] == 1
