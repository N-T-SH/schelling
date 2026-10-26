import numpy as np
import pytest

from schelling.eval.effects import effect_size_agreement

pytestmark = pytest.mark.unit


def test_effect_agreement_sign_correlation_and_false_positives() -> None:
    human = np.array([0.5, -0.4, 0.0, 0.0])  # last two are null effects in humans
    model = np.array([0.4, -0.3, 0.3, 0.0])
    res = effect_size_agreement(
        human,
        model,
        null_mask=np.array([False, False, True, True]),
        model_significant=np.array([True, True, True, False]),
    )
    assert res["sign_agreement"] == 1.0
    assert res["correlation"] > 0.8
    assert res["false_positive_rate"] == pytest.approx(0.5)
