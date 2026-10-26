"""Hand-computed and statistical checks for eval/metrics.py (SPEC "Test plan")."""

import math

import numpy as np
import pytest

from schelling.eval import metrics as m

pytestmark = pytest.mark.unit


def test_nll_hand_computed() -> None:
    assert m.nll([np.array([0.5, 0.5])], np.array([0])) == pytest.approx(math.log(2))
    assert m.nll([np.array([0.8, 0.2]), np.array([0.1, 0.9])], np.array([0, 1])) == pytest.approx(
        -(math.log(0.8) + math.log(0.9)) / 2
    )


def test_nll_is_finite_for_zero_probability() -> None:
    assert math.isfinite(m.nll([np.array([1.0, 0.0])], np.array([1])))


def test_brier_hand_computed() -> None:
    assert m.brier([np.array([0.7, 0.3])], np.array([0])) == pytest.approx(0.18)


def test_accuracy() -> None:
    probs = [np.array([0.6, 0.4]), np.array([0.2, 0.8]), np.array([0.5, 0.5])]
    assert m.accuracy(probs, np.array([0, 0, 0])) == pytest.approx(2 / 3)  # tie -> first option


def test_ece_hand_computed() -> None:
    conf = np.full(10, 0.9)
    correct = np.array([1] * 8 + [0] * 2)
    assert m.ece(conf, correct, n_bins=5) == pytest.approx(0.1)


def test_ece_near_zero_on_calibrated_synthetic_data() -> None:
    rng = np.random.default_rng(0)
    conf = rng.uniform(0.5, 1.0, 100_000)
    correct = (rng.uniform(size=conf.size) < conf).astype(int)
    assert m.ece(conf, correct) < 0.01


def test_calibration_slope_and_intercept_on_calibrated_data() -> None:
    rng = np.random.default_rng(1)
    conf = rng.uniform(0.55, 0.95, 100_000)
    correct = (rng.uniform(size=conf.size) < conf).astype(int)
    slope, intercept = m.calibration_slope_intercept(conf, correct)
    assert slope == pytest.approx(1.0, abs=0.05)
    assert intercept == pytest.approx(0.0, abs=0.05)


def test_calibration_slope_detects_overconfidence() -> None:
    rng = np.random.default_rng(2)
    true_p = rng.uniform(0.55, 0.85, 50_000)
    correct = (rng.uniform(size=true_p.size) < true_p).astype(int)
    # Doubling the logit is the classic over-confident model: the fitted slope should be ~0.5.
    stated = 1 / (1 + np.exp(-2 * np.log(true_p / (1 - true_p))))
    slope, _ = m.calibration_slope_intercept(stated, correct)
    assert slope == pytest.approx(0.5, abs=0.1)


def test_tvd_hand_computed_and_symmetric() -> None:
    p, q = np.array([0.5, 0.5]), np.array([1.0, 0.0])
    assert m.tvd(p, q) == pytest.approx(0.5)
    assert m.tvd(p, q) == m.tvd(q, p)


def test_wasserstein_hand_computed() -> None:
    assert m.wasserstein1(np.array([1.0, 0, 0]), np.array([0, 0, 1.0])) == pytest.approx(2.0)
    # 0.5 of mass moves 0 -> 1 and 0.5 moves 1 -> 2: cost 0.5 + 0.5.
    assert m.wasserstein1(np.array([0.5, 0.5, 0]), np.array([0, 0.5, 0.5])) == pytest.approx(1.0)


def test_sd_and_entropy_ratio_are_one_for_identical_distributions() -> None:
    p = np.array([0.1, 0.2, 0.3, 0.4])
    assert m.sd_ratio(p, p) == pytest.approx(1.0)
    assert m.entropy_ratio(p, p) == pytest.approx(1.0)


def test_sd_ratio_falls_when_distribution_collapses() -> None:
    human = np.array([0.25, 0.25, 0.25, 0.25])
    collapsed = np.array([0.0, 0.0, 1.0, 0.0])
    assert m.sd_ratio(collapsed, human) == 0.0
    assert m.entropy_ratio(collapsed, human) == 0.0
