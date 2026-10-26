import numpy as np
import pandas as pd
import pytest

from schelling.train.sampler import StudySampler

pytestmark = pytest.mark.unit


def _studies() -> pd.Series:  # type: ignore[type-arg]
    return pd.Series(["big"] * 10_000 + ["small"] * 100)


def test_sampling_follows_square_root_of_study_size() -> None:
    idx = StudySampler(_studies(), temperature=0.5, seed=0).sample(200_000)
    share_small = float(np.mean(idx >= 10_000))
    expected = 100**0.5 / (100**0.5 + 10_000**0.5)
    assert share_small == pytest.approx(expected, abs=0.005)


def test_temperature_one_recovers_natural_proportions_and_zero_equalises() -> None:
    nat = float(np.mean(StudySampler(_studies(), 1.0, 0).sample(200_000) >= 10_000))
    eq = float(np.mean(StudySampler(_studies(), 0.0, 0).sample(200_000) >= 10_000))
    assert nat == pytest.approx(100 / 10_100, abs=0.002) and eq == pytest.approx(0.5, abs=0.005)


def test_within_a_study_sampling_is_uniform() -> None:
    idx = StudySampler(_studies(), 0.5, 0).sample(400_000)
    small = idx[idx >= 10_000] - 10_000
    counts = np.bincount(small, minlength=100)
    assert counts.min() > 0.7 * counts.mean() and counts.max() < 1.3 * counts.mean()


def test_deterministic_given_seed() -> None:
    a = StudySampler(_studies(), 0.5, 3).sample(1000)
    assert (a == StudySampler(_studies(), 0.5, 3).sample(1000)).all()
    assert not (a == StudySampler(_studies(), 0.5, 4).sample(1000)).all()
