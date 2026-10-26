import pytest
import torch

from schelling.model import calibration as C

pytestmark = pytest.mark.unit
GEN = torch.Generator().manual_seed(0)


def _labels_from(probs: torch.Tensor) -> torch.Tensor:
    return torch.multinomial(probs, 1, generator=GEN).squeeze(1)


def test_overconfident_choice_logits_get_temperature_above_one_and_lower_nll() -> None:
    true = torch.randn(20_000, 4, generator=GEN)
    y = _labels_from(torch.softmax(true, -1))
    stated, mask = 2.0 * true, torch.ones(20_000, 4, dtype=torch.bool)
    t = C.fit_temperature("choice", stated, mask, y)
    assert t == pytest.approx(2.0, abs=0.15)
    assert C.nll_at_temperature("choice", stated, mask, y, t) < C.nll_at_temperature(
        "choice", stated, mask, y, 1.0
    )


def test_underconfident_logits_get_temperature_below_one() -> None:
    true = 2 * torch.randn(20_000, 3, generator=GEN)
    y = _labels_from(torch.softmax(true, -1))
    t = C.fit_temperature("choice", 0.5 * true, torch.ones(20_000, 3, dtype=torch.bool), y)
    assert 0 < t < 1 and t == pytest.approx(0.5, abs=0.1)


def test_noul_temperature() -> None:
    true = 2 * torch.randn(20_000, 1, generator=GEN)
    y = (torch.rand(20_000, 1, generator=GEN) < torch.sigmoid(true)).long().squeeze(1)
    t = C.fit_temperature("noul", 3 * true, None, y)
    assert t == pytest.approx(3.0, abs=0.35)


def test_score_temperature() -> None:
    t_true = torch.randn(20_000, 3, generator=GEN)
    cond = torch.sigmoid(t_true)
    surv = torch.cumprod(cond, 1)
    probs = torch.cat([1 - surv[:, :1], surv[:, :-1] - surv[:, 1:], surv[:, -1:]], 1)
    y = _labels_from(probs)
    mask = torch.ones(20_000, 4, dtype=torch.bool)
    t = C.fit_temperature("score", 2.0 * t_true, mask, y)
    assert t == pytest.approx(2.0, abs=0.25)


def test_temperature_is_always_positive_even_for_uninformative_logits() -> None:
    y = torch.randint(0, 3, (2000,))
    assert (
        C.fit_temperature("choice", torch.randn(2000, 3), torch.ones(2000, 3, dtype=torch.bool), y)
        > 0
    )


def test_padded_options_are_ignored_when_fitting() -> None:
    logits = torch.randn(500, 4)
    mask = torch.tensor([[True, True, False, False]] * 500)
    y = torch.randint(0, 2, (500,))
    assert C.nll_at_temperature("choice", logits, mask, y, 1.0) == pytest.approx(
        C.nll_at_temperature("choice", logits[:, :2], torch.ones(500, 2, dtype=torch.bool), y, 1.0),
        abs=1e-6,
    )
