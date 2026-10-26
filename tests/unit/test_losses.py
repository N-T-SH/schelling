import pytest
import torch

from schelling.train import losses as L

pytestmark = pytest.mark.unit


def test_soft_cross_entropy_equals_hard_for_one_hot_targets() -> None:
    probs = torch.softmax(torch.randn(6, 5), dim=-1)
    y = torch.tensor([0, 1, 2, 3, 4, 0])
    onehot = torch.nn.functional.one_hot(y, 5).float()
    assert L.soft_cross_entropy(probs, onehot).item() == pytest.approx(
        L.hard_cross_entropy(probs, y).item(), abs=1e-6
    )


def test_kl_is_zero_for_identical_distributions_and_positive_otherwise() -> None:
    p = torch.softmax(torch.randn(4, 3), dim=-1)
    assert L.kl_divergence(p, p).item() == pytest.approx(0.0, abs=1e-6)
    assert L.kl_divergence(p, torch.softmax(torch.randn(4, 3), dim=-1)).item() > 0


def test_hard_cross_entropy_hand_computed() -> None:
    probs = torch.tensor([[0.5, 0.5], [0.8, 0.2]])
    assert L.hard_cross_entropy(probs, torch.tensor([0, 0])).item() == pytest.approx(
        -(torch.log(torch.tensor(0.5)) + torch.log(torch.tensor(0.8))).item() / 2
    )


def test_soft_targets_minimised_when_prediction_equals_target() -> None:
    target = torch.tensor([[0.7, 0.2, 0.1]])
    at_target = L.soft_cross_entropy(target, target).item()
    assert at_target < L.soft_cross_entropy(torch.tensor([[0.4, 0.4, 0.2]]), target).item()


def test_loss_is_finite_for_zero_probability_predictions() -> None:
    assert torch.isfinite(L.hard_cross_entropy(torch.tensor([[1.0, 0.0]]), torch.tensor([1])))


def test_corn_loss_equals_negative_log_likelihood_of_the_implied_distribution() -> None:
    cond = torch.sigmoid(torch.randn(8, 4))  # P(y > k | y >= k), 5 levels
    surv = torch.cumprod(cond, dim=1)
    probs = torch.cat([1 - surv[:, :1], surv[:, :-1] - surv[:, 1:], surv[:, -1:]], dim=1)
    y = torch.randint(0, 5, (8,))
    nll = -torch.log(probs.gather(1, y[:, None]).squeeze(1)).mean()
    assert L.corn_loss(cond, y).item() == pytest.approx(nll.item(), abs=1e-5)


def test_corn_loss_ignores_padded_levels() -> None:
    # Level 2 is padding for the first row: its thresholds are forced to zero probability.
    cond = torch.tensor([[0.7, 0.0, 0.0], [0.6, 0.5, 0.4]])
    y = torch.tensor([1, 3])
    full = L.corn_loss(cond, y).item()
    assert torch.isfinite(torch.tensor(full)) and full > 0
