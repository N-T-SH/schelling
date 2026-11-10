"""C01 / I01: the history tower of the two-tower heads."""

import pytest
import torch

from schelling.model.heads_tt import TwoTowerHeads

pytestmark = pytest.mark.unit
H, D, B, K, A = 12, 8, 5, 6, 3


def _heads(history: bool = True, residual: float = 0.5, seed: int = 0) -> TwoTowerHeads:
    torch.manual_seed(seed)
    h = TwoTowerHeads(H, D, 2, 0.0, True, use_history=history)
    h.reset()
    if residual:
        h.init_residual(residual)
    return h.eval()


def _inputs(seed: int = 1):  # type: ignore[no-untyped-def]
    g = torch.Generator().manual_seed(seed)
    return dict(
        task="choice",
        u=torch.randn(B, H, generator=g),
        attr_vecs=torch.randn(B, A, H, generator=g),
        attr_mask=torch.ones(B, A, dtype=torch.bool),
        e=torch.randn(B, 4, H, generator=g),
        option_mask=torch.ones(B, 4, dtype=torch.bool),
    )


def _hist(seed: int = 2, mask=None):  # type: ignore[no-untyped-def]
    g = torch.Generator().manual_seed(seed)
    return dict(
        hist_u=torch.randn(B, K, H, generator=g),
        hist_e=torch.randn(B, K, H, generator=g),
        hist_mask=torch.ones(B, K, dtype=torch.bool) if mask is None else mask,
    )


@torch.no_grad()
def test_history_changes_the_prediction_once_the_residual_is_non_zero() -> None:
    h, x = _heads(), _inputs()
    assert not torch.allclose(h(**x).logits, h(**x, **_hist()).logits)


@torch.no_grad()
def test_a_fresh_model_ignores_history_because_the_residual_starts_at_zero() -> None:
    h, x = _heads(residual=0.0), _inputs()
    assert torch.equal(h(**x).logits, h(**x, **_hist()).logits)


@torch.no_grad()
def test_no_history_equals_an_all_masked_history() -> None:
    h, x = _heads(), _inputs()
    empty = _hist(mask=torch.zeros(B, K, dtype=torch.bool))
    assert torch.allclose(h(**x).logits, h(**x, **empty).logits, atol=1e-6)


@torch.no_grad()
def test_history_order_does_not_matter() -> None:
    h, x, hist = _heads(), _inputs(), _hist()
    perm = torch.randperm(K, generator=torch.Generator().manual_seed(3))
    shuffled = {k: v[:, perm] for k, v in hist.items()}
    assert torch.allclose(h(**x, **hist).logits, h(**x, **shuffled).logits, atol=1e-5)


@torch.no_grad()
def test_masked_history_slots_have_no_influence() -> None:
    h, x = _heads(), _inputs()
    mask = torch.arange(K).unsqueeze(0).expand(B, K) < 3
    a = _hist(seed=2, mask=mask)
    b = {**a, "hist_u": a["hist_u"].clone(), "hist_e": a["hist_e"].clone()}
    b["hist_u"][:, 3:] = 99.0
    b["hist_e"][:, 3:] = -99.0
    assert torch.allclose(h(**x, **a).logits, h(**x, **b).logits, atol=1e-6)


@torch.no_grad()
@pytest.mark.parametrize("task", ["noul", "score"])
def test_history_reaches_noul_and_score_heads(task: str) -> None:
    h, x = _heads(), _inputs()
    x["task"] = task
    if task == "noul":
        x["e"], x["option_mask"] = None, None
    assert not torch.allclose(h(**x).logits, h(**x, **_hist()).logits)


def test_disabled_history_keeps_the_m4b_parameter_layout() -> None:
    h = _heads(history=False)
    assert not any("hist" in name for name, _ in h.named_parameters())
    assert h.mix[0].in_features == 3 * D
    assert _heads(history=True).mix[0].in_features == 5 * D


def test_the_history_tower_is_trained_by_the_loss() -> None:
    h, x = _heads(), _inputs()
    h.train()
    h(**x, **_hist()).logits.sum().backward()
    grads = [p.grad for n, p in h.named_parameters() if "hist" in n]
    assert grads and all(g is not None and g.abs().sum() > 0 for g in grads)


@torch.no_grad()
def test_table_and_vector_histories_give_identical_logits() -> None:
    h, x = _heads(), _inputs()
    g = torch.Generator().manual_seed(9)
    ut, et = torch.randn(7, H, generator=g), torch.randn(5, H, generator=g)
    qi = torch.randint(0, 7, (B, K), generator=g)
    oi = torch.randint(0, 5, (B, K), generator=g)
    mask = torch.rand(B, K, generator=g) > 0.3
    vec = dict(hist_u=ut[qi], hist_e=et[oi], hist_mask=mask)
    tab = dict(hist_tables=(ut, et), hist_idx=(qi, oi), hist_mask=mask)
    assert torch.allclose(h(**x, **vec).logits, h(**x, **tab).logits, atol=1e-5)
