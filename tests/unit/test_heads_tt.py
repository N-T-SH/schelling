"""Tensor-level behaviour of the two-tower combiner (decision 0004)."""

import pytest
import torch
from tests.helpers import tiny_tt_model

pytestmark = pytest.mark.unit
H = 32


def comb(m, task, **x):  # type: ignore[no-untyped-def]
    with torch.no_grad():
        return m.combine(task, **x)


def _inputs(b: int = 5, a: int = 4, k: int = 4, seed: int = 0):  # type: ignore[no-untyped-def]
    g = torch.Generator().manual_seed(seed)
    return {
        "u": torch.randn(b, H, generator=g),
        "attr_vecs": torch.randn(b, a, H, generator=g),
        "attr_mask": torch.ones(b, a, dtype=torch.bool),
        "e": torch.randn(b, k, H, generator=g),
        "option_mask": torch.ones(b, k, dtype=torch.bool),
    }


@pytest.mark.parametrize("task", ["noul", "choice", "score"])
def test_residual_starts_at_exactly_zero_so_the_model_starts_as_its_prior(task: str) -> None:
    out = tiny_tt_model().combine(task, **_inputs())
    assert out.residual.abs().max().item() == 0.0
    assert torch.equal(out.logits, out.prior)


def test_person_changes_the_answer_only_through_the_residual() -> None:
    m = tiny_tt_model(residual_std=0.5)
    x = _inputs()
    base = comb(m, "choice", **x)
    other = comb(m, "choice", **{**x, "attr_vecs": torch.randn_like(x["attr_vecs"])})
    assert torch.equal(base.prior, other.prior)
    assert not torch.allclose(base.residual, other.residual)


def test_prior_does_not_depend_on_the_person_at_all() -> None:
    m = tiny_tt_model(residual_std=0.5)
    x = _inputs()
    a = comb(m, "score", **x).prior
    b = comb(m, "score", **{**x, "attr_vecs": torch.zeros_like(x["attr_vecs"])}).prior
    assert torch.equal(a, b)


def test_choice_logits_are_equivariant_to_option_order() -> None:
    m = tiny_tt_model(residual_std=0.5)
    x = _inputs()
    perm = torch.tensor([2, 0, 3, 1])
    a = comb(m, "choice", **x).logits
    b = comb(m, "choice", **{**x, "e": x["e"][:, perm]}).logits
    assert b == pytest.approx(a[:, perm], abs=1e-5)


def test_person_vector_is_invariant_to_attribute_order() -> None:
    m = tiny_tt_model(residual_std=0.5)
    x = _inputs()
    perm = torch.tensor([3, 1, 0, 2])
    a = comb(m, "noul", **x).logits
    b = comb(m, "noul", **{**x, "attr_vecs": x["attr_vecs"][:, perm]}).logits
    assert b == pytest.approx(a, abs=1e-5)


def test_padded_attributes_are_ignored() -> None:
    m = tiny_tt_model(residual_std=0.5)
    x = _inputs()
    mask = x["attr_mask"].clone()
    mask[:, 2:] = False
    a = comb(m, "choice", **{**x, "attr_mask": mask}).logits
    noisy = x["attr_vecs"].clone()
    noisy[:, 2:] = 100.0 * torch.randn_like(noisy[:, 2:])
    b = comb(m, "choice", **{**x, "attr_mask": mask, "attr_vecs": noisy}).logits
    assert b == pytest.approx(a, abs=1e-5)


def test_person_with_no_attributes_is_valid_and_finite() -> None:
    m = tiny_tt_model(residual_std=0.5)
    x = _inputs()
    for task in ("noul", "choice", "score"):
        out = comb(m, task, **{**x, "attr_mask": torch.zeros_like(x["attr_mask"])})
        assert torch.isfinite(out.logits).all()


def test_padded_options_get_zero_probability_for_choice_and_score() -> None:
    from schelling.model.heads import choice_probs, score_conditionals, score_probs

    m = tiny_tt_model(residual_std=0.5)
    x = _inputs(k=5)
    x["option_mask"][:, 3:] = False
    c = comb(m, "choice", **x)
    assert choice_probs(c.logits, x["option_mask"], 1.0)[:, 3:].abs().max().item() == 0.0
    s = comb(m, "score", **x)
    probs = score_probs(score_conditionals(s.logits, x["option_mask"], 1.0))
    assert probs[:, 3:].abs().max().item() == 0.0 and probs.sum(-1) == pytest.approx(
        torch.ones(5), abs=1e-5
    )


def test_score_survival_is_monotone() -> None:
    from schelling.model.heads import score_conditionals

    m = tiny_tt_model(residual_std=0.5)
    x = _inputs(k=6)
    cond = score_conditionals(comb(m, "score", **x).logits, x["option_mask"], 1.0)
    surv = torch.cumprod(cond, dim=1)
    assert (surv[:, :-1] >= surv[:, 1:] - 1e-7).all()


def test_the_prior_separates_items_that_differ_only_in_their_options() -> None:
    """Bipolar rows share their question text; the option-set summary must tell them apart."""
    m = tiny_tt_model()
    x = _inputs(b=2, k=3)
    x["u"][1] = x["u"][0]  # identical question vectors
    x["e"][1] = torch.randn(3, H, generator=torch.Generator().manual_seed(9))  # different options
    p = comb(m, "choice", **x).prior
    assert not torch.allclose(p[0], p[1])


def test_use_person_false_disables_the_residual_branch() -> None:
    m = tiny_tt_model(residual_std=0.5, use_person=False)
    x = _inputs()
    out = comb(m, "choice", **x)
    assert out.residual.abs().max().item() == 0.0
