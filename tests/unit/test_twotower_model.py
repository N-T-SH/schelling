"""Text-level behaviour of the two-tower model through the Predictor (decision 0004)."""

import pytest
import torch
from tests.helpers import tiny_tokenizer, tiny_tt_model

from schelling.infer.predictor import Predictor
from schelling.schema import Choice, DecisionRequest, Noul, PersonContext, Score

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def pred() -> Predictor:
    return Predictor(tiny_tt_model(seed=1, residual_std=0.5), tiny_tokenizer())


def _req(
    person: PersonContext | None = None, state: object = "a situation", **q: object
) -> DecisionRequest:
    questions = q or {
        "c": Choice(prompt="pick", options=["a", "b", "c"]),
        "n": Noul(proposition="p"),
        "s": Score(prompt="rate", levels=["lo", "mid", "hi"]),
    }
    return DecisionRequest(person=person, state=state, questions=questions)  # type: ignore[arg-type]


def test_all_three_types_give_valid_distributions(pred: Predictor) -> None:
    out = pred.decide(_req(PersonContext(attributes={"age": "30", "sex": "f"})))
    assert sum(out["c"].probs) == pytest.approx(1.0, abs=1e-5) and len(out["c"].probs) == 3
    assert 0 < out["n"].p_yes < 1  # type: ignore[operator]
    assert sum(out["s"].probs) == pytest.approx(1.0, abs=1e-5)


def test_the_person_changes_answers_once_the_residual_is_non_zero(pred: Predictor) -> None:
    a = pred.decide(_req(PersonContext(attributes={"age": "20"})))["c"].probs
    b = pred.decide(_req(PersonContext(attributes={"age": "70", "sex": "m"})))["c"].probs
    assert a != b


def test_a_freshly_initialised_model_ignores_the_person() -> None:
    p = Predictor(tiny_tt_model(seed=2), tiny_tokenizer())  # residual is zero at init
    a = p.decide(_req(PersonContext(attributes={"age": "20"})))["c"].probs
    b = p.decide(_req(PersonContext(attributes={"age": "70"})))["c"].probs
    assert a == b


def test_attribute_dict_order_does_not_matter(pred: Predictor) -> None:
    a = pred.decide(_req(PersonContext(attributes={"age": "20", "sex": "f", "city": "x"})))[
        "c"
    ].probs
    b = pred.decide(_req(PersonContext(attributes={"city": "x", "sex": "f", "age": "20"})))[
        "c"
    ].probs
    assert a == pytest.approx(b, abs=1e-6)


def test_batch_and_single_inference_agree(pred: Predictor) -> None:
    reqs = [
        _req(PersonContext(attributes={"age": str(20 + i)}), state="s" * (i + 1)) for i in range(5)
    ]
    batched = pred.decide_batch(reqs, batch_size=2)
    for r, got in zip(reqs, batched, strict=True):
        single = pred.decide(r)
        for k in single:
            assert got[k].probs == pytest.approx(single[k].probs, abs=1e-5)


def test_cache_on_and_off_give_identical_answers() -> None:
    model = tiny_tt_model(seed=3, residual_std=0.5)
    reqs = [_req(PersonContext(attributes={"age": str(20 + i % 3)})) for i in range(6)]
    model.use_cache = True
    cached = Predictor(model, tiny_tokenizer()).decide_batch(reqs)
    cached_again = Predictor(model, tiny_tokenizer()).decide_batch(
        reqs
    )  # second pass hits the cache
    model.use_cache = False
    fresh = Predictor(model, tiny_tokenizer()).decide_batch(reqs)
    for a, b, c in zip(cached, cached_again, fresh, strict=True):
        for k in a:
            assert b[k].probs == pytest.approx(a[k].probs, abs=1e-6)
            assert c[k].probs == pytest.approx(a[k].probs, abs=1e-6)


def test_cache_is_not_used_while_training() -> None:
    model = tiny_tt_model(seed=4)
    model.train()
    ids = torch.tensor([[2, 5, 6, 3, 0]])
    mask = torch.tensor([[1, 1, 1, 1, 0]])
    model.encode_texts(ids, mask)
    assert len(model._cache) == 0


def test_empty_and_oversized_inputs_stay_finite(pred: Predictor) -> None:
    for person in (
        None,
        PersonContext(),
        PersonContext(attributes={f"k{i}": "v " * 50 for i in range(40)}),
    ):
        out = pred.decide(_req(person, state="word " * 3000))
        for ans in out.values():
            assert all(torch.isfinite(torch.tensor(ans.probs)))


def test_options_may_be_permuted_without_changing_their_probabilities(pred: Predictor) -> None:
    person = PersonContext(attributes={"age": "30"})
    a = pred.decide(_req(person, c=Choice(prompt="pick", options=["red", "green", "blue"])))[
        "c"
    ].probs
    b = pred.decide(_req(person, c=Choice(prompt="pick", options=["blue", "red", "green"])))[
        "c"
    ].probs
    assert b == pytest.approx([a[2], a[0], a[1]], abs=1e-5)


def test_untrained_model_is_not_degenerate(pred: Predictor) -> None:
    out = pred.decide(
        _req(
            PersonContext(attributes={"age": "30"}),
            c=Choice(prompt="pick", options=["alpha", "beta", "gamma"]),
        )
    )["c"]
    assert max(out.probs) - min(out.probs) > 1e-4
