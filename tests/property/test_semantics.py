"""Semantics the interface promises (SPEC "Typed interface"), checked with Hypothesis."""

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from tests.helpers import tiny_model, tiny_tokenizer

from schelling.infer.predictor import Predictor
from schelling.schema import Choice, DecisionRequest, Noul, PersonContext, Score

pytestmark = pytest.mark.property
ATOL = 1e-5
SETTINGS = settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])

_predictor = Predictor(tiny_model(seed=1), tiny_tokenizer())
text = st.text(alphabet="abcdefghij klmnop", min_size=1, max_size=20).map(str.strip).filter(bool)
options = st.lists(text, min_size=2, max_size=6, unique=True)
attrs = st.dictionaries(
    st.sampled_from(["age", "sex", "city"]), st.one_of(text, st.integers(0, 90)), max_size=3
)


@SETTINGS
@given(state=text, prompt=text, opts=options, data=st.data())
def test_permuting_options_permutes_probabilities(
    state: str, prompt: str, opts: list[str], data: st.DataObject
) -> None:
    perm = data.draw(st.permutations(range(len(opts))))
    a = _predictor.decide(
        DecisionRequest(state=state, questions={"q": Choice(prompt=prompt, options=opts)})
    )["q"]
    b = _predictor.decide(
        DecisionRequest(
            state=state, questions={"q": Choice(prompt=prompt, options=[opts[i] for i in perm])}
        )
    )["q"]
    for new_pos, old_pos in enumerate(perm):
        assert b.probs[new_pos] == pytest.approx(a.probs[old_pos], abs=ATOL)


@SETTINGS
@given(state=text, p_a=text, o_a=options, p_b=text, o_b=options, prop=text, a=attrs)
def test_answer_to_a_question_does_not_depend_on_other_questions(
    state: str, p_a: str, o_a: list[str], p_b: str, o_b: list[str], prop: str, a: dict
) -> None:
    person = PersonContext(attributes=a)
    qa = Choice(prompt=p_a, options=o_a)
    alone = _predictor.decide(DecisionRequest(person=person, state=state, questions={"a": qa}))["a"]
    together = _predictor.decide(
        DecisionRequest(
            person=person,
            state=state,
            questions={"a": qa, "b": Choice(prompt=p_b, options=o_b), "c": Noul(proposition=prop)},
        )
    )["a"]
    assert together.probs == pytest.approx(alone.probs, abs=ATOL)


@SETTINGS
@given(
    states=st.lists(text, min_size=2, max_size=5),
    prompts=st.lists(text, min_size=2, max_size=5),
    levels=st.lists(text, min_size=2, max_size=5, unique=True),
)
def test_batch_and_single_inference_agree(
    states: list[str], prompts: list[str], levels: list[str]
) -> None:
    reqs = [
        DecisionRequest(
            state=s,
            questions={
                "n": Noul(proposition=p),
                "s": Score(prompt=p, levels=levels),
                "c": Choice(prompt=p, options=levels),
            },
        )
        for s, p in zip(states, prompts, strict=False)
    ]
    batched = _predictor.decide_batch(reqs, batch_size=3)
    for req, got in zip(reqs, batched, strict=True):
        single = _predictor.decide(req)
        for key in single:
            assert got[key].probs == pytest.approx(single[key].probs, abs=ATOL)
