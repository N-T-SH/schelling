"""The interface promises of SPEC "Typed interface", for the two-tower model."""

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from tests.helpers import tiny_tokenizer, tiny_tt_model

from schelling.infer.predictor import Predictor
from schelling.schema import Choice, DecisionRequest, Noul, PersonContext

pytestmark = pytest.mark.property
ATOL = 1e-5
SETTINGS = settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
_p = Predictor(tiny_tt_model(seed=1, residual_std=0.5), tiny_tokenizer())
text = st.text(alphabet="abcdefghij klmnop", min_size=1, max_size=20).map(str.strip).filter(bool)
options = st.lists(text, min_size=2, max_size=6, unique=True)
attrs = st.dictionaries(
    st.sampled_from(["age", "sex", "city"]),
    st.one_of(text, st.integers(0, 90).map(str)),
    max_size=3,
)


@SETTINGS
@given(state=text, prompt=text, opts=options, a=attrs, data=st.data())
def test_permuting_options_permutes_probabilities(
    state: str, prompt: str, opts: list[str], a: dict, data: st.DataObject
) -> None:
    perm = data.draw(st.permutations(range(len(opts))))
    person = PersonContext(attributes=a)
    x = _p.decide(
        DecisionRequest(
            person=person, state=state, questions={"q": Choice(prompt=prompt, options=opts)}
        )
    )["q"]
    y = _p.decide(
        DecisionRequest(
            person=person,
            state=state,
            questions={"q": Choice(prompt=prompt, options=[opts[i] for i in perm])},
        )
    )["q"]
    for new_pos, old_pos in enumerate(perm):
        assert y.probs[new_pos] == pytest.approx(x.probs[old_pos], abs=ATOL)


@SETTINGS
@given(state=text, p_a=text, o_a=options, p_b=text, o_b=options, prop=text, a=attrs)
def test_other_questions_do_not_change_an_answer(
    state: str, p_a: str, o_a: list[str], p_b: str, o_b: list[str], prop: str, a: dict
) -> None:
    person = PersonContext(attributes=a)
    qa = Choice(prompt=p_a, options=o_a)
    alone = _p.decide(DecisionRequest(person=person, state=state, questions={"a": qa}))["a"]
    together = _p.decide(
        DecisionRequest(
            person=person,
            state=state,
            questions={"a": qa, "b": Choice(prompt=p_b, options=o_b), "c": Noul(proposition=prop)},
        )
    )["a"]
    assert together.probs == pytest.approx(alone.probs, abs=ATOL)


@SETTINGS
@given(a=attrs, state=text, prompt=text, opts=options, data=st.data())
def test_attribute_order_never_matters(
    a: dict, state: str, prompt: str, opts: list[str], data: st.DataObject
) -> None:
    keys = data.draw(st.permutations(list(a)))
    shuffled = {k: a[k] for k in keys}
    q = {"q": Choice(prompt=prompt, options=opts)}
    x = _p.decide(DecisionRequest(person=PersonContext(attributes=a), state=state, questions=q))[
        "q"
    ]
    y = _p.decide(
        DecisionRequest(person=PersonContext(attributes=shuffled), state=state, questions=q)
    )["q"]
    assert y.probs == pytest.approx(x.probs, abs=ATOL)
