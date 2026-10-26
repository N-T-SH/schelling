import pytest
from tests.helpers import tiny_tokenizer

from schelling.infer.text import build_context, person_text
from schelling.schema import Choice, HistoryItem, Noul, PersonContext

pytestmark = pytest.mark.unit


def _person(n_history: int = 5) -> PersonContext:
    return PersonContext(
        person_id="x",
        attributes={"age": 41, "sex": "f"},
        history=[
            HistoryItem(question=f"q{i}?", options=["a", "b"], answer=i % 2)
            for i in range(n_history)
        ],
    )


def test_person_text_contains_attributes_and_last_k_history_only() -> None:
    text = person_text(_person(5), k=2)
    assert "age: 41" in text and "sex: f" in text
    assert "q4?" in text and "q3?" in text and "q2?" not in text


def test_person_text_never_contains_the_opaque_person_id() -> None:
    assert "secret-id" not in person_text(PersonContext(person_id="secret-id"), k=3)


def test_empty_person_is_valid() -> None:
    assert isinstance(person_text(None, k=3), str) and isinstance(
        person_text(PersonContext(), 3), str
    )


def test_context_orders_person_state_question() -> None:
    ctx = build_context(
        _person(1), "the state", Noul(proposition="I accept"), tiny_tokenizer(), 128, 5
    )
    assert ctx.index("[PERSON]") < ctx.index("[STATE]") < ctx.index("[QUESTION]")
    assert "i accept" in ctx.lower()


def test_long_history_is_dropped_oldest_first_and_question_survives() -> None:
    tok = tiny_tokenizer()
    person = _person(200)
    ctx = build_context(person, "s", Choice(prompt="PICKME?", options=["a", "b"]), tok, 128, 200)
    assert len(tok(ctx)["input_ids"]) <= 128
    assert "pickme" in ctx.lower() and "q199?" in ctx and "q0?" not in ctx


def test_oversized_state_is_truncated_not_the_question() -> None:
    tok = tiny_tokenizer()
    ctx = build_context(None, "word " * 500, Noul(proposition="LASTWORD"), tok, 128, 5)
    assert len(tok(ctx)["input_ids"]) <= 128 and "lastword" in ctx.lower()
