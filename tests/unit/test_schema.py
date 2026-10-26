"""Schema tests required by SPEC "Test plan" (M0)."""

import json
from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError

from schelling.schema import (
    Answer,
    Choice,
    DecisionRequest,
    Noul,
    PersonContext,
    Score,
    export_json_schemas,
)

pytestmark = pytest.mark.unit

SCHEMA_DIR = Path(__file__).resolve().parents[2] / "schemas"


def test_choice_rejects_too_few_options() -> None:
    with pytest.raises(ValidationError):
        Choice(prompt="pick", options=["a"])


def test_choice_rejects_too_many_options() -> None:
    with pytest.raises(ValidationError):
        Choice(prompt="pick", options=[str(i) for i in range(21)])


def test_choice_accepts_bounds() -> None:
    Choice(prompt="pick", options=["a", "b"])
    Choice(prompt="pick", options=[str(i) for i in range(20)])


def test_choice_rejects_duplicate_options() -> None:
    with pytest.raises(ValidationError):
        Choice(prompt="pick", options=["a", "a"])


def test_score_rejects_bad_level_counts() -> None:
    with pytest.raises(ValidationError):
        Score(prompt="rate", levels=["low"])
    with pytest.raises(ValidationError):
        Score(prompt="rate", levels=[str(i) for i in range(11)])


def test_score_accepts_bounds() -> None:
    Score(prompt="rate", levels=["low", "high"])
    Score(prompt="rate", levels=[str(i) for i in range(10)])


def test_noul_requires_proposition() -> None:
    with pytest.raises(ValidationError):
        Noul()  # type: ignore[call-arg]


def test_unknown_question_type_rejected() -> None:
    with pytest.raises(ValidationError):
        DecisionRequest.model_validate(
            {"state": "s", "questions": {"q": {"type": "essay", "prompt": "x"}}}
        )


def test_questions_discriminated_on_type() -> None:
    req = DecisionRequest.model_validate(
        {
            "state": "s",
            "questions": {
                "a": {"type": "noul", "proposition": "I accept"},
                "b": {"type": "choice", "prompt": "p", "options": ["x", "y"]},
                "c": {"type": "score", "prompt": "p", "levels": ["lo", "hi"]},
            },
        }
    )
    assert isinstance(req.questions["a"], Noul)
    assert isinstance(req.questions["b"], Choice)
    assert isinstance(req.questions["c"], Score)


def test_request_json_round_trip_is_lossless() -> None:
    req = DecisionRequest(
        person=PersonContext(person_id="p1", attributes={"age": 30, "sex": "f", "x": 0.5}),
        state=["line one", "line two"],
        questions={
            "a": Noul(proposition="I accept", true_description="yes", false_description="no"),
            "b": Choice(prompt="p", options=["x", "y", "z"]),
            "c": Score(prompt="p", levels=["lo", "mid", "hi"]),
        },
    )
    again = DecisionRequest.model_validate_json(req.model_dump_json())
    assert again == req
    assert again.model_dump_json() == req.model_dump_json()


def test_answer_round_trip() -> None:
    ans = Answer(type="noul", probs=[0.3, 0.7], p_yes=0.7, entropy=0.61)
    assert Answer.model_validate_json(ans.model_dump_json()) == ans


def test_answer_probs_must_sum_to_one() -> None:
    with pytest.raises(ValidationError):
        Answer(type="choice", probs=[0.5, 0.2], entropy=0.0)


def test_noul_answer_shape() -> None:
    with pytest.raises(ValidationError):
        Answer(type="noul", probs=[0.2, 0.3, 0.5], p_yes=0.5, entropy=0.0)
    with pytest.raises(ValidationError):
        Answer(type="noul", probs=[0.3, 0.7], p_yes=0.1, entropy=0.0)


def test_person_defaults_are_not_shared() -> None:
    a, b = PersonContext(), PersonContext()
    a.attributes["k"] = 1
    assert b.attributes == {}


def test_exported_schemas_match_committed_files(tmp_path: Path) -> None:
    export_json_schemas(tmp_path)
    committed = sorted(p.name for p in SCHEMA_DIR.glob("*.json"))
    assert committed == sorted(p.name for p in tmp_path.glob("*.json"))
    for name in committed:
        assert json.loads((SCHEMA_DIR / name).read_text()) == json.loads(
            (tmp_path / name).read_text()
        ), f"{name} is stale: run `uv run python scripts/export_schemas.py`"


def test_question_union_adapter_exists() -> None:
    TypeAdapter(Noul | Choice | Score)
