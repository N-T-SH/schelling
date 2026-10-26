"""Request and response models for the typed decision interface (SPEC "Typed interface")."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field, TypeAdapter, field_validator, model_validator

SCALAR = str | float | int
_PROB_ATOL = 1e-5


class HistoryItem(BaseModel):
    """A prior (question, options, answer) triple from the same person."""

    question: str
    options: list[str]
    answer: int = Field(ge=0)

    @model_validator(mode="after")
    def _answer_in_range(self) -> HistoryItem:
        if self.answer >= len(self.options):
            raise ValueError("answer index out of range for options")
        return self


class PersonContext(BaseModel):
    person_id: str | None = None  # opaque, never a real identifier
    attributes: dict[str, SCALAR] = Field(default_factory=dict)
    history: list[HistoryItem] = Field(default_factory=list)


def _unique(values: list[str], what: str) -> list[str]:
    if len(set(values)) != len(values):
        raise ValueError(f"{what} must be unique")
    return values


class Noul(BaseModel):
    type: Literal["noul"] = "noul"
    proposition: str = Field(min_length=1)
    true_description: str | None = None
    false_description: str | None = None


class Choice(BaseModel):
    type: Literal["choice"] = "choice"
    prompt: str
    options: list[str] = Field(min_length=2, max_length=20)

    @field_validator("options")
    @classmethod
    def _unique_options(cls, v: list[str]) -> list[str]:
        return _unique(v, "options")


class Score(BaseModel):
    type: Literal["score"] = "score"
    prompt: str
    levels: list[str] = Field(min_length=2, max_length=10)  # ordered low -> high


Question = Annotated[Noul | Choice | Score, Field(discriminator="type")]


class DecisionRequest(BaseModel):
    person: PersonContext | None = None
    state: str | dict[str, object] | list[str]
    questions: dict[str, Question]


class Answer(BaseModel):
    type: Literal["noul", "choice", "score"]
    probs: list[float]  # noul: [p_no, p_yes]
    p_yes: float | None = None  # noul only: share of similar people answering yes
    expected_level: float | None = None  # score only, in [0, L-1]
    entropy: float = Field(ge=0)

    @model_validator(mode="after")
    def _check_distribution(self) -> Answer:
        if not self.probs or any(p < 0 for p in self.probs):
            raise ValueError("probs must be non-empty and non-negative")
        if not math.isclose(sum(self.probs), 1.0, abs_tol=_PROB_ATOL):
            raise ValueError("probs must sum to 1")
        if self.type == "noul":
            if len(self.probs) != 2 or self.p_yes is None:
                raise ValueError("noul answers need probs=[p_no, p_yes] and p_yes")
            if not math.isclose(self.p_yes, self.probs[1], abs_tol=_PROB_ATOL):
                raise ValueError("p_yes must equal probs[1]")
        return self


def export_json_schemas(out_dir: Path) -> list[Path]:
    """Write request, response and question schemas; CI compares them to the committed copies."""
    out_dir.mkdir(parents=True, exist_ok=True)
    from schelling.eval.schema import MetricsFile

    models: dict[str, type[BaseModel] | TypeAdapter[object]] = {
        "request": DecisionRequest,
        "response": Answer,
        "metrics": MetricsFile,
    }
    written = []
    for name, model in models.items():
        schema = model.model_json_schema() if isinstance(model, type) else model.json_schema()
        path = out_dir / f"{name}.json"
        path.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n")
        written.append(path)
    return written
