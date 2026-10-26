"""Unified record: one row per observed response (SPEC "Data")."""

from __future__ import annotations

import hashlib
from typing import Literal

from pydantic import BaseModel, model_validator


def make_record_id(
    source: str, study_id: str, person_id: str, item_id: str, trial_idx: int | None
) -> str:
    key = "\x1f".join([source, study_id, person_id, item_id, str(trial_idx)])
    return hashlib.sha1(key.encode()).hexdigest()


class DecisionRecord(BaseModel):
    record_id: str
    source: str
    study_id: str
    item_id: str
    person_id: str
    trial_idx: int | None = None  # order within person, for sequential tasks
    qtype: Literal["noul", "choice", "score"]
    question: str
    options: list[str] | None = None
    levels: list[str] | None = None
    state: str = ""
    person_ref: str
    response: int | None = None  # observed index (individual records)
    response_dist: list[float] | None = None  # aggregate records only
    condition: str | None = None  # experimental arm, for treatment effects
    wave: int | None = None  # for test-retest

    @model_validator(mode="after")
    def _check(self) -> DecisionRecord:
        if (self.response is None) == (self.response_dist is None):
            raise ValueError("exactly one of response and response_dist must be set")
        n = {"noul": 2, "choice": len(self.options or []), "score": len(self.levels or [])}[
            self.qtype
        ]
        if self.qtype == "choice" and not 2 <= n <= 20:
            raise ValueError("choice records need 2..20 options")
        if self.qtype == "score" and not 2 <= n <= 10:
            raise ValueError("score records need 2..10 levels")
        if self.response is not None and not 0 <= self.response < n:
            raise ValueError("response index out of range")
        if self.response_dist is not None and len(self.response_dist) != n:
            raise ValueError("response_dist has the wrong length")
        return self
