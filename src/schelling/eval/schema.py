"""Schema of `metrics.json`. Every file carries its provenance (AGENTS.md rule 4)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class MetricsFile(BaseModel):
    git_sha: str
    dirty: bool
    config_hash: str
    split_manifest_hash: str
    seed: int
    created_at: str
    model: str
    split: str
    partition: str
    config: dict[str, Any]
    n_rows: int
    individual: dict[str, Any]
    population: dict[str, Any]
    retest: dict[str, Any] | None
    robustness: dict[str, Any]
    efficiency: dict[str, Any]
    not_run: list[str]
