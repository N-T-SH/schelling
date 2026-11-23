"""A1 (fork D): a cached teacher distribution per question, looked up by question text and
options. The teacher was asked with the question and its options only (no person, no state), so
those are the key. The table is part of the model and is saved beside its weights."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

FILENAME = "teacher_prior.json"


def teacher_key(qtype: str, text: str, labels: list[str]) -> str:
    """labels: the options (choice) or levels (score) in data order; empty for noul."""
    raw = json.dumps([qtype, text, list(labels)], ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def load_teacher(path: str | Path) -> dict[str, list[float]]:
    path = Path(path)
    if not path.exists():
        return {}
    return {k: [float(x) for x in v] for k, v in json.loads(path.read_text())["table"].items()}


def save_teacher(path: str | Path, table: dict[str, list[float]], meta: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps({"meta": meta, "table": table}, indent=1) + "\n")


def teacher_tensors(
    keys: list[str], table: dict[str, list[float]], width: int
) -> dict[str, torch.Tensor]:
    """(B, width) zero-padded probabilities and a (B,) mask of the questions the table knows."""
    probs = torch.zeros(len(keys), width)
    has = torch.zeros(len(keys), dtype=torch.bool)
    for i, k in enumerate(keys):
        p = table.get(k)
        if p is not None and len(p) <= width:
            probs[i, : len(p)] = torch.tensor(p)
            has[i] = True
    return {"teacher_probs": probs, "teacher_has": has}
