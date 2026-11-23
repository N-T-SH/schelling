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


def _order(qtype: str, labels: list[str]) -> list[int]:
    """Positions in canonical order. Choice options have no order, so they are sorted; score
    levels and noul answers keep theirs."""
    n = 2 if qtype == "noul" else len(labels)
    return sorted(range(n), key=lambda i: labels[i]) if qtype == "choice" else list(range(n))


def teacher_key(qtype: str, text: str, labels: list[str]) -> str:
    """labels: the options (choice) or levels (score); empty for noul."""
    canon = [labels[i] for i in _order(qtype, labels)] if qtype != "noul" else []
    raw = json.dumps([qtype, text, canon], ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def to_canonical(qtype: str, labels: list[str], probs: list[float]) -> list[float]:
    """Probabilities given in `labels` order -> the order stored in the table."""
    return [float(probs[i]) for i in _order(qtype, labels)]


def teacher_lookup(
    table: dict[str, list[float]], qtype: str, text: str, labels: list[str]
) -> list[float] | None:
    """The teacher's probabilities in `labels` order, or None if the table lacks the question."""
    stored = table.get(teacher_key(qtype, text, labels))
    order = _order(qtype, labels)
    if stored is None or len(stored) != len(order):
        return None
    out = [0.0] * len(order)
    for j, i in enumerate(order):
        out[i] = stored[j]
    return out


def load_teacher(path: str | Path) -> dict[str, list[float]]:
    path = Path(path)
    if not path.exists():
        return {}
    return {k: [float(x) for x in v] for k, v in json.loads(path.read_text())["table"].items()}


def save_teacher(path: str | Path, table: dict[str, list[float]], meta: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps({"meta": meta, "table": table}, indent=1) + "\n")


def teacher_tensors(
    questions: list[tuple[str, str, list[str]]], table: dict[str, list[float]], width: int
) -> dict[str, torch.Tensor]:
    """(B, width) zero-padded probabilities in each question's own option order and a (B,) mask
    of the questions the table knows. questions: (qtype, text, labels)."""
    probs = torch.zeros(len(questions), width)
    has = torch.zeros(len(questions), dtype=torch.bool)
    for i, (qtype, text, labels) in enumerate(questions):
        p = teacher_lookup(table, qtype, text, labels)
        if p is not None and len(p) <= width:
            probs[i, : len(p)] = torch.tensor(p)
            has[i] = True
    return {"teacher_probs": probs, "teacher_has": has}
