"""A saved model behind the same `predict_proba` protocol as the baselines, so the unchanged
evaluation harness can score it (`schelling eval --model <dir>`)."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from schelling.infer.batching import Question, build_tensors, options_of
from schelling.infer.predictor import Predictor
from schelling.schema import Choice, DecisionRequest, Noul, PersonContext, Score

Item = tuple[DecisionRequest, Question]


def make_question(
    qtype: str, text: str, options: list[str] | None, levels: list[str] | None
) -> Question:
    # model_construct: the records were validated when they were built.
    if qtype == "noul":
        return Noul.model_construct(type="noul", proposition=text)
    if qtype == "choice":
        return Choice.model_construct(type="choice", prompt=text, options=list(options or []))
    return Score.model_construct(type="score", prompt=text, levels=list(levels or []))


def make_item(
    qtype: str,
    text: str,
    options: list[str] | None,
    levels: list[str] | None,
    state: str,
    attributes: Mapping[str, Any] | None,
) -> Item:
    person = PersonContext.model_construct(
        person_id=None, attributes=dict(attributes or {}), history=[]
    )
    q = make_question(qtype, text, options, levels)
    return DecisionRequest.model_construct(person=person, state=state, questions={}), q


@torch.inference_mode()
def predict_items(
    model: Any, tokenizer: Any, items: list[Item], batch_size: int = 64, precision: str = "fp32"
) -> list[np.ndarray]:
    """Probabilities for each item, in input order. Items are grouped by task type."""
    model.eval()
    device = next(model.parameters()).device
    out: list[np.ndarray | None] = [None] * len(items)
    for task in ("noul", "choice", "score"):
        idx = [i for i, (_, q) in enumerate(items) if q.type == task]
        for lo in range(0, len(idx), batch_size):
            chunk = idx[lo : lo + batch_size]
            tensors = {
                k: v.to(device)
                for k, v in build_tensors(
                    tokenizer, model.config, task, [items[i] for i in chunk]
                ).items()
            }
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=precision == "bf16"):
                probs = model(task, **tensors).float().cpu().numpy()
            for i, p in zip(chunk, probs, strict=True):
                n = 2 if task == "noul" else len(options_of(items[i][1]))
                out[i] = p[:n].astype(np.float64)
    assert all(o is not None for o in out)
    return out  # type: ignore[return-value]


def frame_items(frame: pd.DataFrame, attributes: Mapping[str, Mapping[str, Any]]) -> list[Item]:
    items = []
    for pid, qtype, text, opts, lv, state in zip(
        frame["person_id"],
        frame["qtype"],
        frame["question"],
        frame["options"],
        frame["levels"],
        frame["state"],
        strict=True,
    ):
        items.append(
            make_item(
                qtype,
                text,
                None if opts is None else list(opts),
                None if lv is None else list(lv),
                state or "",
                attributes.get(str(pid)),
            )
        )
    return items


def person_attributes(persons: pd.DataFrame) -> dict[str, dict[str, Any]]:
    cols = [c for c in persons.columns if c != "person_id"]
    return {
        str(pid): {c: v for c, v in zip(cols, vals, strict=True) if pd.notna(v)}
        for pid, *vals in persons[["person_id", *cols]].itertuples(index=False)
    }


class ModelBaseline:
    name = "model"

    def __init__(self, predictor: Predictor, batch_size: int = 64) -> None:
        self.predictor, self.batch_size = predictor, batch_size
        self.attributes: dict[str, dict[str, Any]] = {}

    @classmethod
    def from_path(cls, path: str | Path, batch_size: int = 64) -> ModelBaseline:
        return cls(Predictor.from_pretrained(path), batch_size)

    def fit(self, train: pd.DataFrame, persons: pd.DataFrame) -> ModelBaseline:
        """No fitting: the model is already trained. Only remembers person attributes."""
        self.attributes = person_attributes(persons) if "person_id" in persons.columns else {}
        return self

    def predict_proba(self, frame: pd.DataFrame) -> list[np.ndarray]:
        return predict_items(
            self.predictor.model,
            self.predictor.tokenizer,
            frame_items(frame, self.attributes),
            self.batch_size,
        )


def iter_chunks(items: Iterable[Any], size: int) -> Iterable[list[Any]]:
    chunk: list[Any] = []
    for it in items:
        chunk.append(it)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk
