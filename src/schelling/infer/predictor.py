"""Typed prediction API (SPEC "Public API")."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from schelling.infer.text import build_context, person_text
from schelling.model import SchellingModel
from schelling.schema import Answer, Choice, DecisionRequest, Noul, PersonContext, Score

Item = tuple[int, str, DecisionRequest, "Noul | Choice | Score"]


@dataclass(frozen=True)
class PersonEmbedding:
    """v0: the serialised person text. Cached encoder states arrive with person encoder v1 (M5)."""

    text: str


def _entropy(probs: list[float]) -> float:
    return float(-sum(p * math.log(p) for p in probs if p > 0))


class Predictor:
    def __init__(self, model: SchellingModel, tokenizer: Any) -> None:
        self.model = model.eval()
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, repo_or_path: str | Path, revision: str | None = None) -> Predictor:
        model = SchellingModel.from_pretrained(str(repo_or_path), revision=revision or "main")
        tokenizer = AutoTokenizer.from_pretrained(str(repo_or_path), revision=revision or "main")
        return cls(model, tokenizer)

    def save_pretrained(self, path: str | Path) -> None:
        self.model.save_pretrained(str(path), safe_serialization=True)
        self.tokenizer.save_pretrained(str(path))

    def encode_person(self, person: PersonContext | None) -> PersonEmbedding:
        return PersonEmbedding(person_text(person, self.model.config.history_items))

    # -- batching ---------------------------------------------------------------------------

    def _options_of(self, q: Noul | Choice | Score) -> list[str]:
        return q.options if isinstance(q, Choice) else q.levels if isinstance(q, Score) else []

    def _tensors(self, task: str, items: list[Item]) -> dict[str, torch.Tensor]:
        cfg, tok = self.model.config, self.tokenizer
        contexts = [
            build_context(req.person, req.state, q, tok, cfg.max_context_tokens, cfg.history_items)
            for _, _, req, q in items
        ]
        enc = tok(
            contexts,
            padding=True,
            truncation=True,
            max_length=cfg.max_context_tokens,
            return_tensors="pt",
        )
        out: dict[str, torch.Tensor] = {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
        }
        if task == "noul":
            return out
        opts = [self._options_of(q) for _, _, _, q in items]
        k = max(len(o) for o in opts)
        flat = [o[i] if i < len(o) else "" for o in opts for i in range(k)]
        oenc = tok(
            flat,
            padding="max_length",
            truncation=True,
            max_length=cfg.max_option_tokens,
            return_tensors="pt",
        )
        out["option_input_ids"] = oenc["input_ids"].reshape(len(items), k, -1)
        out["option_attention_mask"] = oenc["attention_mask"].reshape(len(items), k, -1)
        out["option_mask"] = torch.tensor([[i < len(o) for i in range(k)] for o in opts])
        return out

    @torch.no_grad()
    def decide_batch(
        self, requests: list[DecisionRequest], batch_size: int = 32
    ) -> list[dict[str, Answer]]:
        """Each question is scored independently; batching only groups equal task types."""
        results: list[dict[str, Answer]] = [{} for _ in requests]
        by_task: dict[str, list[Item]] = {"noul": [], "choice": [], "score": []}
        for i, req in enumerate(requests):
            for qid, q in req.questions.items():
                by_task[q.type].append((i, qid, req, q))
        for task, items in by_task.items():
            for lo in range(0, len(items), batch_size):
                chunk = items[lo : lo + batch_size]
                probs = self.model(task, **self._tensors(task, chunk)).tolist()
                for (i, qid, _, q), p in zip(chunk, probs, strict=True):
                    n = 2 if task == "noul" else len(self._options_of(q))
                    p = p[:n]
                    results[i][qid] = Answer(
                        type=task,
                        probs=p,
                        p_yes=p[1] if task == "noul" else None,
                        expected_level=sum(j * x for j, x in enumerate(p))
                        if task == "score"
                        else None,
                        entropy=_entropy(p),
                    )
        return results

    def decide(self, request: DecisionRequest) -> dict[str, Answer]:
        return self.decide_batch([request], batch_size=32)[0]
