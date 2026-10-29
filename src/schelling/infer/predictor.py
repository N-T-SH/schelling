"""Typed prediction API (SPEC "Public API")."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer, PretrainedConfig

from schelling.infer.batching import build_tensors, options_of
from schelling.infer.text import person_text
from schelling.model import SchellingModel, SchellingTwoTower
from schelling.schema import Answer, Choice, DecisionRequest, Noul, PersonContext, Score

Item = tuple[int, str, DecisionRequest, "Noul | Choice | Score"]


@dataclass(frozen=True)
class PersonEmbedding:
    """v0: the serialised person text. Cached encoder states arrive with person encoder v1 (M5)."""

    text: str


def _entropy(probs: list[float]) -> float:
    return float(-sum(p * math.log(p) for p in probs if p > 0))


class Predictor:
    def __init__(self, model: SchellingModel | SchellingTwoTower, tokenizer: Any) -> None:
        self.model = model.eval()
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, repo_or_path: str | Path, revision: str | None = None) -> Predictor:
        cfg, _ = PretrainedConfig.get_config_dict(str(repo_or_path), revision=revision or "main")
        cls_ = SchellingTwoTower if cfg.get("model_type") == "schelling_tt" else SchellingModel
        model = cls_.from_pretrained(str(repo_or_path), revision=revision or "main")
        tokenizer = AutoTokenizer.from_pretrained(
            str(repo_or_path), revision=revision or "main", trust_remote_code=False
        )
        return cls(model, tokenizer)

    def save_pretrained(self, path: str | Path) -> None:
        self.model.save_pretrained(str(path), safe_serialization=True)
        self.tokenizer.save_pretrained(str(path))

    def encode_person(self, person: PersonContext | None) -> PersonEmbedding:
        return PersonEmbedding(person_text(person, getattr(self.model.config, "history_items", 0)))

    # -- batching ---------------------------------------------------------------------------

    def _options_of(self, q: Noul | Choice | Score) -> list[str]:
        return options_of(q)

    def _tensors(self, task: str, items: list[Item]) -> dict[str, torch.Tensor]:
        return build_tensors(
            self.tokenizer, self.model.config, task, [(req, q) for _, _, req, q in items]
        )

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
