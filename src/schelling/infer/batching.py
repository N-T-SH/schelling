"""Turn (request, question) pairs of one task type into model tensors.

Shared by inference and training so the two cannot drift apart.
"""

from __future__ import annotations

from typing import Any

import torch

from schelling.infer.text import build_context
from schelling.schema import Choice, DecisionRequest, Noul, Score

Question = Noul | Choice | Score


def options_of(q: Question) -> list[str]:
    return q.options if isinstance(q, Choice) else q.levels if isinstance(q, Score) else []


def build_tensors(
    tokenizer: Any, config: Any, task: str, items: list[tuple[DecisionRequest, Question]]
) -> dict[str, torch.Tensor]:
    contexts = [
        build_context(
            req.person, req.state, q, tokenizer, config.max_context_tokens, config.history_items
        )
        for req, q in items
    ]
    enc = tokenizer(
        contexts,
        padding=True,
        truncation=True,
        max_length=config.max_context_tokens,
        return_tensors="pt",
    )
    out: dict[str, torch.Tensor] = {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
    }
    if task == "noul":
        return out
    opts = [options_of(q) for _, q in items]
    k = max(len(o) for o in opts)
    flat = [o[i] if i < len(o) else "" for o in opts for i in range(k)]
    oenc = tokenizer(
        flat,
        padding="max_length",
        truncation=True,
        max_length=config.max_option_tokens,
        return_tensors="pt",
    )
    out["option_input_ids"] = oenc["input_ids"].reshape(len(items), k, -1)
    out["option_attention_mask"] = oenc["attention_mask"].reshape(len(items), k, -1)
    out["option_mask"] = torch.tensor([[i < len(o) for i in range(k)] for o in opts])
    return out
