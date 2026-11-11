"""Turn (request, question) pairs of one task type into model tensors.

Shared by inference and training so the two cannot drift apart.
"""

from __future__ import annotations

from typing import Any

import torch

from schelling.infer.text import attribute_strings, build_context, build_question_context
from schelling.schema import Choice, DecisionRequest, HistoryItem, Noul, Score

Question = Noul | Choice | Score


def options_of(q: Question) -> list[str]:
    return q.options if isinstance(q, Choice) else q.levels if isinstance(q, Score) else []


def _options(
    tokenizer: Any, max_tokens: int, items: list[tuple[DecisionRequest, Question]]
) -> dict[str, torch.Tensor]:
    opts = [options_of(q) for _, q in items]
    k = max(len(o) for o in opts)
    flat = [o[i] if i < len(o) else "" for o in opts for i in range(k)]
    enc = tokenizer(
        flat, padding="max_length", truncation=True, max_length=max_tokens, return_tensors="pt"
    )
    return {
        "option_input_ids": enc["input_ids"].reshape(len(items), k, -1),
        "option_attention_mask": enc["attention_mask"].reshape(len(items), k, -1),
        "option_mask": torch.tensor([[i < len(o) for i in range(k)] for o in opts]),
    }


def history_question_context(question: str, tokenizer: Any, max_tokens: int) -> str:
    """The question-tower text for an earlier answer. Same function and empty state as for a
    scored item, so the vector equals the one cached for that item."""
    return build_question_context(
        "", Noul.model_construct(type="noul", proposition=question), tokenizer, max_tokens
    )


def _history(
    tokenizer: Any, config: Any, items: list[tuple[DecisionRequest, Question]]
) -> dict[str, torch.Tensor]:
    """Earlier answers of each person: question texts and the text of the option chosen."""
    hists: list[list[HistoryItem]] = [
        (req.person.history[-config.history_items :] if req.person else []) for req, _ in items
    ]
    length = max(1, max(len(h) for h in hists))
    ctx: dict[str, int] = {}
    opt: dict[str, int] = {}
    qi = [[-1] * length for _ in items]
    oi = [[-1] * length for _ in items]
    for b, h in enumerate(hists):
        for j, it in enumerate(h):
            c = history_question_context(it.question, tokenizer, config.max_question_tokens)
            qi[b][j] = ctx.setdefault(c, len(ctx))
            oi[b][j] = opt.setdefault(it.options[it.answer], len(opt))
    q_enc = tokenizer(
        [*ctx, ""],
        padding="max_length",
        truncation=True,
        max_length=config.max_question_tokens,
        return_tensors="pt",
    )
    o_enc = tokenizer(
        [*opt, ""],
        padding="max_length",
        truncation=True,
        max_length=config.max_option_tokens,
        return_tensors="pt",
    )
    qix, oix = torch.tensor(qi), torch.tensor(oi)
    return {
        "hist_input_ids": q_enc["input_ids"][qix],  # -1 picks the blank last row
        "hist_attention_mask": q_enc["attention_mask"][qix],
        "hist_option_input_ids": o_enc["input_ids"][oix],
        "hist_option_attention_mask": o_enc["attention_mask"][oix],
        "hist_mask": qix >= 0,
    }


def _build_two_tower(
    tokenizer: Any, config: Any, task: str, items: list[tuple[DecisionRequest, Question]]
) -> dict[str, torch.Tensor]:
    contexts = [
        build_question_context(req.state, q, tokenizer, config.max_question_tokens)
        for req, q in items
    ]
    enc = tokenizer(
        contexts,
        padding=True,
        truncation=True,
        max_length=config.max_question_tokens,
        return_tensors="pt",
    )
    attrs = [attribute_strings(req.person, config.max_attributes) for req, _ in items]
    a = max(1, max(len(x) for x in attrs))
    flat = [x[i] if i < len(x) else "" for x in attrs for i in range(a)]
    aenc = tokenizer(
        flat,
        padding="max_length",
        truncation=True,
        max_length=config.max_attr_tokens,
        return_tensors="pt",
    )
    out = {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "attr_input_ids": aenc["input_ids"].reshape(len(items), a, -1),
        "attr_attention_mask": aenc["attention_mask"].reshape(len(items), a, -1),
        "attr_mask": torch.tensor([[i < len(x) for i in range(a)] for x in attrs]),
    }
    if task != "noul":
        out.update(_options(tokenizer, config.max_option_tokens, items))
    if getattr(config, "history_items", 0) > 0:
        out.update(_history(tokenizer, config, items))
    return out


def build_tensors(
    tokenizer: Any, config: Any, task: str, items: list[tuple[DecisionRequest, Question]]
) -> dict[str, torch.Tensor]:
    if getattr(config, "model_type", "") == "schelling_tt":
        return _build_two_tower(tokenizer, config, task, items)
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
    if task != "noul":
        out.update(_options(tokenizer, config.max_option_tokens, items))
    return out
