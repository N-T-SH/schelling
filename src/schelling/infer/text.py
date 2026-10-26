"""Serialise a person, state and question into the model's context text (person encoder v0)."""

from __future__ import annotations

import re
from typing import Any

from schelling.schema import Choice, Noul, PersonContext, Score

_PARENS = re.compile(r"\s*\([^)]*\)")


def _short(value: Any) -> str:
    """Drop parenthetical asides ("Midwest (ND, SD, ...)") that cost tokens and add nothing."""
    return _PARENS.sub("", str(value)).strip()


def _attr_text(attrs: dict[str, Any]) -> str:
    return "; ".join(f"{k}: {_short(v)}" for k, v in sorted(attrs.items()))


def _history_line(item: Any) -> str:
    return f"{item.question} [{' | '.join(item.options)}] -> {item.options[item.answer]}"


def person_text(person: PersonContext | None, k: int, n_attributes: int | None = None) -> str:
    """Attributes (the first `n_attributes` alphabetically, default all) plus the last `k`
    history items. The opaque person id is never included."""
    if person is None:
        return ""
    parts = []
    attrs = dict(sorted(person.attributes.items())[:n_attributes])
    if attrs:
        parts.append(_attr_text(attrs))
    for item in person.history[-k:] if k > 0 else []:
        parts.append(_history_line(item))
    return " ; ".join(parts)


def state_text(state: str | dict[str, object] | list[str]) -> str:
    if isinstance(state, str):
        return state
    if isinstance(state, dict):
        return "; ".join(f"{k}: {v}" for k, v in state.items())
    return " ".join(state)


def question_text(q: Noul | Choice | Score) -> str:
    if isinstance(q, Noul):
        extra = " ".join(x for x in (q.true_description, q.false_description) if x)
        return f"{q.proposition} {extra}".strip()
    return q.prompt


def build_context(
    person: PersonContext | None,
    state: str | dict[str, object] | list[str],
    question: Noul | Choice | Score,
    tokenizer: Any,
    max_tokens: int,
    history_items: int,
) -> str:
    """`[PERSON] ... [STATE] ... [QUESTION] ...`, fitted to `max_tokens`.

    When too long, the oldest history items go first, then attributes, then the start of the
    state (its end is the most recent context). The question is never cut.
    """
    q = question_text(question)
    s = state_text(state)

    def render(p: str, st: str) -> str:
        return f"[PERSON] {p} [STATE] {st} [QUESTION] {q}"

    def n_tokens(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=True)["input_ids"])

    k = history_items
    while True:
        text = render(person_text(person, k), s)
        if n_tokens(text) <= max_tokens or k == 0:
            break
        k = k // 2 if k > 8 else k - 1
    n_attr = len(person.attributes) if person is not None else 0
    while n_tokens(text) > max_tokens and n_attr > 0:
        n_attr = n_attr // 2 if n_attr > 4 else n_attr - 1
        text = render(person_text(person, 0, n_attr), s)
    if n_tokens(text) > max_tokens:
        lo, hi = 0, len(s)
        while lo < hi:  # longest state suffix that fits
            mid = (lo + hi + 1) // 2
            fits = n_tokens(render(person_text(person, 0, n_attr), s[len(s) - mid :])) <= max_tokens
            lo, hi = (mid, hi) if fits else (lo, mid - 1)
        text = render(person_text(person, 0, n_attr), s[len(s) - lo :] if lo else "")
    return text
