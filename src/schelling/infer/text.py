"""Serialise a person, state and question into the model's context text (person encoder v0)."""

from __future__ import annotations

from typing import Any

from schelling.schema import Choice, Noul, PersonContext, Score


def _attr_text(attrs: dict[str, Any]) -> str:
    return "; ".join(f"{k}: {v}" for k, v in sorted(attrs.items()))


def _history_line(item: Any) -> str:
    return f"{item.question} [{' | '.join(item.options)}] -> {item.options[item.answer]}"


def person_text(person: PersonContext | None, k: int) -> str:
    """Attributes plus the last `k` history items. The opaque person id is never included."""
    if person is None:
        return ""
    parts = []
    if person.attributes:
        parts.append(_attr_text(person.attributes))
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

    When too long, the oldest history items go first, then the tail of the state; the question
    is never cut.
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
    if n_tokens(text) > max_tokens:  # attributes alone or the state are too long: cut the state
        lo, hi = 0, len(s)
        while lo < hi:  # longest state prefix that fits
            mid = (lo + hi + 1) // 2
            lo, hi = (
                (mid, hi)
                if n_tokens(render(person_text(person, 0), s[:mid])) <= max_tokens
                else (lo, mid - 1)
            )
        text = render(person_text(person, 0), s[:lo])
    return text
