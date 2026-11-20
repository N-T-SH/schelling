"""Prompt records for D01-lite, in two formats from one item.

- `chat`: messages asking for a probability per option as JSON (API models, stated probabilities).
- `completion`: a Psych-101-style transcript ending in `<<`, so a model can be scored by the
  log-probability of each option's key (Psych-101 fine-tunes and any base model).

Options keep the order respondents saw. The only person information is the attribute table
(the 14 demographic fields); persona text fields never enter a prompt (A04).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pandas as pd

from schelling.baselines.base import labels_of
from schelling.baselines.llm_prompt import build_messages

KEYS = tuple("ABCDEFGHIJKLMNOPQRST")  # schema allows at most 20 options
TEMPLATE_VERSION = "d01-lite-v1"

_INTRO = (
    "In this study you answer a decision question. Each option has a key. "
    "You choose an option by pressing its key."
)


def labels(qtype: str, options: Any, levels: Any) -> list[str]:
    return labels_of(
        qtype,
        None if options is None else list(options),
        None if levels is None else list(levels),
    )


def _profile(attrs: dict[str, Any]) -> str:
    return "\n".join(f"  {k}: {v}" for k, v in attrs.items())


def completion_prompt(question: str, opts: list[str], attrs: dict[str, Any] | None) -> str:
    parts = [_INTRO]
    if attrs is not None:
        parts.append("Participant:\n" + _profile(attrs))
    parts.append(f"Question: {question}")
    parts.append("\n".join(f"Press {k} for: {lab}" for k, lab in zip(KEYS, opts, strict=False)))
    return "\n\n".join(parts) + "\n\nYou press <<"


def chat_messages(
    question: str, opts: list[str], attrs: dict[str, Any] | None
) -> list[dict[str, str]]:
    return build_messages(question, opts, attrs, wording=0)


def _id(payload: dict[str, Any]) -> str:
    raw = json.dumps({**payload, "v": TEMPLATE_VERSION}, sort_keys=True, default=str)
    return hashlib.sha1(raw.encode()).hexdigest()


def build_records(
    items: pd.DataFrame, persons: pd.DataFrame, person_ids: list[str]
) -> list[dict[str, Any]]:
    cols = [c for c in persons.columns if c != "person_id"]
    attrs_of = {
        str(r["person_id"]): {c: r[c] for c in cols if pd.notna(r[c])}
        for r in persons.to_dict("records")
    }
    out: list[dict[str, Any]] = []
    for it in items.to_dict("records"):
        opts = labels(it["qtype"], it["options"], it["levels"])
        if len(opts) > len(KEYS):
            raise ValueError(f"{it['item_id']}: more than {len(KEYS)} options")
        base = {
            "item_id": it["item_id"],
            "study_id": it["study_id"],
            "task": it["task"],
            "qtype": it["qtype"],
            "labels": opts,
            "keys": list(KEYS[: len(opts)]),
        }
        for pid in [None, *person_ids]:
            attrs = None if pid is None else attrs_of.get(pid, {})
            rec = {
                **base,
                "arm": "population" if pid is None else "persona",
                "person_id": pid,
                "chat": chat_messages(it["question"], opts, attrs),
                "completion": completion_prompt(it["question"], opts, attrs),
            }
            rec["prompt_id"] = _id({k: rec[k] for k in ("item_id", "person_id", "chat")})
            out.append(rec)
    return out
