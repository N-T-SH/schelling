"""Per-row answer history built from a scored frame (campaign C01).

The frame holds every row of the people being scored, including the target's own response, so
masking is the whole safeguard (assumption A04). For each row the history is the same person's
other first-pass answers: the target's item id, its wave-4 repeats and any other row with the
same question text are dropped, and wave-4 rows are never used as evidence.
"""

from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd

from schelling.schema import HistoryItem

RETEST_WAVE = 4


def _options(qtype: str, options: Any, levels: Any) -> list[str]:
    if qtype == "noul":
        return ["no", "yes"]
    return [str(o) for o in (options if qtype == "choice" else levels)]


def _rank(record_id: str, item_id: str) -> str:
    # A fixed pseudo-random order, so a capped history is neither the first nor the last items.
    return hashlib.md5(f"{record_id}|{item_id}".encode()).hexdigest()


def history_for_frame(frame: pd.DataFrame, k: int) -> list[list[HistoryItem]]:
    """History for each row of `frame`, in row order. A row's history ignores row order."""
    if not frame.index.is_unique:
        raise ValueError("the frame index must be unique")
    out: list[list[HistoryItem]] = [[] for _ in range(len(frame))]
    if k <= 0 or frame.empty:
        return out
    evidence = frame[frame["wave"] != RETEST_WAVE]
    by_person = {pid: g for pid, g in evidence.groupby("person_id", sort=False)}
    for pos, (rid, pid, item, question) in enumerate(
        zip(
            frame["record_id"],
            frame["person_id"],
            frame["item_id"],
            frame["question"],
            strict=True,
        )
    ):
        g = by_person.get(pid)
        if g is None:
            continue
        keep = g[(g["item_id"] != item) & (g["question"] != question)]
        if keep.empty:
            continue
        ranked = sorted(
            zip(keep["item_id"], keep.index, strict=True), key=lambda t: _rank(rid, t[0])
        )
        chosen = sorted(ranked[:k], key=lambda t: t[0])
        rows = keep.loc[[i for _, i in chosen]]
        out[pos] = [
            HistoryItem(
                question=str(q),
                options=_options(t, o, lv),
                answer=int(r),
            )
            for q, t, o, lv, r in zip(
                rows["question"],
                rows["qtype"],
                rows["options"],
                rows["levels"],
                rows["response"],
                strict=True,
            )
        ]
    return out
