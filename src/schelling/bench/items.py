"""Which Twin-2K-500 items count as decisions, and which persons are asked.

Decision items: economic preferences (price lists, ultimatum, trust, dictator, mental
accounting), purchase decisions at a price, the non-experimental heuristics-and-biases block,
and every between-subject heuristics-and-biases experiment. Personality, ability tests and the
false-consensus attitude block are not decisions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DECISION_STUDIES = frozenset(
    {
        "Economic preferences",
        "Product Preferences - Pricing",
        "Non-experimental heuristics and biases",
    }
)
NOT_DECISIONS = frozenset({"False consensus"})
PRICE_LIST_MIN_ROWS = 5  # a task with this many rows is a multiple price list
ROWS_PER_PRICE_LIST = 3
ROWS_PER_SMALL_TASK = 3
PRICING_ITEMS = 12


def is_decision(study: str, condition: object) -> bool:
    if study in NOT_DECISIONS:
        return False
    has_condition = isinstance(condition, str) and condition != ""
    return study in DECISION_STUDIES or has_condition


def _row_index(item_id: str) -> int:
    tail = item_id.split("_")[-1]
    return int(tail) if "_" in item_id and tail.isdigit() else 0


def _spread(ids: list[str], k: int) -> list[str]:
    """First, middle and last rows (for k = 3): the switch point of a price list lies between."""
    ordered = sorted(ids, key=_row_index)
    if len(ordered) <= k:
        return ordered
    picks = np.linspace(0, len(ordered) - 1, k).round().astype(int)
    return [ordered[i] for i in sorted(set(picks))]


def select_items(frame: pd.DataFrame, usable: set[str], seed: int = 0) -> pd.DataFrame:
    """One row per selected item, with a `task` column (the item id before the row suffix).

    `usable` excludes items that must stay untouched (the sealed `heldout_item` test items).
    """
    items = frame.drop_duplicates("item_id")
    mask = [is_decision(s, c) for s, c in zip(items["study_id"], items["condition"], strict=True)]
    items = items[np.array(mask, dtype=bool) & items["item_id"].isin(usable).to_numpy()].copy()
    items["task"] = items["item_id"].str.split("_").str[0]
    keep: list[str] = []
    rng = np.random.default_rng(seed)
    for (study, _task), group in items.groupby(["study_id", "task"], sort=True):
        ids = list(group["item_id"])
        if study == "Product Preferences - Pricing":
            k = min(PRICING_ITEMS, len(ids))
            keep += sorted(rng.choice(sorted(ids), size=k, replace=False).tolist())
        elif len(ids) >= PRICE_LIST_MIN_ROWS:
            keep += _spread(ids, ROWS_PER_PRICE_LIST)
        else:
            keep += _spread(ids, ROWS_PER_SMALL_TASK)
    out = items[items["item_id"].isin(keep)]
    return out.sort_values(["study_id", "task", "item_id"]).reset_index(drop=True)


def select_persons(person_ids: list[str], n: int, seed: int = 0) -> list[str]:
    ids = sorted(set(map(str, person_ids)))
    if n >= len(ids):
        return ids
    return sorted(np.random.default_rng(seed).choice(ids, size=n, replace=False).tolist())
