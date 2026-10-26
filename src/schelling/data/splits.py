"""Deterministic, hash-ranked splits (SPEC "Splits").

Units (persons, items, studies) are ranked by a seeded hash within each source and the first
10% become `val`, the next 10% `test`. Ranking, not thresholding, keeps sizes within 1pp.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Collection
from pathlib import Path
from typing import Any

import pandas as pd

SPLIT_NAMES = (
    "heldout_person",
    "heldout_item",
    "heldout_study",
    "heldout_domain",
    "temporal",
    "retest",
    # Leave-one-experiment-out: fold k holds out one of the 8 experiments in effects_pairs.yaml.
    *(f"heldout_experiment_f{k}" for k in range(8)),
)
EXPERIMENT_PREFIX = "heldout_experiment_f"
VAL_FRAC = 0.10
TEST_FRAC = 0.10


def fold_experiment(fold: int, seed: int) -> str:
    """The experiment held out in LOEO fold `fold` (experiments ranked by seeded hash)."""
    from schelling.eval.effects import load_pairs

    names = sorted({p["experiment"] for p in load_pairs()}, key=lambda e: _h(seed, "loeo", e))
    return names[fold]


def _h(seed: int, salt: str, key: str) -> str:
    return hashlib.sha256(f"{seed}\x1f{salt}\x1f{key}".encode()).hexdigest()


def _rank_assign(units: pd.Series, source: pd.Series, seed: int, salt: str) -> pd.Series:
    """Map each row to train/val/test by its unit's hash rank within the row's source."""
    out = pd.Series("train", index=units.index, dtype=object)
    for src in source.unique():
        mask = source == src
        uniq = sorted(units[mask].unique(), key=lambda u: _h(seed, salt, str(u)))
        n_val, n_test = round(VAL_FRAC * len(uniq)), round(TEST_FRAC * len(uniq))
        label = {u: "val" for u in uniq[:n_val]}
        label |= {u: "test" for u in uniq[n_val : n_val + n_test]}
        out[mask] = units[mask].map(lambda u, lab=label: lab.get(u, "train"))
    return out


def assign_partitions(
    frame: pd.DataFrame,
    name: str,
    seed: int,
    direction: str = "lab_to_survey",
    exclude_item_ids: Collection[str] = (),
) -> pd.Series:
    """Return train/val/test (or None for rows the split excludes), indexed like `frame`.

    `exclude_item_ids` are removed from val and test (contamination removal); train is untouched.
    """
    out = _assign(frame, name, seed, direction)
    if exclude_item_ids:
        out = out.mask(
            frame["item_id"].isin(set(exclude_item_ids)) & out.isin(["val", "test"]), None
        )
    return out


def _assign(frame: pd.DataFrame, name: str, seed: int, direction: str) -> pd.Series:
    if name == "heldout_person":
        return _rank_assign(frame["person_id"], frame["source"], seed, name)
    if name == "heldout_item":
        return _rank_assign(frame["item_id"], frame["source"], seed, name)
    if name == "heldout_study":
        return _rank_assign(frame["study_id"], frame["source"], seed, name)
    if name == "heldout_domain":
        train_src, held_src = ("psych101", "twin2k")
        if direction == "survey_to_lab":
            train_src, held_src = held_src, train_src
        out = pd.Series(None, index=frame.index, dtype=object)
        out[frame["source"] == train_src] = "train"
        held = frame["source"] == held_src
        # Held-out domain is split 50/50 into val/test by person hash.
        persons = sorted(frame.loc[held, "person_id"].unique(), key=lambda p: _h(seed, name, p))
        label = {p: ("val" if i < len(persons) // 2 else "test") for i, p in enumerate(persons)}
        out[held] = frame.loc[held, "person_id"].map(label)
        return out
    if name == "temporal":
        out = pd.Series(None, index=frame.index, dtype=object)
        seq = frame[frame["trial_idx"].notna()]
        for _, grp in seq.groupby(["source", "study_id", "person_id"], sort=False):
            order = grp.sort_values("trial_idx").index
            n = len(order)
            a, b = round(0.8 * n), round(0.9 * n)
            out[order[:a]] = "train"
            out[order[a:b]] = "val"
            out[order[b:]] = "test"
        return out
    if name.startswith(EXPERIMENT_PREFIX):
        # Fold k: experiment e_k is absent from train for everyone. Its val and test rows are the
        # val and test persons of heldout_person; every other row of val and test persons is
        # excluded, so val/test contain only unseen-experiment, unseen-person rows.
        from schelling.eval.effects import experiment_of_item

        exp = frame["item_id"].map(experiment_of_item())
        held_exp = fold_experiment(int(name[len(EXPERIMENT_PREFIX) :]), seed)
        person_part = _rank_assign(frame["person_id"], frame["source"], seed, "heldout_person")
        out = pd.Series(None, index=frame.index, dtype=object)
        is_held = exp == held_exp
        out[(person_part == "train") & ~is_held] = "train"
        out[is_held & (person_part == "val")] = "val"
        out[is_held & (person_part == "test")] = "test"
        return out
    if name == "retest":
        # Wave-4 answers only; used to measure the test-retest ceiling, never to train.
        out = pd.Series(None, index=frame.index, dtype=object)
        w4 = frame["wave"] == 4
        persons = sorted(frame.loc[w4, "person_id"].unique(), key=lambda p: _h(seed, name, p))
        label = {p: ("val" if i < len(persons) // 2 else "test") for i, p in enumerate(persons)}
        out[w4] = frame.loc[w4, "person_id"].map(label)
        return out
    raise ValueError(f"unknown split {name!r}; choose from {SPLIT_NAMES}")


def split_manifest(
    frame: pd.DataFrame, name: str, seed: int, exclude_item_ids: Collection[str] = ()
) -> dict[str, Any]:
    part = assign_partitions(frame, name, seed, exclude_item_ids=exclude_item_ids)
    counts = part.fillna("excluded").value_counts().to_dict()
    joined = "\n".join(
        f"{rid}:{p}" for rid, p in sorted(zip(frame["record_id"], part.fillna("-"), strict=True))
    )
    body = {
        "name": name,
        "seed": seed,
        "fractions": {"val": VAL_FRAC, "test": TEST_FRAC},
        "counts": {k: int(counts.get(k, 0)) for k in ("train", "val", "test", "excluded")},
        "removed_item_ids": sorted(exclude_item_ids),
        "assignment_hash": hashlib.sha256(joined.encode()).hexdigest(),
    }
    if name.startswith(EXPERIMENT_PREFIX):
        body["held_out_experiment"] = fold_experiment(int(name[len(EXPERIMENT_PREFIX) :]), seed)
    body["manifest_hash"] = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    return body


def write_split(manifest: dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{manifest['name']}.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path
