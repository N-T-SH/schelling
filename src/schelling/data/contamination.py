"""Exact and near-duplicate question text between train and held-out sets (MinHash + LSH)."""

from __future__ import annotations

import hashlib
import re
from typing import Any

import numpy as np
import pandas as pd

_NUM_PERM = 128
_PRIME = (1 << 61) - 1
_RNG = np.random.default_rng(12345)  # fixed: signatures must be reproducible
_A = _RNG.integers(1, _PRIME, size=_NUM_PERM, dtype=np.uint64)
_B = _RNG.integers(0, _PRIME, size=_NUM_PERM, dtype=np.uint64)


def normalise(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def shingles(text: str, n: int = 3) -> set[str]:
    toks = normalise(text).split()
    if len(toks) <= n:
        return {" ".join(toks)}
    return {" ".join(toks[i : i + n]) for i in range(len(toks) - n + 1)}


def jaccard_shingles(a: str, b: str) -> float:
    sa, sb = shingles(a), shingles(b)
    return len(sa & sb) / len(sa | sb) if sa | sb else 1.0


def _signature(sh: set[str]) -> np.ndarray:
    hv = np.array(
        [int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "big") for s in sh],
        dtype=np.uint64,
    )
    # (a*h + b) mod p with Python ints would be slow; uint64 wraparound is fine for LSH hashing.
    sig: np.ndarray = ((hv[:, None] * _A[None, :] + _B[None, :]) % np.uint64(_PRIME)).min(axis=0)
    return sig


def contamination_report(
    train: pd.DataFrame, held: pd.DataFrame, threshold: float = 0.8
) -> dict[str, Any]:
    """Item ids of held-out questions that duplicate a train question.

    Both frames need `item_id` and `question`. Near duplicates are candidates from LSH bands
    that are confirmed with an exact shingle Jaccard >= threshold.
    """
    t = train.drop_duplicates("item_id")
    h = held.drop_duplicates("item_id")
    train_norm = {normalise(q) for q in t["question"]}
    exact = sorted(
        i for i, q in zip(h["item_id"], h["question"], strict=True) if normalise(q) in train_norm
    )

    rows, bands = 8, _NUM_PERM // 8
    buckets: dict[tuple[int, bytes], list[str]] = {}
    for q in t["question"].unique():
        sig = _signature(shingles(q))
        for b in range(bands):
            buckets.setdefault((b, sig[b * rows : (b + 1) * rows].tobytes()), []).append(q)
    exact_set = set(exact)
    near = []
    for i, q in zip(h["item_id"], h["question"], strict=True):
        if i in exact_set:
            continue
        sig = _signature(shingles(q))
        cands = {
            c
            for b in range(bands)
            for c in buckets.get((b, sig[b * rows : (b + 1) * rows].tobytes()), [])
        }
        if any(jaccard_shingles(q, c) >= threshold for c in cands):
            near.append(i)
    return {
        "threshold": threshold,
        "n_held_items": len(h),
        "exact": exact,
        "near": sorted(near),
        "remove_item_ids": sorted(set(exact) | set(near)),
    }
