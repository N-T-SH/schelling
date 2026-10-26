"""Partition access. The `test` partition is sealed (AGENTS.md rule 3)."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from schelling.data.splits import assign_partitions

PARTITIONS = ("train", "val", "test")
ALLOW_TEST_ENV = "SCHELLING_ALLOW_TEST"


class SealedPartitionError(PermissionError):
    pass


def select_partition(frame: pd.DataFrame, partition: str) -> pd.DataFrame:
    if partition not in PARTITIONS:
        raise ValueError(f"partition must be one of {PARTITIONS}, got {partition!r}")
    if partition == "test" and os.environ.get(ALLOW_TEST_ENV) != "1":
        raise SealedPartitionError(
            f"the test partition is sealed; only the human sets {ALLOW_TEST_ENV}=1"
        )
    return frame[frame["partition"] == partition]


def load_records(processed_dir: Path) -> pd.DataFrame:
    parts = sorted(processed_dir.glob("records_*.parquet"))
    if not parts:
        raise FileNotFoundError(f"no records_*.parquet in {processed_dir}")
    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)


def load_partition(processed_dir: Path, split: str, partition: str, seed: int = 0) -> pd.DataFrame:
    # Check the guard before touching any data.
    if partition == "test" and os.environ.get(ALLOW_TEST_ENV) != "1":
        select_partition(pd.DataFrame({"partition": []}), partition)
    frame = load_records(processed_dir)
    frame["partition"] = assign_partitions(frame, split, seed)
    return select_partition(frame, partition)
