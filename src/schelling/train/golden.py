"""Frozen mini-benchmark for the regression test: a tiny model trained on the fixtures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from schelling.baselines import make_baseline
from schelling.data.loader import load_records, partition_frame, select_partition
from schelling.eval import metrics as m
from schelling.train.config import TrainConfig
from schelling.train.trainer import train

VOCAB = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "tiny_vocab.txt"


def golden_config(tmp: Path, proc: Path, splits: Path) -> Path:
    cfg = {
        "name": "golden",
        "seed": 0,
        "split": "heldout_person",
        "processed_dir": str(proc),
        "splits_dir": str(splits),
        "out_dir": str(tmp / "golden"),
        "device": "cpu",
        "deterministic": True,
        "model": {
            "tiny_vocab": str(VOCAB),
            "history_items": 0,
            "max_context_tokens": 96,
            "max_option_tokens": 8,
        },
        "optim": {
            "steps": 50,
            "batch_size": 8,
            "lr_backbone": 3e-3,
            "lr_head": 3e-3,
            "warmup_frac": 0.1,
        },
        "eval": {"every": 25, "max_rows": 400, "patience": 5},
    }
    path = tmp / "golden.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def run_golden(cfg: TrainConfig) -> dict[str, Any]:
    """Train, then score the calibrated model on the whole validation partition."""
    result = train(cfg)
    proc, splits = Path(cfg.processed_dir), Path(cfg.splits_dir)
    parted = partition_frame(load_records(proc), cfg.split, splits)
    val = select_partition(parted, "val").reset_index(drop=True)
    persons = pd.read_parquet(proc / "persons_twin2k.parquet")
    model = make_baseline(str(result.run_dir / "final")).fit(
        select_partition(parted, "train"), persons
    )
    probs = model.predict_proba(val)
    y = val["response"].to_numpy()
    conf, correct = m.top_label(probs, y)
    return {
        "val_nll": m.nll(probs, y),
        "val_ece": m.ece(conf, correct),
        "n_val": len(y),
        "steps": cfg.optim.steps,
        "seed": cfg.seed,
        "mean_conf": float(np.mean(conf)),
    }
