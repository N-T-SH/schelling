"""Training configuration: pydantic models loaded from YAML, overridden with `--set key=value`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelCfg(_Strict):
    backbone: str = "answerdotai/ModernBERT-base"
    tiny_vocab: str | None = None  # path to a vocab file: build a tiny random BERT (tests, CI)
    history_items: int = 0
    max_context_tokens: int = 128
    max_option_tokens: int = 16
    head_hidden: int = 256
    freeze_layers: int = 0  # freeze the embeddings and this many lowest encoder layers


class DataCfg(_Strict):
    sources: list[str] = ["twin2k", "psych101"]
    max_train_rows: int | None = None
    study_temperature: float = 0.5
    augment_option_permutation: bool = True


class OptimCfg(_Strict):
    steps: int = 1000
    batch_size: int = 32
    lr_backbone: float = 3e-5
    lr_head: float = 1e-3
    weight_decay: float = 0.01
    warmup_frac: float = 0.05
    grad_clip: float = 1.0
    precision: str = "fp32"  # fp32 | bf16 (autocast)


class EvalCfg(_Strict):
    every: int = 250
    max_rows: int = 5000
    patience: int = 4
    calibration_rows: int = 5000


class TrainConfig(_Strict):
    name: str
    seed: int = 0
    split: str = "heldout_person"
    processed_dir: str = "data/processed"
    splits_dir: str = "splits"
    out_dir: str = "runs/run"
    device: str = "auto"  # auto | cpu | mps | cuda
    deterministic: bool = False  # torch deterministic algorithms and one thread (CPU)
    tensorboard: bool = False
    model: ModelCfg = ModelCfg()
    data: DataCfg = DataCfg()
    optim: OptimCfg = OptimCfg()
    eval: EvalCfg = EvalCfg()


def _set(d: dict[str, Any], dotted: str, value: Any) -> None:
    *path, last = dotted.split(".")
    for key in path:
        d = d.setdefault(key, {})
    d[last] = value


def load_config(path: Path, overrides: list[str]) -> TrainConfig:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override must look like key=value, got {item!r}")
        key, value = item.split("=", 1)
        _set(raw, key.strip(), yaml.safe_load(value))
    return TrainConfig.model_validate(raw)
