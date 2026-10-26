"""Train a SchellingModel on human responses with proper scoring rules, then calibrate it."""

from __future__ import annotations

import json
import math
import platform
import sys
import time
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from transformers import AutoTokenizer, BertTokenizer

from schelling.data.loader import load_records, partition_frame, select_partition
from schelling.data.splits import split_manifest
from schelling.eval import provenance
from schelling.infer.baseline_adapter import frame_items, person_attributes, predict_items
from schelling.infer.batching import build_tensors
from schelling.infer.predictor import Predictor
from schelling.model import SchellingConfig, SchellingModel, calibration
from schelling.train import losses
from schelling.train.config import TrainConfig
from schelling.train.sampler import StudySampler

TASKS = ("noul", "choice", "score")


@dataclass
class TrainResult:
    run_dir: Path
    best_val_nll: float


def _seed_all(seed: int, deterministic: bool) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(1)


def _device(cfg: TrainConfig) -> torch.device:
    if cfg.device != "auto":
        return torch.device(cfg.device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _tiny(cfg: TrainConfig) -> tuple[SchellingModel, Any]:
    vocab_path = Path(cfg.model.tiny_vocab or "")
    vocab = {tok: i for i, tok in enumerate(vocab_path.read_text().split())}
    backbone = {
        "model_type": "bert",
        "vocab_size": len(vocab),
        "hidden_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "intermediate_size": 64,
        "max_position_embeddings": 512,
        "hidden_dropout_prob": 0.0,
        "attention_probs_dropout_prob": 0.0,
    }
    config = SchellingConfig(
        backbone_config=backbone,
        head_hidden=16,
        max_context_tokens=cfg.model.max_context_tokens,
        max_option_tokens=cfg.model.max_option_tokens,
        history_items=cfg.model.history_items,
    )
    return SchellingModel(config), BertTokenizer(vocab=vocab, do_lower_case=True)


def _build(cfg: TrainConfig) -> tuple[SchellingModel, Any]:
    if cfg.model.tiny_vocab:
        return _tiny(cfg)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.backbone)
    model = SchellingModel.from_backbone(
        cfg.model.backbone,
        head_hidden=cfg.model.head_hidden,
        max_context_tokens=cfg.model.max_context_tokens,
        max_option_tokens=cfg.model.max_option_tokens,
        history_items=cfg.model.history_items,
    )
    return model, tokenizer


def _freeze_lower(model: SchellingModel, n_layers: int) -> None:
    if n_layers <= 0:
        return
    backbone = model.backbone
    layers = getattr(backbone, "layers", None) or backbone.encoder.layer
    for p in backbone.embeddings.parameters():
        p.requires_grad = False
    for layer in list(layers)[:n_layers]:
        for p in layer.parameters():
            p.requires_grad = False


def _lr_lambda(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


class _Rows:
    """Columns of a record frame as plain lists, for cheap per-example access."""

    def __init__(self, frame: pd.DataFrame, attributes: dict[str, dict[str, Any]]) -> None:
        self.n = len(frame)
        self.cols = {
            c: frame[c].tolist()
            for c in (
                "person_id",
                "qtype",
                "question",
                "options",
                "levels",
                "state",
                "response",
                "source",
            )
        }
        self.attributes = attributes

    def item(self, i: int, options: list[str] | None = None):  # type: ignore[no-untyped-def]
        from schelling.infer.baseline_adapter import make_item

        c = self.cols
        opts = (
            options
            if options is not None
            else (None if c["options"][i] is None else list(c["options"][i]))
        )
        lv = None if c["levels"][i] is None else list(c["levels"][i])
        return make_item(
            c["qtype"][i],
            c["question"][i],
            opts,
            lv,
            c["state"][i] or "",
            self.attributes.get(str(c["person_id"][i])),
        )


def _val_subset(val: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(val) <= n:
        return val.reset_index(drop=True)
    return val.sample(n=n, random_state=seed).reset_index(drop=True)


def _eval_nll(
    model: SchellingModel,
    tokenizer: Any,
    rows: pd.DataFrame,
    attrs: dict[str, dict[str, Any]],
    batch_size: int,
    precision: str,
) -> tuple[float, dict[str, float]]:
    probs = predict_items(model, tokenizer, frame_items(rows, attrs), batch_size, precision)
    y = rows["response"].to_numpy()
    nll = np.array([-math.log(max(float(p[i]), 1e-12)) for p, i in zip(probs, y, strict=True)])
    by_source = {
        s: float(nll[(rows["source"] == s).to_numpy()].mean())
        for s in sorted(rows["source"].unique())
    }
    return float(nll.mean()), by_source


def _calibrate(
    model: SchellingModel,
    tokenizer: Any,
    rows: pd.DataFrame,
    attrs: dict[str, dict[str, Any]],
    batch_size: int,
) -> dict[str, Any]:
    """Fit one temperature per head type on `rows` (validation only) and store it in the config."""
    device = next(model.parameters()).device
    items = frame_items(rows, attrs)
    y_all = rows["response"].to_numpy()
    result: dict[str, Any] = {"n": {}, "temperature": {}}
    cache: dict[str, tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]] = {}
    model.eval()
    with torch.inference_mode():
        for task in TASKS:
            idx = [i for i, (_, q) in enumerate(items) if q.type == task]
            if not idx:
                continue
            logits, masks, ys = [], [], []
            for lo in range(0, len(idx), batch_size):
                chunk = idx[lo : lo + batch_size]
                t = {
                    k: v.to(device)
                    for k, v in build_tensors(
                        tokenizer, model.config, task, [items[i] for i in chunk]
                    ).items()
                }
                lg, mk = model.raw(task, **t)
                logits.append(lg.float().cpu().reshape(len(chunk), -1))
                if mk is not None:
                    masks.append(mk.cpu())
                ys.append(torch.tensor(y_all[chunk], dtype=torch.long))
            lg_all = calibration.pad_stack(logits, 0.0)
            mk_all = calibration.pad_stack(masks, False) if masks else None
            cache[task] = (lg_all, mk_all, torch.cat(ys))
            result["n"][task] = len(idx)
    before = after = 0.0
    total = sum(len(v[2]) for v in cache.values())
    for task, (lg, mk, y) in cache.items():
        t0 = getattr(model.config, f"temperature_{task}")
        t_fit = calibration.fit_temperature(task, lg, mk, y) if len(y) >= 50 else 1.0
        before += calibration.nll_at_temperature(task, lg, mk, y, t0) * len(y)
        after += calibration.nll_at_temperature(task, lg, mk, y, t_fit) * len(y)
        setattr(model.config, f"temperature_{task}", t_fit)
        result["temperature"][task] = t_fit
    result["val_nll_before"], result["val_nll_after"] = before / total, after / total
    return result


def _collate(rows: _Rows, idx: np.ndarray, rng: np.random.Generator, augment: bool):  # type: ignore[no-untyped-def]
    """Group a sampled batch by task type -> {task: (items, targets, example positions)}."""
    groups: dict[str, tuple[list[Any], list[int], list[int]]] = {t: ([], [], []) for t in TASKS}
    for pos, i in enumerate(idx):
        qtype = rows.cols["qtype"][i]
        y = int(rows.cols["response"][i])
        options = None
        if qtype == "choice" and augment:
            opts = rows.cols["options"][i]
            perm = rng.permutation(len(opts))
            options = [opts[j] for j in perm]
            y = int(np.where(perm == y)[0][0])
        items, ys, poss = groups[qtype]
        items.append(rows.item(i, options))
        ys.append(y)
        poss.append(pos)
    return groups


def train(cfg: TrainConfig) -> TrainResult:
    t_start = time.time()
    _seed_all(cfg.seed, cfg.deterministic)
    device = _device(cfg)
    run_dir = Path(cfg.out_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    frame = load_records(Path(cfg.processed_dir))
    frame = frame[frame["source"].isin(cfg.data.sources)].reset_index(drop=True)
    persons_path = Path(cfg.processed_dir) / "persons_twin2k.parquet"
    attrs = person_attributes(pd.read_parquet(persons_path)) if persons_path.exists() else {}
    splits_dir = Path(cfg.splits_dir)
    parted = partition_frame(frame, cfg.split, splits_dir)
    train_df = select_partition(parted, "train").reset_index(drop=True)  # test is never touched
    val_df = select_partition(parted, "val").reset_index(drop=True)
    if cfg.data.max_train_rows and len(train_df) > cfg.data.max_train_rows:
        train_df = train_df.sample(n=cfg.data.max_train_rows, random_state=cfg.seed).reset_index(
            drop=True
        )
    manifest_path = splits_dir / f"{cfg.split}.json"
    manifest_hash = (
        json.loads(manifest_path.read_text())["manifest_hash"]
        if manifest_path.exists()
        else split_manifest(frame, cfg.split, 0)["manifest_hash"]
    )

    model, tokenizer = _build(cfg)
    _freeze_lower(model, cfg.model.freeze_layers)
    model = model.to(device)  # type: ignore[arg-type]
    head_params = [
        p for n, p in model.named_parameters() if not n.startswith("backbone.") and p.requires_grad
    ]
    back_params = [
        p for n, p in model.named_parameters() if n.startswith("backbone.") and p.requires_grad
    ]
    opt = torch.optim.AdamW(
        [
            {"params": back_params, "lr": cfg.optim.lr_backbone},
            {"params": head_params, "lr": cfg.optim.lr_head},
        ],
        weight_decay=cfg.optim.weight_decay,
    )
    warmup = max(1, int(cfg.optim.warmup_frac * cfg.optim.steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: _lr_lambda(s, cfg.optim.steps, warmup))

    rows = _Rows(train_df, attrs)
    sampler = StudySampler(train_df["study_id"], cfg.data.study_temperature, cfg.seed)
    aug_rng = np.random.default_rng(cfg.seed + 1)
    val_rows = _val_subset(val_df, cfg.eval.max_rows, cfg.seed)
    cal_rows = _val_subset(val_df, cfg.eval.calibration_rows, cfg.seed + 7)
    sources = sorted(train_df["source"].unique())

    writer = None
    if cfg.tensorboard:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(str(run_dir / "tb"))

    curve: list[float] = []
    by_source: dict[str, list[float | None]] = {s: [] for s in sources}
    val_curve: list[dict[str, Any]] = []
    best_nll, best_step, bad_evals = float("inf"), -1, 0
    autocast = torch.autocast(
        device.type, dtype=torch.bfloat16, enabled=cfg.optim.precision == "bf16"
    )

    def evaluate(step: int) -> None:
        nonlocal best_nll, best_step, bad_evals
        nll, per_source = _eval_nll(
            model, tokenizer, val_rows, attrs, cfg.optim.batch_size * 2, cfg.optim.precision
        )
        val_curve.append({"step": step, "val_nll": nll, "by_source": per_source})
        print(f"[eval] step {step} val_nll {nll:.4f} {per_source}", flush=True)
        if writer:
            writer.add_scalar("val/nll", nll, step)
        if nll < best_nll:
            best_nll, best_step, bad_evals = nll, step, 0
            Predictor(model, tokenizer).save_pretrained(run_dir / "best")
        else:
            bad_evals += 1

    for step in range(cfg.optim.steps):
        model.train()
        idx = sampler.sample(cfg.optim.batch_size)
        groups = _collate(rows, idx, aug_rng, cfg.data.augment_option_permutation)
        per_example = torch.zeros(len(idx), device=device)
        for task, (items, ys, poss) in groups.items():
            if not items:
                continue
            # Micro-batches of similar length: padding a short Twin question to a long Psych-101
            # transcript wastes most of the compute on this hardware. The maths is unchanged.
            order = sorted(
                range(len(items)),
                key=lambda j: (
                    len(items[j][0].state)
                    + len(
                        items[j][1].prompt
                        if hasattr(items[j][1], "prompt")
                        else items[j][1].proposition
                    )
                ),
            )
            for lo in range(0, len(order), cfg.optim.micro_batch):
                sel = order[lo : lo + cfg.optim.micro_batch]
                t = {
                    k: v.to(device)
                    for k, v in build_tensors(
                        tokenizer, model.config, task, [items[j] for j in sel]
                    ).items()
                }
                y = torch.tensor([ys[j] for j in sel], device=device)
                with autocast:
                    if task == "score":
                        cond = model.score_conditional_probs(**t)
                        loss_vec = losses.corn_loss(cond.float(), y, reduction="none")
                    else:
                        probs = model(task, **t)
                        loss_vec = losses.hard_cross_entropy(probs.float(), y, reduction="none")
                per_example = per_example.index_put(
                    (torch.tensor([poss[j] for j in sel], device=device),), loss_vec
                )
        loss = per_example.mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
        opt.step()
        sched.step()
        curve.append(float(loss))
        src = np.array([rows.cols["source"][i] for i in idx])
        pe = per_example.detach().cpu().numpy()
        for s in sources:
            by_source[s].append(float(pe[src == s].mean()) if (src == s).any() else None)
        if writer:
            writer.add_scalar("train/loss", curve[-1], step)
        if (step + 1) % 25 == 0:
            print(
                f"[train] step {step + 1}/{cfg.optim.steps} loss {np.mean(curve[-25:]):.4f} "
                f"elapsed {time.time() - t_start:.0f}s",
                flush=True,
            )
        if (step + 1) % cfg.eval.every == 0 or step + 1 == cfg.optim.steps:
            evaluate(step + 1)
            if bad_evals >= cfg.eval.patience:
                print(f"[early stop] no improvement for {bad_evals} evaluations", flush=True)
                break

    # Calibrate the best checkpoint on validation rows only, then save the deployable model.
    best = Predictor.from_pretrained(run_dir / "best")
    best.model.to(device=device)
    cal = _calibrate(best.model, best.tokenizer, cal_rows, attrs, cfg.optim.batch_size * 2)
    best.save_pretrained(run_dir / "final")

    config_dict = json.loads(cfg.model_dump_json())
    base = {
        "git_sha": provenance.git_sha(),
        "dirty": provenance.git_dirty(),
        "config_hash": provenance.config_hash(
            {
                k: v for k, v in config_dict.items() if k not in ("name", "out_dir")
            }  # labels, not science
        ),
        "split_manifest_hash": manifest_hash,
        "seed": cfg.seed,
        "created_at": provenance.now_iso(),
    }
    metrics = {
        **base,
        "train_loss_curve": curve,
        "train_loss_by_source": by_source,
        "val_curve": val_curve,
        "best_step": best_step,
        "best_val_nll": best_nll,
        "calibration": cal,
        "steps_run": len(curve),
        "wall_clock_s": time.time() - t_start,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    n_train = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    run_meta = {
        **base,
        "config": config_dict,
        "device": str(device),
        "partitions_read": ["train", "val"],
        "n_train_rows": len(train_df),
        "n_val_rows": len(val_df),
        "trainable_parameters": n_train,
        "package_versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": version("transformers"),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "schelling": version("schelling"),
        },
        "command": " ".join(sys.argv),
    }
    (run_dir / "run.json").write_text(json.dumps(run_meta, indent=2, sort_keys=True) + "\n")
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config_dict))
    if writer:
        writer.close()
    return TrainResult(run_dir, best_nll)
