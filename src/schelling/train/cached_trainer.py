"""Stage A of the two-tower model (decision 0004): frozen backbone, cached vectors.

Question, option and attribute texts are encoded by the frozen backbone once; the towers' small
networks are then trained on the cached vectors, so a pass over a million rows takes seconds.
The cached path and live inference share `SchellingTwoTower.combine`, and a test pins them equal.
"""

from __future__ import annotations

import copy
import json
import platform
import sys
import time
from collections.abc import Callable
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
from schelling.infer.baseline_adapter import make_item, person_attributes
from schelling.infer.batching import options_of
from schelling.infer.predictor import Predictor
from schelling.infer.text import attribute_strings, build_question_context
from schelling.model import SchellingTwoTower, SchellingTwoTowerConfig, calibration
from schelling.model.heads import score_conditionals
from schelling.train import losses
from schelling.train.config import TrainConfig
from schelling.train.sampler import StudySampler
from schelling.train.trainer import TrainResult, _device, _seed_all

TASKS = ("noul", "choice", "score")
TASK_ID = {t: i for i, t in enumerate(TASKS)}


@dataclass
class CachedData:
    U: torch.Tensor  # (n_questions, H) question-tower vectors
    E: torch.Tensor  # (n_option_texts, H)
    A: torch.Tensor  # (n_attribute_strings, H)
    q_idx: torch.Tensor  # (N,) row -> question vector
    person_idx: torch.Tensor  # (N,) row -> person
    task: torch.Tensor  # (N,) 0 noul, 1 choice, 2 score
    y: torch.Tensor  # (N,) observed answer index
    opt_idx: torch.Tensor  # (N, K) option or level vectors, -1 padding
    attr_idx: torch.Tensor  # (P, Amax) attribute vectors per person, -1 padding
    study: np.ndarray
    source: np.ndarray

    @property
    def n(self) -> int:
        return int(self.q_idx.shape[0])


def _vectors(
    model: SchellingTwoTower, tokenizer: Any, texts: list[str], max_len: int, device: torch.device
) -> torch.Tensor:
    if not texts:
        return torch.zeros(0, model.config.backbone_hidden_size, device=device)
    was_training = model.training
    model.eval()
    out = []
    with torch.inference_mode():
        for lo in range(0, len(texts), 128):
            enc = tokenizer(
                texts[lo : lo + 128],
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            )
            out.append(
                model._encode(enc["input_ids"].to(device), enc["attention_mask"].to(device)).clone()
            )
    model.train(was_training)
    return torch.cat(out)


def build_cache(
    model: SchellingTwoTower,
    tokenizer: Any,
    frame: pd.DataFrame,
    attrs: dict[str, dict[str, Any]],
    device: torch.device,
) -> CachedData:
    cfg = model.config
    q_text: dict[str, int] = {}
    ctx_memo: dict[tuple[str, str, str], str] = {}
    o_text: dict[str, int] = {}
    a_text: dict[str, int] = {}
    person_ix: dict[str, int] = {}
    person_attr: list[list[int]] = []
    q_idx, p_idx, task, y, opts = [], [], [], [], []
    for pid, qtype, text, o, lv, state, resp in zip(
        frame["person_id"],
        frame["qtype"],
        frame["question"],
        frame["options"],
        frame["levels"],
        frame["state"],
        frame["response"],
        strict=True,
    ):
        req, q = make_item(
            qtype,
            text,
            None if o is None else list(o),
            None if lv is None else list(lv),
            state or "",
            attrs.get(str(pid)),
        )
        mkey = (qtype, text, state or "")
        if mkey not in ctx_memo:  # the same item recurs for every person
            ctx_memo[mkey] = build_question_context(
                req.state, q, tokenizer, cfg.max_question_tokens
            )
        q_idx.append(q_text.setdefault(ctx_memo[mkey], len(q_text)))
        if str(pid) not in person_ix:
            person_ix[str(pid)] = len(person_ix)
            person_attr.append(
                [
                    a_text.setdefault(s, len(a_text))
                    for s in attribute_strings(req.person, cfg.max_attributes)
                ]
            )
        p_idx.append(person_ix[str(pid)])
        task.append(TASK_ID[qtype])
        y.append(int(resp))
        opts.append([o_text.setdefault(t, len(o_text)) for t in options_of(q)])
    k = max((len(o) for o in opts), default=1) or 1
    amax = max((len(a) for a in person_attr), default=1) or 1
    opt_idx = torch.full((len(opts), k), -1, dtype=torch.long)
    for i, o in enumerate(opts):
        opt_idx[i, : len(o)] = torch.tensor(o, dtype=torch.long)
    attr_idx = torch.full((len(person_attr), amax), -1, dtype=torch.long)
    for i, a in enumerate(person_attr):
        if a:
            attr_idx[i, : len(a)] = torch.tensor(a, dtype=torch.long)
    dev = device
    return CachedData(
        U=_vectors(model, tokenizer, list(q_text), cfg.max_question_tokens, dev),
        E=_vectors(model, tokenizer, list(o_text), cfg.max_option_tokens, dev),
        A=_vectors(model, tokenizer, list(a_text), cfg.max_attr_tokens, dev),
        q_idx=torch.tensor(q_idx, device=dev),
        person_idx=torch.tensor(p_idx, device=dev),
        task=torch.tensor(task, device=dev),
        y=torch.tensor(y, device=dev),
        opt_idx=opt_idx.to(dev),
        attr_idx=attr_idx.to(dev),
        study=frame["study_id"].to_numpy(),
        source=frame["source"].to_numpy(),
    )


def _gather(data: CachedData, rows: torch.Tensor) -> dict[str, torch.Tensor | None]:
    u = data.U[data.q_idx[rows]]
    ai = data.attr_idx[data.person_idx[rows]]
    attr_mask = ai >= 0
    attr_vecs = (
        data.A[ai.clamp(min=0)]
        if data.A.shape[0]
        else torch.zeros(len(rows), ai.shape[1], u.shape[-1], device=u.device)
    )
    oi = data.opt_idx[rows]
    k = int((oi >= 0).sum(1).max()) if len(rows) else 0
    oi = oi[:, : max(k, 1)]
    option_mask = oi >= 0
    e = data.E[oi.clamp(min=0)] if data.E.shape[0] else None
    return {
        "u": u,
        "attr_vecs": attr_vecs,
        "attr_mask": attr_mask,
        "e": e,
        "option_mask": option_mask,
    }


def _task_rows(data: CachedData, rows: torch.Tensor) -> dict[str, torch.Tensor]:
    t = data.task[rows]
    return {name: rows[t == i] for name, i in TASK_ID.items() if bool((t == i).any())}


def _nll_vec(
    model: SchellingTwoTower,
    task: str,
    logits: torch.Tensor,
    mask: torch.Tensor | None,
    y: torch.Tensor,
) -> torch.Tensor:
    if task == "score":
        assert mask is not None
        cond = score_conditionals(logits, mask, model.config.temperature_score)
        return losses.corn_loss(cond, y, reduction="none")
    return losses.hard_cross_entropy(model.probs_from(task, logits, mask), y, reduction="none")


def _combine(model: SchellingTwoTower, data: CachedData, task: str, rows: torch.Tensor):  # type: ignore[no-untyped-def]
    g = _gather(data, rows)
    if task == "noul":
        g["e"], g["option_mask"] = None, None
    return model.combine(task, **g)  # type: ignore[arg-type]


@torch.inference_mode()
def cached_probs(
    model: SchellingTwoTower, data: CachedData, rows: np.ndarray, batch_size: int = 2048
) -> list[np.ndarray]:
    was_training = model.training
    model.eval()
    out: list[np.ndarray | None] = [None] * len(rows)
    idx = torch.tensor(rows, device=data.q_idx.device)
    pos = {int(r): i for i, r in enumerate(rows)}  # row id -> output position
    for task, sel in _task_rows(data, idx).items():
        for lo in range(0, len(sel), batch_size):
            chunk = sel[lo : lo + batch_size]
            o = _combine(model, data, task, chunk)
            probs = model.probs_from(task, o.logits, o.mask).float().cpu().numpy()
            counts = (data.opt_idx[chunk] >= 0).sum(1).cpu().numpy()
            for j, r in enumerate(chunk.cpu().numpy()):
                m = 2 if task == "noul" else int(counts[j])
                out[pos[int(r)]] = probs[j, :m].astype(np.float64)
    model.train(was_training)
    return out  # type: ignore[return-value]


@torch.inference_mode()
def cached_val_nll(
    model: SchellingTwoTower, data: CachedData, batch_size: int = 4096
) -> tuple[float, float]:
    """Mean NLL over all rows of the full output and of the item-prior branch alone."""
    was_training = model.training
    model.eval()
    full = prior = 0.0
    all_rows = torch.arange(data.n, device=data.q_idx.device)
    for task, sel in _task_rows(data, all_rows).items():
        for lo in range(0, len(sel), batch_size):
            chunk = sel[lo : lo + batch_size]
            o = _combine(model, data, task, chunk)
            y = data.y[chunk]
            full += float(_nll_vec(model, task, o.logits, o.mask, y).sum())
            prior += float(_nll_vec(model, task, o.prior, o.mask, y).sum())
    model.train(was_training)
    return full / data.n, prior / data.n


def fit_combiner(
    model: SchellingTwoTower,
    data: CachedData,
    steps: int,
    batch_size: int,
    lr: float,
    prior_weight: float,
    seed: int,
    study_temperature: float,
    weight_decay: float = 0.01,
    warmup_frac: float = 0.05,
    eval_every: int = 0,
    on_eval: Callable[[int], bool] | None = None,
) -> list[float]:
    """Train the towers' small networks on cached vectors. Returns the per-step training loss."""
    torch.manual_seed(seed)
    for p in model.backbone.parameters():
        p.requires_grad = False
    params = [p for p in model.heads.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    warmup = max(1, int(warmup_frac * steps))

    def lam(s: int) -> float:
        if s < warmup:
            return (s + 1) / warmup
        return 0.5 * (1 + float(np.cos(np.pi * min(1.0, (s - warmup) / max(1, steps - warmup)))))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lam)
    sampler = StudySampler(pd.Series(data.study), study_temperature, seed)
    device = data.q_idx.device
    curve: list[float] = []
    model.train()
    for step in range(steps):
        rows = torch.tensor(sampler.sample(batch_size), device=device)
        total = torch.zeros((), device=device)
        full_sum = torch.zeros((), device=device)
        for task, sel in _task_rows(data, rows).items():
            o = _combine(model, data, task, sel)
            y = data.y[sel]
            f = _nll_vec(model, task, o.logits, o.mask, y)
            full_sum = full_sum + f.sum()
            total = total + f.sum()
            if prior_weight > 0:
                total = total + prior_weight * _nll_vec(model, task, o.prior, o.mask, y).sum()
        loss = total / batch_size
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        curve.append(float((full_sum / batch_size).detach()))
        if eval_every and on_eval and (step + 1) % eval_every == 0:
            model.eval()
            stop = on_eval(step + 1)
            model.train()
            if stop:
                break
    return curve


def _build_model(cfg: TrainConfig) -> tuple[SchellingTwoTower, Any]:
    m = cfg.model
    kw: dict[str, Any] = dict(
        dim=m.dim,
        person_heads=m.person_heads,
        dropout=m.dropout,
        use_person=m.use_person,
        max_question_tokens=m.max_question_tokens,
        max_option_tokens=m.max_option_tokens,
        max_attr_tokens=m.max_attr_tokens,
        max_attributes=m.max_attributes,
    )
    if m.tiny_vocab:
        vocab = {tok: i for i, tok in enumerate(Path(m.tiny_vocab).read_text().split())}
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
        return SchellingTwoTower(
            SchellingTwoTowerConfig(backbone_config=backbone, **kw)
        ), BertTokenizer(vocab=vocab, do_lower_case=True)
    return SchellingTwoTower.from_backbone(m.backbone, **kw), AutoTokenizer.from_pretrained(
        m.backbone
    )


def _subset(data: CachedData, n: int, seed: int) -> CachedData:
    """A random subset of rows that shares the vector tables."""
    if data.n <= n:
        return data
    keep = torch.tensor(
        np.random.default_rng(seed).choice(data.n, n, replace=False), device=data.q_idx.device
    )
    return CachedData(
        U=data.U,
        E=data.E,
        A=data.A,
        q_idx=data.q_idx[keep],
        person_idx=data.person_idx[keep],
        task=data.task[keep],
        y=data.y[keep],
        opt_idx=data.opt_idx[keep],
        attr_idx=data.attr_idx,
        study=data.study[keep.cpu().numpy()],
        source=data.source[keep.cpu().numpy()],
    )


@torch.inference_mode()
def _calibrate(model: SchellingTwoTower, data: CachedData) -> dict[str, Any]:
    model.eval()
    res: dict[str, Any] = {"n": {}, "temperature": {}}
    cache: dict[str, tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]] = {}
    rows = torch.arange(data.n, device=data.q_idx.device)
    for task, sel in _task_rows(data, rows).items():
        lg, mk, ys = [], [], []
        for lo in range(0, len(sel), 4096):
            chunk = sel[lo : lo + 4096]
            o = _combine(model, data, task, chunk)
            lg.append(o.logits.float().cpu().reshape(len(chunk), -1))
            if o.mask is not None:
                mk.append(o.mask.cpu())
            ys.append(data.y[chunk].cpu())
        cache[task] = (
            calibration.pad_stack(lg, 0.0),
            calibration.pad_stack(mk, False) if mk else None,
            torch.cat(ys),
        )
        res["n"][task] = len(sel)
    before = after = 0.0
    total = sum(len(v[2]) for v in cache.values())
    for task, (lg_all, mk_all, y_all) in cache.items():
        t0 = getattr(model.config, f"temperature_{task}")
        t_fit = (
            calibration.fit_temperature(task, lg_all, mk_all, y_all) if len(y_all) >= 50 else 1.0
        )
        before += calibration.nll_at_temperature(task, lg_all, mk_all, y_all, t0) * len(y_all)
        after += calibration.nll_at_temperature(task, lg_all, mk_all, y_all, t_fit) * len(y_all)
        setattr(model.config, f"temperature_{task}", t_fit)
        res["temperature"][task] = t_fit
    res["val_nll_before"], res["val_nll_after"] = before / total, after / total
    return res


def train_cached(cfg: TrainConfig) -> TrainResult:
    t0 = time.time()
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

    model, tokenizer = _build_model(cfg)
    model = model.to(device)  # type: ignore[arg-type]
    train_data = build_cache(model, tokenizer, train_df, attrs, device)
    val_data = build_cache(model, tokenizer, val_df, attrs, device)
    model.heads.set_input_stats(train_data.U, train_data.E, train_data.A)
    monitor = _subset(val_data, cfg.eval.max_rows, cfg.seed)
    print(
        f"[cache] train rows {train_data.n}, question vectors {train_data.U.shape[0]}, "
        f"option vectors {train_data.E.shape[0]}, attribute vectors {train_data.A.shape[0]}",
        flush=True,
    )

    val_curve: list[dict[str, Any]] = []
    best: dict[str, Any] = {"nll": float("inf"), "step": -1, "bad": 0, "state": None}

    def on_eval(step: int) -> bool:
        full, prior = cached_val_nll(model, monitor)
        val_curve.append({"step": step, "val_nll": full, "val_nll_prior_only": prior})
        print(f"[eval] step {step} val_nll {full:.4f} prior-only {prior:.4f}", flush=True)
        if full < best["nll"]:
            best.update(nll=full, step=step, bad=0, state=copy.deepcopy(model.heads.state_dict()))
        else:
            best["bad"] += 1
        return best["bad"] >= cfg.eval.patience

    curve = fit_combiner(
        model,
        train_data,
        cfg.optim.steps,
        cfg.optim.batch_size,
        cfg.optim.lr_head,
        cfg.model.prior_weight,
        cfg.seed,
        cfg.data.study_temperature,
        cfg.optim.weight_decay,
        cfg.optim.warmup_frac,
        cfg.eval.every,
        on_eval,
    )
    if best["state"] is not None:
        model.heads.load_state_dict(best["state"])
    cal = _calibrate(model, _subset(val_data, cfg.eval.calibration_rows, cfg.seed + 7))
    full_nll, prior_nll = cached_val_nll(model, val_data)
    Predictor(model, tokenizer).save_pretrained(run_dir / "final")

    config_dict = json.loads(cfg.model_dump_json())
    base = {
        "git_sha": provenance.git_sha(),
        "dirty": provenance.git_dirty(),
        "config_hash": provenance.config_hash(
            {k: v for k, v in config_dict.items() if k not in ("name", "out_dir")}
        ),
        "split_manifest_hash": manifest_hash,
        "seed": cfg.seed,
        "created_at": provenance.now_iso(),
    }
    metrics = {
        **base,
        "train_loss_curve": curve,
        "val_curve": val_curve,
        "best_step": best["step"],
        "best_val_nll": best["nll"],
        "calibration": cal,
        "final_val_nll": full_nll,
        "final_val_nll_prior_only": prior_nll,
        "steps_run": len(curve),
        "wall_clock_s": time.time() - t0,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    run_meta = {
        **base,
        "config": config_dict,
        "device": str(device),
        "partitions_read": ["train", "val"],
        "n_train_rows": int(train_data.n),
        "n_val_rows": int(val_data.n),
        "trainable_parameters": int(
            sum(p.numel() for p in model.heads.parameters() if p.requires_grad)
        ),
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
    return TrainResult(run_dir, best["nll"])
