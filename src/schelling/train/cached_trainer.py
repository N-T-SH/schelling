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
from schelling.infer.batching import history_question_context, options_of
from schelling.infer.history import history_for_frame
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
    # Answer history (C01). Training samples a person's other answers per step from the evidence
    # tables; evaluation uses fixed histories identical to `infer.history.history_for_frame`.
    history_k: int = 0
    ev_q: torch.Tensor | None = None  # (P, L) question vector of each earlier answer, -1 padding
    ev_o: torch.Tensor | None = None  # (P, L) vector of the option chosen
    ev_key: torch.Tensor | None = None  # (P, L) id of the question text, for masking the target
    row_key: torch.Tensor | None = None  # (N,) id of each row's own question text
    hist_q: torch.Tensor | None = None  # (N, Kfix) fixed history, -1 padding
    hist_o: torch.Tensor | None = None
    q_ctx: list[str] | None = None  # text behind each question vector (tests, debugging)
    o_txt: list[str] | None = None

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
    history_k: int = 0,
    fixed_history: bool = False,
) -> CachedData:
    """`history_k` > 0 adds answer history: evidence tables for sampling at training time, or,
    with `fixed_history`, the exact history `history_for_frame` gives each row (evaluation)."""
    cfg = model.config
    q_text: dict[str, int] = {}
    ctx_memo: dict[tuple[str, str, str], str] = {}
    o_text: dict[str, int] = {}
    a_text: dict[str, int] = {}
    person_ix: dict[str, int] = {}
    person_attr: list[list[int]] = []
    q_idx, p_idx, task, y, opts = [], [], [], [], []
    qkey: dict[str, int] = {}
    row_key: list[int] = []
    person_ev: dict[int, list[tuple[int, int, int]]] = {}
    hctx: dict[str, int] = {}  # question text -> vector index of its history context
    sampled = history_k > 0 and not fixed_history

    def hist_q(text: str) -> int:
        if text not in hctx:
            ctx = history_question_context(text, tokenizer, cfg.max_question_tokens)
            hctx[text] = q_text.setdefault(ctx, len(q_text))
        return hctx[text]

    for pid, qtype, text, o, lv, state, resp, wave in zip(
        frame["person_id"],
        frame["qtype"],
        frame["question"],
        frame["options"],
        frame["levels"],
        frame["state"],
        frame["response"],
        frame["wave"],
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
        if sampled:
            row_key.append(qkey.setdefault(text, len(qkey)))
            if wave != 4:  # retest repeats are never evidence
                chosen = ("no", "yes")[int(resp)] if qtype == "noul" else options_of(q)[int(resp)]
                person_ev.setdefault(person_ix[str(pid)], []).append(
                    (hist_q(text), o_text.setdefault(chosen, len(o_text)), qkey[text])
                )
    fixed: list[list[tuple[int, int]]] = []
    if history_k > 0 and fixed_history:
        for hist in history_for_frame(frame, history_k):
            fixed.append(
                [
                    (hist_q(h.question), o_text.setdefault(h.options[h.answer], len(o_text)))
                    for h in hist
                ]
            )
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
    extra: dict[str, Any] = {}
    if sampled:
        width = max((len(v) for v in person_ev.values()), default=1)
        ev = torch.full((3, len(person_ix), width), -1, dtype=torch.long)
        for pi, items in person_ev.items():
            ev[:, pi, : len(items)] = torch.tensor(items, dtype=torch.long).T
        extra = {
            "history_k": history_k,
            "ev_q": ev[0].to(dev),
            "ev_o": ev[1].to(dev),
            "ev_key": ev[2].to(dev),
            "row_key": torch.tensor(row_key, device=dev),
        }
    elif fixed:
        width = max((len(h) for h in fixed), default=1) or 1
        fq = torch.full((len(fixed), width), -1, dtype=torch.long)
        fo = torch.full((len(fixed), width), -1, dtype=torch.long)
        for i, h in enumerate(fixed):
            if h:
                fq[i, : len(h)] = torch.tensor([a for a, _ in h])
                fo[i, : len(h)] = torch.tensor([b for _, b in h])
        extra = {"history_k": history_k, "hist_q": fq.to(dev), "hist_o": fo.to(dev)}
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
        q_ctx=list(q_text),
        o_txt=list(o_text),
        **extra,
    )


def sample_history(
    data: CachedData, rows: torch.Tensor, gen: torch.Generator, ks: tuple[int, ...]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A random history per row: the person's other first-pass answers, never the row's own
    question (so never its answer), of a size drawn from `ks`. Returns (question vector index,
    chosen-option vector index, validity mask), each (B, L)."""
    assert data.ev_q is not None and data.ev_o is not None and data.ev_key is not None
    assert data.row_key is not None
    dev = data.q_idx.device
    pi = data.person_idx[rows]
    evq, evo, evk = data.ev_q[pi], data.ev_o[pi], data.ev_key[pi]
    valid = (evq >= 0) & (evk != data.row_key[rows].unsqueeze(1))
    width = min(data.history_k, evq.shape[1])
    noise = torch.rand(evq.shape, generator=gen).to(dev).masked_fill(~valid, 2.0)
    top = noise.topk(width, dim=1, largest=False).indices
    size = torch.tensor([min(k, width) for k in ks])[
        torch.randint(len(ks), (len(rows),), generator=gen)
    ]
    keep = (
        torch.arange(width, device=dev).unsqueeze(0) < size.to(dev).unsqueeze(1)
    ) & valid.gather(1, top)
    hq = evq.gather(1, top).masked_fill(~keep, -1)
    ho = evo.gather(1, top).masked_fill(~keep, -1)
    return hq, ho, keep


def _hist_tensors(
    data: CachedData, hq: torch.Tensor, ho: torch.Tensor, keep: torch.Tensor
) -> dict[str, Any]:
    # Tables plus indices: the heads project the shared vectors once instead of every history row.
    return {"hist_tables": (data.U, data.E), "hist_idx": (hq, ho), "hist_mask": keep}


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


def _combine(  # type: ignore[no-untyped-def]
    model: SchellingTwoTower,
    data: CachedData,
    task: str,
    rows: torch.Tensor,
    hist: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
):
    g = _gather(data, rows)
    if task == "noul":
        g["e"], g["option_mask"] = None, None
    if model.config.history_items > 0:
        if hist is None and data.hist_q is not None and data.hist_o is not None:
            fq, fo = data.hist_q[rows], data.hist_o[rows]  # fixed history (evaluation)
            width = max(1, int((fq >= 0).sum(1).max()))
            hist = (fq[:, :width], fo[:, :width], fq[:, :width] >= 0)
        if hist is not None:
            g.update(_hist_tensors(data, *hist))
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
    history_sizes: tuple[int, ...] = (0, 5, 20, 50),
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
    use_history = model.config.history_items > 0 and data.ev_q is not None
    hist_gen = torch.Generator().manual_seed(seed + 1)
    model.train()
    for step in range(steps):
        rows = torch.tensor(sampler.sample(batch_size), device=device)
        total = torch.zeros((), device=device)
        full_sum = torch.zeros((), device=device)
        for task, sel in _task_rows(data, rows).items():
            hist = sample_history(data, sel, hist_gen, history_sizes) if use_history else None
            o = _combine(model, data, task, sel, hist)
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
        history_items=m.history_items,
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
        history_k=data.history_k,
        ev_q=data.ev_q,
        ev_o=data.ev_o,
        ev_key=data.ev_key,
        row_key=None if data.row_key is None else data.row_key[keep],
        hist_q=None if data.hist_q is None else data.hist_q[keep],
        hist_o=None if data.hist_o is None else data.hist_o[keep],
        q_ctx=data.q_ctx,
        o_txt=data.o_txt,
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
    hk = cfg.model.history_items
    train_data = build_cache(model, tokenizer, train_df, attrs, device, history_k=hk)
    val_data = build_cache(
        model, tokenizer, val_df, attrs, device, history_k=hk, fixed_history=hk > 0
    )
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
