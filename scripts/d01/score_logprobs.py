"""D01-lite: score prompts by the log-probability of each option key (standalone).

Runs anywhere with torch + transformers (+ peft for adapters, + bitsandbytes for 4-bit):
Google Colab, a rented GPU, or this Mac. It reads the `completion` field of each prompt record
(a Psych-101-style transcript ending in `<<`) and writes, per prompt, the log-probability of
each option key and the probabilities renormalised over the keys.

    python score_logprobs.py --prompts d01_prompts.jsonl --base unsloth/Qwen3-4B-Base \\
        --adapter socius/Qwentaur-4B-LoRA-r16 --variants adapter,base --out results/

Resumable: results are appended in small part files under <out>/<tag>/, and prompts already
present there are skipped, so re-running the same command after a disconnect continues.
With --variants adapter,base the base model is scored with the adapter switched off, so both
use the same weights in memory and the same precision.

No dependency on the schelling package, so the file can be copied into a notebook.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

SHARD = 200  # prompts per part file; small so a disconnect loses little


# ---- pure helpers (unit-tested without a model) --------------------------------------------


def fast_key_ids(prompt_ids: list[int], with_key_ids: list[list[int]]) -> list[int] | None:
    """If every key adds exactly one distinct token after the prompt, return those token ids.

    Then one forward pass gives all option probabilities. Otherwise return None and the caller
    scores each key's full continuation instead.
    """
    out = []
    n = len(prompt_ids)
    for ids in with_key_ids:
        if len(ids) != n + 1 or ids[:n] != prompt_ids:
            return None
        out.append(ids[n])
    return out if len(set(out)) == len(out) else None


def common_prefix(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        n += 1
    return n


def normalise(logps: list[float]) -> list[float]:
    m = max(logps)
    exps = [pow(2.718281828459045, x - m) for x in logps]
    s = sum(exps)
    return [e / s for e in exps]


def done_ids(tag_dir: Path) -> set[str]:
    seen: set[str] = set()
    for part in sorted(tag_dir.glob("part-*.jsonl")):
        for line in part.read_text().splitlines():
            try:
                seen.add(json.loads(line)["prompt_id"])
            except (json.JSONDecodeError, KeyError):
                continue  # a line cut by a disconnect; that prompt is simply redone
    return seen


def read_prompts(path: Path, arm: str) -> list[dict[str, Any]]:
    recs = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    return [r for r in recs if arm == "all" or r["arm"] == arm]


def batches(items: list[Any], size: int) -> Iterator[list[Any]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---- model scoring -------------------------------------------------------------------------


def _device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load(base: str, adapter: str | None, dtype: str, load_4bit: bool) -> tuple[Any, Any, str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = _device()
    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    kwargs: dict[str, Any] = {"torch_dtype": getattr(torch, dtype)}
    if load_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=getattr(torch, dtype),
        )
        kwargs["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(base, **kwargs)
    if not load_4bit:
        model = model.to(dev)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tok, dev


def _forward(model: Any, seqs: list[list[int]], pad_id: int, dev: str, left: bool) -> Any:
    """Log-softmax over the vocabulary at every position, for a padded batch."""
    import torch

    width = max(len(s) for s in seqs)
    ids, mask = [], []
    for s in seqs:
        pad = [pad_id] * (width - len(s))
        ones, zeros = [1] * len(s), [0] * (width - len(s))
        ids.append(pad + s if left else s + pad)
        mask.append(zeros + ones if left else ones + zeros)
    input_ids = torch.tensor(ids, device=dev)
    attn = torch.tensor(mask, device=dev)
    # Explicit positions so left padding does not shift them.
    pos = (attn.cumsum(-1) - 1).clamp(min=0)
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attn, position_ids=pos).logits
    return torch.log_softmax(logits.float(), dim=-1)


def score_batch(model: Any, tok: Any, recs: list[dict[str, Any]], dev: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any] | None] = [None] * len(recs)
    fast, slow = [], []
    for i, r in enumerate(recs):
        p_ids = tok(r["completion"]).input_ids
        k_ids = fast_key_ids(p_ids, [tok(r["completion"] + k).input_ids for k in r["keys"]])
        (fast if k_ids is not None else slow).append((i, p_ids, k_ids))
    if fast:
        lp = _forward(model, [p for _, p, _ in fast], tok.pad_token_id, dev, left=True)
        for row, (i, _, k_ids) in enumerate(fast):
            vals = [float(lp[row, -1, t]) for t in k_ids]
            out[i] = {
                "logps": vals,
                "fast": True,
                "key_mass": sum(pow(2.718281828459045, v) for v in vals),
            }
    for i, p_ids, _ in slow:
        r = recs[i]
        seqs = [tok(r["completion"] + k + ">>").input_ids for k in r["keys"]]
        lp = _forward(model, seqs, tok.pad_token_id, dev, left=False)
        vals = []
        for row, s in enumerate(seqs):
            start = common_prefix(p_ids, s)
            vals.append(float(sum(lp[row, t - 1, s[t]] for t in range(max(start, 1), len(s)))))
        out[i] = {"logps": vals, "fast": False, "key_mass": None}
    results = []
    for r, o in zip(recs, out, strict=True):
        assert o is not None
        results.append({"prompt_id": r["prompt_id"], "probs": normalise(o["logps"]), **o})
    return results


def run_variant(
    model: Any, tok: Any, dev: str, recs: list[dict[str, Any]], tag_dir: Path, batch_size: int
) -> None:
    tag_dir.mkdir(parents=True, exist_ok=True)
    seen = done_ids(tag_dir)
    todo = [r for r in recs if r["prompt_id"] not in seen]
    todo.sort(key=lambda r: len(r["completion"]))  # less padding per batch
    print(f"[{tag_dir.name}] {len(seen)} done, {len(todo)} to go", flush=True)
    start, n_done = time.time(), 0
    for shard in batches(todo, SHARD):
        lines = []
        for b in batches(shard, batch_size):
            lines += [json.dumps(x) for x in score_batch(model, tok, b, dev)]
        part = tag_dir / f"part-{int(time.time() * 1000)}.jsonl"
        tmp = part.with_suffix(".tmp")
        tmp.write_text("\n".join(lines) + "\n")
        os.replace(tmp, part)  # a part file is complete or absent
        n_done += len(shard)
        rate = n_done / (time.time() - start)
        eta = (len(todo) - n_done) / rate if rate else float("nan")
        print(
            f"[{tag_dir.name}] {n_done}/{len(todo)}  {rate:.1f}/s  eta {eta / 60:.0f} min",
            flush=True,
        )


def _tag(name: str) -> str:
    return name.split("/")[-1].lower()


def main(argv: Iterable[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", type=Path, required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--variants", default="adapter", help="comma list of adapter, base")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--arm", default="all", choices=["all", "population", "persona"])
    ap.add_argument("--limit", type=int, default=0, help="score only the first N prompts (timing)")
    args = ap.parse_args(list(argv) if argv is not None else None)

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    if "adapter" in variants and not args.adapter:
        ap.error("--variants adapter needs --adapter")
    recs = read_prompts(args.prompts, args.arm)
    if args.limit:
        recs = recs[: args.limit]
    model, tok, dev = load(args.base, args.adapter, args.dtype, args.load_4bit)
    precision = "4bit-nf4" if args.load_4bit else args.dtype
    for v in variants:
        name = _tag(args.adapter) if v == "adapter" else _tag(args.base)
        tag_dir = args.out / f"{name}-{precision}"
        tag_dir.mkdir(parents=True, exist_ok=True)
        import torch
        import transformers

        meta = {
            "variant": v,
            "base": args.base,
            "adapter": args.adapter if v == "adapter" else None,
            "precision": precision,
            "device": dev,
            "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else platform.processor(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "prompts_sha256": sha256_file(args.prompts),
            "arm": args.arm,
            "limit": args.limit,
        }
        (tag_dir / "run.json").write_text(json.dumps(meta, indent=2) + "\n")
        if v == "base" and args.adapter:
            with model.disable_adapter():
                run_variant(model, tok, dev, recs, tag_dir, args.batch_size)
        else:
            run_variant(model, tok, dev, recs, tag_dir, args.batch_size)
    print("ALLDONE", flush=True)


if __name__ == "__main__":
    main()
