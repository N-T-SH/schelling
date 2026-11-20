"""Fork D, D01b: does an LLM's edge on unseen questions survive rewording? (pre-registered in
research/forks/distill/campaigns/D01-external-benchmark/D01b-probe.md)

    # paid: ask the teacher about every paraphrase (cached; stops at the cap)
    uv run python scripts/fd_paraphrase_probe.py fetch --spec SPEC.json --cap-usd 1
    # offline: score teacher and M4b control on reworded questions
    uv run python scripts/fd_paraphrase_probe.py score --spec SPEC.json --out reports/FD-d01b/probe.json

Reads the `val` partition only. Options and levels stay as they are; only the question text changes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import pandas as pd
import typer

from schelling import compare
from schelling.augment.paraphrases import question_key, spend_usd
from schelling.baselines.llm_prompt import LLMPromptBaseline, LLMSpec, ParseError, parse_probs
from schelling.data.loader import load_partition
from schelling.eval import provenance
from schelling.eval.metrics import EPS
from schelling.llm.fireworks import OPENROUTER_URL, FireworksClient, load_dotenv

app = typer.Typer(add_completion=False)

PARAPHRASES = Path("data/paraphrases/twin2k-gpt-oss-120b-v1.jsonl")


def reworded_frames(frame: pd.DataFrame, store: Path) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    """The rows of items that have paraphrases, and one copy of them per paraphrase slot (an item
    with fewer paraphrases drops out of the later slots)."""
    by_key: dict[str, list[str]] = {}
    for line in store.read_text().splitlines():
        row = json.loads(line)
        by_key[row["key"]] = list(row["paraphrases"])
    texts = {
        item: by_key.get(question_key(q), [])
        for item, q in frame.drop_duplicates("item_id")[["item_id", "question"]].itertuples(
            index=False
        )
    }
    keep = frame[frame["item_id"].map(lambda i: len(texts[i]) > 0)].reset_index(drop=True)
    slots = []
    for k in range(max(len(v) for v in texts.values())):
        part = keep[keep["item_id"].map(lambda i, k=k: len(texts[i]) > k)].copy()
        part["question"] = part["item_id"].map(lambda i, k=k: texts[i][k])
        slots.append(part.reset_index(drop=True))
    return keep, slots


def row_nll(probs: list[np.ndarray], y: np.ndarray) -> np.ndarray:
    p = np.array([q[i] for q, i in zip(probs, y, strict=True)])
    out: np.ndarray = -np.log(np.clip(p, EPS, 1.0))
    return out


def item_means(frame: pd.DataFrame, nll: np.ndarray) -> pd.Series:
    return pd.Series(nll, index=frame.index).groupby(frame["item_id"].to_numpy()).mean()


def saved_item_nll(frame: pd.DataFrame, run_dir: Path) -> pd.Series:
    pred = compare.load_predictions(run_dir)
    rows = frame[["record_id", "item_id", "response"]].merge(pred, on="record_id", validate="1:1")
    nll = row_nll(list(rows["probs"]), rows["response"].to_numpy(dtype=int))
    return pd.Series(nll).groupby(rows["item_id"].to_numpy()).mean()


def _boot(a: pd.Series, b: pd.Series, n_boot: int, seed: int) -> dict[str, Any]:
    units = pd.DataFrame({"a": a, "b": b}).dropna()
    return compare.paired_bootstrap(units, n_boot, seed)


def _val(processed: Path, splits_dir: Path) -> pd.DataFrame:
    frame = load_partition(processed, "heldout_item", "val", splits_dir=splits_dir)
    return frame[frame["source"] == "twin2k"].reset_index(drop=True)


@app.command()
def fetch(
    spec: Annotated[Path, typer.Option()],
    cap_usd: Annotated[float, typer.Option()],
    in_per_m: Annotated[float, typer.Option()] = 0.30,
    out_per_m: Annotated[float, typer.Option()] = 1.20,
    processed: Annotated[Path, typer.Option()] = Path("data/processed"),
    splits_dir: Annotated[Path, typer.Option()] = Path("splits"),
    store: Annotated[Path, typer.Option()] = PARAPHRASES,
    env: Annotated[Path, typer.Option()] = Path(".env"),
) -> None:
    """Paid: one teacher call per paraphrase, through the cache, stopping at the cap."""
    cfg = LLMSpec.model_validate_json(spec.read_text())
    load_dotenv(env)
    client = FireworksClient(
        cache_dir=Path(cfg.cache_dir),
        base_url=OPENROUTER_URL,
        key_env="OPENROUTER_API_KEY",
        retries=5,
    )
    base = LLMPromptBaseline(cfg, client=client)
    base.fit(pd.DataFrame(), pd.read_parquet(processed / "persons_twin2k.parquet"))
    _, slots = reworded_frames(_val(processed, splits_dir), store)
    reqs: dict[str, tuple[str, list[str], Any]] = {}
    for part in slots:
        for r in base.requests(part.drop_duplicates("item_id")):
            reqs[json.dumps(r, sort_keys=True, default=str)] = r
    spent, fresh, empty, failed = 0.0, 0, 0, 0
    for question, labels, attrs in reqs.values():
        if spent >= cap_usd:
            break
        try:
            resp = base.fetch(question, labels, attrs)
        except RuntimeError as err:
            typer.echo(f"request failed: {err}", err=True)
            failed += 1
            continue
        if not resp["cached"]:
            fresh += 1
            spent += spend_usd(resp["usage"], in_per_m, out_per_m)
        try:
            parse_probs(resp["text"], len(labels), cfg.eps)
        except ParseError:
            empty += 1
    summary = {"requests": len(reqs), "network_calls": fresh, "unparsed": empty, "failed": failed}
    typer.echo(json.dumps({**summary, "spent_usd": round(spent, 4), "cap_usd": cap_usd}))


@app.command()
def score(
    spec: Annotated[Path, typer.Option()],
    out: Annotated[Path, typer.Option()],
    control: Annotated[Path, typer.Option(help="saved M4b model directory")] = Path(
        "runs/tt-item/final"
    ),
    control_run: Annotated[Path, typer.Option()] = Path("reports/M4b-tt-item-val"),
    teacher_run: Annotated[Path, typer.Option()] = Path(
        "reports/FD-gate-deepseek-high16k-item-val"
    ),
    b1_run: Annotated[Path, typer.Option()] = Path("reports/FD-gate-b1-item-val"),
    processed: Annotated[Path, typer.Option()] = Path("data/processed"),
    splits_dir: Annotated[Path, typer.Option()] = Path("splits"),
    store: Annotated[Path, typer.Option()] = PARAPHRASES,
    n_boot: Annotated[int, typer.Option()] = 10_000,
    seed: Annotated[int, typer.Option()] = 0,
) -> None:
    """Offline: teacher (from the cache) and M4b control on every paraphrase slot."""
    from schelling.baselines import make_baseline

    persons = pd.read_parquet(processed / "persons_twin2k.parquet")
    keep, slots = reworded_frames(_val(processed, splits_dir), store)
    teacher = LLMPromptBaseline.from_spec(spec)
    teacher.fit(pd.DataFrame(), persons)
    m4b = make_baseline(str(control)).fit(pd.DataFrame(), persons)
    para: dict[str, list[pd.Series]] = {"teacher": [], "m4b": []}
    unanswered = 0
    for part in slots:
        y = part["response"].to_numpy(dtype=int)
        probs = teacher.predict_proba(part)
        unanswered += teacher.stats["parse_failures"]
        para["teacher"].append(item_means(part, row_nll(probs, y)))
        para["m4b"].append(item_means(part, row_nll(m4b.predict_proba(part), y)))
    t_para = pd.concat(para["teacher"], axis=1).mean(axis=1)
    m_para = pd.concat(para["m4b"], axis=1).mean(axis=1)
    t_orig = saved_item_nll(keep, teacher_run)
    m_orig = saved_item_nll(keep, control_run)
    b1 = saved_item_nll(keep, b1_run)
    items = sorted(t_para.index)
    edge_orig = float((b1[items] - t_orig[items]).mean())
    edge_para = float((b1[items] - t_para[items]).mean())
    result = {
        "split": "heldout_item",
        "partition": "val",
        "source": "twin2k",
        "spec": str(spec),
        "control": str(control),
        "paraphrase_store": str(store),
        "n_items": len(items),
        "n_paraphrases": int(sum(len(s.drop_duplicates("item_id")) for s in slots)),
        "teacher_unanswered_paraphrases": int(unanswered),
        "edge_orig": edge_orig,
        "edge_para": edge_para,
        "retention": edge_para / edge_orig,
        "teacher_para_vs_b1": _boot(t_para[items], b1[items], n_boot, seed),
        "teacher_para_vs_orig": _boot(t_para[items], t_orig[items], n_boot, seed),
        "m4b_para_vs_orig": _boot(m_para[items], m_orig[items], n_boot, seed),
        "per_item": {
            i: {
                "b1": float(b1[i]),
                "teacher_orig": float(t_orig[i]),
                "teacher_para": float(t_para[i]),
                "m4b_orig": float(m_orig[i]),
                "m4b_para": float(m_para[i]),
            }
            for i in items
        },
        "git_sha": provenance.git_sha(),
        "dirty": provenance.git_dirty(),
        "seed": seed,
        "created_at": provenance.now_iso(),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    keys = ("n_items", "n_paraphrases", "edge_orig", "edge_para", "retention")
    typer.echo(json.dumps({k: result[k] for k in keys}))


if __name__ == "__main__":
    app()
