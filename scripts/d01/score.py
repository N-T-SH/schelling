"""D01-lite: score every model's results against the human answers.

    uv run python scripts/d01/score.py --prompts data/d01/d01_prompts.jsonl \\
        --results results/ --processed <main checkout>/data/processed --out reports/FD-d01lite/metrics.json

`--results` holds one folder per model (from score_logprobs.py or run_api.py), each with
part-*.jsonl lines {"prompt_id", "probs"}. Human answers: the `heldout_person` val rows.
Human effect sizes for the bias test: `heldout_person` train and val rows (never test).
"""

from __future__ import annotations

import hashlib
import json
from itertools import combinations
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import pandas as pd
import typer

from schelling.bench.score import compare_items, score_model
from schelling.data.loader import load_partition
from schelling.eval import provenance
from schelling.eval.effects import load_pairs

app = typer.Typer(add_completion=False)


def load_results(folder: Path) -> dict[str, Any]:
    probs: dict[str, Any] = {}
    for part in sorted(folder.glob("part-*.jsonl")):
        for line in part.read_text().splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            probs[rec["prompt_id"]] = None if rec["probs"] is None else np.asarray(rec["probs"])
    return probs


def retest_agreement(human: pd.DataFrame) -> dict[str, float]:
    """How often a person repeats their own wave-1 answer in wave 4: a ceiling on
    individual-level prediction for the retested items."""
    both = human.pivot_table(
        index=["person_id", "item_id"], columns="wave", values="response", aggfunc="first"
    )
    if 1 not in both or 4 not in both:
        return {"n_pairs": 0.0}
    both = both[[1, 4]].dropna()
    return {
        "n_pairs": float(len(both)),
        "n_items": float(both.index.get_level_values("item_id").nunique()),
        "exact_agreement": float((both[1] == both[4]).mean()),
    }


@app.command()
def main(
    prompts: Annotated[Path, typer.Option()],
    results: Annotated[Path, typer.Option()],
    out: Annotated[Path, typer.Option()],
    processed: Annotated[Path, typer.Option()] = Path("data/processed"),
    splits_dir: Annotated[Path, typer.Option()] = Path("splits"),
    n_splits: Annotated[int, typer.Option()] = 200,
    n_boot: Annotated[int, typer.Option()] = 2000,
    seed: Annotated[int, typer.Option()] = 0,
    waves: Annotated[
        str, typer.Option(help="'first': each person's wave-1 answer only; 'all': pool retests")
    ] = "first",
) -> None:
    recs = pd.DataFrame([json.loads(x) for x in prompts.read_text().splitlines() if x.strip()])
    items = set(recs["item_id"])
    val = load_partition(processed, "heldout_person", "val", splits_dir=splits_dir)
    human = val[(val["source"] == "twin2k") & val["item_id"].isin(items)].copy()
    ref = pd.concat(
        [
            load_partition(processed, "heldout_person", part, splits_dir=splits_dir)
            for part in ("train", "val")
        ]
    )
    ref = ref[(ref["source"] == "twin2k") & ref["item_id"].isin(items)]
    for frame in (human, ref):
        frame["person_id"] = frame["person_id"].astype(str)
    retest = retest_agreement(human)
    if waves == "first":
        # 54 decision items were asked again in wave 4; pooling would count those people twice.
        human = human[human["wave"] == 1]
        ref = ref[ref["wave"] == 1]
    elif waves != "all":
        raise typer.BadParameter("--waves must be first or all")
    pairs = load_pairs()

    models: dict[str, Any] = {}
    for folder in sorted(p for p in results.iterdir() if p.is_dir()):
        probs = load_results(folder)
        if not probs:
            continue
        typer.echo(f"scoring {folder.name} ({len(probs)} results)")
        models[folder.name] = score_model(recs, probs, human, ref, pairs, n_splits, n_boot, seed)
        run = folder / "run.json"
        models[folder.name]["run"] = json.loads(run.read_text()) if run.exists() else None

    comparisons = []
    for a, b in combinations(sorted(models), 2):
        for key in ("persona_acc", "population_acc"):
            cmp = compare_items(models[a]["per_item"], models[b]["per_item"], key, n_boot, seed)
            comparisons.append({"a": a, "b": b, **cmp})

    result = {
        "prompts": str(prompts),
        "prompts_sha256": hashlib.sha256(prompts.read_bytes()).hexdigest(),
        "n_prompts": len(recs),
        "waves": waves,
        "human_retest": retest,
        "models": models,
        "comparisons": comparisons,
        "git_sha": provenance.git_sha(),
        "dirty": provenance.git_dirty(),
        "seed": seed,
        "n_splits": n_splits,
        "n_boot": n_boot,
        "created_at": provenance.now_iso(),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=float) + "\n")
    for name, m in models.items():
        line = {
            arm: round(m[arm]["accuracy_over_ceiling"], 3)
            for arm in ("population", "persona")
            if arm in m
        }
        typer.echo(f"{name}: accuracy/ceiling {line}  coverage {m['coverage']}")
    typer.echo(f"wrote {out}")


if __name__ == "__main__":
    app()
