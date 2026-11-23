"""Fork D: per-question paired comparison pooled over item folds (decision 0007).

    uv run python scripts/fd_pooled.py \
        --arm "a0=reports/FD-a0-f{f}-s{s}-val" --arm "a1=reports/FD-a1-f{f}-s{s}-val" \
        --arm "b1=reports/FD-a1-b1-f{f}-val" --pair a1,a0 --pair a1,b1 \
        --out reports/FD-a1/pooled.json

An arm pattern with `{s}` is a trained model (one run per seed, averaged per question); without
it, a reference scored once per fold. Each question is val in at most one fold, so questions are
pooled and resampled as units. Reads `val` only. Negative mean_diff favours the first arm.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import pandas as pd
import typer

from schelling import compare
from schelling.data.loader import load_partition
from schelling.eval import provenance

app = typer.Typer(add_completion=False)


def _get(d: dict[str, Any], path: str) -> float:
    for k in path.split("."):
        d = d[k]
    return float(d)  # type: ignore[arg-type]


GUARDS = {
    "nll": "individual.by_source.{src}.nll",
    "ece": "individual.by_source.{src}.ece",
    "sd_ratio": "population.{src}.summary.sd_ratio.mean",
    "tvd": "population.{src}.summary.tvd.mean",
    "permutation_max_abs_dp": "robustness.permutation_max_abs_dp",
}


def _summary(values: list[float]) -> dict[str, float]:
    a = np.asarray(values, dtype=float)
    return {
        "mean": float(a.mean()),
        "sd": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
        "min": float(a.min()),
        "max": float(a.max()),
        "n": len(a),
    }


@app.command()
def main(
    arm: Annotated[list[str], typer.Option(help="name=pattern with {f} and optionally {s}")],
    pair: Annotated[list[str], typer.Option(help="a,b: report a - b")],
    out: Annotated[Path, typer.Option()],
    folds: Annotated[str, typer.Option()] = "0,1,2,3,4",
    seeds: Annotated[str, typer.Option()] = "0,1,2",
    split_prefix: Annotated[str, typer.Option()] = "heldout_item_f",
    source: Annotated[str, typer.Option()] = "twin2k",
    processed: Annotated[Path, typer.Option()] = Path("data/processed"),
    splits_dir: Annotated[Path, typer.Option()] = Path("splits"),
    n_boot: Annotated[int, typer.Option()] = 10_000,
    seed: Annotated[int, typer.Option()] = 0,
) -> None:
    arms = dict(a.split("=", 1) for a in arm)
    fold_ids = [int(f) for f in folds.split(",")]
    seed_ids = [int(s) for s in seeds.split(",")]
    per_item: dict[str, list[pd.Series]] = {name: [] for name in arms}
    runs: dict[str, list[dict[str, Any]]] = {name: [] for name in arms}
    for f in fold_ids:
        split = f"{split_prefix}{f}"
        records = load_partition(processed, split, "val", splits_dir=splits_dir)
        records = records[records["source"] == source][["record_id", "item_id", "response"]]
        for name, pattern in arms.items():
            dirs = (
                [Path(pattern.format(f=f, s=s)) for s in seed_ids]
                if "{s}" in pattern
                else [Path(pattern.format(f=f))]
            )
            per_seed = []
            for d in dirs:
                per_seed.append(compare.item_nll(records, compare.load_predictions(d)))
                m = json.loads((d / "metrics.json").read_text())
                row: dict[str, Any] = {"run": str(d), "fold": f, "git_sha": m["git_sha"]}
                row["dirty"] = m["dirty"]
                for key, path in GUARDS.items():
                    row[key] = _get(m, path.format(src=source))
                runs[name].append(row)
            per_item[name].append(compare.seed_mean(per_seed))
        del records
    pooled = {name: pd.concat(parts).sort_index() for name, parts in per_item.items()}
    for name, s in pooled.items():
        if s.index.has_duplicates:
            raise ValueError(f"{name}: a question is val in more than one fold")
    result: dict[str, Any] = {
        "split_prefix": split_prefix,
        "folds": fold_ids,
        "seeds": seed_ids,
        "partition": "val",
        "source": source,
        "arms": {
            name: {
                "pattern": arms[name],
                "n_items": len(pooled[name]),
                "item_mean_nll": float(pooled[name].mean()),
                "runs": runs[name],
                **{key: _summary([r[key] for r in runs[name]]) for key in GUARDS},
                "any_dirty": any(r["dirty"] for r in runs[name]),
            }
            for name in arms
        },
        "pairs": {},
        "seed": seed,
        "n_boot": n_boot,
    }
    for p in pair:
        a, b = p.split(",")
        units = pd.DataFrame({"a": pooled[a], "b": pooled[b]})
        if units.isna().any().any():
            raise ValueError(f"{a} and {b} cover different questions")
        res = compare.paired_bootstrap(units, n_boot, seed)
        res["items_a_better"] = int((units["a"] < units["b"]).sum())
        res["relative_change"] = res["mean_diff"] / res["mean_b"]
        result["pairs"][f"{a}_vs_{b}"] = res
    result["per_item"] = {name: s.to_dict() for name, s in pooled.items()}
    result |= {
        "git_sha": provenance.git_sha(),
        "dirty": provenance.git_dirty(),
        "created_at": provenance.now_iso(),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    typer.echo(json.dumps({k: v for k, v in result["pairs"].items()}, indent=2))


if __name__ == "__main__":
    app()
