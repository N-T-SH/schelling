"""Fork D: paired NLL comparison of two harness runs from their saved predictions.

    uv run python scripts/fd_paired.py --a reports/X --b reports/Y --split heldout_item \
        --unit item_id --out reports/FD-paired/x-vs-y.json

Reads the `val` partition only (the loader refuses `test`). Negative mean_diff favours --a.
"""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer

from schelling import compare
from schelling.data.loader import load_partition
from schelling.eval import metrics

app = typer.Typer(add_completion=False)


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()


@app.command()
def main(
    a: Annotated[Path, typer.Option(help="run directory of model A")],
    b: Annotated[Path, typer.Option(help="run directory of model B")],
    split: Annotated[str, typer.Option()],
    out: Annotated[Path, typer.Option()],
    unit: Annotated[str, typer.Option(help="item_id or person_id: the bootstrap unit")] = "item_id",
    source: Annotated[str, typer.Option()] = "twin2k",
    processed: Annotated[Path, typer.Option()] = Path("data/processed"),
    splits_dir: Annotated[Path, typer.Option()] = Path("splits"),
    persons: Annotated[int, typer.Option(help="seeded person subsample; 0 keeps everyone")] = 0,
    seed: Annotated[int, typer.Option()] = 0,
    n_boot: Annotated[int, typer.Option()] = 10_000,
) -> None:
    records = load_partition(processed, split, "val", splits_dir=splits_dir)
    records = records[records["source"] == source]
    rows = compare.join_rows(records, compare.load_predictions(a), compare.load_predictions(b))
    if persons:
        rows = compare.subsample_persons(rows, persons, seed)
    y = rows["response"].to_numpy(dtype=int)
    result = {
        "a": str(a),
        "b": str(b),
        "split": split,
        "partition": "val",
        "source": source,
        "unit": unit,
        "n_rows": len(rows),
        "n_persons": int(rows["person_id"].nunique()),
        "n_items": int(rows["item_id"].nunique()),
        # Row-weighted NLL, comparable with metrics.json#individual.by_source.<source>.nll.
        "row_nll_a": metrics.nll(list(rows["probs_a"]), y),
        "row_nll_b": metrics.nll(list(rows["probs_b"]), y),
        "paired": compare.paired_bootstrap(compare.unit_nll(rows, unit), n_boot, seed),
        "a_metrics_sha": json.loads((a / "metrics.json").read_text())["git_sha"],
        "b_metrics_sha": json.loads((b / "metrics.json").read_text())["git_sha"],
        "git_sha": _git("rev-parse", "HEAD"),
        "dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "created_at": datetime.now(UTC).isoformat(),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    typer.echo(json.dumps(result["paired"]))


if __name__ == "__main__":
    app()
