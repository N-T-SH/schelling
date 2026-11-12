"""Fork D: mean and SD across seeds of harness metrics, written with provenance.

    uv run python scripts/fd_summarise.py --out reports/FD-d01/m4b-item-summary.json \\
        reports/FD-m4b-item-s0-val reports/FD-m4b-item-s1-val reports/FD-m4b-item-s2-val
"""

from __future__ import annotations

import json
import statistics
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import typer

app = typer.Typer(add_completion=False)

PATHS = {
    "nll": "individual.by_source.twin2k.nll",
    "ece": "individual.by_source.twin2k.ece",
    "tvd": "population.twin2k.summary.tvd.mean",
    "w1": "population.twin2k.summary.w1.mean",
    "sd_ratio": "population.twin2k.summary.sd_ratio.mean",
}


def _get(doc: dict[str, Any], path: str) -> float:
    node: Any = doc
    for part in path.split("."):
        node = node[part]
    return float(node)


@app.command()
def main(runs: list[Path], out: Annotated[Path, typer.Option()]) -> None:
    docs = [json.loads((r / "metrics.json").read_text()) for r in runs]
    idents = {(d["git_sha"], d["dirty"], d["split"], d["partition"]) for d in docs}
    if len(idents) != 1:
        raise typer.BadParameter(f"runs differ in sha/dirty/split/partition: {idents}")
    sha, dirty, split, partition = next(iter(idents))
    result: dict[str, Any] = {
        "runs": [str(r) for r in runs],
        "runs_git_sha": sha,
        "runs_dirty": dirty,
        "split": split,
        "partition": partition,
        "n_seeds": len(docs),
        "metrics": {},
        "paths": PATHS,
        "git_sha": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip(),
        "created_at": datetime.now(UTC).isoformat(),
    }
    for name, path in PATHS.items():
        values = [_get(d, path) for d in docs]
        result["metrics"][name] = {
            "values": values,
            "mean": statistics.mean(values),
            "sd": statistics.stdev(values) if len(values) > 1 else None,
            "min": min(values),
            "max": max(values),
        }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    typer.echo(json.dumps({k: (v["mean"], v["sd"]) for k, v in result["metrics"].items()}))


if __name__ == "__main__":
    app()
