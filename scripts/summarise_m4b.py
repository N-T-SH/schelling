"""Collect the M4b training metrics and cross-run summaries into reports/M4b-train/.

Copies each run's metrics.json (training-time cached validation NLL, Twin-2K-500 only) and
writes summary.json with means, SDs and fold comparisons so the report can cite them.
Usage: python scripts/summarise_m4b.py
"""

from __future__ import annotations

import json
import shutil
import statistics as st
import subprocess
from datetime import UTC, datetime
from pathlib import Path

RUNS, OUT = Path("runs"), Path("reports/M4b-train")
SEEDS = ["tt-a2", "tt-s1", "tt-s2", "tt-s3", "tt-s4"]
NO_PERSON = [f"tt-np-s{s}" for s in range(3)]
NO_PRIOR = [f"tt-nopr-s{s}" for s in range(3)]
FOLDS = range(8)


def nll(name: str) -> float:
    return float(json.loads((RUNS / name / "metrics.json").read_text())["final_val_nll"])


def spread(names: list[str]) -> dict[str, float]:
    v = [nll(n) for n in names]
    return {"mean": st.mean(v), "sd": st.stdev(v), "n": len(v)}


def main() -> None:
    names = ["tt-a", *SEEDS, *NO_PERSON, *NO_PRIOR, "tt-item", *[f"tt-f{k}" for k in FOLDS]]
    for n in names:
        (OUT / n).mkdir(parents=True, exist_ok=True)
        shutil.copy(RUNS / n / "metrics.json", OUT / n / "metrics.json")
    folds = {}
    for k in FOLDS:
        tt = json.loads(Path(f"reports/M4b-tt-f{k}-val/metrics.json").read_text())
        b1 = json.loads(Path(f"reports/EC1-b1-f{k}-val/metrics.json").read_text())
        folds[f"f{k}"] = {
            "tt_nll": tt["individual"]["overall"]["nll"],
            "b1_nll": b1["individual"]["overall"]["nll"],
            "delta": tt["individual"]["overall"]["nll"] - b1["individual"]["overall"]["nll"],
        }
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    summary = {
        "git_sha": sha,
        "created_at": datetime.now(UTC).isoformat(),
        "seeds": spread(SEEDS),
        "no_person": spread(NO_PERSON),
        "no_prior_loss": spread(NO_PRIOR),
        "folds": folds,
        "folds_better_than_b1": sum(v["delta"] < 0 for v in folds.values()),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
