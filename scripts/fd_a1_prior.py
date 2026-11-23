"""Fork D, A1: the teacher's cached population distribution for every non-test Twin-2K-500
question, as a table the two-tower model reads (`model.teacher_prior_path`).

    uv run python scripts/fd_a1_prior.py --spec configs/fork-distill/a1-deepseek-high32k.json \
        --out runs/fd-a1/teacher_prior.json

Offline: reads the response cache only (a miss is an error). Reads `train` and `val` of every
item fold (a fold drops some questions from its train to avoid contamination, so one fold is not
enough); never `test`. Questions whose reply does not
parse are left out, so the model gives them its plain prior.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from schelling.baselines.llm_prompt import LLMPromptBaseline
from schelling.data.loader import load_partition
from schelling.eval import provenance
from schelling.infer.teacher import save_teacher, teacher_key, to_canonical

app = typer.Typer(add_completion=False)


@app.command()
def main(
    spec: Annotated[Path, typer.Option()],
    out: Annotated[Path, typer.Option()],
    splits: Annotated[str, typer.Option(help="comma-separated")] = ",".join(
        f"heldout_item_f{i}" for i in range(5)
    ),
    processed: Annotated[Path, typer.Option()] = Path("data/processed"),
    splits_dir: Annotated[Path, typer.Option()] = Path("splits"),
) -> None:
    cols = ["qtype", "question", "options", "levels", "source", "person_id", "item_id"]
    parts = []
    for split in splits.split(","):
        for part in ("train", "val"):
            f = load_partition(processed, split, part, splits_dir=splits_dir)[cols]
            parts.append(f[f["source"] == "twin2k"].drop_duplicates("item_id"))
            del f
    items = pd.concat(parts).drop_duplicates("item_id").reset_index(drop=True)
    teacher = LLMPromptBaseline.from_spec(spec)
    teacher.fit(pd.DataFrame(), pd.DataFrame({"person_id": []}))
    table: dict[str, list[float]] = {}
    unparsed: list[str] = []
    for i in range(len(items)):
        row = items.iloc[[i]]
        probs = teacher.predict_proba(row)[0]
        qtype, text, o, lv = row[["qtype", "question", "options", "levels"]].iloc[0]
        if teacher.stats["parse_failures"]:
            unparsed.append(str(row["item_id"].iloc[0]))
            continue
        labels = [] if qtype == "noul" else list(o if qtype == "choice" else lv)
        table[teacher_key(qtype, text, labels)] = to_canonical(qtype, labels, list(probs))
    meta = {
        "spec": str(spec),
        "splits": splits.split(","),
        "partitions": ["train", "val"],
        "source": "twin2k",
        "items": len(items),
        "questions": len(table),
        "unparsed_items": unparsed,
        "git_sha": provenance.git_sha(),
        "dirty": provenance.git_dirty(),
        "created_at": provenance.now_iso(),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    save_teacher(out, table, meta)
    typer.echo(json.dumps({k: meta[k] for k in ("items", "questions", "unparsed_items")}))


if __name__ == "__main__":
    app()
