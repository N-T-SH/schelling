"""D01-lite: build the prompt file every model is run on.

    uv run python scripts/d01/build_prompts.py --processed <main checkout>/data/processed

People: a seeded sample of `heldout_person` val persons (never used for training).
Items: Twin-2K-500 decision items (schelling.bench.items), excluding anything outside the
`heldout_item` train and val partitions, so the sealed item-test questions stay untouched.
The test partition of any split is never loaded.

Writes data/d01/d01_prompts.jsonl (git-ignored: it contains dataset text) and a manifest
with counts and hashes next to it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from schelling.bench.items import select_items, select_persons
from schelling.bench.prompts import TEMPLATE_VERSION, build_records
from schelling.data.loader import load_partition
from schelling.eval import provenance

app = typer.Typer(add_completion=False)


@app.command()
def main(
    processed: Annotated[Path, typer.Option()] = Path("data/processed"),
    splits_dir: Annotated[Path, typer.Option()] = Path("splits"),
    out: Annotated[Path, typer.Option()] = Path("data/d01/d01_prompts.jsonl"),
    persons_n: Annotated[int, typer.Option()] = 100,
    seed: Annotated[int, typer.Option()] = 0,
) -> None:
    val = load_partition(processed, "heldout_person", "val", splits_dir=splits_dir)
    val = val[val["source"] == "twin2k"]
    usable: set[str] = set()
    for part in ("train", "val"):  # heldout_item test items are left out by construction
        usable |= set(
            load_partition(processed, "heldout_item", part, splits_dir=splits_dir)["item_id"]
        )
    items = select_items(val, usable, seed=seed)
    people = select_persons(list(val["person_id"].astype(str).unique()), persons_n, seed=seed)
    attrs = pd.read_parquet(processed / "persons_twin2k.parquet")
    attrs["person_id"] = attrs["person_id"].astype(str)
    recs = build_records(items, attrs[attrs["person_id"].isin(people)], people)

    out.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(json.dumps(r) for r in recs) + "\n"
    out.write_text(body)
    manifest = {
        "prompts": str(out),
        "sha256": hashlib.sha256(body.encode()).hexdigest(),
        "template": TEMPLATE_VERSION,
        "seed": seed,
        "split_people": "heldout_person/val",
        "n_persons": len(people),
        "n_items": len(items),
        "n_prompts": len(recs),
        "items_by_study": items["study_id"].value_counts().sort_index().to_dict(),
        "persons": people,
        "item_ids": list(items["item_id"]),
        "git_sha": provenance.git_sha(),
        "dirty": provenance.git_dirty(),
        "created_at": provenance.now_iso(),
    }
    out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    typer.echo(
        json.dumps({k: manifest[k] for k in ("n_persons", "n_items", "n_prompts", "sha256")})
    )


if __name__ == "__main__":
    app()
