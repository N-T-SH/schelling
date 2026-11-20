"""Fork D: list the items an LLM baseline has no usable cached answer for (offline, no spend).

    uv run python scripts/fd_unanswered.py --spec configs/fork-distill/gate-deepseek-high16k.json \\
        --split heldout_item --out reports/FD-gate/high16k-unanswered.json

An item is unanswered when its reply is empty (reasoning ran out of tokens), does not parse,
or was never asked.
The list feeds `fd_paired.py --exclude-items` for an answered-only comparison.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from schelling.baselines.llm_prompt import (
    CacheMissError,
    LLMPromptBaseline,
    LLMSpec,
    ParseError,
    parse_probs,
)
from schelling.data.loader import load_partition

app = typer.Typer(add_completion=False)


@app.command()
def main(
    spec: Annotated[Path, typer.Option(help="LLMSpec JSON")],
    split: Annotated[str, typer.Option()],
    out: Annotated[Path, typer.Option()],
    processed: Annotated[Path, typer.Option()] = Path("data/processed"),
    splits_dir: Annotated[Path, typer.Option()] = Path("splits"),
) -> None:
    cfg = LLMSpec.model_validate_json(spec.read_text())
    frame = load_partition(processed, split, "val", splits_dir=splits_dir)
    base = LLMPromptBaseline(cfg)  # offline: a cache miss raises
    base.fit(frame.iloc[:0], pd.read_parquet(processed / "persons_twin2k.parquet"))
    firsts = frame.drop_duplicates("item_id").reset_index(drop=True)
    missing: dict[str, list[str]] = {}
    for (question, labels, attrs), item, source in zip(
        base.requests(firsts), firsts["item_id"], firsts["source"], strict=True
    ):
        try:
            parse_probs(base.fetch(question, labels, attrs)["text"], len(labels), cfg.eps)
        except (ParseError, CacheMissError):
            # Never asked at this limit (e.g. the spend cap stopped the re-ask) counts too.
            missing.setdefault(str(source), []).append(str(item))
    result = {
        "spec": str(spec),
        "split": split,
        "partition": "val",
        "n_items": len(firsts),
        "n_unanswered": sum(len(v) for v in missing.values()),
        "by_source": {k: sorted(v) for k, v in missing.items()},
        "items": sorted(i for v in missing.values() for i in v),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    typer.echo(json.dumps({k: result[k] for k in ("n_items", "n_unanswered", "by_source")}))


if __name__ == "__main__":
    app()
