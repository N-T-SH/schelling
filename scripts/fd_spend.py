"""Fork D: billed spend of an LLM baseline on a split's val partition, read from the cache.

    uv run python scripts/fd_spend.py --out reports/FD-gate/spend.json SPEC.json [SPEC.json ...]

Sums the provider-reported `cost` of every cached response the specs' requests map to (no
network). With `reuse_max_tokens`, answers reused from the lower limit are counted under the
spec that made them, so pass every spec of a run once and nothing is counted twice.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from schelling.baselines.llm_prompt import LLMPromptBaseline, LLMSpec, build_messages
from schelling.data.loader import load_partition
from schelling.eval import provenance
from schelling.llm.fireworks import request_key

app = typer.Typer(add_completion=False)


@app.command()
def main(
    specs: Annotated[list[Path], typer.Argument()],
    out: Annotated[Path, typer.Option()],
    split: Annotated[str, typer.Option()] = "heldout_item",
    processed: Annotated[Path, typer.Option()] = Path("data/processed"),
    splits_dir: Annotated[Path, typer.Option()] = Path("splits"),
) -> None:
    frame = load_partition(processed, split, "val", splits_dir=splits_dir)
    persons = pd.read_parquet(processed / "persons_twin2k.parquet")
    rows: dict[str, dict[str, float]] = {}
    for spec in specs:
        cfg = LLMSpec.model_validate_json(spec.read_text())
        base = LLMPromptBaseline(cfg)
        base.fit(frame.iloc[:0], persons)
        params = {"max_tokens": cfg.max_tokens, "temperature": 0.0, **cfg.extra}
        seen: set[str] = set()
        usd = calls = empty = 0.0
        for question, labels, attrs in base.requests(frame):
            key = request_key(
                cfg.model, build_messages(question, labels, attrs, cfg.wording), params
            )
            path = Path(cfg.cache_dir) / key[:2] / f"{key}.json"
            if key in seen or not path.exists():
                continue
            seen.add(key)
            hit = json.loads(path.read_text())
            usd += float(hit["usage"].get("cost", 0.0))
            calls += 1
            empty += not (isinstance(hit["text"], str) and hit["text"].strip())
        rows[cfg.name] = {"billed_usd": usd, "cached_responses": calls, "empty_replies": empty}
    result = {
        "split": split,
        "partition": "val",
        # A list, in argument order: spec names contain dots, which JSON paths cannot address.
        "specs": [
            {"spec": str(s), "name": n, **r} for s, (n, r) in zip(specs, rows.items(), strict=True)
        ],
        "total_billed_usd": sum(r["billed_usd"] for r in rows.values()),
        "git_sha": provenance.git_sha(),
        "created_at": provenance.now_iso(),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    typer.echo(json.dumps(result, indent=2))


if __name__ == "__main__":
    app()
