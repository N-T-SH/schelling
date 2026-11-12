"""Fork D: size, and (only with --go) make, the LLM calls an external-model baseline needs.

Dry run (default, no network, no key needed): counts distinct requests for a split's `val`
partition and estimates input tokens as characters / 4. That ratio is a rough rule of thumb,
not a tokenizer; output tokens per call are a stated guess until a smoke test measures them.

    uv run python scripts/fd_llm_calls.py --spec configs/fork-distill/e3-population.json \\
        --split heldout_item --processed <main checkout>/data/processed

With --go it calls the model through the disk cache and stops at --cap-usd. The human
approves the cap first (AGENTS.md rule 7).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from schelling.augment.paraphrases import spend_usd
from schelling.baselines.llm_prompt import (
    LLMPromptBaseline,
    LLMSpec,
    ParseError,
    build_messages,
    parse_probs,
)
from schelling.data.loader import load_partition
from schelling.llm.fireworks import FireworksClient, load_dotenv

app = typer.Typer(add_completion=False)


@app.command()
def main(
    spec: Annotated[Path, typer.Option(help="LLMSpec JSON")],
    split: Annotated[str, typer.Option()],
    in_per_m: Annotated[float, typer.Option(help="USD per million input tokens")],
    out_per_m: Annotated[float, typer.Option(help="USD per million output tokens")],
    processed: Annotated[Path, typer.Option()] = Path("data/processed"),
    splits_dir: Annotated[Path, typer.Option()] = Path("splits"),
    source: Annotated[str, typer.Option()] = "twin2k",
    out_tokens: Annotated[int, typer.Option(help="guessed output tokens per call")] = 400,
    go: Annotated[bool, typer.Option(help="make the paid calls")] = False,
    cap_usd: Annotated[float, typer.Option()] = 0.0,
    limit: Annotated[int, typer.Option(help="at most this many requests (smoke test)")] = 0,
    env: Annotated[Path, typer.Option(help=".env holding the API key")] = Path(".env"),
) -> None:
    cfg = LLMSpec.model_validate_json(spec.read_text())
    frame = load_partition(processed, split, "val", splits_dir=splits_dir)
    frame = frame[frame["source"] == source].reset_index(drop=True)
    persons = pd.read_parquet(processed / "persons_twin2k.parquet")
    client = None
    if go:
        if cap_usd <= 0:
            raise typer.BadParameter("--go needs --cap-usd")
        load_dotenv(env)
        client = FireworksClient(cache_dir=Path(cfg.cache_dir))
    base = LLMPromptBaseline(cfg, client=client)
    base.fit(frame.iloc[:0], persons)
    unique = {json.dumps(r, sort_keys=True, default=str): r for r in base.requests(frame)}
    reqs = list(unique.values())[: limit or None]
    chars = sum(
        sum(len(m["content"]) for m in build_messages(q, labels, attrs, cfg.wording))
        for q, labels, attrs in reqs
    )
    est_in = chars / 4
    est = (est_in * in_per_m + len(reqs) * out_tokens * out_per_m) / 1e6
    summary: dict[str, object] = {
        "spec": cfg.name,
        "model": cfg.model,
        "form": cfg.form,
        "split": split,
        "rows": len(frame),
        "requests": len(reqs),
        "est_input_tokens": round(est_in),
        "assumed_output_tokens_per_call": out_tokens,
        "est_usd": round(est, 4),
    }
    if go:
        spent, done, bad, fresh = 0.0, 0, 0, 0
        for q, labels, attrs in reqs:
            if spent >= cap_usd:
                break
            resp = base.fetch(q, labels, attrs)
            if not resp["cached"]:
                fresh += 1
                spent += spend_usd(resp["usage"], in_per_m, out_per_m)
            try:
                parse_probs(resp["text"], len(labels), cfg.eps)
            except ParseError:
                bad += 1
            done += 1
        summary |= {
            "done": done,
            "network_calls": fresh,
            "parse_failures": bad,
            "spent_usd": round(spent, 4),
            "cap_reached": done < len(reqs),
        }
    typer.echo(json.dumps(summary, indent=2))


if __name__ == "__main__":
    app()
