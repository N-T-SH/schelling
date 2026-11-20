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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.compute as pc  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import typer

from schelling.augment.paraphrases import spend_usd
from schelling.baselines.llm_prompt import (
    CacheMissError,
    LLMPromptBaseline,
    LLMSpec,
    ParseError,
    build_messages,
    parse_probs,
)
from schelling.data.loader import load_records, partition_frame, select_partition
from schelling.llm.fireworks import OPENROUTER_URL, FireworksClient, load_dotenv

app = typer.Typer(add_completion=False)


def _request_rows(
    processed: Path, splits_dir: Path, source: str, split: str, partitions: str, form: str
) -> tuple[pd.DataFrame, int]:
    """The rows that need a request, loaded once and kept small.

    Loads only the source's records, once, and reduces each partition to one row per item for
    the population form (its request does not depend on the person). Loading every split and
    partition in full, in four parallel processes, used about 4 GB each and crashed a 16 GB
    machine on 2026-11-20.
    """
    path = processed / f"records_{source}.parquet"
    light = ["record_id", "source", "study_id", "item_id", "person_id", "trial_idx", "wave"]
    text = ["qtype", "question", "options", "levels"]
    # Split assignment needs only the small columns; the question text (a copy per row) is
    # read afterwards for the rows that need a request.
    records = (
        pd.read_parquet(path, columns=light, dtype_backend="pyarrow")
        if path.exists()
        else load_records(processed)
    )
    records = records[records["source"] == source]
    parts, n_rows = [], 0
    for s in split.split(","):
        parted = partition_frame(records, s, splits_dir)
        for part in partitions.split(","):
            rows = select_partition(parted, part)
            n_rows += len(rows)
            parts.append(rows.drop_duplicates("item_id") if form == "population" else rows)
        del parted
    frame = pd.concat(parts).drop_duplicates("record_id").reset_index(drop=True)
    if path.exists():
        # Dictionary-encoded: each distinct question is held once, not once per row.
        table = pq.read_table(
            path, columns=["record_id", *text], read_dictionary=["question", "qtype"]
        )
        keep = pc.is_in(table["record_id"], value_set=pa.array(frame["record_id"].tolist()))
        texts = table.filter(keep).to_pandas()
        del table
        texts["question"] = texts["question"].astype(str)
        texts["qtype"] = texts["qtype"].astype(str)
        frame = frame[light].merge(texts, on="record_id", validate="1:1")
    return frame, n_rows


@app.command()
def main(
    spec: Annotated[Path, typer.Option(help="LLMSpec JSON")],
    split: Annotated[str, typer.Option(help="one split, or several joined by commas")],
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
    provider: Annotated[str, typer.Option(help="fireworks or openrouter")] = "fireworks",
    partitions: Annotated[
        str, typer.Option(help="train and/or val, joined by commas (test is sealed)")
    ] = "val",
    shard: Annotated[
        str, typer.Option(help="i/n: only every n-th request from i (parallel runs)")
    ] = "0/1",
    workers: Annotated[int, typer.Option(help="parallel requests in this one process")] = 1,
) -> None:
    cfg = LLMSpec.model_validate_json(spec.read_text())
    frame, n_rows = _request_rows(processed, splits_dir, source, split, partitions, cfg.form)
    persons = pd.read_parquet(processed / "persons_twin2k.parquet")
    client = None
    if go:
        if cap_usd <= 0:
            raise typer.BadParameter("--go needs --cap-usd")
        load_dotenv(env)
        if provider == "openrouter":
            client = FireworksClient(
                cache_dir=Path(cfg.cache_dir),
                base_url=OPENROUTER_URL,
                key_env="OPENROUTER_API_KEY",
                retries=5,
            )
        elif provider == "fireworks":
            client = FireworksClient(cache_dir=Path(cfg.cache_dir))
        else:
            raise typer.BadParameter("--provider must be fireworks or openrouter")
    base = LLMPromptBaseline(cfg, client=client)
    base.fit(frame.iloc[:0], persons)
    unique = {json.dumps(r, sort_keys=True, default=str): r for r in base.requests(frame)}
    i, n = (int(x) for x in shard.split("/"))
    reqs = list(unique.values())[i::n][: limit or None]
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
        "partitions": partitions,
        "rows": n_rows,
        "requests": len(reqs),
        "est_input_tokens": round(est_in),
        "assumed_output_tokens_per_call": out_tokens,
        "est_usd": round(est, 4),
    }
    if not go:
        # Progress from the cache, no network: how many requests already have a response.
        cached, cached_usd = 0, 0.0
        for q, labels, attrs in reqs:
            try:
                hit = base.fetch(q, labels, attrs)
            except CacheMissError:
                continue
            cached += 1
            cached_usd += spend_usd(hit["usage"], in_per_m, out_per_m)
        summary |= {"cached": cached, "cached_billed_usd": round(cached_usd, 4)}
    if go:
        spent, done, bad, fresh = 0.0, 0, 0, 0

        def one(req: tuple[str, list[str], dict[str, Any] | None]) -> tuple[Any, int]:
            q, labels, attrs = req
            try:
                return base.fetch(q, labels, attrs), len(labels)
            except RuntimeError as err:
                typer.echo(f"request failed: {err}", err=True)
                return None, len(labels)

        # The cap is checked between chunks, so it can be passed by at most one chunk.
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for lo in range(0, len(reqs), workers):
                if spent >= cap_usd:
                    break
                for resp, n_labels in pool.map(one, reqs[lo : lo + workers]):
                    done += 1
                    if resp is None:
                        bad += 1
                        continue
                    if not resp["cached"]:
                        fresh += 1
                        spent += spend_usd(resp["usage"], in_per_m, out_per_m)
                    try:
                        parse_probs(resp["text"], n_labels, cfg.eps)
                    except ParseError:
                        bad += 1
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
