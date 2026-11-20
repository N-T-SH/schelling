"""D01-lite: ask an API model (OpenRouter, Fireworks or any OpenAI-compatible endpoint) for a
probability per option, for every prompt record. Dry run by default.

    # dry run: count requests and estimate cost, no network
    uv run python scripts/d01/run_api.py --prompts data/d01/d01_prompts.jsonl \\
        --model deepseek/deepseek-v4.1-flash-1017 --in-per-m 0.30 --out-per-m 1.20

    # paid run, only after the human approved the cap (AGENTS.md rule 7)
    uv run python scripts/d01/run_api.py ... --go --cap-usd 3

Every response is cached on disk (cache/llm), so re-running continues where it stopped and
costs nothing for finished prompts. Results go to results/<tag>/part-api.jsonl in the format
score.py reads: {"prompt_id", "probs"} with probs null when the reply did not parse.
Prices are per million tokens and must be read from the provider on the day of the run.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Annotated

import typer

from schelling.augment.paraphrases import spend_usd
from schelling.baselines.llm_prompt import ParseError, parse_probs
from schelling.llm.fireworks import BASE_URL, OPENROUTER_URL, FireworksClient, load_dotenv

app = typer.Typer(add_completion=False)
ENDPOINTS = {
    "openrouter": (OPENROUTER_URL, "OPENROUTER_API_KEY"),
    "fireworks": (BASE_URL, "FIREWORKS_API_KEY"),
}
EPS = 1e-4


def billed_usd(usage: dict[str, object], in_per_m: float, out_per_m: float) -> float:
    """OpenRouter reports the charge per call (`cost`, after time-of-day discounts); use it
    when present, else list price from token counts."""
    cost = usage.get("cost")
    if isinstance(cost, int | float):
        return float(cost)
    return spend_usd(usage, in_per_m, out_per_m)


@app.command()
def main(
    prompts: Annotated[Path, typer.Option()],
    model: Annotated[str, typer.Option(help="provider model id")],
    in_per_m: Annotated[float, typer.Option(help="USD per million input tokens, read today")],
    out_per_m: Annotated[float, typer.Option(help="USD per million output tokens, read today")],
    endpoint: Annotated[str, typer.Option(help="openrouter or fireworks")] = "openrouter",
    out: Annotated[Path, typer.Option()] = Path("results"),
    arm: Annotated[str, typer.Option(help="all, population or persona")] = "all",
    extra: Annotated[
        str, typer.Option(help='JSON request params, e.g. {"reasoning": {"effort": "low"}}')
    ] = "{}",
    max_tokens: Annotated[int, typer.Option()] = 2000,
    out_tokens_guess: Annotated[int, typer.Option(help="dry-run guess per call")] = 300,
    go: Annotated[bool, typer.Option(help="make paid calls")] = False,
    cap_usd: Annotated[float, typer.Option()] = 0.0,
    limit: Annotated[int, typer.Option(help="first N prompts only (smoke test)")] = 0,
    env: Annotated[Path, typer.Option()] = Path(".env"),
    workers: Annotated[int, typer.Option(help="parallel requests")] = 1,
) -> None:
    recs = [json.loads(x) for x in prompts.read_text().splitlines() if x.strip()]
    recs = [r for r in recs if arm == "all" or r["arm"] == arm][: limit or None]
    chars = sum(sum(len(m["content"]) for m in r["chat"]) for r in recs)
    est = (chars / 4 * in_per_m + len(recs) * out_tokens_guess * out_per_m) / 1e6
    summary: dict[str, object] = {
        "model": model,
        "requests": len(recs),
        "est_input_tokens": round(chars / 4),
        "assumed_output_tokens_per_call": out_tokens_guess,
        "est_usd": round(est, 2),
    }
    if not go:
        typer.echo(json.dumps(summary, indent=2))
        return
    if cap_usd <= 0:
        raise typer.BadParameter("--go needs --cap-usd")
    load_dotenv(env)
    url, key_env = ENDPOINTS[endpoint]
    client = FireworksClient(base_url=url, key_env=key_env, retries=5)
    params = json.loads(extra)
    tag_dir = out / (model.split("/")[-1].lower() + "-api")
    tag_dir.mkdir(parents=True, exist_ok=True)
    spent, bad, fresh, done, failed = 0.0, 0, 0, 0, 0
    lines = []

    def ask(r: dict[str, object]) -> dict[str, object] | None:
        try:
            return client.chat(model, r["chat"], max_tokens=max_tokens, temperature=0.0, **params)  # type: ignore[arg-type]
        except RuntimeError:
            return None  # retries exhausted: recorded as unparsed, retried on the next run

    chunk = max(1, workers) * 20
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        # The cap is checked between chunks, so it can be overshot by at most one chunk.
        for start in range(0, len(recs), chunk):
            if spent >= cap_usd:
                break
            batch = recs[start : start + chunk]
            for r, resp in zip(batch, pool.map(ask, batch), strict=True):
                probs: list[float] | None = None
                if resp is None:
                    failed += 1
                else:
                    if not resp["cached"]:
                        fresh += 1
                        spent += billed_usd(resp["usage"], in_per_m, out_per_m)
                    try:
                        probs = parse_probs(str(resp["text"] or ""), len(r["labels"]), EPS).tolist()
                    except ParseError:
                        bad += 1
                lines.append(json.dumps({"prompt_id": r["prompt_id"], "probs": probs}))
                done += 1
            if start // chunk % 25 == 0:
                typer.echo(f"{done}/{len(recs)} done, ${spent:.3f} spent", err=True)
    # Rewritten in full each run: the cache makes earlier prompts free to replay.
    (tag_dir / "part-api.jsonl").write_text("\n".join(lines) + "\n")
    meta = {**summary, "endpoint": endpoint, "params": params, "max_tokens": max_tokens}
    (tag_dir / "run.json").write_text(json.dumps(meta, indent=2) + "\n")
    summary |= {
        "done": done,
        "network_calls": fresh,
        "parse_failures": bad,
        "request_failures": failed,
        "spent_usd": round(spent, 4),
        "cap_reached": done < len(recs),
    }
    typer.echo(json.dumps(summary, indent=2))


if __name__ == "__main__":
    app()
