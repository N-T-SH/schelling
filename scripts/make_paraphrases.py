"""Generate question paraphrases once, offline, under a hard spend cap (M5, rule 7).

    uv run python scripts/make_paraphrases.py --limit 5            # smoke
    uv run python scripts/make_paraphrases.py --cap-usd 2.0        # full run

Resumable: questions already in the store are skipped, responses are cached on disk.
Stops before the next call once the estimated spend reaches the cap.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from schelling.augment.paraphrases import (
    ParaphraseStore,
    build_messages,
    filter_candidates,
    parse_json_list,
    spend_usd,
)
from schelling.llm.fireworks import FireworksClient, load_dotenv

MODEL = "accounts/fireworks/models/gpt-oss-120b"
IN_PER_M, OUT_PER_M = 0.15, 0.60  # https://docs.fireworks.ai/serverless/pricing.md, 2026-11-11


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default="data/processed/records_twin2k.parquet")
    ap.add_argument("--out", default="data/paraphrases/twin2k-gpt-oss-120b-v1.jsonl")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--cap-usd", type=float, default=2.0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    load_dotenv()
    texts = sorted(pd.read_parquet(args.records, columns=["question"])["question"].unique())
    store = ParaphraseStore(Path(args.out), generator=f"{MODEL}@v1")
    todo = [t for t in texts if t not in store]
    if args.limit:
        todo = todo[: args.limit]
    client = FireworksClient()
    spent, done, short = 0.0, 0, 0
    for text in todo:
        if spent >= args.cap_usd:
            print(f"cap reached at ${spent:.4f}; stopping with {len(todo) - done} left")
            break
        resp = client.chat(
            MODEL,
            build_messages(text, args.n),
            max_tokens=1500,
            temperature=0.8,
            reasoning_effort="low",
        )
        if not resp["cached"]:
            spent += spend_usd(resp["usage"], IN_PER_M, OUT_PER_M)
        kept = filter_candidates(text, parse_json_list(resp["text"]))
        short += len(kept) < 2
        store.add(text, kept)
        done += 1
    print(
        f"questions {len(texts)}  done this run {done}  fewer than 2 kept {short}  "
        f"est spend ${spent:.4f}  network calls {client.network_calls}"
    )


if __name__ == "__main__":
    main()
