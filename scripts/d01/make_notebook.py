"""Write notebooks/d01_colab_logprobs.ipynb with score_logprobs.py embedded.

The scorer has one source (scripts/d01/score_logprobs.py); re-run this after changing it:

    uv run python scripts/d01/make_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / "scripts" / "d01" / "score_logprobs.py"
OUT = ROOT / "notebooks" / "d01_colab_logprobs.ipynb"


def md(text: str) -> dict[str, object]:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text.strip("\n").splitlines(True),
    }


def cells() -> list[dict[str, object]]:
    scorer = SCRIPT.read_text()
    return [
        md("""
# D01-lite: score Psych-101 fine-tunes on Twin-2K-500 decision prompts

Free-tier target: one T4 GPU (16 GB). Scores **Qwentaur-4B** (socius, LoRA on
`unsloth/Qwen3-4B-Base`) and the **same base model with the adapter switched off**, in fp16.

**Before you start**
1. Runtime → Change runtime type → **T4 GPU**.
2. Put `d01_prompts.jsonl` (made by `scripts/d01/build_prompts.py`) in
   `My Drive/schelling-d01/`.

**Disconnects.** Results go to Drive in small part files. If the session drops, reconnect and
run all cells again: finished prompts are skipped. Nothing else needs changing.

**Bring back** the folder `My Drive/schelling-d01/results/` and score it with
`scripts/d01/score.py`.
"""),
        code("""
!nvidia-smi --query-gpu=name,memory.total --format=csv
"""),
        code("""
# transformers, peft and accelerate; bitsandbytes only matters for the optional 4-bit cell.
!pip -q install "transformers>=4.51" "peft>=0.15" accelerate bitsandbytes
"""),
        code("""
from google.colab import drive
drive.mount('/content/drive')
import os
WORK = '/content/drive/MyDrive/schelling-d01'
PROMPTS = f'{WORK}/d01_prompts.jsonl'
RESULTS = f'{WORK}/results'
assert os.path.exists(PROMPTS), f'upload d01_prompts.jsonl to {WORK} first'
os.makedirs(RESULTS, exist_ok=True)
"""),
        md("The scorer (generated from `scripts/d01/score_logprobs.py`; do not edit here)."),
        code("%%writefile /content/score_logprobs.py\n" + scorer),
        md("""
**Timing check** (about a minute): scores 64 prompts into a separate folder, so the main run
is not affected. Read the `/s` rate it prints to estimate the full run.
"""),
        code("""
!python /content/score_logprobs.py --prompts $PROMPTS --out /content/timing \\
    --base unsloth/Qwen3-4B-Base --adapter socius/Qwentaur-4B-LoRA-r16 \\
    --variants adapter --dtype float16 --batch-size 8 --limit 64
"""),
        md("""
**Main run: Qwentaur-4B and its base model, fp16.** Re-run this cell after a disconnect.
"""),
        code("""
!python /content/score_logprobs.py --prompts $PROMPTS --out $RESULTS \\
    --base unsloth/Qwen3-4B-Base --adapter socius/Qwentaur-4B-LoRA-r16 \\
    --variants adapter,base --dtype float16 --batch-size 8
"""),
        md("""
**Optional scale check: Qwentaur-14B and its base, 4-bit** (fits a T4 only quantised).
Both variants use the same 4-bit weights, so the comparison stays fair; do not compare these
numbers with the fp16 4B run as if precision were equal.
"""),
        code("""
!python /content/score_logprobs.py --prompts $PROMPTS --out $RESULTS \\
    --base unsloth/Qwen3-14B-Base --adapter socius/Qwentaur-14B-LoRA-r16 \\
    --variants adapter,base --load-4bit --batch-size 4
"""),
        md("Progress: prompts finished per model folder."),
        code("""
import glob, json
total = sum(1 for _ in open(PROMPTS))
for d in sorted(glob.glob(f'{RESULTS}/*/')):
    n = sum(len(open(p).read().splitlines()) for p in glob.glob(d + 'part-*.jsonl'))
    print(f'{d.split("/")[-2]:40s} {n:6d} / {total}')
"""),
    ]


def main() -> None:
    nb = {
        "cells": cells(),
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(nb, indent=1) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
