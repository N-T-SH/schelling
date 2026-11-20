# D01-lite: do decision-tuned models predict people better than general LLMs?

**Status:** approved by the human on 2026-11-20 as an amendment to the D01 contract. Tooling is
built and tested; no model has been scored and nothing has been spent. Still needed for the API
models: an OpenRouter key in `.env` and a spend cap. The Colab runs are free and can start now.

**Dry-run cost estimate (2026-11-20, all 10,807 prompts, no network):** about 3.1 M input tokens
per model. Assuming 300 output tokens per call: DeepSeek V4.1 Flash about $4.82, Qwen3.8-27B
about $7.32 (list prices above, not batch). The output length is a guess; reasoning output can
make it several times larger, so a 50-prompt paid smoke run measures it before the full run.

## Question

On Twin-2K-500 **decision** questions, how close do these models get to how real people answer:

| Model | Kind | Runs on | Readout |
|---|---|---|---|
| Qwentaur-4B (socius, LoRA r16 on Qwen3-4B-Base, Psych-101) | Decision-tuned | Colab free T4, fp16 | Log-probability of each option key |
| Qwen3-4B-Base, same weights with the adapter off | Its untuned base | Same session | Same |
| Qwentaur-14B and its base, optional | Scale check | Colab T4, 4-bit | Same |
| DeepSeek V4.1 Flash | General, open weights | OpenRouter | Stated probabilities (JSON) |
| Qwen3.8-27B | General, open weights | OpenRouter | Stated probabilities (JSON) |

The adapter-against-base pair is the cleanest test: same size, same family, same precision; the
only difference is fine-tuning on human decisions.

## Data (validation only)

- **People:** 100 seeded persons from `heldout_person` val (no model of ours trained on them).
- **Items:** 107 decision items: economic preferences (3 rows per multiple price list, plus
  ultimatum, trust, dictator and mental-accounting items), 12 purchase decisions, the
  non-experimental heuristics block, and the between-subject bias experiments. Personality,
  ability tests and attitudes are out. Items in the sealed `heldout_item` test set are out.
- **Prompts:** 10,807 per model: 107 population prompts (no person) and 10,700 persona prompts
  (the 14 demographic attributes). Persona text fields are never used (A04).

## Metrics

1. **Crowd:** distribution accuracy (1 − TVD) per item, divided by a human noise ceiling:
   split the human respondents into halves many times; the model is compared with one half,
   the other half with the same half, so both carry the same sampling noise.
2. **Spread:** median SD ratio of predicted to human answer distribution (too uniform < 1).
3. **Person:** NLL of each sampled person's real answer, persona prompt against population
   prompt, bootstrapped over persons.
4. **Bias test:** 13 of the 16 pre-registered contrasts of decision 0002 (three are dropped
   because an arm is a sealed `heldout_item` test item). Human effect sizes from
   `heldout_person` train and val people; correlation and sign agreement with the model.

Comparisons between models are paired over items (bootstrap, 95 % interval).

## How to run

```bash
# 1. prompts (once; git-ignored output in data/d01/)
uv run python scripts/d01/build_prompts.py --processed <main checkout>/data/processed

# 2a. Psych-101 fine-tunes: upload data/d01/d01_prompts.jsonl to My Drive/schelling-d01/,
#     open notebooks/d01_colab_logprobs.ipynb in Colab (T4) and run all cells.
#     The same scorer runs anywhere else:
python scripts/d01/score_logprobs.py --prompts data/d01/d01_prompts.jsonl --out results \
    --base unsloth/Qwen3-4B-Base --adapter socius/Qwentaur-4B-LoRA-r16 --variants adapter,base

# 2b. API models: dry run first (no network), then the paid run after approval
uv run python scripts/d01/run_api.py --prompts data/d01/d01_prompts.jsonl --out results \
    --model deepseek/deepseek-v4.1-flash-1017 --in-per-m 0.30 --out-per-m 1.20
uv run python scripts/d01/run_api.py ... --go --cap-usd <approved>

# 3. score every folder in results/
uv run python scripts/d01/score.py --prompts data/d01/d01_prompts.jsonl --results results \
    --processed <main checkout>/data/processed --out reports/FD-d01lite/metrics.json
```

Any other model works if it writes `results/<name>/part-*.jsonl` lines of
`{"prompt_id", "probs"}` in the prompt's option order.

## Checks done (no results)

- Unit tests for item selection, prompts, the scorer's resume logic and the metrics (known-answer
  cases: true distribution reaches the ceiling, uniform does not).
- End-to-end smoke run on this Mac with the smallest adapter, Smoltaur-0.1B, on 400 prompts:
  loads base plus adapter, scores both variants, resumes, and the scorer reads the output. Every
  prompt took the one-forward-pass route. Its numbers are not results (4 % coverage).
- A diagnostic worth keeping: the share of next-token probability that lands on the option keys
  (`key_mass` in each result line) was about 0.16 for the untuned SmolLM2 base and about 1.0 for
  its adapter. A low share means the base model is unsure of the format, not of the decision;
  report it beside the base model's scores.

## Known limits

- Twin-2K-500 has been public since 2025-05-15 and the bias experiments are textbook cases, so
  every model may have seen them; the contamination probes (paraphrase gap, completion) are a
  separate step, not yet built.
- Psych-101 transcripts carry no demographics, so the persona arm is new ground for Qwentaur.
- Stated probabilities (API models) and log-probabilities (Qwentaur) are different readouts; the
  API models' log-probability readout is not built.
- 4-bit 14B numbers are comparable with their own base only, not with fp16 4B.
