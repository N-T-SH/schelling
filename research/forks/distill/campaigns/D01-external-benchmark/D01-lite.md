# D01-lite: do decision-tuned models predict people better than general LLMs?

**Status:** closed by the human on 2026-11-20. Approved the same day as an amendment to the D01
contract. DeepSeek V4.1 Flash and Qwentaur-4B (with its base) scored; results below. Qwentaur-14B
skipped by the human; Qwen3.8-27B dropped for budget; contamination probes not built. Conclusions
and recommended fork updates are at the end.

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

## Results so far (2026-11-20)

DeepSeek V4.1 Flash and Qwentaur-4B with its base model. Validation people only, each
person's first answer (wave 1). Accuracy = 1 - TVD between predicted and human answer
distributions, averaged over items, 95 % bootstrap intervals over items. References: uniform
guess, and the training crowd (each item's answer distribution among `heldout_person` train
people; the best possible prediction that ignores who is asked). Split-half human comparison
on these items: 0.921.

**Full run, reasoning off** (10,807 prompts, 100 people, every reply parsed).
Source: `reports/FD-d01lite/nothink-vs-refs/metrics.json`, git d926ef81, clean.

| Predictor | Accuracy, no persona | Accuracy, persona | Spread (SD ratio) |
|---|---|---|---|
| Uniform | 0.713 | 0.713 | 1.24 |
| DeepSeek, reasoning off | 0.779 [0.752, 0.805] | 0.776 [0.748, 0.803] | 1.08 |
| Training crowd | 0.942 | 0.942 | 1.00 |

- DeepSeek minus uniform: +0.066 [0.038, 0.096]; minus training crowd: -0.163 [-0.192, -0.135]
  (paired over items).
- Persona prompts make individual answers slightly less likely: NLL +0.018 [0.010, 0.027].
- Bias test (13 contrasts, 10 with a clear human effect): correlation 0.70 and 6 of 10 signs
  right without persona; 0.88 and 10 of 10 with persona.
- People repeat their own answer on retest 68 % of the time (54 items, 7,215 pairs), which
  limits individual-level prediction for anyone.
- Pooling retest answers instead (`nothink-vs-refs-allwaves`) changes accuracy by 0.003 or less.

**Reasoning on against off** (2,247 prompts: all 107 population prompts plus 20 people; reasoning
"high", 8,000-token limit). 134 reasoning replies (6 %) never answered, mostly by thinking past
8,000 tokens, concentrated on a few items (QID189 to 191 have 40). Main comparison on the 2,113
prompts both settings answered. Source: `reports/FD-d01lite/think-vs-nothink-answered/metrics.json`,
git 9d597f99, clean; the version with unanswered prompts scored as uniform is
`think-vs-nothink-subset` and agrees.

| Predictor | Accuracy, no persona | Accuracy, persona | Spread | Persona minus population NLL |
|---|---|---|---|---|
| DeepSeek, reasoning off | 0.787 [0.759, 0.815] | 0.775 [0.749, 0.803] | 1.07 | +0.021 [0.003, 0.038] |
| DeepSeek, reasoning high | 0.857 [0.835, 0.877] | 0.852 [0.831, 0.871] | 1.01 | +0.028 [0.007, 0.049] |
| Training crowd | 0.943 | 0.942 | 1.00 | 0 |

- Reasoning raises accuracy by 0.071 [0.038, 0.104] without persona (93 items) and 0.076
  [0.048, 0.106] with persona (107 items), paired. It closes about 45 % of the gap from
  reasoning-off to the training crowd, and its spread matches the human spread.
- Persona prompts still do not help individual prediction, with or without reasoning.
- Bias test with persona: correlation 0.93 with reasoning, 0.82 without; signs right 8 of 10
  with reasoning, 10 of 10 without. With 10 contrasts this is not a clear difference.

**Spend:** $3.96 billed by OpenRouter for all DeepSeek calls (from the `cost` field of each
response in `cache/llm/usage.jsonl`), plus about $0.03 for the first smoke test whose records
were lost with the deleted worktree. Reasoning off costs about $0.0001 per prompt; reasoning
high about $0.0017.

**Qwentaur-4B against its base** (all 10,807 prompts, fp16 on a Colab T4, log-probabilities over
the answer keys, renormalised). Scored together with DeepSeek reasoning-off and the references;
the DeepSeek and reference numbers reproduce the table above.
Source: `reports/FD-d01lite/qwentaur4b-vs-all/metrics.json`, git 25a7010f, clean.

| Predictor | Accuracy, no persona | Accuracy, persona | Spread | Persona minus population NLL |
|---|---|---|---|---|
| Qwen3-4B base | 0.663 [0.623, 0.704] | 0.668 [0.631, 0.706] | 1.19 / 1.25 | -0.021 [-0.027, -0.015] |
| Uniform | 0.713 | 0.713 | 1.24 | 0 |
| Qwentaur-4B | 0.719 [0.689, 0.750] | 0.723 [0.696, 0.750] | 1.10 / 1.11 | -0.018 [-0.023, -0.012] |
| DeepSeek, reasoning off | 0.779 | 0.776 | 1.08 / 1.09 | +0.018 |
| Training crowd | 0.942 | 0.942 | 1.00 | 0 |

- Fine-tuning on Psych-101 helps: Qwentaur minus its base +0.056 [0.024, 0.089] without persona,
  +0.054 [0.020, 0.089] with persona (paired over items). But that only lifts the base from below
  a uniform guess to level with it: Qwentaur minus uniform +0.005 [-0.027, 0.039] and +0.009
  [-0.019, 0.039].
- DeepSeek (reasoning off) beats Qwentaur by 0.060 [0.027, 0.091] without persona and 0.053
  [0.027, 0.079] with persona.
- Persona prompts lower Qwentaur's NLL on a person's real answers (the opposite of DeepSeek), but
  the base model shows the same drop, and Psych-101 has no demographics, so this is probably the
  prompt format rather than learned use of demographics. Qwentaur's NLL with persona (0.975) is
  barely below uniform's (0.999) and far above the training crowd's (0.737).
- Bias test: correlation with the human effects 0.24 without persona and 0.23 with (signs right 8
  and 6 of 10); the base 0.37 and 0.19. DeepSeek: 0.70 and 0.88.
- The base model puts a median 25 % of its next-token probability on the answer keys (Qwentaur
  over 99.9 %), so the base readout is a renormalised minority of its mass.

**Reading so far:** on these decision items no model comes near the simple training crowd.
Reasoning helps DeepSeek a lot. Decision tuning helps Qwen3-4B but only up to a uniform guess,
and it stays behind a general model without reasoning. Demographic personas do not help any
model predict a given person in a way that can be credited to the demographics.

## Known limits

- Twin-2K-500 has been public since 2025-05-15 and the bias experiments are textbook cases, so
  every model may have seen them; the contamination probes (paraphrase gap, completion) are a
  separate step, not yet built.
- Psych-101 transcripts carry no demographics, so the persona arm is new ground for Qwentaur.
- Stated probabilities (API models) and log-probabilities (Qwentaur) are different readouts; the
  API models' log-probability readout is not built.
- Only the 4B Qwentaur was run; whether scale changes the picture is untested.

## Closed: conclusions and recommended fork updates (2026-11-20)

Scope reminder: D01-lite covers **seen** questions only (decision items of `heldout_person`, where a
training crowd exists). The fork's headline regime, new questions (`heldout_item`), was not tested.
Recommendations below are proposals; none of the D02 to D05 contracts is edited until the human
approves.

### What D01-lite established

1. **On seen questions the training crowd is far ahead of every model** (accuracy 0.942 against at
   most 0.857; NLL of a person's real answer 0.737 against 0.928 for DeepSeek reasoning-off; sources
   above). The training crowd is the same idea as B1, so this matches D01 H1 on the decision items.
2. **A general LLM carries real but modest signal without reasoning** (0.779, a quarter of the way
   from uniform 0.713 to the crowd 0.942), and **reasoning adds a lot** (+0.071 to +0.076 paired).
3. **Demographic personas do not give an LLM individual-level signal.** DeepSeek's persona prompts
   make real answers less likely (NLL +0.018). Qwen's improvement is shared by its untuned base and
   leaves it at about uniform (0.975 against 0.999).
4. **Psych-101 decision tuning does not transfer to these items.** Qwentaur-4B beats its base but
   only ties a uniform guess.
5. **No variance collapse at the population level.** Median SD ratios are 1.01 to 1.10 for DeepSeek
   (slightly too spread, not too narrow). This argues against D01 H3 and FA6, with a caveat: it is the
   spread of the predicted distribution, not the spec's sampled SD ratio.
6. **An LLM knows the direction of classic effects.** On the 10 bias contrasts with a clear human
   effect, DeepSeek with persona correlates 0.88 (10 of 10 signs right). M4b on unseen experiments
   is at chance (correlation 0.0248 <!-- src: reports/M4b-pooled-tt/effects.json#effects.summary.correlation.mean -->).
   These are textbook effects, so memorisation is not ruled out.
7. **Operations:** reasoning-off calls cost about $0.0001 each and always parsed. Reasoning-high calls
   cost about $0.0017, and 6 % never answered within 8,000 tokens, clustered on a few items.

### Recommended updates, in priority order

1. **Run the real D02 gate now, cheaply.** Population prompts for the 28 `heldout_item` val items,
   DeepSeek reasoning-high and reasoning-off, scored through `llm_prompt.py` and the sealed harness
   against B1 and M4b on NLL. That is 56 calls, well under $0.25 by arithmetic, inside the remaining
   OpenRouter credit, but it is paid, so it needs approval. This answers the fork's go/no-go directly;
   everything below depends on it.
2. **Make the teacher population-only and send it to the prior path.** Finding 3 says
   persona-conditioned teacher labels carry no person signal, so drop D02 idea 5 (K = 8 persona
   labels) and make idea 3 (teacher trains the question-to-prior path, people learned from human data)
   the default design. This cuts the label calls 9-fold.
3. **Use reasoning for teacher labels and re-cost the bank.** At the measured $0.0017 per call, a
   10 k population-only bank is about $17 and 50 k about $85 (arithmetic, before a dry run). Add an
   answer-rate guard (drop or re-ask questions that exceed the token limit) and track billed cost from
   the response, as `run_api.py` now does.
4. **Add DeepSeek V4.1 Flash as a candidate teacher beside `gpt-oss-120b`,** chosen by the gate run.
   Its licence for training on outputs is unverified; record it as a new assumption and check it
   before any bank is generated.
5. **Raise D04 (treatment effects) in priority, behind a contamination probe.** Finding 6 is the
   clearest sign that an LLM knows something M4b cannot learn from 435 questions. Before relying on
   it, build the paraphrase-gap probe the D01 contract already lists, using the M5 paraphrases.
6. **Drop the Psych-101 models (Centaur B6, Qwentaur) as teachers or headline comparators.** Keep
   finding 4 as a reported result.
7. **Lower the D03 risk estimate.** Update FA6 with finding 5 and keep D03 as a check, not a main
   campaign. Note that the opposite problem (too spread) is what was seen.
8. **Keep the fork's target on new questions and experiments.** On seen questions the crowd wins
   by a wide margin, so a distilled model can only add value where no crowd exists. That is already
   the design; D01-lite supports it.

