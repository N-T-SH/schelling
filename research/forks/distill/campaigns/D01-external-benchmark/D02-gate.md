# D02 gate: does an LLM teacher know unseen questions better than B1?

**Status:** run 2026-11-20 on `val` only, approved by the human ("Approve, run it", with a cap of 25 cents).
**Answer: yes.** DeepSeek V4.1 Flash with reasoning "high", asked once per question with no person
information, beats B1 on the unseen Twin-2K-500 questions of `heldout_item` val, with the item-bootstrap
interval excluding 0. It also beats the M4b two-tower model. The gate as written in the D01 contract
named `gpt-oss-120b` and a temperature fitted on a dev slice; this run uses DeepSeek (D01-lite showed
it works) and **no** temperature, which can only make the teacher look worse. Read the
contamination caveat before building on it.

## Setup

- **Split:** `heldout_item` val, Twin-2K-500 rows: 28 questions nobody in our training data answered,
  65804 rows (2058 persons). Psych-101 rows are outside the claim (decision 0001).
- **Model:** `deepseek/deepseek-v4.1-flash-1017` on OpenRouter, population prompt (question and options
  only), stated probabilities as JSON, `src/schelling/baselines/llm_prompt.py`, wording 0. Scored by the
  sealed harness (`schelling eval`), offline from the response cache.
- **Two settings:** reasoning off (`configs/fork-distill/gate-deepseek-off.json`) and reasoning high
  (`gate-deepseek-high16k.json`: 8,000-token limit, with the one empty Twin-2K-500 reply, QID74, asked
  again at 16,000; it then answered).
- **References:** B1 (item mean of the training people, re-run at this fork's SHA) and M4b seed 0
  (`reports/M4b-tt-item-val`, the only M4b run on this split whose predictions survive; the fork's
  three-seed re-run kept only its metrics).
- **Paired test:** per-item mean NLL, 10,000 bootstrap resamples of the 28 items
  (`scripts/fd_paired.py`). Negative favours the LLM.

## Results (Twin-2K-500 rows)

| Model | NLL | ECE (spec limit applies) | Population TVD | SD ratio |
| --- | --- | --- | --- | --- |
| B1 item mean | 1.3534 <!-- src: reports/FD-gate-b1-item-val/metrics.json#individual.by_source.twin2k.nll --> | 0.0479 <!-- src: reports/FD-gate-b1-item-val/metrics.json#individual.by_source.twin2k.ece --> | 0.2316 <!-- src: reports/FD-gate-b1-item-val/metrics.json#population.twin2k.summary.tvd.mean --> | 1.1811 <!-- src: reports/FD-gate-b1-item-val/metrics.json#population.twin2k.summary.sd_ratio.mean --> |
| M4b two-tower, seed 0 | 1.2939 <!-- src: reports/M4b-tt-item-val/metrics.json#individual.by_source.twin2k.nll --> | 0.0582 <!-- src: reports/M4b-tt-item-val/metrics.json#individual.by_source.twin2k.ece --> | 0.1813 <!-- src: reports/M4b-tt-item-val/metrics.json#population.twin2k.summary.tvd.mean --> | 1.0347 <!-- src: reports/M4b-tt-item-val/metrics.json#population.twin2k.summary.sd_ratio.mean --> |
| DeepSeek, reasoning off | 1.2692 <!-- src: reports/FD-gate-deepseek-off-item-val/metrics.json#individual.by_source.twin2k.nll --> | 0.0917 <!-- src: reports/FD-gate-deepseek-off-item-val/metrics.json#individual.by_source.twin2k.ece --> | 0.1860 <!-- src: reports/FD-gate-deepseek-off-item-val/metrics.json#population.twin2k.summary.tvd.mean --> | 0.9856 <!-- src: reports/FD-gate-deepseek-off-item-val/metrics.json#population.twin2k.summary.sd_ratio.mean --> |
| **DeepSeek, reasoning high** | 1.1874 <!-- src: reports/FD-gate-deepseek-high16k-item-val/metrics.json#individual.by_source.twin2k.nll --> | 0.0835 <!-- src: reports/FD-gate-deepseek-high16k-item-val/metrics.json#individual.by_source.twin2k.ece --> | 0.1604 <!-- src: reports/FD-gate-deepseek-high16k-item-val/metrics.json#population.twin2k.summary.tvd.mean --> | 0.9811 <!-- src: reports/FD-gate-deepseek-high16k-item-val/metrics.json#population.twin2k.summary.sd_ratio.mean --> |

The fork's three-seed M4b mean on this split is 1.2825
<!-- src: reports/FD-d01/m4b-item-summary.json#metrics.nll.mean -->, so seed 0 is a fair stand-in.

Paired differences in per-item NLL (95 % interval):

| Comparison | Difference | Interval | Items the LLM wins |
| --- | --- | --- | --- |
| Reasoning high − B1 | -0.2006 <!-- src: reports/FD-gate/high16k-vs-FD-gate-b1-item-val.json#paired.mean_diff --> | -0.2938 <!-- src: reports/FD-gate/high16k-vs-FD-gate-b1-item-val.json#paired.ci_low --> to -0.1125 <!-- src: reports/FD-gate/high16k-vs-FD-gate-b1-item-val.json#paired.ci_high --> | 23 of 28 |
| Reasoning high − M4b seed 0 | -0.1268 <!-- src: reports/FD-gate/high16k-vs-M4b-tt-item-val.json#paired.mean_diff --> | -0.2257 <!-- src: reports/FD-gate/high16k-vs-M4b-tt-item-val.json#paired.ci_low --> to -0.0400 <!-- src: reports/FD-gate/high16k-vs-M4b-tt-item-val.json#paired.ci_high --> | 15 of 28 |
| Reasoning high − reasoning off | -0.0980 <!-- src: reports/FD-gate/high16k-vs-FD-gate-deepseek-off-item-val.json#paired.mean_diff --> | -0.1958 <!-- src: reports/FD-gate/high16k-vs-FD-gate-deepseek-off-item-val.json#paired.ci_low --> to -0.0240 <!-- src: reports/FD-gate/high16k-vs-FD-gate-deepseek-off-item-val.json#paired.ci_high --> | 20 of 28 |
| Reasoning off − B1 | -0.1026 <!-- src: reports/FD-gate/off-vs-FD-gate-b1-item-val.json#paired.mean_diff --> | -0.2203 <!-- src: reports/FD-gate/off-vs-FD-gate-b1-item-val.json#paired.ci_low --> to 0.0133 <!-- src: reports/FD-gate/off-vs-FD-gate-b1-item-val.json#paired.ci_high --> | 18 of 28 |
| Reasoning off − M4b seed 0 | -0.0288 <!-- src: reports/FD-gate/off-vs-M4b-tt-item-val.json#paired.mean_diff --> | -0.1518 <!-- src: reports/FD-gate/off-vs-M4b-tt-item-val.json#paired.ci_low --> to 0.0837 <!-- src: reports/FD-gate/off-vs-M4b-tt-item-val.json#paired.ci_high --> | 14 of 28 |

## Reading

- **Gate passed with reasoning high.** Reasoning off is level with M4b and does not clear B1 with
  confidence; the reasoning is what makes the teacher useful.
- **Population fidelity is the bigger win for the simulator:** reasoning high has the lowest TVD and
  an SD ratio near 1, so the predicted spread is right, not collapsed.
- **Calibration is off:** ECE is above the spec limit for both LLM settings. A per-head temperature
  (planned in D02 Stage C, and in the gate contract) is still needed.
- **Unanswered items:** every Twin-2K-500 question was answered, so the answered-only comparison is
  the same as the one above. Of the 44 Psych-101 questions in this split, 35 have no reasoning-high
  answer (20 thought past 8,000 tokens; the spend cap stopped 15 before they were asked); the harness
  gives those rows a uniform guess, and they are outside the claim
  (`reports/FD-gate/high16k-unanswered.json`). Long, trial-by-trial Psych-101 prompts are a poor fit
  for a reasoning teacher.

## Caveats

1. **Contamination is not ruled out.** Twin-2K-500 has been public since 2025-05-15. The drop against
   B1 is about an eighth of B1's NLL, below the 20 % stop rule but large. The paraphrase probe (score the
   same questions reworded with the M5 paraphrases) must run before D02 relies on this teacher.
2. **28 items.** The intervals are wide; the reasoning-high result is resolved, but small differences
   between candidate teachers will not be.
3. **Reasoning replies are not deterministic** even at temperature 0 (QID74 thought past 8,000 tokens
   once and answered in a few hundred the second time). The cache is the record; a re-run can differ.
4. **M4b is one seed, from an older SHA.** A three-seed re-run at the fork SHA should precede any D02
   contract.

## Spend

0.1251 <!-- src: reports/FD-gate/spend.json#total_billed_usd --> dollars billed by OpenRouter
(provider-reported cost of every cached response, `scripts/fd_spend.py`): reasoning off 0.0025
<!-- src: reports/FD-gate/spend.json#specs.0.billed_usd -->,
reasoning high 0.1225
<!-- src: reports/FD-gate/spend.json#specs.1.billed_usd -->,
most of it on Psych-101 replies that never answered.
