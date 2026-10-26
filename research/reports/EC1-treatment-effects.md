# Eval change EC1: treatment-effect metrics and leave-one-experiment-out folds

**Branch:** `eval-change/treatment-effects`. Decisions implemented: `docs/decisions/0002-treatment-effects.md` and `docs/decisions/0003-loeo-folds.md` (authorised to merge by the human on 2026-10-26).

**Result:** with leave-one-experiment-out folds, all 16 contrasts are scored by a model that never saw its experiment. The four baselines, which cannot know any effect, sit at noise level on effect correlation and sign agreement, and their placebo false-positive rate is near the nominal level. On `heldout_person`, where the baselines have seen the arms, they reproduce the effects, which only shows the runner is wired correctly. All runs: validation partition, seed 0, 20 population seeds, code revision c6e6121 with a clean tree; the test partition was never read.

## What changed

- `src/schelling/eval/effects_pairs.yaml`: 16 contrasts in 8 between-subject experiments.
- `src/schelling/eval/effects.py`: `run_effects` (human effect with person-bootstrap interval from the evaluated partition only; synthetic effect per seed; Holm correction within experiment; 20 placebo contrasts per arm item; two false-positive rates) and `pool_runs`.
- `src/schelling/data/splits.py`: splits `heldout_experiment_f0` to `heldout_experiment_f7`; `splits/heldout_experiment_f*.json`. The other split files are unchanged.
- CLI: `schelling effects-pool` pools folds into one `effects.json` with provenance.
- `metrics.json` gains an optional `effects` section (schema regenerated); fixtures gain two disease-frame items. Tests changed only where the sealed surface changed (split-name set, number of files written by `data splits`).

## Results: pooled over the eight folds

16 contrasts, 6 null in humans, 400 placebo contrasts, summed over folds.

| baseline | correlation | sign agreement | FP rate, pairs only (iteration) | FP rate, pairs + placebo |
| --- | --- | --- | --- | --- |
| B0 uniform | -0.129 <!-- src: reports/EC1-pooled-b0/effects.json#effects.summary.correlation.mean --> | 0.435 <!-- src: reports/EC1-pooled-b0/effects.json#effects.summary.sign_agreement.mean --> | 0.025 <!-- src: reports/EC1-pooled-b0/effects.json#effects.summary.false_positive_rate_pairs.mean --> | 0.050 <!-- src: reports/EC1-pooled-b0/effects.json#effects.summary.false_positive_rate_pairs_and_placebo.mean --> |
| B1 item mean | -0.129 <!-- src: reports/EC1-pooled-b1/effects.json#effects.summary.correlation.mean --> | 0.435 <!-- src: reports/EC1-pooled-b1/effects.json#effects.summary.sign_agreement.mean --> | 0.025 <!-- src: reports/EC1-pooled-b1/effects.json#effects.summary.false_positive_rate_pairs.mean --> | 0.050 <!-- src: reports/EC1-pooled-b1/effects.json#effects.summary.false_positive_rate_pairs_and_placebo.mean --> |
| B2 demographic cell | -0.129 <!-- src: reports/EC1-pooled-b2/effects.json#effects.summary.correlation.mean --> | 0.435 <!-- src: reports/EC1-pooled-b2/effects.json#effects.summary.sign_agreement.mean --> | 0.025 <!-- src: reports/EC1-pooled-b2/effects.json#effects.summary.false_positive_rate_pairs.mean --> | 0.050 <!-- src: reports/EC1-pooled-b2/effects.json#effects.summary.false_positive_rate_pairs_and_placebo.mean --> |
| B3 LightGBM | -0.105 <!-- src: reports/EC1-pooled-b3/effects.json#effects.summary.correlation.mean --> | 0.425 <!-- src: reports/EC1-pooled-b3/effects.json#effects.summary.sign_agreement.mean --> | 0.017 <!-- src: reports/EC1-pooled-b3/effects.json#effects.summary.false_positive_rate_pairs.mean --> | 0.049 <!-- src: reports/EC1-pooled-b3/effects.json#effects.summary.false_positive_rate_pairs_and_placebo.mean --> |

Human effects by contrast, with the 95 per cent person-bootstrap interval (identical for every baseline, taken from the B1 pooled file):

| pair | experiment | persons A | persons B | human d | interval low | interval high | null in humans |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E01 | framing_disease | 100 | 106 | 0.43 <!-- src: reports/EC1-pooled-b1/effects.json#effects.pairs.E01.human_d --> | 0.16 <!-- src: reports/EC1-pooled-b1/effects.json#effects.pairs.E01.human_ci.0 --> | 0.70 <!-- src: reports/EC1-pooled-b1/effects.json#effects.pairs.E01.human_ci.1 --> | False |
| E02 | outcome_bias | 110 | 96 | -0.37 <!-- src: reports/EC1-pooled-b1/effects.json#effects.pairs.E02.human_d --> | -0.64 <!-- src: reports/EC1-pooled-b1/effects.json#effects.pairs.E02.human_ci.0 --> | -0.10 <!-- src: reports/EC1-pooled-b1/effects.json#effects.pairs.E02.human_ci.1 --> | False |
| E03 | absolute_relative | 106 | 100 | -0.89 <!-- src: reports/EC1-pooled-b1/effects.json#effects.pairs.E03.human_d --> | -1.16 <!-- src: reports/EC1-pooled-b1/effects.json#effects.pairs.E03.human_ci.0 --> | -0.62 <!-- src: reports/EC1-pooled-b1/effects.json#effects.pairs.E03.human_ci.1 --> | False |
| E04 | myside | 105 | 101 | 0.19 <!-- src: reports/EC1-pooled-b1/effects.json#effects.pairs.E04.human_d --> | -0.08 <!-- src: reports/EC1-pooled-b1/effects.json#effects.pairs.E04.human_ci.0 --> | 0.45 <!-- src: reports/EC1-pooled-b1/effects.json#effects.pairs.E04.human_ci.1 --> | True |
| E05 | less_is_more | 81 | 57 | 0.91 <!-- src: reports/EC1-pooled-b1/effects.json#effects.pairs.E05.human_d --> | 0.56 <!-- src: reports/EC1-pooled-b1/effects.json#effects.pairs.E05.human_ci.0 --> | 1.25 <!-- src: reports/EC1-pooled-b1/effects.json#effects.pairs.E05.human_ci.1 --> | False |
| E06 | less_is_more | 57 | 68 | -0.24 <!-- src: reports/EC1-pooled-b1/effects.json#effects.pairs.E06.human_d --> | -0.59 <!-- src: reports/EC1-pooled-b1/effects.json#effects.pairs.E06.human_ci.0 --> | 0.09 <!-- src: reports/EC1-pooled-b1/effects.json#effects.pairs.E06.human_ci.1 --> | True |
| E07 | less_is_more | 81 | 68 | 0.60 <!-- src: reports/EC1-pooled-b1/effects.json#effects.pairs.E07.human_d --> | 0.28 <!-- src: reports/EC1-pooled-b1/effects.json#effects.pairs.E07.human_ci.0 --> | 0.91 <!-- src: reports/EC1-pooled-b1/effects.json#effects.pairs.E07.human_ci.1 --> | False |
| E08 | proportion_dominance_1 | 81 | 57 | 0.10 <!-- src: reports/EC1-pooled-b1/effects.json#effects.pairs.E08.human_d --> | -0.23 <!-- src: reports/EC1-pooled-b1/effects.json#effects.pairs.E08.human_ci.0 --> | 0.42 <!-- src: reports/EC1-pooled-b1/effects.json#effects.pairs.E08.human_ci.1 --> | True |
| E09 | proportion_dominance_1 | 57 | 68 | 0.25 <!-- src: reports/EC1-pooled-b1/effects.json#effects.pairs.E09.human_d --> | -0.08 <!-- src: reports/EC1-pooled-b1/effects.json#effects.pairs.E09.human_ci.0 --> | 0.59 <!-- src: reports/EC1-pooled-b1/effects.json#effects.pairs.E09.human_ci.1 --> | True |
| E10 | proportion_dominance_1 | 81 | 68 | 0.33 <!-- src: reports/EC1-pooled-b1/effects.json#effects.pairs.E10.human_d --> | 0.01 <!-- src: reports/EC1-pooled-b1/effects.json#effects.pairs.E10.human_ci.0 --> | 0.66 <!-- src: reports/EC1-pooled-b1/effects.json#effects.pairs.E10.human_ci.1 --> | False |
| E11 | proportion_dominance_2 | 81 | 57 | 0.25 <!-- src: reports/EC1-pooled-b1/effects.json#effects.pairs.E11.human_d --> | -0.08 <!-- src: reports/EC1-pooled-b1/effects.json#effects.pairs.E11.human_ci.0 --> | 0.59 <!-- src: reports/EC1-pooled-b1/effects.json#effects.pairs.E11.human_ci.1 --> | True |
| E12 | proportion_dominance_2 | 57 | 68 | 0.13 <!-- src: reports/EC1-pooled-b1/effects.json#effects.pairs.E12.human_d --> | -0.20 <!-- src: reports/EC1-pooled-b1/effects.json#effects.pairs.E12.human_ci.0 --> | 0.49 <!-- src: reports/EC1-pooled-b1/effects.json#effects.pairs.E12.human_ci.1 --> | True |
| E13 | proportion_dominance_2 | 81 | 68 | 0.38 <!-- src: reports/EC1-pooled-b1/effects.json#effects.pairs.E13.human_d --> | 0.04 <!-- src: reports/EC1-pooled-b1/effects.json#effects.pairs.E13.human_ci.0 --> | 0.70 <!-- src: reports/EC1-pooled-b1/effects.json#effects.pairs.E13.human_ci.1 --> | False |
| E14 | wta_wtp | 66 | 74 | 1.44 <!-- src: reports/EC1-pooled-b1/effects.json#effects.pairs.E14.human_d --> | 1.12 <!-- src: reports/EC1-pooled-b1/effects.json#effects.pairs.E14.human_ci.0 --> | 1.75 <!-- src: reports/EC1-pooled-b1/effects.json#effects.pairs.E14.human_ci.1 --> | False |
| E15 | wta_wtp | 66 | 66 | 0.77 <!-- src: reports/EC1-pooled-b1/effects.json#effects.pairs.E15.human_d --> | 0.44 <!-- src: reports/EC1-pooled-b1/effects.json#effects.pairs.E15.human_ci.0 --> | 1.10 <!-- src: reports/EC1-pooled-b1/effects.json#effects.pairs.E15.human_ci.1 --> | False |
| E16 | wta_wtp | 66 | 74 | 2.40 <!-- src: reports/EC1-pooled-b1/effects.json#effects.pairs.E16.human_d --> | 2.08 <!-- src: reports/EC1-pooled-b1/effects.json#effects.pairs.E16.human_ci.0 --> | 2.70 <!-- src: reports/EC1-pooled-b1/effects.json#effects.pairs.E16.human_ci.1 --> | False |

For a model that knows nothing, sign agreement should hover around one half and correlation around zero, which is what the baselines show.

## Results: `heldout_person` (baselines have seen the arms; runner check only)

| baseline | correlation | sign agreement | FP rate, pairs only (iteration) | FP rate, pairs + placebo |
| --- | --- | --- | --- | --- |
| B0 uniform | -0.018 <!-- src: reports/EC1-b0-heldout_person-val/metrics.json#effects.summary.correlation.mean --> | 0.550 <!-- src: reports/EC1-b0-heldout_person-val/metrics.json#effects.summary.sign_agreement.mean --> | 0.017 <!-- src: reports/EC1-b0-heldout_person-val/metrics.json#effects.summary.false_positive_rate_pairs.mean --> | 0.053 <!-- src: reports/EC1-b0-heldout_person-val/metrics.json#effects.summary.false_positive_rate_pairs_and_placebo.mean --> |
| B1 item mean | 0.956 <!-- src: reports/EC1-b1-heldout_person-val/metrics.json#effects.summary.correlation.mean --> | 0.990 <!-- src: reports/EC1-b1-heldout_person-val/metrics.json#effects.summary.sign_agreement.mean --> | 0.150 <!-- src: reports/EC1-b1-heldout_person-val/metrics.json#effects.summary.false_positive_rate_pairs.mean --> | 0.054 <!-- src: reports/EC1-b1-heldout_person-val/metrics.json#effects.summary.false_positive_rate_pairs_and_placebo.mean --> |
| B2 demographic cell | 0.952 <!-- src: reports/EC1-b2-heldout_person-val/metrics.json#effects.summary.correlation.mean --> | 0.985 <!-- src: reports/EC1-b2-heldout_person-val/metrics.json#effects.summary.sign_agreement.mean --> | 0.150 <!-- src: reports/EC1-b2-heldout_person-val/metrics.json#effects.summary.false_positive_rate_pairs.mean --> | 0.051 <!-- src: reports/EC1-b2-heldout_person-val/metrics.json#effects.summary.false_positive_rate_pairs_and_placebo.mean --> |
| B3 LightGBM | 0.955 <!-- src: reports/EC1-b3-heldout_person-val/metrics.json#effects.summary.correlation.mean --> | 0.995 <!-- src: reports/EC1-b3-heldout_person-val/metrics.json#effects.summary.sign_agreement.mean --> | 0.125 <!-- src: reports/EC1-b3-heldout_person-val/metrics.json#effects.summary.false_positive_rate_pairs.mean --> | 0.051 <!-- src: reports/EC1-b3-heldout_person-val/metrics.json#effects.summary.false_positive_rate_pairs_and_placebo.mean --> |

The pairs-only false-positive rate has a small denominator (the number of human-null pairs), so one extra significant pair moves it by a large step; the pairs-plus-placebo version is steadier and sits near the nominal level. The release target of at most one in ten applies to a model that has not seen the arms, so only the pooled table is relevant to it.

## Limitations

- **Power.** Human effects come from the evaluated partition only (a tenth of the persons), so each contrast has about one hundred persons per arm and wide intervals; six of the contrasts are null in humans largely for lack of power.
- **Cost.** A trained model needs eight training runs for a full effect evaluation. The released model is trained on all data; fold models exist only to measure effects (to be stated in the model card).
- **Iteration metric.** The pairs-only false-positive rate (`iteration_metric`), per the human's reading in decision 0002.
- **Direction.** Option order in the catalogue defines the response index; signs are only compared between human and synthetic.
- **Not covered:** free-text and slider outcomes, Linda and Allais (reasons in decision 0002); regression sign agreement is still not run.

## Verification

| Command | Result |
| --- | --- |
| `uv run pytest -m "not slow and not gpu and not network" -q` | 125 passed |
| `uv run ruff check . && uv run ruff format --check .` | all checks passed |
| `uv run mypy src/` | no issues found in 31 source files |
| `uv run schelling data splits` | adds the eight fold files; every other split file is unchanged |
