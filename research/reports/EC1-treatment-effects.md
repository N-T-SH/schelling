# Eval change EC1: treatment-effect metrics and `heldout_experiment` split

**Branch:** `eval-change/treatment-effects` (not merged). **Needs your approval to merge**, because it adds to the sealed evaluator (`src/schelling/eval/`, `splits/`). Decisions implemented: `docs/decisions/0002-treatment-effects.md`.

**Result:** the runner works and behaves sensibly on the four baselines. On `heldout_experiment`, where no baseline has seen either arm, effect correlation and sign agreement sit at noise level, as they should. On `heldout_person`, where the baselines have seen the arms, they reproduce the effects almost perfectly, which only shows the runner is wired correctly. Placebo contrasts are significant about as often as the nominal level, so the significance test is calibrated. All runs: validation partition, seed 0, 20 population seeds, code revision 4c7093f with a clean tree; the test partition was never read.

## What changed

- `src/schelling/eval/effects_pairs.yaml`: 16 contrasts in 8 between-subject experiments.
- `src/schelling/eval/effects.py`: `run_effects` (human effect with person-bootstrap interval from the evaluated partition only; synthetic effect per population seed; Holm correction within experiment; 20 placebo contrasts per arm item; two false-positive rates).
- `src/schelling/data/splits.py`: new split `heldout_experiment` (two experiments in `val`, two in `test`, rest `train`); `splits/heldout_experiment.json`. The other six split files are byte-identical to before.
- `metrics.json` gains an optional `effects` section; `schemas/metrics.json` regenerated; `scripts/make_fixtures.py` gains two disease-frame items. Tests changed only where the sealed surface changed (split-name set, number of files written by `data splits`).

## Results: `heldout_experiment` (no baseline has seen either arm)

The validation partition holds 4 contrasts (two experiments), of which 1 is null in humans, plus 100 placebo contrasts.

| baseline | correlation | sign agreement | FP rate, pairs only (iteration) | FP rate, pairs + placebo |
| --- | --- | --- | --- | --- |
| B0 uniform | -0.208 <!-- src: reports/EC1-b0-heldout_experiment-val/metrics.json#effects.summary.correlation.mean --> | 0.400 <!-- src: reports/EC1-b0-heldout_experiment-val/metrics.json#effects.summary.sign_agreement.mean --> | 0.000 <!-- src: reports/EC1-b0-heldout_experiment-val/metrics.json#effects.summary.false_positive_rate_pairs.mean --> | 0.052 <!-- src: reports/EC1-b0-heldout_experiment-val/metrics.json#effects.summary.false_positive_rate_pairs_and_placebo.mean --> |
| B1 item mean | -0.208 <!-- src: reports/EC1-b1-heldout_experiment-val/metrics.json#effects.summary.correlation.mean --> | 0.400 <!-- src: reports/EC1-b1-heldout_experiment-val/metrics.json#effects.summary.sign_agreement.mean --> | 0.000 <!-- src: reports/EC1-b1-heldout_experiment-val/metrics.json#effects.summary.false_positive_rate_pairs.mean --> | 0.052 <!-- src: reports/EC1-b1-heldout_experiment-val/metrics.json#effects.summary.false_positive_rate_pairs_and_placebo.mean --> |
| B2 demographic cell | -0.208 <!-- src: reports/EC1-b2-heldout_experiment-val/metrics.json#effects.summary.correlation.mean --> | 0.400 <!-- src: reports/EC1-b2-heldout_experiment-val/metrics.json#effects.summary.sign_agreement.mean --> | 0.000 <!-- src: reports/EC1-b2-heldout_experiment-val/metrics.json#effects.summary.false_positive_rate_pairs.mean --> | 0.052 <!-- src: reports/EC1-b2-heldout_experiment-val/metrics.json#effects.summary.false_positive_rate_pairs_and_placebo.mean --> |
| B3 LightGBM | -0.237 <!-- src: reports/EC1-b3-heldout_experiment-val/metrics.json#effects.summary.correlation.mean --> | 0.417 <!-- src: reports/EC1-b3-heldout_experiment-val/metrics.json#effects.summary.sign_agreement.mean --> | 0.000 <!-- src: reports/EC1-b3-heldout_experiment-val/metrics.json#effects.summary.false_positive_rate_pairs.mean --> | 0.049 <!-- src: reports/EC1-b3-heldout_experiment-val/metrics.json#effects.summary.false_positive_rate_pairs_and_placebo.mean --> |

Human effects in this partition (from the B1 run; the human side is identical for every baseline):

| pair | experiment | persons A | persons B | human d | 95 per cent interval low | 95 per cent interval high | null in humans |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E04 | myside | 1015 | 1043 | 0.26 <!-- src: reports/EC1-b1-heldout_experiment-val/metrics.json#effects.pairs.E04.human_d --> | 0.17 <!-- src: reports/EC1-b1-heldout_experiment-val/metrics.json#effects.pairs.E04.human_ci.0 --> | 0.35 <!-- src: reports/EC1-b1-heldout_experiment-val/metrics.json#effects.pairs.E04.human_ci.1 --> | False |
| E05 | less_is_more | 735 | 672 | 0.63 <!-- src: reports/EC1-b1-heldout_experiment-val/metrics.json#effects.pairs.E05.human_d --> | 0.53 <!-- src: reports/EC1-b1-heldout_experiment-val/metrics.json#effects.pairs.E05.human_ci.0 --> | 0.73 <!-- src: reports/EC1-b1-heldout_experiment-val/metrics.json#effects.pairs.E05.human_ci.1 --> | False |
| E06 | less_is_more | 672 | 651 | -0.02 <!-- src: reports/EC1-b1-heldout_experiment-val/metrics.json#effects.pairs.E06.human_d --> | -0.13 <!-- src: reports/EC1-b1-heldout_experiment-val/metrics.json#effects.pairs.E06.human_ci.0 --> | 0.09 <!-- src: reports/EC1-b1-heldout_experiment-val/metrics.json#effects.pairs.E06.human_ci.1 --> | True |
| E07 | less_is_more | 735 | 651 | 0.63 <!-- src: reports/EC1-b1-heldout_experiment-val/metrics.json#effects.pairs.E07.human_d --> | 0.53 <!-- src: reports/EC1-b1-heldout_experiment-val/metrics.json#effects.pairs.E07.human_ci.0 --> | 0.73 <!-- src: reports/EC1-b1-heldout_experiment-val/metrics.json#effects.pairs.E07.human_ci.1 --> | False |

With so few contrasts the correlation and sign-agreement values are noisy: for a model that knows nothing, sign agreement should hover around one half. Do not read anything into differences between baselines here.

## Results: `heldout_person` (baselines have seen the arms; runner check only)

All 16 contrasts are in the partition, 6 null in humans, 400 placebo contrasts.

| baseline | correlation | sign agreement | FP rate, pairs only (iteration) | FP rate, pairs + placebo |
| --- | --- | --- | --- | --- |
| B0 uniform | -0.018 <!-- src: reports/EC1-b0-heldout_person-val/metrics.json#effects.summary.correlation.mean --> | 0.550 <!-- src: reports/EC1-b0-heldout_person-val/metrics.json#effects.summary.sign_agreement.mean --> | 0.017 <!-- src: reports/EC1-b0-heldout_person-val/metrics.json#effects.summary.false_positive_rate_pairs.mean --> | 0.053 <!-- src: reports/EC1-b0-heldout_person-val/metrics.json#effects.summary.false_positive_rate_pairs_and_placebo.mean --> |
| B1 item mean | 0.956 <!-- src: reports/EC1-b1-heldout_person-val/metrics.json#effects.summary.correlation.mean --> | 0.990 <!-- src: reports/EC1-b1-heldout_person-val/metrics.json#effects.summary.sign_agreement.mean --> | 0.150 <!-- src: reports/EC1-b1-heldout_person-val/metrics.json#effects.summary.false_positive_rate_pairs.mean --> | 0.054 <!-- src: reports/EC1-b1-heldout_person-val/metrics.json#effects.summary.false_positive_rate_pairs_and_placebo.mean --> |
| B2 demographic cell | 0.952 <!-- src: reports/EC1-b2-heldout_person-val/metrics.json#effects.summary.correlation.mean --> | 0.985 <!-- src: reports/EC1-b2-heldout_person-val/metrics.json#effects.summary.sign_agreement.mean --> | 0.150 <!-- src: reports/EC1-b2-heldout_person-val/metrics.json#effects.summary.false_positive_rate_pairs.mean --> | 0.051 <!-- src: reports/EC1-b2-heldout_person-val/metrics.json#effects.summary.false_positive_rate_pairs_and_placebo.mean --> |
| B3 LightGBM | 0.955 <!-- src: reports/EC1-b3-heldout_person-val/metrics.json#effects.summary.correlation.mean --> | 0.995 <!-- src: reports/EC1-b3-heldout_person-val/metrics.json#effects.summary.sign_agreement.mean --> | 0.125 <!-- src: reports/EC1-b3-heldout_person-val/metrics.json#effects.summary.false_positive_rate_pairs.mean --> | 0.051 <!-- src: reports/EC1-b3-heldout_person-val/metrics.json#effects.summary.false_positive_rate_pairs_and_placebo.mean --> |

The pairs-only false-positive rate has a very small denominator (the number of human-null pairs above), so one extra significant pair moves it by a large step; the pairs-plus-placebo version is steadier and sits near the nominal level for every baseline. The release target for the false-positive rate is at most one in ten on a model that has not seen the arms, so only the `heldout_experiment` rows are relevant to it.

## Limitations and decisions to check

- **Power.** Human effects come from the evaluated partition only (your choice), so the intervals are wide and the experiments in `val` are few. Effect metrics will be descriptive until the model is large enough to make a difference visible; they should not be used as a headline claim without the test partition and a bootstrap over experiments.
- **Four experiments in the final test.** `heldout_experiment` test has two experiments, so the test-set effect metrics will rest on very few contrasts. If that is too thin, the alternative is leave-one-experiment-out folds over all eight, at the cost of training a model per fold; that would be a further eval change.
- **Iteration metric.** I read "use latter" as the pairs-only false-positive rate (`iteration_metric` in `metrics.json`). Tell me if you meant the combined one.
- **Direction of effects.** Option order in the catalogue defines the response index, so a positive effect means "later option in the list". Signs are only compared between human and synthetic, never interpreted.
- **Standardisation.** Synthetic effects use the human pooled SD, so a model that collapses variance shows a shrunken effect, not an inflated one.
- **Not covered:** free-text and slider outcomes, the Linda conjunction items and the Allais pair (reasons in decision 0002); regression sign agreement is still not run.
- **Split metadata.** The new split's manifest records experiment counts rather than fractions, because ten per cent of eight experiments is less than one.

## Verification

| Command | Result |
| --- | --- |
| `uv run pytest -m "not slow and not gpu and not network" -q` | 121 passed |
| `uv run ruff check . && uv run ruff format --check .` | all checks passed |
| `uv run mypy src/` | no issues found in 31 source files |
| `uv run schelling data splits` | only `splits/heldout_experiment.json` is new; every other split file is unchanged |

## Approval needed

Merge `eval-change/treatment-effects` into `main`? If yes, the evaluator stays sealed afterwards with these additions included.
