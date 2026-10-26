# 0002: Treatment-effect metric, pairing table and heldout_experiment split

**Date:** 2026-10-26 · **Status:** approved (Nitesh, 2026-10-26), implemented on `eval-change/treatment-effects`, awaiting merge approval

## Context

SPEC level-2 evaluation requires effect-size correlation, sign agreement and a false-positive rate for effects that are null in humans, but no table says which arms to contrast. Draft: `research/drafts/effects-pairing-table.md`. All 16 retained contrasts are between-subject (zero overlap of persons across arms, checked on `wave1_3_response.csv`) and the arms of each experiment share an identical ordered option list (checked on `question_catalog.json`).

## Decision (your answers, plus how I applied "do what is more rigorous")

1. **Human reference effect:** estimated only from the evaluated partition's persons, with a 95% person-bootstrap interval within arm. Consequence: low power (a validation partition has about a tenth of the persons), so many human effects will look null; the table below shows how many.
2. **New split `heldout_experiment`** (eval-change branch): both arms of every contrast land in the same partition. Only 8 experiments exist, so 10% would be less than one; I hold out 2 for `val` and 2 for `test`, chosen by seeded hash rank, and everything else (including all Psych-101) is `train`. This is a deliberate departure from the 10% convention and means effect metrics rest on few experiments.
3. **Pairs (rigour choices):** dropped E03 Allais (the two forms have different option sets, so an effect size is ill-defined), the two anchoring manipulation checks (the outcome is a trivial function of the anchor and would inflate sign agreement), and Linda (no principled arm-to-row pairing; I will not choose one arbitrarily). Dropped the "literature prior" column (unverified and unused by any metric). Kept all pairwise contrasts for multi-arm experiments, with Holm correction within experiment for the model's significance test. Effects are standardised with the human pooled SD for both human and synthetic estimates, so variance collapse shows up as a smaller synthetic effect.
4. **False-positive rate, two versions, both reported:** `false_positive_rate_pairs` (over human-null pairs only) and `false_positive_rate_pairs_and_placebo` (human-null pairs plus 20 hash-split placebo contrasts per arm item). **I read "use latter for iteration" as the pairs-only version** (`iteration_metric` in `metrics.json`); say if you meant the combined one, it is a one-line change.

## Consequences

- New files: `src/schelling/eval/effects_pairs.yaml`, extended `effects.py`; `splits.py` gets `heldout_experiment`; `splits/heldout_experiment.json`; `metrics.json` gains an optional `effects` section (schema regenerated).
- Tests changed because the sealed surface changed: the expected set of split names and the number of files `data splits` writes.
- Fixtures gain two disease-frame items so the runner is exercised on fixtures.
- `scipy` becomes a direct dependency (Welch test, percentiles).
- B1 to B3 cannot predict effects on `heldout_experiment` by construction (they have never seen either arm); on `heldout_person` they reproduce effects because they saw the arms. Both are reported as sanity checks of the runner, not as results about a model.
