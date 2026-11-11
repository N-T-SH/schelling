# Campaign D04: treatment effects on unseen experiments

**Status:** draft. Not approved. Needs the D02 winner.

## Question

Does the distilled model predict the direction and size of experimental effects on experiments it has
not seen better than M4b does?

## Hypothesis

H1 (can fail). Pooled over the 8 leave-one-experiment-out folds, the distilled model's correlation with
human effect sizes is higher than the M4b re-run's, and sign agreement is above 0.5. M4b is at chance:
correlation 0.0248 <!-- src: reports/M4b-pooled-tt/effects.json#effects.summary.correlation.mean -->,
sign agreement 0.4250 <!-- src: reports/M4b-pooled-tt/effects.json#effects.summary.sign_agreement.mean -->.

Why it might work: classic effects (framing, anchoring, defaults) are described in text an LLM has read;
435 survey questions contain almost none of that. Why it might not: the teacher is asked one condition at
a time and may not be sensitive to small wording changes.

## Metrics

- **Primary:** pooled effect-size correlation, `src/schelling/eval/effects.py::run_effects` and
  `pool_runs`, `effects.json#effects.summary.correlation.mean`.
- **Guards (binding):** false-positive rate on human-null pairs ≤ 0.10 (spec acceptance target). M4b is
  at 0.0917 <!-- src: reports/M4b-pooled-tt/effects.json#effects.summary.false_positive_rate_pairs.mean -->.
- **Recorded:** sign agreement, per-fold NLL against B1, and the same effect metrics for E1 to E3 from D01.

## Data

`heldout_experiment_f0` to `f7`, `val` only. The bank must contain no question that is a near-duplicate
of an item in the held-out experiment of each fold (already ensured by the D02 bank filter, which removes
near-duplicates of every Twin-2K-500 item).

## Budget

- One idea cycle (the D02 winner's recipe, unchanged), 3 seeds × 8 folds for the distilled model and the
  M4b re-run. **[proposal]** 12 hours at fork priority. Paid spend: none (the bank is reused).

## Success, failure and early-stop criteria

- **Yes:** pooled correlation higher than M4b with the paired interval (resampling condition pairs)
  excluding 0, sign agreement above 0.5, false-positive guard met.
- **No:** otherwise. With 412 validation rows per fold the test has little power; a null here says
  "not shown", not "absent", and the write-up must say so.
- **Stop early:** false-positive rate above 0.10 on the pooled result of the first seed.

## Frozen references

As D01, plus the 8 LOEO split manifest hashes (filled at approval).
