# Campaign C01: person conditioning on other answers

**Status:** approved by Nitesh on 2026-11-10 (contract approved, then the three confirmations below). Nothing has been run; `SCHELLING_ALLOW_TEST` is not set. The first task is open question 1.

Numbers below carry `<!-- src: path#json.path -->` tags. Paths are relative to the repo root.
Items marked **[proposal]** are my suggested thresholds and need your edit or approval.

## Question

Does conditioning a model on a person's answers to *other* items beat the item-mean baseline B1
on unseen people in Twin-2K-500, without SD-ratio collapse and without making population-level
fidelity worse?

## Hypothesis

H1 (primary, can fail). Adding a history encoder over the person's other Twin-2K-500 answers
(target item masked) to the M4b two-tower model lowers `heldout_person` validation NLL by at
least 0.010 relative to the M4b five-seed mean, with a paired 95 % bootstrap interval over
persons that excludes 0. **[proposal: 0.010 is about 3 seed SDs of M4b and about 1 % of NLL.]**

Why I expect it: M4b's person tower sees only 14 demographic attributes, and its ablation shows
all of its gain over B1 comes from that tower (no-person-tower NLL 0.9991 <!-- src: reports/M4b-train/summary.json#no_person.mean -->
vs full model 0.9892 <!-- src: reports/M4b-train/summary.json#seeds.mean -->). A person's
answers to other items (for example other Likert rows of the same scale) are far more informative
than demographics, so the gain should grow with history size.

H1 is false if the history model's improvement over M4b is below 0.010 or its interval includes 0,
or if it only wins by breaking a guard.

Secondary hypotheses (recorded, not binding; they are the two M4b gaps):

- H2: population TVD and W1 on `heldout_person` val reach B1's level or better (B1: TVD 0.0537
  <!-- src: reports/M2-b1-heldout_person-val/metrics.json#population.twin2k.summary.tvd.mean -->, W1 0.1074
  <!-- src: reports/M2-b1-heldout_person-val/metrics.json#population.twin2k.summary.w1.mean -->;
  M4b seed 0: TVD 0.0573 <!-- src: reports/M4b-tt-a2-heldout_person-val/metrics.json#population.twin2k.summary.tvd.mean -->,
  W1 0.1119 <!-- src: reports/M4b-tt-a2-heldout_person-val/metrics.json#population.twin2k.summary.w1.mean -->).
- H3: model treatment-effect fidelity moves above chance. M4b pooled over the 8 LOEO folds:
  correlation 0.0248 <!-- src: reports/M4b-pooled-tt/effects.json#effects.summary.correlation.mean -->,
  sign agreement 0.4250 <!-- src: reports/M4b-pooled-tt/effects.json#effects.summary.sign_agreement.mean -->.
  I expect little change, because history of *other* items is not what separates experimental
  arms; I include H3 so a null is on the record rather than assumed.

## Metrics

- **Primary:** Twin-2K-500 NLL, computed by `src/schelling/eval/metrics.py::nll`, aggregated per source in
  `src/schelling/eval/report.py::_individual` and read from `metrics.json#individual.by_source.twin2k.nll`.
  Lower is better. Paired comparison: per-person mean NLL, bootstrap over persons (10 000 resamples),
  same persons and rows for every model.
- **Guards (all binding, any idea breaking one is a failure, not a win):**
  - ECE ≤ 0.05 (existing limit; `metrics.py::ece`; path `individual.by_source.twin2k.ece`).
  - SD ratio within [0.85, 1.15] (existing band; `metrics.py::sd_ratio`; path `population.twin2k.summary.sd_ratio.mean`).
  - Option-permutation max |Δp| ≤ 0.01 (`robustness.py::permutation_invariance`; path `robustness.permutation_max_abs_dp`).
  - Person-NLL vs B1: the model must not be worse than B1 on the same rows (decision 0005 row 1).
- **Recorded, not binding (decision 0005 is still a draft):** population TVD
  (`metrics.py::tvd`, path `population.twin2k.summary.tvd.mean`) and W1
  (`metrics.py::wasserstein1`, `...w1.mean`). Reported for every run with the B1 and M4b values beside
  them. If you approve option 1 or 2 of decision 0005 before the final evaluation, TVD becomes binding
  and the success criterion below applies to it; I will not decide that myself.
- **Recorded, not binding:** pooled LOEO treatment-effect correlation, sign agreement and false-positive
  rate (`effects.py::run_effects`, `pool_runs`); the `heldout_person` `effects` block has only 16 pairs
  (6 human-null), so it is too small to carry a claim.
- **Sanity ceiling:** top-choice agreement must not exceed the Twin-2K-500 test-retest ceiling, 0.7149
  <!-- src: reports/M2-b1-retest-val/metrics.json#retest.ceiling -->. Exceeding it is treated as a leak.

## Data

- **Split:** `heldout_person` (Twin-2K-500 rows only); search on `val` only. `test` is never read here.
- **Sources:** Twin-2K-500 only. Psych-101 stays out (decision 0001; its history unit is campaign C04).
- **Context rule (leakage, assumption A04):** when predicting item *j* for a person, item *j*'s answer
  (and its question text) must appear nowhere in the context. Context is built only from the
  `wave1_3_response*.csv`-derived answers with the target masked, never from `persona_text`,
  `persona_json` or `full_persona`.
  Additional rules I propose, because plain masking is not enough in this dataset:
  1. wave-4 repeats of item *j* (a retest of the same question) are also masked;
  2. exact-duplicate question text in another column is masked;
  3. no `val` or `test` person's answers are used in training, and a person's history never includes
     rows from another person.
- **Required leakage tests (written and failing-for-the-right-reason before any idea runs):**
  - extend `tests/unit/test_twin_leakage.py` with sentinel tests for a history-encoder input path: for every
    target, the target's question, options and answer are absent from the encoder input (token ids and cached
    vectors), for varying history size k;
  - a model-level test: change only the target item's answer in the person's raw record and assert the
    prediction for that item is bitwise unchanged;
  - an end-to-end check in the harness path (see open question 1) that the history actually passed to the
    model excludes the target.

## Baselines the winner must beat

All on `heldout_person` val, Twin-2K-500 (`individual.by_source.twin2k.nll`):

| model | NLL | source |
| --- | --- | --- |
| B1 item mean | 0.9976 <!-- src: reports/M2-b1-heldout_person-val/metrics.json#individual.by_source.twin2k.nll --> | must beat (binding) |
| B3 LightGBM | 0.9921 <!-- src: reports/M2-b3-heldout_person-val/metrics.json#individual.by_source.twin2k.nll --> | must beat (binding); M4b did not establish a win over it |
| M4b two-tower (no history), 5-seed mean | 0.9892 <!-- src: reports/M4b-train/summary.json#seeds.mean --> (SD 0.0030 <!-- src: reports/M4b-train/summary.json#seeds.sd -->) | must beat by the H1 margin |

B0 and B2 are re-run in ablations for completeness (B1 and B3 are the binding ones). M4b is re-run at the
frozen SHA with the C01 seeds so the paired bootstrap uses identical persons.

## Budget

- **Idea cycles:** at most 5. Seeds: 3 per search run, 5 per ablation run (seeds 0 to 4; search uses 0 to 2).
- **Compute:** CPU only, on the machine in A12 (Apple M4, 10 cores, 16 GiB, no NVIDIA GPU). Stage A vectors are cached, and
  the history encoder is a small network over cached per-item vectors; the backbone stays frozen, so no GPU
  or paid hardware is needed. Cap: **12 wall-clock hours total for the campaign, 30 minutes per training run
  (timeout, logged as `timeout`)**, at most two concurrent jobs with two threads each (the M4b driver setting).
  **[proposal; see open question 3, because M4b's recorded wall-clock does not give a reliable training time.]**
- **Paid API spend:** none. No Hub push. Ideas needing a paid API or new data go back to you.

## Candidate ideas (not pre-registered; each becomes an `ideas/Ixx` file with a prediction before it runs)

1. **History encoder.** Add a set encoder over the person's other answers (question vector `u_j`, chosen-option
   vector, answer position), target masked, producing a history vector combined with the attribute vector in the
   residual branch.
2. **History-size ablation** (k = 0, 5, 20, 50, all). Tests whether gains come from more evidence or from a few
   informative items, and shows how much history a deployed simulator needs.
3. **Similar-item attention.** Weight history items by question-vector similarity to the target (the same-scale
   items carry most of the signal). Risk: close paraphrases may re-introduce target information; the leakage
   tests cover it.
4. **Population-level loss term.** Add a term that matches the batch-level answer distribution per item (aggregate
   predicted probabilities against the empirical item distribution), aimed at the TVD and W1 gap and at the SD ratio.
5. **Person-latent (collaborative filtering) variant.** Infer a low-dimensional latent per held-out person from
   history by amortised inference (no person-id table, so unseen people still work). A contrast to idea 1, not a
   precursor.

Not planned: any use of `persona_*` fields, person identifiers, test rows, or Psych-101.

## Success, failure and early-stop criteria

- **Yes:** the best idea, run with 5 seeds, has (a) mean `heldout_person` val NLL at least 0.010 below the
  M4b 5-seed mean and below B3, with the paired 95 % bootstrap interval over persons excluding 0 against B1,
  B3 and M4b; (b) all binding guards pass on every seed; (c) the leakage tests pass; (d) the gain is not
  concentrated in a handful of items (checked in Phase 4, reported either way). **[proposal]**
  A "yes" does not require TVD, W1 or treatment effects to improve; those are reported as the secondary
  findings H2 and H3. If you make TVD binding before Phase 2, criterion (b) includes it.
- **No:** after the full budget no idea beats M4b by the margin with an interval excluding 0, or every
  winner breaks a binding guard. A history that does not help is a result worth writing up.
- **Partial:** NLL margin met but a binding guard is broken, or the margin is met only against B1: reported as
  such, and the winner does not go to final evaluation without your decision.
- **Stop early** (write the reason in `research/STATE.md` under Blocked and wait):
  - any leakage test fails, or agreement exceeds the retest ceiling, or NLL drops by more than 20 % in one change
    (AGENTS.md §5); assume a leak until shown otherwise;
  - two consecutive idea cycles fail a binding guard in the same way;
  - the same failure recurs three times after attempted fixes;
  - the 12-hour compute cap is reached;
  - the harness cannot pass history without an evaluator change (open question 1).

## Frozen references

- Evaluator git SHA (`src/schelling/eval/`): `40f458c14f943034ca2c3afba0d2ac6145241ed2` (HEAD of `main` at approval, after the guards change in PR 3).
- Split manifest hash (`heldout_person`): `c77fe1b963008c710d8f9740ad977a54cd69b19e01943ec96095e509e2b00297`
  (`splits/heldout_person.json#manifest_hash`; same value in `split_manifest_hash` of the B1, B3 and M4b `metrics.json` files).
- Data manifest hashes (SHA-256 of the files in `data/manifests/`):
  - `twin2k.json`: `6d0f099d415efeabef3853ec8af311f354f89c975c02a5a7d5985afc6efabcec`
  - `psych101.json`: `3fd612ac3067cb2ed47e83c95219b5aedef20e28ed7a3fd5bc4535902b03087d` (not used in C01, recorded for completeness)
- Sealed during the campaign: `src/schelling/eval/`, `tests/golden/`, `splits/`, `data/manifests/`. Any evaluator change goes through an `eval-change/<topic>` branch and your approval.

## Confirmed at approval (2026-11-10)

Nitesh approved the contract and confirmed, in chat, three things:

1. The **[proposal]** items are accepted as written: the 0.010 NLL margin against the M4b mean, masking wave-4 repeats and duplicate questions, treatment effects recorded-only (H3), and TVD and W1 recorded-only (decision 0005, option 3, merged in PR 3).
2. The compute cap is 12 hours in total and 30 minutes per run. The large `wall_clock_s` in `reports/M4b-train/tt-s1/metrics.json` is elapsed time including about six days paused with SIGSTOP; an unpaused seed-0 run took 815 s <!-- src: reports/M4b-train/tt-a2/metrics.json#wall_clock_s -->.
3. The first task is to verify that answer history can reach the model without changing `src/schelling/eval/`. If it cannot, stop and write it under Blocked; an `eval-change/person-history` branch would need its own approval.

The provenance gap in question 7 is fixed by PR 4 (`reports/M4b-train/summary.json` tracked).

## Final-evaluation commands (filled before Phase 3, run by the human)

Not prepared yet: filled after Phase 2 with the frozen winner, seeds and config hashes.

```bash
SCHELLING_ALLOW_TEST=1 uv run schelling eval --model <path> --split heldout_person --partition test --seed <n> --out reports/<run_id>
```
