# Campaign D02: distilled prior on unseen questions

**Status:** draft. Not approved. Blocked on the D01 gate. Items marked **[proposal]** need your edit or approval.

## Question

Does pretraining the two-tower model on calibrated teacher labels for a broad synthetic question bank,
then fine-tuning on human answers, lower `heldout_item` validation NLL relative to M4b, without
breaking a guard?

## Hypothesis

H1 (primary, can fail). The distilled model's Twin-2K-500 `heldout_item` val NLL is lower than the M4b
re-run's by at least three M4b seed SDs **[proposal; the seed SD on this split is not measured yet, so
task 1 measures it and the margin is fixed in numbers before any distilled run]**, with the paired 95 %
item-bootstrap interval excluding 0.

Why I expect it: M4b already beats B1 on unseen questions using text alone (1.2939
<!-- src: reports/M4b-tt-item-val/metrics.json#individual.by_source.twin2k.nll --> against 1.3534
<!-- src: reports/M4b-b1-heldout_item-val/metrics.json#individual.by_source.twin2k.nll -->, one seed)
after seeing only 435 questions. Thousands of labelled questions should help the question-to-prior path
most.

H2 (secondary). The gain grows with bank size (2 k < 10 k < 50 k questions).

H1 is false if the margin is missed, the interval includes 0, or the win needs a broken guard.

## Metrics

- **Primary:** Twin-2K-500 NLL on `heldout_item` val, `src/schelling/eval/metrics.py::nll`,
  `metrics.json#individual.by_source.twin2k.nll`. Paired by item, 10 000 resamples.
- **Guards (binding):**
  - ECE ≤ 0.05 on `heldout_person` val. On `heldout_item` val M4b itself is at 0.0582
    <!-- src: reports/M4b-tt-item-val/metrics.json#individual.by_source.twin2k.ece -->, so there the
    guard is "not worse than the M4b re-run" **[proposal]**.
  - SD ratio within [0.85, 1.15] on both splits.
  - Option-permutation max |Δp| ≤ 0.01.
  - `heldout_person` val NLL not worse than the M4b re-run (interval for "worse" must include 0) and not
    worse than B1.
- **Recorded:** population TVD and W1, entropy ratio, mode-collapse rate, per-question-type NLL.
- **Sanity ceiling:** top-choice agreement must not exceed the test-retest ceiling 0.7149
  <!-- src: reports/M2-b1-retest-val/metrics.json#retest.ceiling -->.

## Data

- **Splits:** `heldout_item` (primary) and `heldout_person` (guard); `val` only.
- **Human training data:** the split's `train` partition, as M4b.
- **Synthetic bank:** `data/synth/bank-<generator>-v1.jsonl`, git-ignored, hash recorded in every run.
- **Leakage rules:**
  1. bank questions that are near-duplicates (MinHash, Jaccard ≥ 0.8) of any Twin-2K-500 or Psych-101
     question are removed (needs your answer to DESIGN open question 1);
  2. personas for teacher labels are sampled from `train` persons only;
  3. teacher calibration temperature is fitted on `train` items only;
  4. M5 paraphrases of `heldout_item` val or test items are never used in training.
- **Required tests, failing for the right reason before any run:** bank filter removes a planted
  near-duplicate of a held-out item; pretraining off gives bit-identical M4b training; soft-label loss
  equals the hand-computed cross-entropy on a fixture; persona sampler never returns a val or test person.

## Baselines the winner must beat

| model | `heldout_item` val NLL (Twin-2K-500) | status |
| --- | --- | --- |
| B1 | 1.3534 <!-- src: reports/M4b-b1-heldout_item-val/metrics.json#individual.by_source.twin2k.nll --> | binding |
| M4b two-tower | 1.2939 <!-- src: reports/M4b-tt-item-val/metrics.json#individual.by_source.twin2k.nll --> (one seed, older SHA) | binding by the H1 margin, after a 3-seed re-run at the fork SHA |
| E3 teacher, calibrated | from D01 | reported; beating the teacher is the interesting result |

## Budget

- **Idea cycles:** at most 5. Seeds: 3 in search, 5 in ablations.
- **Compute:** CPU only (A12), backbone frozen, vectors cached. **[proposal]** 16 wall-clock hours for
  the campaign at fork priority (one job, two threads, paused while a mainline job runs).
- **Paid API spend:** **[proposal]** $30 cap for bank generation and labels on `gpt-oss-120b`, replaced
  by the dry-run figure before any call.

## Candidate ideas (each pre-registered in `ideas/` before it runs)

1. **Pretrain then fine-tune** on the 10 k bank with calibrated teacher labels (the main design).
2. **Teacher calibration ablation:** raw teacher labels against temperature-mapped ones.
3. **Prior-path-only distillation:** teacher labels train the question-to-prior path; the person residual
   learns from human data only.
4. **Bank size:** 2 k, 10 k, 50 k (H2).
5. **Persona-conditioned labels** (K = 8 per question) against population labels only.

Not planned here: unfreezing the encoder (needs a compute decision), closed-model teachers (FA3).

## Success, failure and early-stop criteria

- **Yes:** H1 holds with 5 seeds, every binding guard passes on every seed, and the leakage tests pass.
- **No:** no idea meets the margin after the budget. Written up as a finding.
- **Partial:** margin met but a guard broken, or `heldout_person` regresses. Reported; no further step
  without your decision.
- **Stop early:** any leakage test fails; NLL drops more than 20 % in one change; agreement exceeds the
  retest ceiling; SD ratio leaves the band in two consecutive ideas; a budget is reached.

## Frozen references

Same evaluator SHA, split and data manifest hashes as D01; re-checked and written here at approval.

## Final-evaluation commands

Not prepared. The fork's single test evaluation happens in D05.
