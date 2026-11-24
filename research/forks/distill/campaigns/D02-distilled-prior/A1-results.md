# A1 probe: results

**Answer: A1 passes the pre-registered rule.** On 282 unseen Twin-2K-500 questions (val of the five
item folds), A1 lowers per-question NLL against A0 by
-0.0920 <!-- src: reports/FD-a1/summary.json#pairs.a1_vs_a0.mean_diff -->
(95 % interval -0.1134 <!-- src: reports/FD-a1/summary.json#pairs.a1_vs_a0.ci_low --> to
-0.0718 <!-- src: reports/FD-a1/summary.json#pairs.a1_vs_a0.ci_high -->), a relative change of
-7.0% <!-- src: reports/FD-a1/summary.json#pairs.a1_vs_a0.relative_change -->, and passes all three
guards. Under the rule, A1 becomes the default design for unseen questions, pending the
`heldout_person` no-regression check. But A1 is not clearly better than the teacher alone (below).

Pre-registration: `A1-probe.md` (above its line, unchanged). Every run: git SHA e9191c9, clean
(`any_dirty` false for all four arms in `reports/FD-a1/summary.json`). Pooling script:
`scripts/fd_pooled.py`, 10,000 item-bootstrap resamples, seed 0.

## Primary comparison and references

Mean per-question NLL (seed-averaged for A0 and A1, pooled over folds, 282 questions):

| Arm | NLL | vs A1 (A1 − arm) | 95 % interval | Questions where A1 is better |
|---|---|---|---|---|
| A1 | 1.2278 <!-- src: reports/FD-a1/summary.json#arms.a1.item_mean_nll --> | | | |
| A0 (M4b) | 1.3198 <!-- src: reports/FD-a1/summary.json#arms.a0.item_mean_nll --> | -0.0920 <!-- src: reports/FD-a1/summary.json#pairs.a1_vs_a0.mean_diff --> | -0.1134 <!-- src: reports/FD-a1/summary.json#pairs.a1_vs_a0.ci_low --> to -0.0718 <!-- src: reports/FD-a1/summary.json#pairs.a1_vs_a0.ci_high --> | 212 <!-- src: reports/FD-a1/summary.json#pairs.a1_vs_a0.items_a_better --> |
| Teacher alone | 1.2384 <!-- src: reports/FD-a1/summary.json#arms.teacher.item_mean_nll --> | -0.0106 <!-- src: reports/FD-a1/summary.json#pairs.a1_vs_teacher.mean_diff --> | -0.0256 <!-- src: reports/FD-a1/summary.json#pairs.a1_vs_teacher.ci_low --> to 0.0036 <!-- src: reports/FD-a1/summary.json#pairs.a1_vs_teacher.ci_high --> | 172 <!-- src: reports/FD-a1/summary.json#pairs.a1_vs_teacher.items_a_better --> |
| B1 item mean | 1.4037 <!-- src: reports/FD-a1/summary.json#arms.b1.item_mean_nll --> | -0.1759 <!-- src: reports/FD-a1/summary.json#pairs.a1_vs_b1.mean_diff --> | -0.1997 <!-- src: reports/FD-a1/summary.json#pairs.a1_vs_b1.ci_low --> to -0.1523 <!-- src: reports/FD-a1/summary.json#pairs.a1_vs_b1.ci_high --> | 235 <!-- src: reports/FD-a1/summary.json#pairs.a1_vs_b1.items_a_better --> |

- **Too-good check:** the drop is well under the 20 % stop line, so no leak hunt was triggered.
- **A1 vs teacher alone:** the interval includes 0. On unseen questions, the gain over M4b comes
  almost entirely from the teacher's answer; the person tower and the learned correction add little
  to pooled NLL. This was not required to pass, but it is the main caveat.

## Guards (mean over the 15 runs per arm)

| Guard | Limit | A1 | A0 |
|---|---|---|---|
| Permutation max abs Δp | `≤ 0.01` | `1.5e-07` | `7.9e-08` |
| SD ratio | `0.85 to 1.15` | 1.0386 <!-- src: reports/FD-a1/summary.json#arms.a1.sd_ratio.mean --> | 1.0573 <!-- src: reports/FD-a1/summary.json#arms.a0.sd_ratio.mean --> |
| ECE | `≤ 0.05`, or not above A0 | 0.0404 <!-- src: reports/FD-a1/summary.json#arms.a1.ece.mean --> | 0.0429 <!-- src: reports/FD-a1/summary.json#arms.a0.ece.mean --> |

- Permutation values are shown in scientific notation, which the audit does not parse; both are in
  `arms.{a1,a0}.permutation_max_abs_dp.mean` of `reports/FD-a1/summary.json`.
- The SD ratio is the harness's population summary (`population.twin2k.summary.sd_ratio.mean`), the
  field the sealed guard (`src/schelling/eval/guards.py`) reads for "median SD ratio".
- ECE varies more across A1 runs: the worst A1 run reaches
  0.0668 <!-- src: reports/FD-a1/summary.json#arms.a1.ece.max --> (A0's worst:
  0.0505 <!-- src: reports/FD-a1/summary.json#arms.a0.ece.max -->). The rule reads the mean, which
  passes; the spread is worth watching in the `heldout_person` check.
- The teacher alone is less well calibrated: ECE
  0.0621 <!-- src: reports/FD-a1/summary.json#arms.teacher.ece.mean --> over the five folds. A1 keeps
  the teacher's accuracy while bringing calibration back within the limit.
- Population fit also improves: mean TVD
  0.1163 <!-- src: reports/FD-a1/summary.json#arms.a1.tvd.mean --> for A1 against
  0.1658 <!-- src: reports/FD-a1/summary.json#arms.a0.tvd.mean --> for A0 and
  0.1260 <!-- src: reports/FD-a1/summary.json#arms.teacher.tvd.mean --> for the teacher.

## Per fold (harness row NLL, seed mean)

| Fold | A0 | A1 | A1 − A0 | Teacher | B1 |
|---|---|---|---|---|---|
| 0 | 1.3860 <!-- src: reports/FD-a1-folds/summary.json#folds.0.a0.mean --> | 1.2556 <!-- src: reports/FD-a1-folds/summary.json#folds.0.a1.mean --> | -0.1304 <!-- src: reports/FD-a1-folds/summary.json#folds.0.a1_minus_a0 --> | 1.2349 <!-- src: reports/FD-a1-folds/summary.json#folds.0.teacher.mean --> | 1.4219 <!-- src: reports/FD-a1-folds/summary.json#folds.0.b1.mean --> |
| 1 | 1.2294 <!-- src: reports/FD-a1-folds/summary.json#folds.1.a0.mean --> | 1.1162 <!-- src: reports/FD-a1-folds/summary.json#folds.1.a1.mean --> | -0.1132 <!-- src: reports/FD-a1-folds/summary.json#folds.1.a1_minus_a0 --> | 1.1439 <!-- src: reports/FD-a1-folds/summary.json#folds.1.teacher.mean --> | 1.3443 <!-- src: reports/FD-a1-folds/summary.json#folds.1.b1.mean --> |
| 2 | 1.2156 <!-- src: reports/FD-a1-folds/summary.json#folds.2.a0.mean --> | 1.1654 <!-- src: reports/FD-a1-folds/summary.json#folds.2.a1.mean --> | -0.0503 <!-- src: reports/FD-a1-folds/summary.json#folds.2.a1_minus_a0 --> | 1.1999 <!-- src: reports/FD-a1-folds/summary.json#folds.2.teacher.mean --> | 1.2824 <!-- src: reports/FD-a1-folds/summary.json#folds.2.b1.mean --> |
| 3 | 1.2584 <!-- src: reports/FD-a1-folds/summary.json#folds.3.a0.mean --> | 1.2257 <!-- src: reports/FD-a1-folds/summary.json#folds.3.a1.mean --> | -0.0327 <!-- src: reports/FD-a1-folds/summary.json#folds.3.a1_minus_a0 --> | 1.2594 <!-- src: reports/FD-a1-folds/summary.json#folds.3.teacher.mean --> | 1.3506 <!-- src: reports/FD-a1-folds/summary.json#folds.3.b1.mean --> |
| 4 | 1.2397 <!-- src: reports/FD-a1-folds/summary.json#folds.4.a0.mean --> | 1.1807 <!-- src: reports/FD-a1-folds/summary.json#folds.4.a1.mean --> | -0.0589 <!-- src: reports/FD-a1-folds/summary.json#folds.4.a1_minus_a0 --> | 1.2088 <!-- src: reports/FD-a1-folds/summary.json#folds.4.teacher.mean --> | 1.3061 <!-- src: reports/FD-a1-folds/summary.json#folds.4.b1.mean --> |

A1 beats A0 on every fold. It beats the teacher alone on folds 1 to 4 but not on fold 0. Largest
seed SD within a fold: A0 0.0115 <!-- src: reports/FD-a1-folds/summary.json#max_seed_sd.a0 -->,
A1 0.0056 <!-- src: reports/FD-a1-folds/summary.json#max_seed_sd.a1 -->. Fold rows weight
questions by their number of answers, so they differ from the per-question pooled means above.

## Learned teacher weights (start 1)

Mean over the 15 runs, with the range:

- yes/no: 1.0257 <!-- src: reports/FD-a1-weights/summary.json#mean.0 -->
  (1.0036 <!-- src: reports/FD-a1-weights/summary.json#min.0 --> to
  1.0498 <!-- src: reports/FD-a1-weights/summary.json#max.0 -->)
- choice: 0.8593 <!-- src: reports/FD-a1-weights/summary.json#mean.1 -->
  (0.8109 <!-- src: reports/FD-a1-weights/summary.json#min.1 --> to
  0.9350 <!-- src: reports/FD-a1-weights/summary.json#max.1 -->)
- score: 0.9304 <!-- src: reports/FD-a1-weights/summary.json#mean.2 -->
  (0.9068 <!-- src: reports/FD-a1-weights/summary.json#min.2 --> to
  0.9684 <!-- src: reports/FD-a1-weights/summary.json#max.2 -->)

Training keeps the teacher at close to full strength and slightly damps it for choice and score
questions. The trainer found a teacher answer for 279 of 281 distinct val questions (as it counts
them, by question text and options).

## What this does and does not show

- It shows that the cached teacher answer, added to M4b's prior, makes M4b clearly better on
  questions it has never seen, without breaking calibration, spread or option shuffling.
- It does not show that M4b adds much on top of the teacher for unseen questions.
- It does not rule out that the teacher knows Twin-2K-500 answers (D01b limit). The teacher alone
  scoring close to A1 makes this limit matter more, not less.
- It says nothing yet about seen questions: the `heldout_person` check is next.
- Test was not touched.

## Files

- `reports/FD-a1/summary.json`: the pooled comparison, guards and per-question values.
- `reports/FD-a1-folds/summary.json`: per-fold means and seed SDs, derived from the file above.
- `reports/FD-a1-weights/summary.json`: learned weights and coverage, from `runs/fd-a1-f*-s*/run.json`.
- `reports/FD-a{0,1}-f*-s*-val/metrics.json`, `reports/FD-a1-b1-f*-val/metrics.json`: harness outputs.
  The teacher-alone `metrics.json` files are not committed because they embed a local absolute
  path; their numbers are carried in `reports/FD-a1/summary.json`.
- Absolute path prefixes were stripped from the two summary files after the run (values unchanged).
