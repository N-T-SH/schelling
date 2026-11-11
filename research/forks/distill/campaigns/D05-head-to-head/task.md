# Campaign D05: head to head against the frontier model

**Status:** draft. Not approved. Needs D01, D02 and D03. This is the fork's only use of the test partition.

## Question

Is the distilled model at least as good as the frontier model (E1/E2) at predicting people, at a small
fraction of the cost per decision?

## Hypothesis

H1 (can fail). On Twin-2K-500:

1. `heldout_item`: distilled NLL is lower than the best calibrated frontier variant, paired
   item-bootstrap interval excluding 0;
2. `heldout_person`: distilled NLL is lower than the frontier model's, paired person-bootstrap interval
   excluding 0;
3. population TVD is not worse than the frontier model's on either split, and the SD ratio is inside
   [0.85, 1.15];
4. cost per million decisions is at least 100 times lower **[proposal]**.

The claim "outperforms a frontier model" needs 1 to 4. Meeting only 2 to 4 is the weaker claim "matches
it where the questions are known, and is far cheaper".

## Metrics

- **Primary:** Twin-2K-500 NLL, `src/schelling/eval/metrics.py::nll`, per split.
- **Guards (binding):** ECE, SD ratio, permutation invariance as in D02; false-positive rate ≤ 0.10.
- **Cost:** frontier dollars per million decisions from the D01 usage log; Schelling decisions per second
  from the harness `efficiency` block, converted with a stated machine cost. Latency p50 and p95 for both.
- **Fairness record:** the frozen prompt, elicitation form and temperature chosen in D01 are used
  unchanged. No new prompt tuning after seeing D02 results.

## Data

`heldout_person` and `heldout_item`; search and ablations on `val`. Persona comparisons use the D01
persona subsample. Test partition: once, by the human.

## Budget

- No idea cycles. 5 seeds of the frozen distilled model. Paid spend on `val`: none new (D01 cache).
- **Test evaluation needs new paid calls** for the frontier model on test rows. The dry-run cost is shown
  to you first; you run it.

## Success, failure and early-stop criteria

- **Yes:** H1 parts 1 to 4 on `val`, then confirmed on `test`.
- **Partial:** parts 2 to 4 only. Reported as the weaker claim.
- **No:** the frontier model is better on NLL in both regimes, or the distilled model needs a broken guard.
- **Stop early:** a val-to-test gap larger than the val interval width (report, do not re-run); any sign
  the frontier model has memorised items (D01 probe).

## Frozen references

As D01, plus the distilled model's config hash and bank hash (filled before Phase 3).

## Final-evaluation commands (filled before Phase 3, run by the human)

```bash
SCHELLING_ALLOW_TEST=1 uv run schelling eval --model <path> --split heldout_item --partition test --seed <n> --out reports/FD-final-<model>-item
SCHELLING_ALLOW_TEST=1 uv run schelling eval --model <path> --split heldout_person --partition test --seed <n> --out reports/FD-final-<model>-person
```
