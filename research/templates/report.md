# Campaign <ID> report: <short title>

**Answer:** _one sentence answering the campaign question, with its key number._

## Claims

Every number carries a source tag that `scripts/audit_report.py` checks against the file.

| Claim | Value | Source |
| --- | --- | --- |
| Winner primary metric on test | 1.234 <!-- src: reports/C01-final-winner/metrics.json#nll --> | final evaluation |
| B1 primary metric on test | 1.301 <!-- src: reports/C01-final-b1/metrics.json#nll --> | final evaluation |

_(Example rows: replace them.)_

## What was tried

_Ideas in order, one line each, with outcome. Link `ideas/`._

## Ablations

_Summary of `ablations.md`: which components carry the effect._

## Post-hoc findings

_Subgroups, calibration, failure cases, gaming checks, val-vs-test gap._

## Limitations

_Data, scope, anything that would change the conclusion._

## Next question

_The single most useful follow-up campaign._
