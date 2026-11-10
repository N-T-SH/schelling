# 0005: Population TVD and held-out-person NLL become guard metrics

**Date:** 2026-11-10 · **Status:** draft. Nitesh agreed in principle on 2026-11-10; the definitions and thresholds below need approval before any evaluator change is built.

## Context

M4b (`research/reports/M4b.md`) met the M4 criterion on held-out-person NLL by a small margin, while population TVD and W1 came out slightly worse than B1 and treatment effects sat at chance. The evaluator reports these numbers but nothing checks them, so a model can improve the headline and degrade the thing the project is for without any signal. The evaluator is sealed (AGENTS.md rule 2), so adding checks needs an `eval-change/guard-metrics` branch and your approval.

## Decision (proposed)

Add a `guards` block to `metrics.json`, computed from numbers the harness already produces, and listed in every report. A guard fails or passes; it never changes a metric.

| guard | definition | pass condition (proposed) |
| --- | --- | --- |
| person NLL | Twin-2K-500 `heldout_person` NLL against B1 on the same rows | model NLL at most B1 NLL |
| population TVD | mean population TVD (20 sampling seeds) against B1 | model TVD at most B1 TVD plus a tolerance |
| ECE | adaptive ECE | at most 0.05 (existing limit) |
| SD ratio | population SD ratio | between 0.85 and 1.15 (existing band) |
| permutation | option-permutation maximum change | at most 0.01 (existing limit) |

## Options for the TVD tolerance (needs your choice)

1. **No tolerance (strict):** TVD at most B1. M4b seed 0 would fail it (TVD 0.0573 against 0.0537).
2. **Tolerance of one seed SD of B1's TVD** (about 0.002). M4b would still fail.
3. **Record only for now, make it binding when C01 finishes.** M4b is unaffected.

My recommendation is option 1 as the target and option 3 as the interim: report the guard now so it is visible, and make it binding once a campaign has had a chance to fix it. A binding guard that the current best model already fails would block every campaign.

## Consequences

- Changes only `src/schelling/eval/report.py`, a golden metrics entry and its test, on `eval-change/guard-metrics`; no existing metric definition changes.
- Existing `metrics.json` files lack the block. Reports compute guards from the numbers they already contain, so no model needs re-evaluation.
- Evaluation stays on CPU (decided 2026-11-10); the efficiency module is not touched.
