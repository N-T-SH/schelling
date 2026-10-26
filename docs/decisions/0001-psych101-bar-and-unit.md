# 0001: Psych-101 is excluded from the M4 baseline bar until C04

**Date:** 2026-10-26 · **Status:** approved (Nitesh, 2026-10-26)

## Context

M2 ran B0 to B3 on `heldout_person` validation (`research/reports/M2.md`). On Psych-101 the item-mean baseline B1 is worse than uniform: item ids collapse to one "You press" item per experiment and the options are participant-specific key letters, so train marginals do not transfer to held-out persons (M1 finding 1, confirmed in M2 finding 1). "Beat B1 on every M1 source" (SPEC acceptance table, M4 done-criterion) is therefore not a meaningful bar there.

## Options

1. Keep the adapter; treat Psych-101 baseline results as uninformative; settle the unit of prediction in campaign C04; exclude Psych-101 from the M4 bar.
2. Restrict the adapter to experiments with semantic options now (changes manifests and splits).
3. Keep Psych-101 in the bar as is (trivially met).

## Decision

Option 1. Psych-101 stays in the data build and in reports, but its B0 to B3 numbers are labelled uninformative, and the M4 done-criterion ("beats B1 on `heldout_person` NLL for both sources") applies to Twin-2K-500 only. The Psych-101 bar is defined after C04.

## Consequences

- SPEC milestone M4 and the release acceptance row "NLL vs B1 item mean, `heldout_person`: better on every M1 source" are read as "every M1 source with a meaningful baseline" until C04 closes. The SPEC text is not edited; this record is the deviation.
- C04 (unit of prediction for Psych-101) moves ahead of any Psych-101 claim in M4 or later.
- No change to sealed files (`eval/`, `splits/`, `data/manifests/`).
