# 0006: Revised M5 scope (data levers, not encoder work)

**Date:** 2026-11-11 · **Status:** approved (human, 2026-11-11)

## Context

M5 in `docs/SPEC.md` is "Person encoder v1 with caching, optional diversity regulariser, paraphrase augmentation, survey sources", done when population metrics are reported with an ablation table.

- The person encoder exists: the two-tower model of decision 0004, signed off as M4b (`research/reports/M4b.md`).
- Further person conditioning gave little. C01 idea I01 (history encoder) beat the M4b re-run by 0.0033 NLL, under its 0.005 drop threshold; it was dropped by the human on 2026-11-11 (`research/campaigns/C01-person-conditioning/ideas/I01-history-encoder.md`, `ideas/I01-bootstrap.json`).
- Variance collapse is not visible: SD ratio is 0.987 to 1.006 across the six C01 runs (`reports/C01-guards/*.json`). A08 predicts this when per-person probabilities are calibrated.
- Population fidelity is still weak: M4b population TVD is above B1's and treatment-effect correlation is low (`research/reports/M4b.md`).

## Options

1. **Revised M5:** drop person-encoder and regulariser work; keep paraphrase augmentation and survey sources.
2. **Original M5 as written:** spends effort on parts the evidence says will not help.
3. **Skip to M6, return to M5 later:** earlier external comparison, but needs paid API approval and leaves population fidelity weak.

## Decision

Option 1. M5 becomes "Population fidelity through data":

1. **Survey sources** (GSS, ANES, WVS): verify licence, fields and redistribution terms first (new assumptions A16 to A18). Add an adapter only for sources that pass. Derived-model distribution terms decide whether a source can be used at all.
2. **Paraphrase augmentation:** paraphrases generated offline once, stored with a hash, never regenerated at train time. Generation by a paid LLM API needs the human's approval first (rule 7); a local model is the default to propose.
3. **Ablation table:** with and without each addition, on the existing population metrics, 5 seeds, `val` only.
4. **Diversity regulariser:** not built. A cheap check (campaign C02) may confirm it is not needed; it runs only if the human asks.
5. **Person encoder v1:** treated as done (M4b).

Done criterion (unchanged in spirit): population metrics reported with an ablation table. No metric definition, split or threshold changes. No evaluator change is planned; if survey sources need new split logic, that goes through an `eval-change/` branch.

## Consequences

- `docs/SPEC.md` M5 line gets a pointer to this record once approved.
- First work item is assumption checks only (A16 to A18), no code. If a source fails its licence check, M5 shrinks to augmentation alone and I report that.
- Psych-101 stays excluded from the primary criterion (decision 0001).
- Risk: more data may not move TVD or treatment-effect correlation; a null result is reported as such.

## Addendum (2026-11-11)

Licence checks A16 to A18 cleared none of GSS, ANES or WVS for a released model (`research/ASSUMPTIONS.md`). The human chose augmentation only. Survey sources are out of M5; they return only if written permission is obtained (GSS via NORC) or the ANES terms are supplied and allow it. M5 is now: paraphrase augmentation plus its ablation table.
