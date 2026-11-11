# F001: Fork D, a distilled prior corrected on human data

**Date:** 2026-11-11 · **Status:** proposed

Numbered `F001` rather than `0007` so it cannot collide with mainline decision records. If the fork is
merged it takes the next free mainline number.

## Context

- Mainline levers on seen questions gave little: history encoder 0.0033 NLL (dropped), paraphrase
  augmentation a null (`research/reports/M5.md`). M5 was stopped by the human on 2026-11-11.
- The model trains on 435 distinct questions. On unseen questions its population fidelity and calibration
  are much worse than on seen ones (`research/forks/distill/DESIGN.md` §1, sourced).
- The external comparison (spec B4, B5) has never been run, so the gap to a frontier model is unknown.
- `research/PROTOCOL.md` already lists this idea as planned campaign C05, blocked on API-spend approval.
- The human asked (2026-11-11) for this to be tested as a fork that does not interrupt the mainline.

## Options

1. **Fork D as designed:** benchmark external models first (D01), then distil an open-weight teacher
   into the two-tower model and correct on human data (D02 to D05), on its own branch. Costs API money
   and background CPU; mainline is untouched.
2. **Benchmark only (D01), decide later:** cheapest; answers "how far are we from a frontier model" but
   tests no fix.
3. **Do it in mainline as M6 then C05:** one thread, simpler bookkeeping, but it blocks the mainline on
   approvals and spend, which the human asked to avoid.

## Decision

Proposed: option 1, gated. D01 runs first; D02 starts only if the calibrated teacher beats B1 on
`heldout_item` val.

Deviations from `AGENTS.md` and `docs/SPEC.md` that need the human's explicit approval:

- **Milestone order:** the B5 baseline adapter (M6 work) is built in the fork before M6 is signed off in
  mainline. Mainline M6 can later reuse it.
- **Assumptions register:** fork assumptions are kept in `research/forks/distill/ASSUMPTIONS.md` (ids
  `FA…`) instead of `research/ASSUMPTIONS.md`, to avoid merge conflicts. They are folded in on merge.
- **Training data:** teacher soft labels are a new training source. The spec's loss for aggregate records
  (`response_dist`) is used unchanged. Evaluation data is unchanged.
- **Paid services:** Fireworks (open-weight teacher and E3) and one frontier provider (E1/E2). Each batch
  needs a dry-run cost and an approved cap first; all responses are cached.

Not changed: public interfaces, metric definitions, acceptance thresholds, splits, the evaluator, the
test-partition seal.

## Consequences

- New code (all default-off or additive): `src/schelling/baselines/llm_prompt.py`,
  `src/schelling/augment/question_bank.py`, a pretraining stage in `cached_trainer.py`, scripts and
  configs under fork namespaces. Tests first, as in BUILD mode.
- The fork has its own state file, campaigns `D01` to `D05`, and its own compute and spend budgets
  (`research/forks/distill/README.md`).
- Risk: the teacher imports variance collapse; guards are binding and D03 exists to catch it.
- Risk: about 28 val items on `heldout_item` may be too few to decide; an `eval-change/` proposal may
  follow, which is a separate approval.
- If the fork fails, mainline is unaffected and the result is recorded in `research/FINDINGS.md` on merge
  of the write-up only.
