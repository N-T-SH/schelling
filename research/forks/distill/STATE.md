# Fork D state

The fork's own state file. `research/STATE.md` belongs to the mainline and is not edited here.

## Now

- **Branch:** `fork/distill`, worktree `.claude/worktrees/fork-distill`, cut from `m5/licence-checks` at `463dd4c`.
- **Phase:** design drafted 2026-11-11. Nothing run. Nothing spent.
- **Next step:** human reviews `DESIGN.md` and `docs/decisions/F001-distilled-prior-fork.md`.

## Read first

1. `README.md` (rules for running beside the mainline)
2. `DESIGN.md`
3. `ASSUMPTIONS.md`
4. `campaigns/D0x-*/task.md`

## Blocked: needs the human

1. Approve, edit or reject F001 (and so the fork).
2. The five open questions in `DESIGN.md` §8: sealed question text for the bank filter; item folds if
   intervals are too wide; frontier model and spend caps; Jev in scope or not; PR 9 merge order.
3. Approve the D01 contract. D02 to D05 are approved one at a time, each after the one before it reports.

## What can start with no spend once F001 and D01 are approved

- Assumption checks FA5 and FA7 (read saved predictions only).
- The 3-seed M4b re-run on `heldout_item` val (the run the mainline skipped when M5 was stopped). It
  fixes the D02 margin in numbers.
- Tests and code for `baselines/llm_prompt.py` with fixtures, and the dry-run cost estimate.

## Done

Nothing yet.
