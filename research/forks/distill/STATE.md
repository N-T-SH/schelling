# Fork D state

The fork's own state file. `research/STATE.md` belongs to the mainline and is not edited here.

## Now

- **Branch:** `fork/distill`, worktree `.claude/worktrees/fork-distill`, cut from `m5/licence-checks` at `463dd4c`.
- **Approved by Nitesh, 2026-11-11 (chat: "Approve both"):** decision F001 and the D01 contract. D02 to D05 are not approved.
- **Phase:** D01, no-spend tasks. No paid call has been made.
- **Next step:** human answers the questions under Blocked; then the E3 smoke test (FA2, FA4).

## Read first

1. `README.md` (rules for running beside the mainline)
2. `DESIGN.md`
3. `ASSUMPTIONS.md`
4. `campaigns/D0x-*/task.md`

## Done in D01 so far (no spend)

- FA5 verified, FA7 partly true (`ASSUMPTIONS.md`). Tool: `src/schelling/compare.py`, `scripts/fd_paired.py`.
- External-model baseline: `src/schelling/baselines/llm_prompt.py` (population and persona forms, offline
  cache replay, cache miss raises). `schelling eval --model <spec.json>` scores it through the unchanged
  harness. Specs for E3 in `configs/fork-distill/`.
- Dry-run sizing script `scripts/fd_llm_calls.py` (paid calls only with `--go --cap-usd`).
- M4b reference on `heldout_item` val, seeds 0 to 2, and B1: first set finished but is flagged
  `dirty: true` (files were being edited in the worktree while it ran). Kept as `*-dirty`; a clean
  re-run at the committed SHA was started 2026-11-12 (`scripts/fd_run_d01_m4b.sh`). Do not quote the
  dirty numbers in a report.

## Dry-run estimates for E3 (`gpt-oss-120b`, $0.15 in / $0.60 out per million tokens, A19)

Input tokens are characters / 4 and output is a guessed 400 tokens per call, so these are rough
estimates, not measurements. Command: `scripts/fd_llm_calls.py` at the SHA of this commit.

| split (val) | form | requests | estimate |
| --- | --- | --- | --- |
| `heldout_item` | population | 28 | under $0.01 |
| `heldout_item` | persona, 100 persons | 2770 | about $0.78 |
| `heldout_person` | population | 586 | about $0.16 |
| `heldout_person` | persona, 100 persons | 56774 | about $16 |

The last row is over the proposed $5 cap for E3. Options: fewer persons on `heldout_person`, or a seeded
item subsample there. The LOEO folds are not sized yet.

## Blocked: needs the human

1. Spend approval for the E3 smoke test and runs (cap), and how to shrink the `heldout_person` persona run.
2. Which frontier model is E1/E2 and its cap (FA1 to FA3 are checked after you name it).
3. Jev in scope or not (A10).
4. May the D02 bank filter read sealed question text (never answers)?
5. Merge PR 9 first, or keep the fork stacked on it?
