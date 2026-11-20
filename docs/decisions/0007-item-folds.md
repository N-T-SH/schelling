# 0007: Five item folds for comparing models on unseen questions

**Date:** 2026-11-20 · **Status:** proposed (the human approved writing it, 2026-11-20; approval of
this branch is the decision)

## Context

`heldout_item` val holds 28 Twin-2K-500 questions after contamination removal. Comparisons between
models on unseen questions resample those items, so the intervals are wide: in fork D's D02 gate the
95 % interval for a clear win was about a fifth of a nat wide
(`research/forks/distill/campaigns/D01-external-benchmark/D02-gate.md` on `fork/distill`). Fork D now
needs to tell apart models that may differ by a few hundredths (`research/forks/distill/D02-OPTIONS.md`).
Its DESIGN open question 2 and assumption FA7 anticipated this.

## Options

1. **Keep `heldout_item` only.** No change; small differences stay unresolvable.
2. **Re-split with a larger val fraction.** More val items, but fewer training items for every model,
   and the sealed test partition would move.
3. **K item folds over the non-test items (chosen).** Every non-test item is val in exactly one fold
   and is scored by a model that never saw it; pooled over folds, about ten times as many unseen items.
   Costs K training runs per model and seed. Test stays exactly where it is.

## Decision

- Five new splits, `heldout_item_f0` to `heldout_item_f4`. Existing splits, including `heldout_item`,
  are unchanged byte for byte.
- **Test:** identical to `heldout_item` test (same rows, same contamination removals). It stays sealed;
  the folds give no new access to it.
- **Train and val:** the remaining items are ranked by a seeded hash within each source and dealt into
  five folds of equal size (within one item per source). Fold k: its items are val, all other non-test
  items are train.
- **Contamination:** the `heldout_item` rule per fold. Val items that near-duplicate the fold's own
  train items (MinHash, Jaccard 0.8 or more, on question plus options) leave val, as do
  `heldout_item`'s removals. Summary: `splits/contamination_heldout_item_folds.json`.
- **Reading results:** per-item results are pooled over the five folds, each item scored by its own
  fold's model; paired comparisons resample pooled items. A single fold is never the headline.
- `heldout_item` stays the split for one-model reports and for the single test evaluation.

Sizes from this branch's `schelling data splits` on the current processed data: 52 to 62 Twin-2K-500
val items per fold, 282 pooled (against 28 in `heldout_item` val).

## Consequences

- `src/schelling/data/splits.py`: `ITEM_FOLD_PREFIX`, `N_ITEM_FOLDS`, `_fold_assign`, and a branch in
  `_assign`. `src/schelling/cli.py`: `schelling data splits` applies the per-fold contamination rule.
  Six new tests in `tests/unit/test_splits.py`; the test pinning the split list gains the five names
  (separate commit).
- A full fold comparison of a trained model needs five training runs per seed. Baselines and cached
  LLM predictions cost nothing extra.
- About 60 % of each fold's raw val items are removed as near-duplicates, almost all repeated
  Psych-101 trials; this is the same rule `heldout_item` already applies.
- A pooling tool for per-item paired comparisons across folds is fork-side work
  (`scripts/fd_paired.py` takes one run per side today).
