# Campaign D03: does distillation import variance collapse?

**Status:** draft. Not approved. Needs D02 artifacts. This is the second half of the question planned as
C05 in `research/PROTOCOL.md`.

## Question

Compared with the M4b re-run, does the D02 model produce populations that are too uniform or too
confident, on seen or unseen questions?

## Hypothesis

H1 (can fail). With calibrated teacher labels and human fine-tuning, the SD ratio stays within
[0.85, 1.15] and population TVD is not worse than the M4b re-run on both `heldout_person` and
`heldout_item` val. With **raw** teacher labels and no fine-tuning, the SD ratio falls below 0.85.

The second sentence matters: it shows the defences are doing the work, not luck.

## Metrics

- **Primary:** SD ratio, `src/schelling/eval/metrics.py::sd_ratio`,
  `metrics.json#population.twin2k.summary.sd_ratio.mean`.
- **Also reported:** entropy ratio, mode-collapse rate, population TVD and W1, subgroup worst-group gap,
  ECE, and the per-item SD ratio distribution (not only the mean, which can hide collapse on some items
  offset by over-dispersion on others).

## Data

`heldout_person` and `heldout_item`, `val` only, Twin-2K-500 rows.

## Variants (no new ideas; four fixed models, 5 seeds each)

1. M4b re-run.
2. Teacher-pretrained only, raw labels, no human fine-tuning.
3. Teacher-pretrained, calibrated labels, no human fine-tuning.
4. The D02 winner (calibrated labels plus human fine-tuning).

## Budget

- Compute: evaluation of saved D02 models plus training of variants 2 and 3; **[proposal]** 6 hours at
  fork priority. Paid spend: none.

## Success, failure and early-stop criteria

- **Yes (no collapse):** variant 4 passes the SD-ratio band on every seed and split, and its population
  TVD is not worse than variant 1 (paired interval includes 0 or favours variant 4).
- **No:** variant 4 leaves the band or is worse on TVD. Then D02's result is "partial" whatever its NLL.
- **Gaming check:** if the mean SD ratio passes but more than a quarter of items are individually outside
  the band **[proposal]**, report it as a failure of the guard, not a pass.

## Frozen references

As D01.
