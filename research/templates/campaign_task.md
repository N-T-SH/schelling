# Campaign <ID>: <short title>

**Status:** draft | approved (date, by) | running | closed

## Question

_One question, answerable yes / no / how much._

## Hypothesis

_What you expect and why. Stated so it can turn out false._

## Metrics

- **Primary:** <name> computed by `src/schelling/eval/<module>.py::<function>`; lower/higher is better.
- **Guards (must not regress beyond):** ECE ≤ <x>; SD ratio within [<a>, <b>];
  option-permutation max |Δp| ≤ 0.01; <others>.

## Data

- **Split:** <heldout_person | heldout_item | …>; search on `val` only.
- **Sources:** <Twin-2K-500 | Psych-101 | …>

## Baselines the winner must beat

<B0 … B6, with their current val numbers and source tags>

## Budget

- Idea cycles: <5>
- Seeds: <3> per search run, <5> per ablation run
- Compute: <hours> on <hardware from A12>
- Paid API spend: <none | amount, approved by>

## Success, failure and early-stop criteria

- **Yes:** <e.g. primary improves on B1 by ≥ x with the 95 % bootstrap interval excluding 0, guards hold>
- **No:** <e.g. no idea beats B1 after the full budget>
- **Stop early:** <e.g. suspected leakage; guard broken by every idea>

## Frozen references

- Evaluator git SHA (`src/schelling/eval/`): <sha>
- Split manifest hash: <hash>
- Data manifest hash(es): <hash>

## Final-evaluation commands (filled before Phase 3, run by the human)

```bash
SCHELLING_ALLOW_TEST=1 uv run schelling eval --model <path> --split <split> --partition test --seed <n> --out reports/<run_id>
```
