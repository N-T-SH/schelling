# 0003: Leave-one-experiment-out folds replace the 2+2 `heldout_experiment` split

**Date:** 2026-10-26 · **Status:** approved (Nitesh, 2026-10-26); amends 0002 item 2

## Context

0002 added `heldout_experiment` with 2 experiments in val and 2 in test. EC1 showed that validation then holds only 4 contrasts, so effect metrics were close to uninformative. You chose leave-one-experiment-out (LOEO) folds over all eight experiments instead, accepting one trained model per fold.

## Decision

- The split becomes eight splits, `heldout_experiment_f0` to `heldout_experiment_f7`. Fold k holds out experiment e_k (experiments ranked by a seeded hash, `fold_experiment`).
- Train: train-person rows of every item except e_k. Val and test: the val-person and test-person rows of e_k, with the person assignment identical to `heldout_person`. All other rows are excluded. So val and test are held out in both experiment and person, and a person used in search is never in the sealed test partition.
- Effect metrics are computed per fold, then pooled over the eight folds (`schelling effects-pool`): every one of the 16 contrasts is scored by a model that never saw its experiment, and per-seed correlation, sign agreement and false-positive rates are recomputed over the union.
- Human effects still come only from the evaluated partition's persons (0002 item 1), so each contrast has roughly a tenth of the persons; the pooled metrics, not any single fold, are the ones to read.

## Consequences

- `splits/heldout_experiment.json` is replaced by eight fold files. Other split files are unchanged.
- A full effect evaluation of a trained model needs eight training runs. For baselines it is free.
- The released model for the headline claims is trained on all data; fold models exist only to measure effects. State this in the model card.
- Pooling requires the same population seeds in every fold.
