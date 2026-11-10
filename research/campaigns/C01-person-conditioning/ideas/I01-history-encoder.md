# Idea I01: history encoder over the person's other answers

**Campaign:** C01 · **Branch:** `exp/C01/I01-history-encoder` · **Registered:** 2026-11-10, before any run (status: draft until the human approves; nothing has been run)

## Change

Add a history tower to the M4b two-tower model. Everything else (frozen ModernBERT vectors, item prior, calibration, losses, optimiser, 4000 steps of batch 1024) stays as in `configs/twotower-a.yaml`.

- **History item embedding:** for each of a person's other answers `j`, `x_j = MLP([u_j, e_{j,a_j}])`, where `u_j` is the cached question vector and `e_{j,a_j}` the cached vector of the option (or level) the person chose. Noul answers use the yes or no option vectors. No item id, no position.
- **History vector:** attention pooling over the set `{x_j}` with the target question vector `u` as the query, giving `h`. Order-free, like the attribute encoder.
- **Residual branch:** `r = MLP([p, h, u, p*u, h*u])`, last layer zero-initialised as before, so the model still starts at its prior.
- **Training histories:** for each training row, `k` other first-pass answers of the same person, chosen by the rule in `src/schelling/infer/history.py` (target item, its wave-4 repeats and same-text questions excluded). `k` is drawn per example from {0, 5, 20, 50}, so the model also works with no history and the history-size ablation (idea 2) is a by-product.
- **Evaluation:** `history_items = 50` in the saved config, so `ModelBaseline` feeds 50 history items per row through `history_for_frame` (no evaluator change; `research/campaigns/C01-person-conditioning/history-check.md`).
- **Code:** new tower in `model/heads_tt.py`, history sampling in `train/cached_trainer.py`, `history_items` in `configuration_schelling_tt.py` and `train/config.py`. Cached vectors already exist for every question and option, so no new encoder pass is needed.

## Why it should help

M4b's ablation shows the whole gain over B1 comes from the person tower, which sees only 14 demographics (no person tower 0.9991 <!-- src: reports/M4b-train/summary.json#no_person.mean -->, full model 0.9892 <!-- src: reports/M4b-train/summary.json#seeds.mean -->). A person's answers to related questions, such as other rows of the same Likert scale, say much more about the next answer than age or region do. This is a prior from how survey responses behave, not a result from this project; nothing here has tested it.

## Prediction (written before running)

- **Primary:** `heldout_person` validation NLL (`individual.by_source.twin2k.nll`) falls by about 0.015 against the M4b five-seed mean of 0.9892 <!-- src: reports/M4b-train/summary.json#seeds.mean -->, within a plausible range of 0.005 to 0.040, with the largest gain on Score items (the Likert blocks) and the smallest on Noul. The contract's bar is 0.010.
- **Guards:** ECE stays under 0.05 (expected: roughly unchanged). SD ratio stays in its band; I expect it to rise slightly because predictions become more person-specific. Option-order invariance unchanged (the history carries option text, not indices). Population TVD (recorded, not binding): I expect it to improve slightly or stay level; I would not be surprised if it does not move, since NLL gains do not imply a better population distribution.
- **Treatment effects (recorded, H3):** no change expected, within noise.
- **Drop the idea if:**
  - the gain over the M4b mean is below 0.005 on three seeds, or the paired interval over persons includes 0;
  - any binding guard fails on any seed;
  - NLL falls by more than 20 % in one change, or top-choice agreement exceeds the retest ceiling (treated as a leak, stop and report per `task.md`);
  - any leakage test fails.

## Leakage checks (must pass before the first training run)

- `tests/unit/test_history_plumbing.py` (merged with PR 6).
- New sentinel test: for a tiny tower, the history encoder input for a target contains none of the target's question tokens, option tokens or cached vector, for k = 5, 20, 50.
- New model-level test: change only the target item's answer in the person's record; the prediction for that item is bitwise unchanged.
- Near-paraphrase check: report, for the validation rows, the share whose history contains a question with the same stem and options (Likert blocks). These stay allowed (they are the signal) but the number goes in the write-up.

## Runs

| run_id | seed | primary | guards | status |
| --- | --- | --- | --- | --- |

_Rows generated from `experiments.jsonl`, not typed by hand. Planned: `C01-I01-s0` to `s2` (search) and an M4b re-run at the frozen SHA with the same three seeds, so the paired bootstrap uses identical persons._

## Outcome

_Not run._
