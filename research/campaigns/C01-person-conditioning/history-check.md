# C01 first task: can answer history reach the model without an evaluator change?

**Answer: yes.** No file under `src/schelling/eval/`, `splits/`, `tests/golden/` or `data/manifests/` has to change. Stop condition 1 in `task.md` does not trigger.

## How it works

- The harness calls `baseline.predict_proba(eval_df)` with every row of the scored persons, including each row's own `response` (`src/schelling/eval/report.py`, `run_eval`). A model scored through `schelling eval --model <dir>` is wrapped by `ModelBaseline` in `src/schelling/infer/baseline_adapter.py`, which is not sealed.
- New `src/schelling/infer/history.py::history_for_frame` builds, for each row, the same person's other first-pass answers. It drops the target's item id, every wave-4 row, and any row whose question text equals the target's (the contract's rules 1 and 2), never reads another person's rows, and returns the same history whatever the row order of the frame.
- `frame_items` now passes that history into each request. `ModelBaseline.history_k` is zero unless the loaded model is a two-tower model whose config declares `history_items`. Every model scored so far (the M4 cross-encoder and the M4b two-tower) therefore gets exactly the inputs it always got.

## Evidence

- `tests/unit/test_history_plumbing.py` (10 tests, written before the code): target question and answer absent, wave-4 repeats masked, duplicate text masked under a new item id, no cross-person leakage, changing only the target's answer leaves its history unchanged, row order does not matter, cap `k` is deterministic, noul, score and choice rows become valid history items, scored items carry no target, history off by default.
- Fast suite, ruff and mypy pass on this branch.
- `scripts/c01_history_check.py` on real Twin-2K-500 validation rows (15 persons, 9633 rows, `heldout_person`): no row's history contains its own question text. The data does contain 120 person-and-question pairs where one question text maps to more than one item id, so rule 2 is not hypothetical.

## Limits to carry into Phase 1

- `latency()` calls `predict_proba` on single rows, so those calls see no history and the reported latency understates a history model's cost. `permutation_invariance()` scores a sample of Choice rows, so the history it sees is built from that sample. Both only touch the efficiency and robustness blocks, not NLL. The model-level permutation test for C01 must therefore be written against `history_for_frame` output, not left to the harness.
- Training-time history (from the cached train rows) is a separate path to be built in Phase 1, with the same masking function and the leakage tests from `task.md`.
- No model uses the history yet; nothing here changes any reported number.
