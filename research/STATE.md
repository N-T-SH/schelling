# Project state

Keep this file short and current. Update it at the end of every work session and at every
checkpoint. A fresh agent should be able to resume from it alone.

## Now

- **Mode:** BUILD
- **Milestone:** M4 Real training (run complete; criterion NOT met; awaiting human decision)
- **Campaign:** none
- **Next step:** human chooses among the three options in the M4 report. No work is queued.
- **Eval change:** treatment-effect metrics and LOEO folds merged (`research/reports/EC1-treatment-effects.md`, decisions 0002 and 0003).
- **Done so far:** M0 signed off 2026-10-26. M1 built; real data built (Psych-101 1% sample).

## Plan for the current milestone or phase

M4 Real training (SPEC milestone 5). Standing instruction: continue past M3 when it leaves no decision points (it did). M4 ends at a human checkpoint regardless.

Done when (SPEC, read with decision 0001): trained ModernBERT-base beats B1 on NLL on `heldout_person` val for Twin-2K-500 (Psych-101 excluded from the bar); goldens frozen. If it does not, that is the result: report it plainly.

Hardware reality (A12 + benchmark 2026-10-26, Apple M4, MPS): ModernBERT-base full fine-tune runs at about 2,200 tokens/s (about 16 examples/s at 128 context tokens); freezing the embeddings and lowest 14 of 22 layers doubles that. So the budget is on the order of 100,000 to 150,000 example presentations (about a tenth of one epoch of the Twin train partition) in 1 to 1.5 hours, and the full validation evaluation of about 136,000 rows will take tens of minutes. This is a smoke-scale run of the idea, not a tuned one. Decision for the report: single seed, one config, no tuning on val beyond early stopping.

Design:
- `train/config.py` (pydantic, YAML, `--set key=value`), `train/losses.py` (hard and soft CE, Noul BCE, Choice CE with masks, Score CORN loss which equals the NLL of the distribution; tested), `train/sampler.py` (study temperature sampling, p proportional to n^0.5), `train/trainer.py` (AdamW with separate head and backbone learning rates, cosine schedule with warmup, early stopping on val NLL of `heldout_person`, best and last checkpoints, TensorBoard, `run.json` with config, git sha, dirty flag, split manifest hash, package versions, seeds), `model/calibration.py` (per-head temperature fitted on val logits, stored in the config), `infer/batching.py` (shared tensorisation so training and inference cannot differ), `infer/baseline_adapter.py` (a saved model as a `predict_proba` baseline so `schelling eval --model <dir>` works unchanged).
- Heads expose raw logits so calibration can refit temperatures without re-running the encoder; behaviour unchanged (existing tests guard it).
- Training input: person attributes only (`history_items` 0): conditioning on a person's other answers is campaign C01, not M4. Random option permutation augmentation for Choice. Psych-101 included in training through the sampler; its results stay labelled uninformative.
- Option texts are deduplicated per batch in the model (done, numerically identical).
- Tests first: `test_losses`, `test_calibration`, `test_sampler`, `test_train_config`, integration (50-step fixture run: loss decreases, `run.json` complete, safetensors, two runs bitwise identical = A13, CLI smoke, `eval --model <dir>` works), regression golden (slow).
- Goldens: `tests/golden/metrics.json` created once from the fixture run by `scripts/make_golden.py`; updating needs `[golden-update]` in the commit message.

## Blocked: needs the human

M4 checkpoint (`research/reports/M4.md`): the done-criterion is not met (model worse than B1 on Twin-2K-500 validation NLL). Decide the next step: (1) design record for a two-tower model with an item prior (recommended), (2) approve paid GPU compute for a longer run of the current design, (3) accept and go to C01. M5 is not started.

## Done

| Date | Milestone / phase | Report | Signed off |
| --- | --- | --- | --- |
| 2026-10-26 | M0 Scaffold | research/reports/M0.md | yes |
| 2026-10-26 | M1 Data | research/reports/M1.md | yes (near duplicates removed) |
| 2026-10-26 | M2 Harness + baselines | research/reports/M2.md | yes; Psych-101 excluded from M4 bar (docs/decisions/0001) |
| 2026-10-26 | M3 Model v0 | research/reports/M3.md | accepted under standing instruction (no open decision points) |
| 2026-10-26 | M4 Real training | research/reports/M4.md | no: criterion not met, awaiting decision |
