# Project state

Keep this file short and current. Update it at the end of every work session and at every
checkpoint. A fresh agent should be able to resume from it alone.

## Now

- **Mode:** BUILD
- **Milestone:** M4b two-tower model (decision 0004): signed off 2026-11-10
- **Campaign:** none
- **Next step:** human chooses between a C01 contract and a revised M5 scope; guard-metric eval-change (decision 0005) awaits approval of its definitions. No work is queued.
- **Eval change:** treatment-effect metrics and LOEO folds merged (`research/reports/EC1-treatment-effects.md`, decisions 0002 and 0003).
- **Done so far:** M0 signed off 2026-10-26. M1 built; real data built (Psych-101 1% sample).

## Plan for the current milestone or phase

M4 redesign per `docs/decisions/0004-two-tower-model.md` (human approved 2026-10-29). M5 is NOT started (human instruction). The M4 done-criterion is unchanged: beat B1 on Twin-2K-500 `heldout_person` val NLL; Psych-101 excluded (decision 0001).

Build: `model/{configuration_schelling_tt,heads_tt,modeling_schelling_tt}.py`; `infer/text.py` question context; `infer/batching.py` dispatch on `model_type`; `Predictor` dispatch; `train/cached_trainer.py` (Stage A: frozen backbone, cached vectors, Twin only); `schelling train` dispatch on `model.architecture`.
Tests first: head shapes and zero-init residual, order-equivariance and set-invariance (Hypothesis), batch equals single, cache on/off identical, cached-training path equals live forward, save/load and fresh-process custom-code load, cached trainer integration (loss decreases, run.json, determinism, ablation flags), eval of the saved model through the unchanged harness.
Experiments (validation only, protocol: 5 seeds for the final config): full model; ablations (no person tower, no prior loss); `heldout_person` against the criterion; extra evidence on `heldout_item` and the eight `heldout_experiment` folds with pooled effects; baselines on `heldout_item` for comparison.
Report: `research/reports/M4b.md` with audited numbers; stop at the checkpoint.

## Blocked: needs the human

1. Decision 0005 approved with option 3 (record now, bind later): `eval-change/guard-metrics` adds `schelling guards`; PR awaiting human review. Making the TVD guard binding is a later decision.
2. C01 contract approved 2026-11-10 (`research/campaigns/C01-person-conditioning/task.md`). The first task is done: history can reach the model without an evaluator change (`research/campaigns/C01-person-conditioning/history-check.md`). Phase 1 ideas are not yet pre-registered.
3. M5 is NOT started.
4. Decided 2026-11-10: model evaluation stays on CPU (no eval-change for the device label).

## Done

| Date | Milestone / phase | Report | Signed off |
| --- | --- | --- | --- |
| 2026-10-26 | M0 Scaffold | research/reports/M0.md | yes |
| 2026-10-26 | M1 Data | research/reports/M1.md | yes (near duplicates removed) |
| 2026-10-26 | M2 Harness + baselines | research/reports/M2.md | yes; Psych-101 excluded from M4 bar (docs/decisions/0001) |
| 2026-10-26 | M3 Model v0 | research/reports/M3.md | accepted under standing instruction (no open decision points) |
| 2026-10-26 | M4 Real training | research/reports/M4.md | no: criterion not met; superseded by M4b |
| 2026-11-05 | M4b Two-tower model | research/reports/M4b.md | yes, 2026-11-10 (caveats on record) |
