# Project state

Keep this file short and current. Update it at the end of every work session and at every
checkpoint. A fresh agent should be able to resume from it alone.

## Now

- **Mode:** BUILD
- **Milestone:** M4b two-tower model (decision 0004): signed off 2026-11-10
- **Campaign:** none
- **Next step:** M5 step 4: run the ablation `scripts/run_m5.sh` (aug vs base, seeds 0 to 2, heldout_person val, six trainings and evals, compute only, roughly a day throttled). Trainer option built and tested; paraphrases generated (A19 verified, A20 partly). Awaiting the human's go.
- **Eval change:** treatment-effect metrics and LOEO folds merged (`research/reports/EC1-treatment-effects.md`, decisions 0002 and 0003).
- **Done so far:** M0 signed off 2026-10-26. M1 built; real data built (Psych-101 1% sample).

## Plan for the current milestone or phase

M4 redesign per `docs/decisions/0004-two-tower-model.md` (human approved 2026-10-29). M5 is NOT started (human instruction). The M4 done-criterion is unchanged: beat B1 on Twin-2K-500 `heldout_person` val NLL; Psych-101 excluded (decision 0001).

Build: `model/{configuration_schelling_tt,heads_tt,modeling_schelling_tt}.py`; `infer/text.py` question context; `infer/batching.py` dispatch on `model_type`; `Predictor` dispatch; `train/cached_trainer.py` (Stage A: frozen backbone, cached vectors, Twin only); `schelling train` dispatch on `model.architecture`.
Tests first: head shapes and zero-init residual, order-equivariance and set-invariance (Hypothesis), batch equals single, cache on/off identical, cached-training path equals live forward, save/load and fresh-process custom-code load, cached trainer integration (loss decreases, run.json, determinism, ablation flags), eval of the saved model through the unchanged harness.
Experiments (validation only, protocol: 5 seeds for the final config): full model; ablations (no person tower, no prior loss); `heldout_person` against the criterion; extra evidence on `heldout_item` and the eight `heldout_experiment` folds with pooled effects; baselines on `heldout_item` for comparison.
Report: `research/reports/M4b.md` with audited numbers; stop at the checkpoint.

## M5 plan (augmentation only; awaiting go)

**Idea:** the model sees one wording per question. Frozen-backbone vectors are cached, so paraphrases are cheap: embed K wordings per training question once, and sample one wording per step. Evaluation stays on original wording (evaluator is sealed).

**Steps**
1. Check assumptions first (no code): A19 (now Fireworks, see `research/ASSUMPTIONS.md`): chosen model's licence allows generating training text, then a paid-run budget approved by the human; A20 paraphrases keep meaning (measure: embedding similarity filter plus a hand-read sample of 50, recorded).
2. Tests first: paraphrases only for `train`-partition questions; none for val/test items; option semantics unchanged (Noul yes/no, Choice order, Score scale); store keyed by hash of source text and generator version; cached-trainer determinism unchanged when augmentation is off.
3. Build: `scripts/make_paraphrases.py` (offline, once, cached to disk, never at train time); paraphrase store loader; `cached_trainer` option `paraphrases_per_question` (default 0 keeps M4b behaviour).
4. Run: M4b config with and without augmentation, 5 seeds, harness val on `heldout_person`, plus `heldout_item` (where wording robustness should show). Report population metrics and guards with sources.
5. Report `research/reports/M5.md` through `audit_report.py`; stop at the checkpoint.

**Needs from the human:** go for step 1; approval for any paid API (default is a local model); the ablation table is the done criterion.

## Blocked: needs the human

1. Decision 0005 approved with option 3 (record now, bind later): `eval-change/guard-metrics` adds `schelling guards`; PR awaiting human review. Making the TVD guard binding is a later decision.
2. C01 contract approved 2026-11-10 (`research/campaigns/C01-person-conditioning/task.md`). The first task is done: history can reach the model without an evaluator change (`research/campaigns/C01-person-conditioning/history-check.md`). Idea I01 (history encoder) is drafted in `research/campaigns/C01-person-conditioning/ideas/`; it awaits human approval before any run.
3. M5 is NOT started. Decision 0006 (revised scope) approved 2026-11-11; step 1 (licence checks A16 to A18) is next.
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
