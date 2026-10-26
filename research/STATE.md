# Project state

Keep this file short and current. Update it at the end of every work session and at every
checkpoint. A fresh agent should be able to resume from it alone.

## Now

- **Mode:** BUILD
- **Milestone:** M2 Harness + baselines (complete, awaiting sign-off)
- **Campaign:** none
- **Next step:** wait for human sign-off on `research/reports/M2.md` and decisions on its findings; then plan M3.
- **Done so far:** M0 signed off 2026-10-26. M1 built; real data built (Psych-101 1% sample).

## Plan for the current milestone or phase

M2 Harness + simple baselines. Assumptions: A08 (sampling from correct probs gives SD ratio ~1; tested by simulation), A09 (B1 hard to beat; measured on heldout_person val), A14 (retest coverage), A15 new (LightGBM needs libomp on macOS: installed via `brew install libomp`, import verified).

Design:
- Baselines share one protocol (`baselines/base.py`): `fit(train, persons)` and `predict_proba(frame) -> list[np.ndarray]`. B0 uniform; B1 item mean with Jeffreys pseudo-count 0.5 per option and uniform fallback for unseen items; B2 cell = sex x age_band x education with backoff to B1 below 20 persons (equals B1 on Psych-101: no attributes); B3 LightGBM multiclass per (qtype, n options) group on attributes + item id (categorical).
- `eval/`: `metrics.py` (NLL, Brier, accuracy, adaptive ECE, calibration slope/intercept, TVD, W1, SD and entropy ratio), `population.py` (one draw per held-out person per item, 20 seeds, per-seed and per-subgroup, correlation Frobenius), `retest.py` (fraction of test-retest ceiling), `robustness.py` (option permutation), `efficiency.py`, `report.py` + `provenance.py` (metrics.json with git_sha, dirty, config_hash, split_manifest_hash, seed, created_at; pydantic model exported to `schemas/metrics.json`), `effects.py` (generic effect-size agreement; not run on data, needs a pre-registered pairing table).
- `scripts/audit_report.py` (tag check) with tests.
- Not in M2: paraphrase invariance and memorisation probe (need paraphrases, M5), treatment effects and regression sign agreement on data (need pairing table; will ask), B4 to B6.
- Retest: fit on wave-1 records of persons not evaluated, score wave-4 answers; ceiling = P(wave 1 answer = wave 4 answer); model agreement = mean probability on the wave-4 answer.
- `schelling eval --model b0|b1|b2|b3 --split S --partition val` only; it never sets `SCHELLING_ALLOW_TEST`.
Done when: metric tests pass and `schelling eval` produces a full report for B0 to B3 on both sources.

## Blocked: needs the human

M2 checkpoint: sign-off plus decisions on Psych-101 baseline bar and pairing table for treatment effects (see `research/reports/M2.md`). Signing off seals the evaluator.

## Done

| Date | Milestone / phase | Report | Signed off |
| --- | --- | --- | --- |
| 2026-10-26 | M0 Scaffold | research/reports/M0.md | yes |
| 2026-10-26 | M1 Data | research/reports/M1.md | yes (near duplicates removed) |
