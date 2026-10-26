# Project state

Keep this file short and current. Update it at the end of every work session and at every
checkpoint. A fresh agent should be able to resume from it alone.

## Now

- **Mode:** BUILD
- **Milestone:** M1 Data (complete, awaiting sign-off)
- **Campaign:** none
- **Next step:** wait for human sign-off on `research/reports/M1.md` and decisions on its three findings; then plan M2.
- **Done so far:** M0 signed off 2026-10-26. M1 built; real data built (Psych-101 1% sample).

## Plan for the current milestone or phase

M1 Data. Assumptions: A01, A03, A04, A05 checked 2026-10-26 (see ASSUMPTIONS.md); A02 verified without download.

Design choices (from evidence, flag in report):
- Psych-101: one row = one participant x experiment. Adapter emits one `DecisionRecord` per `<<...>>` span; `state` = transcript text before the span (truncated to the last N chars), `trial_idx` = span order. Options = the closed set of distinct choice strings observed for that experiment (kept if 2..20, else the experiment is dropped and logged). This is a pragmatic reading of "unit of prediction"; campaign C04 tests alternatives.
- Twin-2K-500: build records from `question_catalog.json` + `wave1_3_response*.csv` / `wave4_response*.csv`. Never use `persona_*` fields. Mapping: Likert rows -> score, Bipolar rows and single-select MC -> choice (yes/no 2-option MC -> noul), unmappable (free text, multi-select, slider) dropped and counted.
- `persons.parquet` holds attributes (Twin demographics) and no history; context is assembled at use time with the target item masked.
- Splits: hash-based, deterministic, five splits + retest, train/val/test partitions, manifest hash; test sealed by `SCHELLING_ALLOW_TEST` in the loader.
- Raw data downloads in `data/raw/` (gitignored); manifests with sha256 in `data/manifests/`.

Files: `src/schelling/data/{schema,registry,splits,contamination,manifest,loader}.py`, `data/adapters/{psych101,twin2k}.py`, `scripts/make_fixtures.py`, `tests/fixtures/`, `tests/unit/{test_adapters,test_splits,test_loader_guard,test_contamination,test_twin_leakage}.py`, CLI `schelling data build`.
Done when: adapter, split, contamination, guard and leakage tests pass; `schelling data build` reproduces identical manifests twice.

## Blocked: needs the human

M1 checkpoint: sign-off and three decisions needed (see `research/reports/M1.md`, section Findings).

## Done

| Date | Milestone / phase | Report | Signed off |
| --- | --- | --- | --- |
| 2026-10-26 | M0 Scaffold | research/reports/M0.md | yes |
