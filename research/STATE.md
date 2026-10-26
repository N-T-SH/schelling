# Project state

Keep this file short and current. Update it at the end of every work session and at every
checkpoint. A fresh agent should be able to resume from it alone.

## Now

- **Mode:** BUILD
- **Milestone:** M0 Scaffold (complete, awaiting sign-off)
- **Campaign:** none
- **Next step:** wait for human sign-off on `research/reports/M0.md`; then plan M1.
- **Done so far:** A12 verified (M4, 10 cores, 16 GiB, no CUDA GPU).

## Plan for the current milestone or phase

M0 Scaffold. Assumptions needed: A12 only (verified).

Files: `pyproject.toml` (hatchling, uv, extras train/api-baselines/dev), `.pre-commit-config.yaml`,
`.github/workflows/ci.yml`, `src/schelling/{__init__,schema,cli}.py`, empty subpackages per
SPEC layout, `scripts/export_schemas.py`, `schemas/*.json`.
Tests (written first): `tests/unit/test_schema.py` (rejections, JSON round trip, schema
snapshot), `tests/unit/test_cli.py` (`--help` and stubs exit cleanly).
Done when: fast suite, ruff, mypy --strict pass locally; CI file present.

## Blocked: needs the human

M0 checkpoint: sign-off needed (see `research/reports/M0.md`).

## Done

| Date | Milestone / phase | Report | Signed off |
| --- | --- | --- | --- |
