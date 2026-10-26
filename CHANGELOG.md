# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Project brief: build spec, coding-agent instructions, research protocol, assumptions
  register and templates.
- M0 scaffold: package skeleton, typed request/response schemas with JSON Schema export, CLI stubs, ruff/mypy/pre-commit, CI workflow.
- M1 data: unified `DecisionRecord`, Psych-101 and Twin-2K-500 adapters, manifests with checksums, six splits, MinHash contamination report, sealed-test loader guard, near-duplicate removal from heldout_item, `schelling data build|splits`.
- M2 harness: metrics, population fidelity, retest ceiling, robustness, efficiency, `metrics.json` with provenance, `schelling eval`, baselines B0 to B3, `scripts/audit_report.py`.
- Decision 0001: Psych-101 excluded from the M4 baseline bar until C04; draft treatment-effects pairing table for review.
- Eval change EC1: treatment-effect metrics (16 pre-registered contrasts), leave-one-experiment-out split folds, `schelling effects-pool`.
