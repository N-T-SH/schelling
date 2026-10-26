# Project state

Keep this file short and current. Update it at the end of every work session and at every
checkpoint. A fresh agent should be able to resume from it alone.

## Now

- **Mode:** BUILD
- **Milestone:** M3 Model v0 (planned, not started)
- **Campaign:** none
- **Next step:** human decides on merging the eval-change branch and gives the go for M3; then check A06/A07 and write M3 tests first. M3 work happens on `main`.
- **Done so far:** M0 signed off 2026-10-26. M1 built; real data built (Psych-101 1% sample).

## Plan for the current milestone or phase

M3 Model v0 (SPEC milestone 4). Evaluator sealed since M2 sign-off; nothing below touches `eval/`, `splits/`, `data/manifests/`.

Scope: `SchellingConfig`, `SchellingModel` with Noul, Choice and Score heads on a shared encoder, v0 serialised person context, per-head temperature fields, save/load, Auto-class registration, `Predictor` API, `simulate.sample_population`, `schelling simulate`. Untrained (random head init); training is M4.

Assumptions to check first: A06 (ModernBERT-base and -large load and run one CPU forward pass with the pinned transformers; public download of about 600 MB and 1.6 GB), A07 (`register_for_auto_class` copies the custom files and `AutoModel.from_pretrained(..., trust_remote_code=True)` works in a fresh process). A13 (bitwise determinism) needs the trainer, so I propose moving it to M4 and checking only forward-pass determinism in M3.

Design (decisions flagged in the report if they bind later work):
- Backbone loaded through `AutoModel`; tests use a tiny random BERT built from a committed tiny WordPiece vocab (`scripts/make_fixtures.py` extended), no network.
- Context text = `[PERSON] attributes + last k history items [STATE] ... [QUESTION] ...`; h = pooled (mean over attention mask) representation. Options and levels are encoded separately by the same backbone (mean pooled) so padded options can be masked.
- Noul: MLP(h) to logit, sigmoid, probs = [1 - p, p].
- Choice: logit_i = <W h, e_i> / tau, padded options masked to minus infinity; permutation equivariant by construction.
- Score: CORN-style conditional thresholds. t_k = MLP([h; (e_k + e_{k+1}) / 2]) for k in 0..L-2, P(y > k | y > k-1) = sigmoid(t_k), so P(y > k) is a product of probabilities and is monotone decreasing by construction; P(y = k) from successive differences; expected_level from the distribution.
- Temperatures: three scalar fields in the config (default 1.0), applied to logits; fitting is M4.
- Each question is a separate forward pass, so answers are independent of other questions in the request by construction.
- `Predictor.decide`, `decide_batch` (padding-invariant), `encode_person` returns the serialised person text in v0 (embedding caching arrives with person encoder v1 in M5; stated in the docstring).

Tests first: `tests/unit/test_heads.py` (sums to 1 atol 1e-5, padded options zero, Noul p_yes in (0,1) and probs consistent, Score sums to 1 / expected level in range / monotone cumulative, no NaN or Inf for empty history, empty attributes and maximum-length input), `tests/property/` with Hypothesis (option permutation permutes probabilities, adding question B leaves answer A unchanged, batch equals single, all atol 1e-5), `tests/integration/` (save_pretrained then from_pretrained identical outputs atol 1e-6, safetensors only, fresh-process AutoModel with trust_remote_code, `schelling simulate` smoke test on fixtures, sampler determinism).

Risks and open points:
- CI cost: `uv sync --extra train` pulls CUDA PyTorch wheels on Linux (about 2 GB). Plan: pin the CPU PyTorch index for Linux in `pyproject.toml` (`tool.uv.sources`), and verify CI time; if CI becomes too slow, run torch tests in a separate job.
- Dependency pins: torch, transformers and safetensors are declared with lower bounds only; M3 locks versions in `uv.lock` and records them in A06.
- `PersonContext.history` is serialised as text in v0; long histories are truncated to k (default 20) by recency. Tokeniser max length 512 for the tiny model, 1024 for ModernBERT (to confirm in A06).
- The untrained model is not evaluated with `schelling eval` in M3 (no trained weights; it would only measure random initialisation). Wiring `eval --model <path>` to a `SchellingModel` arrives with M4, and the evaluator needs no change for it because it only needs a `predict_proba` on the shared protocol.

Done when: head, property and round-trip tests pass with the tiny backbone; ruff, mypy --strict and the fast suite are green; CI passes; `research/reports/M3.md` written.

## Blocked: needs the human

Branch `eval-change/treatment-effects` awaits your merge approval (`research/reports/EC1-treatment-effects.md`). M3 not started; waiting for the go.

## Done

| Date | Milestone / phase | Report | Signed off |
| --- | --- | --- | --- |
| 2026-10-26 | M0 Scaffold | research/reports/M0.md | yes |
| 2026-10-26 | M1 Data | research/reports/M1.md | yes (near duplicates removed) |
| 2026-10-26 | M2 Harness + baselines | research/reports/M2.md | yes; Psych-101 excluded from M4 bar (docs/decisions/0001) |
