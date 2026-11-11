# Campaign D01: external benchmark (models not conditioned on human answers)

**Status:** draft. Not approved. Nothing run, nothing spent. Items marked **[proposal]** need your edit or approval.

## Question

On validation data, how well do existing models that were never trained on human answer distributions
(E1 to E4 in `research/forks/distill/DESIGN.md` §2) predict Twin-2K-500 answers, compared with B1 and
the M4b two-tower model, in three regimes: seen questions, new questions, new experiments?

## Hypothesis

- H1 (can fail). On `heldout_person`, no external model beats B1 on NLL, even after temperature
  calibration. Reason: B1 knows each question's real answer distribution (A09).
- H2 (can fail). On `heldout_item`, the best calibrated external model beats B1 and M4b on NLL. Reason:
  no item mean exists there, and the LLM knows far more questions than our 435.
- H3 (can fail). External models are under-dispersed: SD ratio below 0.85 when their per-person
  probabilities are sampled.

H2 is the gate for D02. If H2 is false for the open-weight model E3, there is nothing to distil.

## Metrics

- **Primary:** Twin-2K-500 NLL, `src/schelling/eval/metrics.py::nll`, read from
  `metrics.json#individual.by_source.twin2k.nll`. Lower is better.
- **Recorded for every model:** ECE, population TVD and W1, SD ratio, entropy ratio, permutation
  max |Δp|, pooled treatment-effect correlation and sign agreement, dollars per million decisions
  (from the usage log), p50 latency.
- **Paired comparisons:** bootstrap, 10 000 resamples, 95 % interval. Resample persons on
  `heldout_person`; resample **items** on `heldout_item`.
- No guard is binding in D01: this campaign measures, it does not select a model of ours.

## Data

- **Splits:** `heldout_person`, `heldout_item`, `heldout_experiment_f0` to `f7`; `val` only.
- **Sources:** Twin-2K-500 rows. Psych-101 stays out of the claim (decision 0001).
- **Dev slice for prompt choice and LLM temperature:** a seeded 20 % of `heldout_person` val items
  **[proposal]**. Never the `heldout_item` val items.
- **Persona subsample:** 100 seeded val persons **[proposal]**, the same for every model.
- **Leakage rule:** prompts contain the question, options, and (E2/E3 persona form) the 14 attributes
  only. Never `persona_text`, `persona_json` or `full_persona` (A04), never other answers.

## Models compared

| Model | Source of numbers |
| --- | --- |
| B1 item mean | re-run at the fork SHA |
| M4b two-tower | re-trained at the fork SHA, seeds 0 to 2 |
| E1 frontier LLM, population prompt | new, cached |
| E2 frontier LLM, persona prompt | new, cached, persona subsample |
| E3 `gpt-oss-120b`, both prompts | new, cached |
| E4 Jev | only if you supply a key and A10 is verified; otherwise "not run" |

## Budget

- **Idea cycles:** none (measurement only). Prompt variants: at most 2 elicitation forms × 2 wordings per
  model, chosen on the dev slice, then frozen.
- **Compute:** M4b re-runs only (3 seeds, about the M5 baseline cost), at fork priority.
- **Paid API spend:** **[proposal]** cap of $5 for E3 and a separate cap you set for E1/E2 after the
  dry-run token count. No call before the dry run is shown to you.

## Tasks, in order

1. Verify assumptions FA1 to FA4 (`research/forks/distill/ASSUMPTIONS.md`). No code.
2. Confirm that subsample comparisons can be made from saved `predictions.parquet` with the functions in
   `src/schelling/eval/metrics.py`, without an evaluator change. If not, stop and report.
3. Tests first, then `src/schelling/baselines/llm_prompt.py`: prompt builder, probability parser (rejects
   malformed output, renormalises, floors at a small epsilon recorded in the config), cache replay,
   cache-miss error, option-order seed. Fixtures only; no network in tests.
4. Dry run: token counts and cost per model and split. **Stop for approval.**
5. Paid runs, cached. Then `schelling eval` for each model and split.
6. Memorisation probe with the M5 paraphrases.
7. Report `report.md` through `scripts/audit_report.py`.

## Success, failure and early-stop criteria

- **Done:** a table of every model × regime with sourced numbers and paired intervals, and a yes/no on
  H1 to H3.
- **Gate for D02:** E3, temperature-calibrated on the dev slice, beats B1 on `heldout_item` val NLL with
  the item-bootstrap interval excluding 0 **[proposal]**.
- **Stop early:** an external model beats the test-retest ceiling or beats B1 on `heldout_person` by more
  than 20 % NLL (treat as contamination or a leak, AGENTS.md §5); more than 5 % of responses fail to
  parse; spend would exceed a cap.

## Frozen references

- Evaluator git SHA (`src/schelling/eval/`): `40f458c14f943034ca2c3afba0d2ac6145241ed2` (re-check at approval).
- Split manifest hashes: `heldout_person` `c77fe1b963008c710d8f9740ad977a54cd69b19e01943ec96095e509e2b00297`;
  `heldout_item` `7d11f16086a89ac98dbf8e96f72f82baeb3735f3663cf7f69609fbd77b350a57`; LOEO folds filled at approval.
- Data manifest `twin2k.json`: `6d0f099d415efeabef3853ec8af311f354f89c975c02a5a7d5985afc6efabcec`.

## Final-evaluation commands

None in D01. The test partition is used once, in D05, by the human.
