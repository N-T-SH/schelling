# Fork D assumptions

Same rules as `research/ASSUMPTIONS.md`: unverified until checked, with evidence. Kept here to avoid
merge conflicts with the mainline register; folded in on merge. Mainline assumptions the fork relies on:
A04 (persona leakage), A09, A10 (Jev, unverified), A12 (hardware), A19 (Fireworks, verified), A20
(paraphrase meaning, partly true).

| ID | Assumption | Why it matters | How to check | Needed by | Status | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| FA1 | The frontier model chosen for E1/E2 is reachable through an OpenAI-compatible endpoint, and its current price per token is known. | D01 cost and the cost claim in D05. | Human names the model; read the provider's current pricing and API pages; record URL and date. | D01 | unverified | |
| FA2 | The chosen endpoints return token log-probabilities for answer options. | Second elicitation form in D01; often better calibrated than verbalised numbers. | Read the API docs; one cached smoke call per model after spend approval. | D01 | unverified | |
| FA3 | The terms of the frontier provider restrict training a model on its outputs, so it cannot be the teacher. | Teacher choice. If false, a frontier teacher becomes an option for a later idea. | Read the provider's current terms of service; record URL, clause and date. | D02 | unverified | |
| FA4 | `gpt-oss-120b` on Fireworks follows a strict JSON output format for per-option probabilities at a parse-failure rate under 5 %. | Bank labels and E3. | 50-call cached smoke test after spend approval; count parse failures. | D01 | unverified | |
| FA5 | Per-row predictions saved by `schelling eval` (`predictions.parquet`) are enough to compare models on a fixed person subsample using `src/schelling/eval/metrics.py`, with no evaluator change. | Persona-prompt benchmark at affordable cost. | Read the columns of an existing `predictions.parquet`; recompute NLL for the full file and match `metrics.json`. | D01 | unverified | 2026-11-11: file exists at `reports/M4b-tt-item-val/predictions.parquet`; contents not inspected. |
| FA6 | LLM-simulated answer distributions are under-dispersed relative to people. | Motivates teacher calibration and D03. | Measured directly in D01 (H3). | D02 | unverified | Believed from published work; not checked in this repo. |
| FA7 | `heldout_item` val has enough Twin-2K-500 items for an item-level bootstrap to resolve the D02 margin. | Whether D02 can give a decision at all. | Compute the item-bootstrap interval width for M4b against B1 from saved predictions. | D02 | unverified | 2026-11-11: 28 items carry population metrics (`reports/M4b-tt-item-val/metrics.json#population.twin2k.n_items`); interval width not computed. |
| FA8 | A generated question bank can cover the planned topic grid with under 20 % removed by the duplicate and near-duplicate filters. | Bank cost and breadth. | Generate 200 questions after spend approval; measure removal rate; hand-read 50. | D02 | unverified | |
