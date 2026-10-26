# Research protocol

This adapts the phases of Sakana AI's Agentic Scientist (task contract → investigation →
ablations → sealed final evaluation → post-hoc analysis → paper) to a single coding agent
working in this repository, with human checkpoints. The key idea carried over: **the
evaluator is the scientific contract.** The agent optimises against it, so it is fixed
before the search begins and nobody in the search can change it.

Each campaign answers **one** question and lives in `research/campaigns/<id>-<slug>/`:

```text
research/campaigns/C01-person-conditioning/
├── task.md              # Phase 0 contract (human-approved)
├── ideas/               # Phase 1: one file per idea, pre-registered
│   └── I01-history-cf.md
├── experiments.jsonl    # every run, append-only (one JSON object per line)
├── ablations.md         # Phase 2 results table
├── final.md             # Phase 3: the single test-set result
├── posthoc.md           # Phase 4 analyses
└── report.md            # Phase 5 write-up (must pass audit_report.py)
```

Templates are in `research/templates/`.

---

## Phase 0: Contract (human checkpoint ①)

Copy `research/templates/campaign_task.md` to the campaign folder and fill it in:

- **Question** and **hypothesis**, stated so the result can come out false.
- **Primary metric** and how it is computed (name the function in `src/schelling/eval/`), plus
  guard metrics that must not regress (ECE, SD ratio, permutation invariance).
- **Split and partition**: which split; search uses `val` only.
- **Baselines**: which of B0–B6 the winner must beat.
- **Budget**: maximum idea cycles (default 5), seeds per run (default 3 in search, 5 in
  ablations), maximum compute time.
- **Success and kill criteria**: what result counts as a yes, a no, and "stop early".
- **Frozen references**: git SHA of `src/schelling/eval/`, split manifest hash, data manifest hash.

Stop. The human approves or edits the contract. Nothing runs before approval.

## Phase 1: Investigation (search on `val` only)

Repeat for at most the budgeted number of idea cycles:

1. **Pre-register** the idea in `ideas/Ixx-<slug>.md` *before* running it: the change, why it
   should help, the predicted direction and rough size of the effect on the primary metric,
   and what result would make you drop it.
2. **Implement** on a branch `exp/<campaign>/<idea>`. Model and training code may change;
   the evaluator, splits and data manifests may not.
3. **Run** with the budgeted seeds through the CLI (`schelling train`, `schelling eval --split <s>
   --partition val`). Never write ad-hoc metric code inside the experiment.
4. **Record** every run, including failures and crashes, as one line in `experiments.jsonl`
   (schema below).
5. **Compare** to the current best using the mean across seeds and a paired comparison over
   the same items/persons. Keep the best idea; note what you learned.

Rules: no peeking at `test`; no changing the metric to suit the idea; ideas that need a
paid API or new data source go back to the human first. Optimise the primary metric, but an
idea that improves it while breaking a guard metric is a failure, not a win.

## Phase 2: Ablations (human checkpoint ②)

Run the winner, every required baseline, and one ablated variant per component the winner
added (e.g. without person history, without calibration), with ≥ 5 seeds each, on `val`.
Report in `ablations.md` as a table: variant, primary metric mean ± SD, guard metrics,
paired difference vs winner with a 95 % bootstrap interval (resampling persons).

Stop. The human decides whether the winner goes to final evaluation.

## Phase 3: Final evaluation (human runs it)

The human sets `SCHELLING_ALLOW_TEST=1` and runs the exact frozen commands you prepared in
`final.md` (one command per model, fixed seeds, frozen config hashes). The test set is used
**once** per campaign. No changes to code or configs after this point; if something is
wrong, it is reported, not fixed and re-run.

## Phase 4: Post-hoc analysis (on existing artifacts only)

No new training. Mine the saved predictions and metrics for:

- subgroup results and the worst-group gap;
- calibration (reliability diagram data) and SD-ratio per item;
- the 20 items with the largest error, with a one-line hypothesis each;
- **gaming checks**: is the SD ratio sitting at a guard bound? does the gain come from a
  handful of items? does performance on paraphrased or option-shuffled items hold?
- consistency between `val` and `test` results (a large gap is a finding).

## Phase 5: Write-up

`report.md` (template provided) leads with the answer to the campaign question, then a
**claims table**: each claim, the number, and its source tag
`<!-- src: path/to/metrics.json#json.path -->`. Run `uv run python scripts/audit_report.py
research/campaigns/<id>/report.md`; it must pass. Close with limitations and the next
question. Update `research/FINDINGS.md` with a one-paragraph summary and link.

---

## `experiments.jsonl` schema

One JSON object per line, append-only:

```json
{
  "run_id": "C01-I02-s1",
  "campaign": "C01",
  "idea": "I02",
  "phase": "investigation",
  "git_sha": "abc1234",
  "dirty": false,
  "config_path": "configs/c01/i02.yaml",
  "config_hash": "…",
  "split": "heldout_person",
  "partition": "val",
  "seed": 1,
  "status": "ok",
  "metrics_path": "reports/C01-I02-s1/metrics.json",
  "primary": {"name": "nll", "value": 1.234},
  "guards": {"ece": 0.031, "sd_ratio": 0.94},
  "wall_clock_s": 812,
  "notes": "one line"
}
```

`status` is one of `ok`, `failed`, `crashed`, `timeout`. Values come from the metrics file,
never typed by hand: write a helper (`scripts/log_run.py`) that builds the line from
`metrics.json`.

---

## Planned campaigns (order may change after results)

| ID | Question | Needs |
| --- | --- | --- |
| C01 | Does conditioning on a person's answers to other items beat the item-mean baseline (B1) on unseen people in Twin-2K-500, without SD-ratio collapse? | M2 |
| C02 | Does sampling from calibrated per-person probabilities preserve population variance, or is a diversity regulariser needed? | M4 |
| C03 | Does the model generalise to unseen items (`heldout_item`) using question text? | M4 |
| C04 | What is the right unit of prediction for Psych-101 (single trial vs trial-with-history), judged on `heldout_person` NLL and leakage? | M4 |
| C05 | Does pretraining on LLM-predicted soft labels help or import variance collapse? | Human approval for API spend |
