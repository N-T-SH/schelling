# D01b: is the teacher's edge on unseen questions memorisation?

**Status:** pre-registered 2026-11-20, before any paraphrase call. Approved by the human (cap one
dollar, chat "Approve all three"). Results are appended below the line; nothing above it changes after
the run.

## Question

DeepSeek V4.1 Flash with reasoning high beat B1 on the 28 unseen Twin-2K-500 questions of
`heldout_item` val (`D02-gate.md`). Twin-2K-500 is public. Does the edge survive when the same
questions are reworded?

## Design

- **Items:** the `heldout_item` val Twin-2K-500 questions that have at least one M5 paraphrase in
  `data/paraphrases/twin2k-gpt-oss-120b-v1.jsonl` (24 of 28; 129 paraphrases in all). Options and
  levels are unchanged; only the question text is reworded.
- **Teacher:** the frozen gate prompt and settings (`configs/fork-distill/gate-deepseek-high16k.json`),
  population prompt, one call per paraphrase.
- **Control:** the M4b two-tower model, seed 0 (`runs/tt-item/final`, the model behind
  `reports/M4b-tt-item-val`). It never saw these questions or their answers, so its change under
  rewording measures what rewording costs a model that cannot have memorised them.
- **Per item:** mean row NLL over that item's val rows. For paraphrases, the mean over the item's
  paraphrases of that number. B1 and the original-wording predictions are read from the saved gate runs.

## Metrics

- `edge_orig` = mean over items of (B1 NLL − teacher NLL, original wording).
- `edge_para` = mean over items of (B1 NLL − teacher NLL, paraphrased).
- `retention` = `edge_para` / `edge_orig`.
- Paired item bootstrap (10,000 resamples) for the teacher's paraphrased NLL against B1, and for the
  teacher's and M4b's change under rewording.

## Decision rule

- **Pass** (no sign that the edge is memorisation): `retention` is at least one half **and** the
  teacher's paraphrased NLL beats B1 with the interval excluding 0. A1 proceeds.
- **Fail:** `retention` below one half. Stop A1 and A2 as planned in `D02-OPTIONS.md`; report.
- **Unclear:** `retention` at least one half but the interval against B1 includes 0. Report; the human
  decides.
- The control is reported beside the result. If M4b loses as much as the teacher under rewording, a
  drop is a rewording cost, not memorisation, and the write-up says so.

## Limits

The paraphrases were written by `gpt-oss-120b` and checked only partly for meaning (mainline A20).
Rewording tests verbatim recall of the question; it cannot rule out a teacher that learned the
dataset's answer patterns in a way that survives rewording.

---

## Results

**Pass.** Run 2026-11-20; 129 paraphrase calls for 24 questions, billed about 12 cents (fetch summary,
not a metrics file: `runs/fd-gate/d01b-fetch.json`, git-ignored). 4 paraphrase replies did not parse
and count as a uniform guess, which can only hurt the teacher. Source: `reports/FD-d01b/probe.json`, git e9145d55, clean.

| | Value |
| --- | --- |
| `edge_orig` (B1 minus teacher, original wording) | 0.2120 <!-- src: reports/FD-d01b/probe.json#edge_orig --> |
| `edge_para` (B1 minus teacher, reworded) | 0.2002 <!-- src: reports/FD-d01b/probe.json#edge_para --> |
| `retention` | 0.9441 <!-- src: reports/FD-d01b/probe.json#retention --> |

Paired over the 24 items (negative favours the first model):

| Comparison | Difference | Interval |
| --- | --- | --- |
| Teacher reworded − B1 | -0.2002 <!-- src: reports/FD-d01b/probe.json#teacher_para_vs_b1.mean_diff --> | -0.3052 <!-- src: reports/FD-d01b/probe.json#teacher_para_vs_b1.ci_low --> to -0.1055 <!-- src: reports/FD-d01b/probe.json#teacher_para_vs_b1.ci_high --> |
| Teacher reworded − teacher original | 0.0118 <!-- src: reports/FD-d01b/probe.json#teacher_para_vs_orig.mean_diff --> | -0.0021 <!-- src: reports/FD-d01b/probe.json#teacher_para_vs_orig.ci_low --> to 0.0277 <!-- src: reports/FD-d01b/probe.json#teacher_para_vs_orig.ci_high --> |
| M4b reworded − M4b original (control) | 0.0607 <!-- src: reports/FD-d01b/probe.json#m4b_para_vs_orig.mean_diff --> | 0.0017 <!-- src: reports/FD-d01b/probe.json#m4b_para_vs_orig.ci_low --> to 0.1458 <!-- src: reports/FD-d01b/probe.json#m4b_para_vs_orig.ci_high --> |

**Reading.** Retention is well above one half and the reworded teacher still beats B1 with the
interval excluding 0, so the decision rule says pass: A1 proceeds. The control strengthens it: M4b,
which cannot have memorised these questions, loses more under rewording than the teacher does. The
rewording has a real cost, and the teacher absorbs it; that is the opposite of what verbatim recall
would show. The limit stated above stands: this rules out recall of the wording, not knowledge of the
dataset's answers that survives rewording.
