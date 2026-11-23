# D02 step 2–3, arm A1: does a cached teacher prior make M4b better on unseen questions?

**Status:** pre-registered 2026-11-23, before any A1 run was trained or scored. The human approved
building A1 (chat: "Build A1") and the teacher spend (cap 1.50 dollars, about 1.00 billed). Results
are appended below the line; nothing above it changes after the first A1 number exists.

## Question

Does A1 (the M4b two-tower model with the teacher's cached population distribution added to its
item prior) beat A0 (M4b alone) on unseen Twin-2K-500 questions? This is `D02-OPTIONS.md` step 3.1.

## Arms

- **A0:** `configs/twotower-a.yaml`, `split=heldout_item_f{0..4}`, seeds 0, 1, 2. Unchanged M4b.
- **A1:** the same config and seeds plus `model.teacher_prior_path=runs/fd-a1/teacher_prior.json`.
  The prior for a question becomes the prior network's output plus `w_task × teacher offset`, where
  the offset is the teacher's distribution in the head's logit form and `w` (one per task type, start
  1) is learned on training answers. Code: `src/schelling/model/heads_tt.py`, `infer/teacher.py`
  (commits 302371b, 3358148).
- **Teacher table:** `scripts/fd_a1_prior.py` from the response cache of
  `configs/fork-distill/a1-deepseek-high32k.json` (the frozen gate prompt; replies that ran out of
  tokens re-asked at 32,000). 529 non-test Twin-2K-500 questions, 3 without a parsed reply (QID58,
  QID60, QID278_11), which get the plain prior. Built from `train` and `val` question text only.
- **References, no training:** B1 item mean, and the teacher alone (R-LLM: the same spec scored by
  `schelling eval`, no temperature), each on every fold's val.

Everything else is identical between A0 and A1, including the trainer's temperature fit on a val
subset, which both arms already do.

## Metric and comparison

- **Primary:** Twin-2K-500 rows of each fold's `val`, scored by the sealed harness. Per question, the
  mean row NLL, averaged over the three seeds. Questions pooled over the five folds: each is val in
  at most one fold (decision 0007; near-duplicates of training questions are dropped from val), 282
  Twin-2K-500 questions in all.
- **Paired test:** A1 − A0 per question, 10,000 item-bootstrap resamples. Negative favours A1.
- **Also reported:** A1 − teacher alone, A1 − B1, the learned weights, per-fold NLL, seed SD.

## Decision rule (from `D02-OPTIONS.md` step 3)

- **A1 becomes the default design** if the A1 − A0 interval lies wholly below 0 **and** A1 passes
  the guards, each read as the mean over the 15 runs:
  - option-permutation max |Δp| ≤ 0.01;
  - median population SD ratio within 0.85 to 1.15;
  - ECE ≤ 0.05, **or** no higher than A0's. (The 0.05 limit is set for `heldout_person`; A0 itself
    misses it on unseen questions on two of three seeds, so a design that does not make it worse
    passes here. Both numbers are reported.)
- **A1 fails** if the interval includes 0 or lies above it, or a guard fails. Then step 3.2 applies:
  A2 is considered.
- **Too good:** if A1 lowers pooled NLL by more than 20 % against A0, stop and look for a leak before
  reporting (AGENTS.md §5).
- A1 beating the teacher alone is not required; it is reported because it says whether the person
  tower and human correction add anything to the teacher's per-question answer.

## Limits

- The seen-question no-regression check (`heldout_person`) is not part of this run; it follows if A1
  passes.
- The teacher may know Twin-2K-500 answers in a way that survives rewording (D01b limit).
- Three of 529 questions have no teacher answer; A1 is scored on them with its plain prior.

---

## Results

Not run yet.
