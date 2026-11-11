# Fork D: distilled prior, corrected on human data

**Status:** draft for human review (2026-11-11). Nothing here has been run. No money has been spent.
**Branch:** `fork/distill` (worktree `.claude/worktrees/fork-distill`), cut from `m5/licence-checks` at `463dd4c`.
**Decision record:** `docs/decisions/F001-distilled-prior-fork.md`.

Numbers carry `<!-- src: path#json.path -->` tags; paths are relative to the repo root.

## 1. The bet

The mainline model learns from 435 distinct questions. That is enough to learn people, and too few to
learn *questions*. A large language model has the opposite profile: it has read about almost every kind
of question, but it was never trained on how real people actually spread across the answers.

The fork's bet: **teach a small typed model the breadth of an LLM, then correct it with real human
answers.** If it works, the result is the "Jev-like" object the project wants: a fast, typed, calibrated
predictor that is at least as good as a frontier model at predicting people, at a small fraction of the
cost per decision.

Where the room is. On unseen questions (`heldout_item` val, Twin-2K-500 rows) the two-tower model is
already ahead of the item-mean baseline, but it is far from good:

| | two-tower (M4b, one seed) | B1 |
| --- | --- | --- |
| NLL | 1.2939 <!-- src: reports/M4b-tt-item-val/metrics.json#individual.by_source.twin2k.nll --> | 1.3534 <!-- src: reports/M4b-b1-heldout_item-val/metrics.json#individual.by_source.twin2k.nll --> |
| ECE (limit 0.05) | 0.0582 <!-- src: reports/M4b-tt-item-val/metrics.json#individual.by_source.twin2k.ece --> | 0.0479 <!-- src: reports/M4b-b1-heldout_item-val/metrics.json#individual.by_source.twin2k.ece --> |
| population TVD | 0.1813 <!-- src: reports/M4b-tt-item-val/metrics.json#population.twin2k.summary.tvd.mean --> | 0.2316 <!-- src: reports/M4b-b1-heldout_item-val/metrics.json#population.twin2k.summary.tvd.mean --> |

For comparison, the same model's population TVD on seen questions is 0.0573
<!-- src: reports/M4b-tt-a2-heldout_person-val/metrics.json#population.twin2k.summary.tvd.mean -->. The model
reproduces the answer distribution of a new question about three times worse than that of a known one,
and its calibration is over the limit there. That gap is the fork's target. The mainline levers (history
tower, paraphrases) were tested on seen questions, where the gap to B1 is about 0.01 NLL and there is
little left to win.

## 2. What "non-distribution-conditioned" means here

I read the request as: benchmark against existing models that were **not trained on, or conditioned on,
human answer distributions**. They predict people from general knowledge only.

| ID | Model | Person information | Role |
| --- | --- | --- | --- |
| E1 | Frontier closed LLM, zero-shot, asked for a probability per option | none (population prompt) | headline comparator |
| E2 | Same LLM | the 14 demographic attributes Schelling's person tower sees | headline comparator, per person |
| E3 | Open-weight LLM (`gpt-oss-120b` on Fireworks, A19 verified) | none and 14 attributes | the candidate teacher; also a cheap comparator |
| E4 | Jev (spec baseline B4) | person and situation as `state` | only if the human supplies a key; A10 is unverified |

E1 and E2 are the spec's B5 in two person conditions. Reference points that *are* trained on human
answers, shown beside them but not part of the claim: B1, B3, M4b, and Centaur (B6, Psych-101 only,
needs an 80 GB GPU, so "not run" unless hardware appears).

Which frontier model E1 is, and its price, are the human's choice; I have not verified any closed
model's price or terms (assumption FA1).

## 3. Benchmark design (campaign D01)

Three regimes, all on `val`, all through the sealed harness:

| Regime | Split | What a win means |
| --- | --- | --- |
| Seen questions, new people | `heldout_person` | person-level skill; B1 is strong here |
| New questions | `heldout_item` | the headline: no item mean exists |
| New experiments | 8 `heldout_experiment` folds, pooled | treatment-effect direction and size |

**How the external models enter the harness.** Each becomes a `Baseline` in
`src/schelling/baselines/llm_prompt.py` (the spec's path, not sealed). It replays cached responses; a
cache miss raises. `schelling eval` then scores it exactly like B0 to B3. No evaluator change is planned.

**Cost control.** Two call shapes:

- *Population prompt* (E1, E3): one call per question. `heldout_item` val has 28
  <!-- src: reports/M4b-tt-item-val/metrics.json#population.twin2k.n_items --> Twin-2K-500 items with
  population metrics, so this is tens of calls per model per split. Scored for every row by giving every
  person the same distribution.
- *Persona prompt* (E2, E3): one call per (person, question). The full `heldout_item` val partition is
  65804 <!-- src: reports/M4b-tt-item-val/metrics.json#individual.by_source.twin2k.n --> Twin-2K-500 rows,
  too many for a frontier model. Use a fixed, seeded subsample of persons (proposal: 100), and compare
  every model on those same rows from the saved `predictions.parquet` using the functions in
  `src/schelling/eval/metrics.py`. First task of D01 is to confirm this works without an evaluator change.

A dry run counts tokens and prints the cost before any paid call. The human approves a cap per model.

**Fairness to the LLMs.** A weak prompt would make our win meaningless.

- Two elicitation forms per model: verbalised probabilities per option, and token log-probabilities
  where the endpoint offers them (FA2). The better one is chosen on a *dev slice* (`heldout_person` val
  items not used for the headline) and frozen before `heldout_item` is scored.
- Post-hoc temperature for the LLM, fitted on the same dev slice. Schelling is allowed calibration on
  val, so the LLM gets it too. Both raw and calibrated numbers are reported.
- Option order randomised with a fixed seed, and the permutation guard reported for the LLM too.

**Contamination.** Twin-2K-500 is public, so an LLM may have seen its questions. Check with the spec's
memorisation probe: score each LLM on the original wording and on the paraphrases already generated in
M5 (`data/paraphrases/twin2k-gpt-oss-120b-v1.jsonl`). A large gap flags memorised items. Limits: the
paraphrases were written by E3's model, and A20 (meaning preserved) is only partly checked.

**Reported for every model, per regime:** NLL, ECE, population TVD and W1, SD ratio, entropy ratio,
permutation max |Δp|, treatment-effect correlation and sign agreement, dollars per million decisions,
and p50 latency. Differences are paired, with a 95 % bootstrap interval.

**A statistical warning.** The unit of generalisation on `heldout_item` is the item, and there are only
about 28 Twin-2K-500 val items. The bootstrap there must resample *items*, and its intervals will be wide.
If they are too wide to decide anything, the fix is item-fold cross-validation, which is a split change
and needs an `eval-change/` branch and the human's approval (open question 2).

## 4. Model design (campaign D02)

```text
 synthetic question bank ──► teacher LLM ──► soft labels ──┐
 (broad, typed, filtered)                                   ▼
                                           Stage A: pretrain (soft-label cross-entropy)
 Twin-2K-500 train (human answers) ──────► Stage B: fine-tune (unchanged M4b loss)
 val (human answers) ────────────────────► Stage C: per-head temperature
                                                            ▼
                                     two-tower model, same interface as M4b
```

**Question bank.** Generated offline once by the open-weight model, stored with a hash and generator id
(same pattern as the M5 paraphrase store).

- Typed like the schema: Noul, Choice (2 to 20 options), Score (2 to 10 levels).
- Covers the domains the simulator needs (attitudes, risk and time preferences, social dilemmas,
  consumer choices, everyday decisions) from a written topic grid, so coverage is by design, not by luck.
- Sizes 2 k, 10 k and 50 k questions, nested, for a dose-response curve.
- **Held-out protection:** every bank question that is a near-duplicate (existing MinHash check,
  Jaccard ≥ 0.8) of *any* Twin-2K-500 or Psych-101 question is removed, so no held-out item re-enters
  training in disguise. This needs question *text* of sealed items but never their answers (open
  question 1).

**Teacher labels.** For each bank question: one population distribution, plus K persona-conditioned
distributions (proposal K = 8). Personas are attribute profiles sampled from Twin-2K-500 *train* persons,
so the person tower sees realistic inputs. Labels are stored as `response_dist`; the schema and the
aggregate-record loss already exist in the spec.

**Why the teacher is the open-weight model, not the frontier one.** `gpt-oss-120b` is Apache-2.0 and
cleared for generating training text (A19). Closed providers' terms often restrict training a model on
their outputs; that is unverified (FA3), so a closed model is benchmark-only until the human says
otherwise. Consequence: the student cannot beat the frontier model by copying it. The win has to come
from the human-data correction, which is the scientific claim worth testing.

**The known risk: imported variance collapse.** LLM "silicon samples" are widely reported to be too
confident and too uniform across people. Three defences, each an ablation:

1. *Teacher calibration.* Before training, fit a per-question-type temperature that maps teacher
   distributions to human ones on Twin-2K-500 **train** items, and apply it to all bank labels.
2. *Distil into the prior path first.* The two-tower model has a question-to-prior path and a person
   residual. Breadth matters for the prior; person effects are better learned from real people. Variant:
   teacher labels train the prior path only.
3. *Human data has the last word.* Stage B fine-tunes on human answers and Stage C calibrates on val;
   the SD ratio, entropy ratio and ECE guards are binding.

**Training changes (default off, so M4b behaviour is bit-identical when unused).**
`data.pretrain_path`, `data.pretrain_generator`, `train.pretrain_steps` in the config; Stage A in
`cached_trainer.py` reuses the cached-vector path (the backbone stays frozen, so 50 k questions are
embedded once and training stays on CPU). Unfreezing the top encoder layers is a later idea; on this
machine (A12, no GPU) it may not fit the budget.

**Estimated generation cost.** Roughly 50 k questions × 9 label calls on a model priced at $0.15 in and
$0.60 out per million tokens (A19) is tens of dollars. This is arithmetic, not a measurement; the dry run
replaces it before any spend, and the human sets the cap.

## 5. What would count as success

Pre-registered in the contracts, summarised here:

- **D02 yes:** the distilled model beats the M4b re-run on `heldout_item` val NLL by a margin of at least
  three seed SDs, with the paired interval excluding 0, all binding guards passing, and no regression on
  `heldout_person`. A rising dose-response curve (2 k → 10 k → 50 k) is the supporting evidence.
- **Fork yes (D05):** on val, the distilled model is better than E1/E2 on NLL in the new-question regime
  with the interval excluding 0, not worse on population TVD and SD ratio, and at least 100 times cheaper
  per decision. Then, and only then, the human runs the test partition once.
- **Fork no:** the teacher carries no usable signal (D01), or distillation does not beat M4b after the
  budget (D02), or it only wins by breaking a guard (D03). Each is written up as a finding.

## 6. Campaigns and order

| ID | Question | Needs | Paid spend | Compute |
| --- | --- | --- | --- | --- |
| D01 | How good are non-human-conditioned models (E1 to E4) against B1 and M4b in the three regimes? | F001 approved; model choice and cap | yes | none |
| D02 | Does pretraining on calibrated teacher labels for a broad bank, then fine-tuning on human data, beat M4b on unseen questions? | D01 gate | yes (bank) | CPU |
| D03 | Does distillation import variance collapse or hurt the seen-question regime? | D02 artifacts | none | CPU, eval only |
| D04 | Does it improve treatment-effect direction and size on unseen experiments? | D02 winner | none | CPU, 8 folds |
| D05 | Head to head: distilled model against E1 to E4 on accuracy, population fidelity and cost | D01, D02, D03 | none new | eval only |

Gate after D01: D02 starts only if the calibrated teacher beats B1 on `heldout_item` val NLL. A teacher
that knows less than "no information" has nothing to distil.

## 7. Running beside the mainline without interfering

1. **Own branch and worktree.** All fork work is on `fork/distill` in its own worktree. The main checkout
   and its branch are never switched or edited.
2. **Own namespaces.** Docs in `research/forks/distill/`; configs in `configs/fork-distill/`; reports in
   `reports/FD-*`; runs in `runs/fd-*`; generated data in `data/synth/`. Campaign ids are `D01…`, so they
   cannot collide with the mainline's `C02…`. Assumptions are `FA1…` in this folder's `ASSUMPTIONS.md`.
3. **Mainline files are read-only for the fork:** `research/STATE.md`, `research/PROTOCOL.md`,
   `docs/SPEC.md`, `research/ASSUMPTIONS.md`, `CHANGELOG.md`. Fork state lives in
   `research/forks/distill/STATE.md`.
4. **Sealed things stay sealed:** `src/schelling/eval/`, `tests/golden/`, `splits/`, `data/manifests/`,
   and the test partition. `SCHELLING_ALLOW_TEST` is never set by the agent.
5. **Shared code is additive and default-off.** New modules where possible; any change to
   `cached_trainer.py` or `config.py` must leave existing configs bit-identical, proven by a test.
6. **The baseline is re-run inside the fork.** Every comparison uses M4b re-trained at the fork's SHA
   with the same seeds, so mainline changes cannot move the fork's reference.
7. **Mainline has the machine.** One 16 GiB laptop. Fork training runs at `nice 19`, one job, two
   threads, and is paused whenever a mainline job is running. API calls are network-only and do not
   compete.
8. **Separate money.** A fork spend cap approved by the human, a separate usage log
   (`cache/llm/usage-fork-distill.jsonl`), and a dry run before every paid batch.
9. **Merging is the human's call.** If the fork wins, it comes back through a normal decision record
   and PR. If it loses, the branch is kept as the record and nothing in mainline changes.

## 8. Open questions for the human

1. May the bank's near-duplicate filter read the question *text* (never answers) of sealed test items,
   so held-out questions cannot leak in through the bank? My recommendation: yes, text only, through a
   small audited script.
2. If `heldout_item` intervals over about 28 items are too wide, do you want an `eval-change/item-folds`
   proposal (like the LOEO folds of decision 0003)? My recommendation: decide after D01 shows the widths.
3. Which frontier model is E1/E2, and what is the spend cap for D01 and for the bank?
4. Is Jev (E4) in scope? It needs your key and the A10 check.
5. The fork depends on the Fireworks client and paraphrase store on PR 9. Merge PR 9 first, or keep the
   fork stacked on it?
