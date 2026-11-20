# Fork D: distilled prior, corrected on human data

**Status:** draft for human review. Written 2026-11-11; revised 2026-11-20 after D01-lite and the D02
gate (§3a), with the D01-lite recommendations folded in and a red-team review (§9). The D02 to D05
contracts in `campaigns/` are not yet changed to match; that waits for the human's approval of §9.
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
| E3 | Open-weight LLM (`gpt-oss-120b` on Fireworks, A19 verified) | none and 14 attributes | a candidate teacher; also a cheap comparator |
| E3b | DeepSeek V4.1 Flash (open weights, OpenRouter), reasoning high | none | candidate teacher; passed the D02 gate (§3a); licence for training on its outputs unverified (FA9) |
| E4 | Jev (spec baseline B4) | person and situation as `state` | only if the human supplies a key; A10 is unverified |

Psych-101 fine-tunes (Centaur, B6; Qwentaur) are **dropped** as teachers and headline comparators:
in D01-lite, Qwentaur-4B only tied a uniform guess on Twin-2K-500 decision items. The result stays as
a reported finding.

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

## 3a. What D01 found (2026-11-20)

Two runs, both on `val`. Full write-ups: `campaigns/D01-external-benchmark/D01-lite.md` (seen decision
questions, 100 persons) and `campaigns/D01-external-benchmark/D02-gate.md` (unseen questions).

1. **On unseen questions the LLM is the best model we have.** DeepSeek reasoning high, population
   prompt, no temperature: NLL 1.1874
   <!-- src: reports/FD-gate-deepseek-high16k-item-val/metrics.json#individual.by_source.twin2k.nll -->
   against B1 1.3534 <!-- src: reports/FD-gate-b1-item-val/metrics.json#individual.by_source.twin2k.nll -->
   and the M4b three-seed mean 1.2825 <!-- src: reports/FD-d01/m4b-item-summary.json#metrics.nll.mean -->.
   Paired against B1 the interval excludes 0
   (upper end -0.1125 <!-- src: reports/FD-gate/high16k-vs-FD-gate-b1-item-val.json#paired.ci_high -->).
   Population TVD 0.1604
   <!-- src: reports/FD-gate-deepseek-high16k-item-val/metrics.json#population.twin2k.summary.tvd.mean -->
   against M4b's 0.1813 <!-- src: reports/M4b-tt-item-val/metrics.json#population.twin2k.summary.tvd.mean -->.
   **The D02 gate is passed.**
2. **Reasoning is what makes it work.** Reasoning off is level with M4b and does not clear B1 with
   confidence (gate write-up).
3. **Calibration is off:** ECE 0.0835
   <!-- src: reports/FD-gate-deepseek-high16k-item-val/metrics.json#individual.by_source.twin2k.ece -->,
   over the spec limit. Teacher calibration (below) stays necessary.
4. **No variance collapse.** SD ratio 0.9811
   <!-- src: reports/FD-gate-deepseek-high16k-item-val/metrics.json#population.twin2k.summary.sd_ratio.mean -->
   on unseen questions. On seen decision items D01-lite found the opposite tendency (slightly too spread).
   FA6 is not supported so far.
5. **Personas carry no individual signal through an LLM** (D01-lite): persona prompts made a person's
   real answers less likely for DeepSeek. Person heterogeneity has to come from human data.
6. **On seen questions the training crowd (B1) wins by a wide margin** (D01-lite), so the fork can only
   add value where no crowd exists: new questions and new experiments.
7. **The LLM knows the direction of classic effects** (D01-lite bias test), where M4b on unseen
   experiments is at chance. Memorisation is not ruled out for either finding 1 or 7.
8. **Operations.** Reasoning-high replies sometimes think past the token limit and return nothing:
   1 of 28 Twin-2K-500 questions at 8,000 tokens (answered on a re-ask), and most long Psych-101
   trial prompts. Replies are not deterministic even at temperature 0; the cache is the record.

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

**Teacher labels (revised after D01).** For each bank question: **one population distribution, with
reasoning on.** Persona-conditioned labels are dropped (§3a finding 5), which cuts the label calls
9-fold. Labels are stored as `response_dist`; the schema and the aggregate-record loss already exist in
the spec. Guards on the labelling run: a reply that thinks past the token limit is re-asked once at a
higher limit (`reuse_max_tokens`), then dropped and counted; billed cost is read from the provider's
response (`scripts/fd_spend.py`). Bank questions are survey-style; long trial-by-trial prompts are out
(§3a finding 8).

**Teacher choice.** Two open-weight candidates, `gpt-oss-120b` and DeepSeek V4.1 Flash, chosen on the
gate set before the bank is generated; DeepSeek has passed the gate, `gpt-oss-120b` is not yet run.
DeepSeek's terms for training on its outputs are unverified (FA9) and must be checked first.
`gpt-oss-120b` is Apache-2.0 and cleared for generating training text (A19). Closed providers' terms often restrict training a model on
their outputs; that is unverified (FA3), so a closed model is benchmark-only until the human says
otherwise. Consequence: the student cannot beat the frontier model by copying it. The win has to come
from the human-data correction, which is the scientific claim worth testing.

**The feared risk: imported variance collapse.** LLM "silicon samples" are widely reported to be too
confident and too uniform across people. D01 did not see it at the population level (§3a finding 4),
so this is now a check rather than the main risk. Three defences, each an ablation:

1. *Teacher calibration.* Before training, fit a per-question-type temperature that maps teacher
   distributions to human ones on Twin-2K-500 **train** items, and apply it to all bank labels.
2. *Distil into the prior path only (now the default).* The two-tower model has a question-to-prior
   path and a person residual. Teacher labels train the prior path only; the person residual learns from
   human data only (§3a finding 5). Training both from teacher labels is the ablation.
3. *Human data has the last word.* Stage B fine-tunes on human answers and Stage C calibrates on val;
   the SD ratio, entropy ratio and ECE guards are binding.

**Training changes (default off, so M4b behaviour is bit-identical when unused).**
`data.pretrain_path`, `data.pretrain_generator`, `train.pretrain_steps` in the config; Stage A in
`cached_trainer.py` reuses the cached-vector path (the backbone stays frozen, so 50 k questions are
embedded once and training stays on CPU). Unfreezing the top encoder layers is a later idea; on this
machine (A12, no GPU) it may not fit the budget.

**Estimated generation cost (revised).** One reasoning-high label per question. D01-lite measured
about a sixth of a cent per DeepSeek reasoning-high call, so a 10 k bank is roughly 17 dollars and
50 k roughly 85 (arithmetic from a measured rate on different prompts, not a quote). The dry run
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
| D01 | How good are non-human-conditioned models (E1 to E4) against B1 and M4b in the three regimes? | F001 approved; model choice and cap | yes | none (D01-lite and the gate done, §3a) |
| D01b | Contamination probe: does the teacher's gate result survive paraphrased questions? | M5 paraphrases | under a dollar | none |
| D02 | Does pretraining on calibrated teacher labels for a broad bank, then fine-tuning on human data, beat M4b on unseen questions? | D01 gate | yes (bank) | CPU |
| D04 | Does it improve treatment-effect direction and size on unseen experiments? (raised in priority, §3a finding 7) | D02 winner, D01b | small (teacher labels for 8 folds) | CPU, 8 folds |
| D03 | Does distillation import variance collapse or hurt the seen-question regime? (demoted to a check, §3a finding 4) | D02 artifacts | none | CPU, eval only |
| D05 | Head to head: distilled model against E1 to E4 on accuracy, population fidelity and cost | D01, D02, D03 | none new | eval only |

Gate after D01: D02 starts only if the calibrated teacher beats B1 on `heldout_item` val NLL. A teacher
that knows less than "no information" has nothing to distil. **Passed on 2026-11-20 without
calibration** (§3a). Before D02: D01b, a three-seed M4b re-run at the fork SHA whose predictions are
kept, and the human's decision on §9.

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
   (`cache/llm/usage.jsonl` in the fork worktree), and a dry run before every paid batch.
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

## 9. Red team (2026-11-20, after the D02 gate)

The question: given what D01 found, is "distil an LLM into the two-tower model" still the right way to
reach the goal (realistic population outcomes on questions and experiments we have no data for)?

**Attacks, strongest first.**

1. **The cost argument for distillation mostly disappears.** The winning teacher needs one population
   call per *question*, not per decision: no person information goes in (§3a finding 5). A simulator
   asks each question once for the whole population, so a cached LLM prior costs a sixth of a cent per
   question, shared across every simulated person. D05 part 4 ("100 times cheaper per decision") is met
   by caching alone. What distillation still buys: no network or provider dependency at run time,
   questions generated on the fly without a call, deterministic output, and offline use. Those are real
   but they are product requirements, and nobody has said they are required.
2. **A simpler design has not been tried: use the LLM prior directly.** Feed the teacher's population
   distribution into the two-tower model as a per-question prior (for example a logit offset on the
   question-to-prior path), and let the person tower learn individual deviations from human data. It
   needs no bank, no pretraining stage and no licence for training on outputs, and it keeps the
   teacher's full quality instead of a compressed copy. If it is as good as distillation, distillation is
   not needed; if distillation is better, that is the result worth reporting. Right now D02 cannot tell.
3. **The student may not be able to hold what the teacher knows.** The two-tower model reads questions
   through a small frozen sentence encoder. The teacher's advantage comes from reasoning about the
   question (reasoning off is no better than M4b). Pretraining on 10 k soft labels may teach a frozen
   encoder's head little of that. Cheap check before any bank: fit the prior path to teacher labels on
   the 435 training questions and see how close it gets to the teacher on the 28 val questions.
4. **D02's decision rule cannot be measured on 28 items.** H1 asks for a margin of three M4b seed SDs
   (seed SD 0.0099 <!-- src: reports/FD-d01/m4b-item-summary.json#metrics.nll.sd -->), but a 95 % interval
   over 28 items is several times wider: the gate's against B1 runs from -0.2938
   <!-- src: reports/FD-gate/high16k-vs-FD-gate-b1-item-val.json#paired.ci_low --> to -0.1125
   <!-- src: reports/FD-gate/high16k-vs-FD-gate-b1-item-val.json#paired.ci_high -->. Choosing among
   distillation variants on these items will mostly fit noise. Item-fold cross-validation (open question
   2, an `eval-change/` proposal) is a precondition, not an option.
5. **The headline may be memorisation.** Twin-2K-500 is public. If the teacher's edge shrinks on
   paraphrased questions, distilling it buys recall of a public dataset, not knowledge of questions in
   general. D01b must run first; a bank filtered against Twin-2K-500 text does not protect against a
   teacher that has read the answers.
6. **A fine-tuned decision LLM was not considered either.** LoRA on a small open LLM trained on
   Twin-2K-500 training answers (not Psych-101) combines an LLM's language knowledge with human
   correction in one network-free model. Against it: it answers in one pass without reasoning, the
   setting where the LLM was no better than M4b, and Psych-101 tuning of the same base did not
   transfer. It belongs in the comparison rather than being assumed away.
7. **Persons.** The LLM adds nothing per person, and on seen questions the person tower adds little over
   B1. For the simulator, most value is in population distributions and treatment effects on new
   material. Both designs (distilled or direct prior) put the person tower on human data, so this
   attack does not separate them, but it does limit how much either can claim per person.

**Defences that hold.** Distillation does give a self-contained model, which matters if the simulator
must run without network access or create questions at run time faster than an API allows. The human
data still has the last word in both designs. Calibration and guards apply unchanged.

**Verdict.** Distillation is not clearly the right primary approach any more. The gate shows the
teacher is worth using; it does not show that compressing it is worth the cost and risk. Recommended
revision for the human's approval:

1. **Run first, cheaply:** D01b (contamination probe, under a dollar), the three-seed M4b re-run at the
   fork SHA, and the encoder-capacity check (attack 3, CPU only, no spend).
2. **Reframe D02 as a four-arm comparison** under one contract: M4b (A0); M4b with the cached LLM
   prior injected, no distillation (A1); M4b pretrained on distilled teacher labels (A2); a small LLM
   fine-tuned on Twin-2K-500 training answers (A3). The question becomes "which way of using an LLM is
   best", decided on item folds (attack 4). Pros, cons and decision rules for each arm:
   `research/forks/distill/D02-OPTIONS.md`.
3. **Ask the human** whether offline, network-free running is a requirement. If not, and arm 2 matches
   arm 3, the fork recommends the direct prior and skips the bank.
4. **Move D04 up** with the winning arm, since treatment effects on new experiments are where the
   LLM showed the most that M4b lacks.

