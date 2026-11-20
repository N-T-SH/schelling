# Fork D: status, the four ways to use an LLM, and how to decide

**Date:** 2026-11-20. **For:** the human's review. **Replaces nothing yet:** `DESIGN.md` §9 points here;
the D02 to D05 contracts in `campaigns/` change only after approval.

Numbers carry `<!-- src: path#json.path -->` tags (paths from the repo root) and pass
`scripts/audit_report.py`.

## 1. Where the fork stands

The goal: realistic population outcomes on questions and experiments we have no human data for. On
questions we *do* have data for, the training crowd (B1) is hard to beat, so new material is where the
fork can matter.

| Step | Status | Result |
| --- | --- | --- |
| D01-lite: LLMs on seen decision questions | done, closed | general LLM beats a uniform guess, reasoning helps a lot, personas add nothing per person, Psych-101 tuning does not transfer |
| D02 gate: LLM on unseen questions | **passed** | DeepSeek V4.1 Flash, reasoning high, beats B1 and M4b |
| D01b contamination probe | not built | needed before trusting the gate |
| D02 to D05 | not approved | this document proposes how to run D02 |

The gate, Twin-2K-500 rows of `heldout_item` val (28 unseen questions), NLL (lower is better):

| Model | NLL |
| --- | --- |
| B1 item mean | 1.3534 <!-- src: reports/FD-gate-b1-item-val/metrics.json#individual.by_source.twin2k.nll --> |
| M4b two-tower, three-seed mean | 1.2825 <!-- src: reports/FD-d01/m4b-item-summary.json#metrics.nll.mean --> |
| DeepSeek, reasoning off | 1.2692 <!-- src: reports/FD-gate-deepseek-off-item-val/metrics.json#individual.by_source.twin2k.nll --> |
| DeepSeek, reasoning high | 1.1874 <!-- src: reports/FD-gate-deepseek-high16k-item-val/metrics.json#individual.by_source.twin2k.nll --> |

Two facts shape everything below. **The edge comes from reasoning:** without it the LLM is level with
M4b. **The edge is per question, not per person:** the winning prompt contains no person at all, and
persona prompts did not help (D01-lite).

Spend so far on OpenRouter: about four dollars for D01-lite, and 0.1251
<!-- src: reports/FD-gate/spend.json#total_billed_usd --> for the gate.

## 2. The arms

All arms are scored the same way (§4). "Reference" rows are not candidates; they are what a candidate
must beat.

### A0. M4b two-tower (reference)

A frozen ModernBERT-base encoder turns question, options and person attributes into vectors; small
trained heads give a per-question prior plus a per-person shift. Trained on Twin-2K-500 answers only.

- **Pros:** runs on a laptop CPU, no network, deterministic; questions and people are cached, so each
  decision is a little vector arithmetic; typed, calibrated output; already built and measured.
- **Cons:** understands a new question only through a small frozen encoder and 435 training questions,
  so it is weak exactly where the fork wants to help; its treatment-effect predictions on unseen
  experiments are at chance.

### R-LLM. The teacher alone (reference)

DeepSeek reasoning high, one population call per question, every person gets the same distribution,
with a fitted temperature.

- **Pros:** best NLL on unseen questions so far; nothing to train.
- **Cons:** no person heterogeneity at all; ECE over the limit until a temperature is fitted; needs the
  network; replies vary between runs; some prompts never answer.

### A1. Direct LLM prior (new, recommended first)

Ask the reasoning LLM once per question for the population distribution, cache it, and feed it into
M4b's question-to-prior path (for example as a logit offset with a learned weight). The person tower
still learns individual shifts from human answers. Needs the teacher's distribution for every training
question as well as the scored ones (about 500 questions in all).

- **Pros:** keeps the teacher's full reasoning instead of a compressed copy; cheap (about 500 calls,
  under a dollar at the reasoning-high rate measured in D01-lite); no question bank; no licence needed
  for training on outputs (the outputs are inputs, not training targets, though FA9 should still be
  read); the cost per simulated decision is near zero because the call is shared by everyone who
  answers that question; simple to build on the existing two-tower code.
- **Cons:** a new question at simulation time needs one network call before it can be answered; the
  prior inherits whatever the teacher memorised (D01b decides how much that matters); replies vary, so
  the cache is part of the model and must be versioned with it; the scored test questions need teacher
  calls at final evaluation (question text only, never answers), which the human runs.

### A2. Distilled prior (the original fork plan)

Generate a broad, typed bank of synthetic questions (10 k to 50 k), label each with the teacher's
population distribution, pretrain M4b's prior path on those labels, then fine-tune on human answers.

- **Pros:** the finished model is self-contained: no network, deterministic, fast, any new question
  answered instantly; a bank broader than Twin-2K-500 may help on question types the teacher handles
  well; the original design, with guards and ablations already written.
- **Cons:** costs a bank (roughly 17 dollars for 10 k questions, 85 for 50 k, arithmetic); needs FA9
  (training on DeepSeek outputs) to be true; the small frozen encoder may not be able to hold what the
  teacher knows, because the teacher's edge comes from reasoning (checked cheaply before any bank, §3
  step 2); at best it approaches the teacher, so it only wins over A1 if being network-free matters.

### A3. Fine-tuned decision LLM (new)

Fine-tune a small open LLM (for example Qwen3-4B with a LoRA adapter) on Twin-2K-500 **training**
answers, then read its option log-probabilities, as was done for Qwentaur in D01-lite but trained on
our data instead of Psych-101. Two forms: population-level (one soft-label target per training
question) or person-level (attributes in the prompt, one row per answer).

- **Pros:** one model that combines an LLM's knowledge of language with human correction; open
  weights, no API at run time; the person-level form could learn person effects that prompting could
  not; the D01-lite log-probability scorer and Colab notebook already exist.
- **Cons:** reads the answer in one pass with no reasoning, which is the setting where the LLM was no
  better than M4b (DeepSeek reasoning off, above); Psych-101 tuning of the same base did not transfer
  (Qwentaur-4B accuracy 0.719
  <!-- src: reports/FD-d01lite/qwentaur4b-vs-all/metrics.json#models.qwentaur-4b.population.accuracy.mean -->
  against a uniform guess at 0.713
  <!-- src: reports/FD-d01lite/qwentaur4b-vs-all/metrics.json#models.ref-uniform.population.accuracy.mean -->);
  only 435 training questions, so the population form risks overfitting; needs a GPU (free Colab T4,
  whose quota already ran out once); the person-level form costs one LLM pass per person and question
  when simulating, far slower than M4b; less transparent than a prior plus a shift.

### Summary

| | A0 M4b | A1 direct prior | A2 distilled prior | A3 fine-tuned LLM |
| --- | --- | --- | --- | --- |
| Keeps the teacher's reasoning | no | yes | partly, if the encoder can hold it | no |
| Person heterogeneity from human data | yes | yes | yes | person form only |
| Runs without network | yes | no (one call per new question) | yes | yes |
| Paid spend to try | none | under a dollar | tens of dollars | none (GPU time) |
| Compute to try | CPU | CPU | CPU, plus bank generation | GPU |
| Licence question | none | read FA9 | FA9 must be true | base model licence only |
| Main risk | weak on new questions | memorised prior | encoder too small | no reasoning, too few questions |

## 3. Decision steps

Each step has a rule that decides the next one. Nothing below spends money or touches `test` without the
human's approval.

**Step 0. Human decisions (no work).**

1. Is running without the network a requirement for the simulator? If **no**, A2's main advantage is
   gone and it runs only if A1 disappoints.
2. Approve an `eval-change/item-folds` proposal: 28 items cannot separate arms that differ by a few
   hundredths of NLL (DESIGN §9, attack 4).
3. Approve the spend caps below.

**Step 1. Prerequisites (cheap).**

1. **D01b contamination probe** (under a dollar): score the teacher on the M5 paraphrases of the gate
   questions. If its edge over B1 mostly disappears on paraphrases, the gate result is memorisation:
   stop A1 and A2, report it, and keep only A0 and A3.
2. **M4b three-seed re-run** at the fork SHA, with predictions kept (CPU, no spend). This is A0.
3. **Item folds** built on the approved branch, so every later comparison uses them.

**Step 2. Cheap probes, one per arm.**

1. **A1:** teacher labels for every Twin-2K-500 question in train and val (about 500 calls, cap one
   dollar), then train M4b with the prior input. A day of CPU.
2. **A2 capacity check** (no spend): fit M4b's prior path to teacher labels on the training questions
   only and measure how close it gets to the teacher on the val questions. If it gets less than half
   of the teacher's gain over M4b, drop A2 unless network-free running is required (step 0).
3. **A3 feasibility** (no spend, needs a GPU): population-form LoRA on the 435 training questions on a
   free T4; record training time and the val score.

**Step 3. Choose (on item folds, val only, three seeds each).**

1. If **A1 beats A0** with the interval excluding 0 and passes the guards (ECE, SD ratio,
   permutation), A1 is the default design.
2. **A2** runs in full (bank, five seeds) only if network-free running is required (step 0), or if its
   capacity check came close to the teacher. It then has to match A1 to be worth its cost.
3. **A3** is kept only if it beats A1, or if A1 is ruled out by the contamination probe.
4. If no arm beats A0, that is the finding; write it up.

**Step 4. Then, with the winner.**

1. **D04 treatment effects** on the eight unseen-experiment folds (raised in priority: the LLM knew effect
   directions where M4b is at chance).
2. **D03 dispersion** as a check, not a campaign.
3. **D05 head to head** against the teacher alone, including an honest cost per decision (a cached
   per-question call shared across the population).
4. **Test partition, once, run by the human.**

## 4. How every arm is scored

- Twin-2K-500 rows; NLL primary; ECE, population TVD, SD ratio and option-permutation guards; treatment
  effects in D04.
- Unseen questions (`heldout_item`) are the headline; seen questions (`heldout_person`) are a
  no-regression check.
- Item folds once approved; until then, the 28 `heldout_item` val questions with the item bootstrap,
  read as indicative only.
- Same seeds and the same sealed harness for every arm; the teacher's prompt is frozen as used in the
  gate (`configs/fork-distill/gate-deepseek-high16k.json`).

## 5. Spend to approve

| Item | Cap proposed |
| --- | --- |
| D01b contamination probe | 1 dollar |
| A1 teacher labels (train and val questions) | 1 dollar |
| A2 bank, only if step 3 calls for it | set after a dry run |

About three dollars remain on the OpenRouter account.
