# Draft: pre-registered pairing table for treatment-effect metrics

**Status:** SUPERSEDED by `docs/decisions/0002-treatment-effects.md` (decisions applied; pairs now in `src/schelling/eval/effects_pairs.yaml`). Kept for the record. Note: the outcome-bias and WTA/WTP option lists have 7 and 10 options, not 6 as written below. If approved, it becomes
`src/schelling/eval/effects_pairs.yaml` plus a runner, added on an `eval-change/treatment-effects`
branch (the evaluator is sealed). Contains no result numbers, by design: the table is fixed
before any model sees it.

## What the spec asks for

"Treatment effects (studies with `condition`): effect-size correlation, sign agreement, and the
false-positive rate on effects that are null in humans" (SPEC, Evaluation plan, level 2), with a
release target of false-positive rate at most 0.10.

## Why a table is needed

Twin-2K-500 contains 11 between-subject experiments whose arms are separate Qualtrics blocks
(`condition` = block name in our records). I checked the wave 1-3 file: for every pair below, no
person answered both arms, so each pair is a genuine randomised between-subject contrast. The
metric needs to know which arms to contrast and on which outcome; guessing that at evaluation
time would let the analysis be tuned to the model.

## Proposed definitions (please confirm or change)

1. **Outcome:** the response index (0-based, in catalogue option order) of one item per arm. Only
   items our adapter maps are usable; arms whose outcome is free text or a slider are excluded
   (below).
2. **Effect size:** standardised mean difference d = (mean index in arm B minus arm A) divided by
   the pooled human SD, so binary, 5-point and 6-point outcomes are comparable.
3. **Human reference effect:** estimated on all 2,058 persons (a fixed property of the
   population, used only as the reference, never as model input), with a 95% person-bootstrap
   interval. **Open question 1:** alternatively use only the evaluated partition's persons; this
   is purer but has roughly a tenth of the power, so most human effects would look null.
4. **Synthetic effect:** the same contrast computed on the sampled draws of the held-out persons
   in each arm, per population seed (20 seeds), reported per seed.
5. **Metrics:** Pearson correlation of d across pairs; sign agreement over pairs whose human
   interval excludes 0; false-positive rate = share of null contrasts (below) where the synthetic
   contrast is significant (two-sided Welch test, alpha 0.05), averaged over seeds.
6. **Null contrasts, two kinds:** (a) pairs whose human interval includes 0; (b) 20 placebo
   contrasts per item, splitting persons within one arm by a hash of person id, null by
   construction. Reported separately.
7. **Evaluation split caveat (important):** on `heldout_person` a model has seen the arms' items
   in training, so effects are memorised, not predicted. A meaningful test holds out **both arms
   of a pair together**. That needs a new split (`heldout_experiment`: whole pairs in
   val/test, 10% each). Splits are sealed, so it also goes on the eval-change branch.
   **Open question 2:** approve a new split, or accept the memorisation-prone version for B1 to B3
   only (they are item-keyed, so they will trivially reproduce effects; useful only as a check
   that the runner works).

## Pairs and arms

All pairs are between-subject (zero overlap checked in `wave1_3_response.csv`). "Literature prior"
is informational, not used by any metric (sign agreement uses the human estimate); where I am
not certain of the published direction I wrote "none".

| ID | Experiment | Arm A item | Arm B item | Outcome type | Literature prior |
| --- | --- | --- | --- | --- | --- |
| E01 | Framing (disease) | QID157 gain | QID158 loss | 6-point preference, program A to B | loss frame shifts towards the risky program B |
| E02 | Outcome bias | QID161 success | QID162 failure | 6-point decision quality | success rated better |
| E03 | Allais | QID192 form 1 | QID193 form 2 | 2 options (different sets, so contrast is on the share choosing the sure or higher-probability option; to confirm) | none |
| E04 | Absolute vs relative | QID183 calculator | QID184 jacket | yes / no | none |
| E05 | Myside bias | QID194 Ford | QID195 German | 6-point yes scale | none |
| E06 | Less is more, gamble | QID171 A | QID172 B | 5-point agreement | none |
| E07 | Less is more, gamble | QID172 B | QID173 C | 5-point agreement | none |
| E08 | Less is more, gamble | QID171 A | QID173 C | 5-point agreement | none |
| E09 | Proportion dominance 1 | QID174 A | QID175 B | 5-point agreement | none |
| E10 | Proportion dominance 1 | QID175 B | QID176 C | 5-point agreement | none |
| E11 | Proportion dominance 1 | QID174 A | QID176 C | 5-point agreement | none |
| E12 | Proportion dominance 2 | QID177 A | QID178 B | 6-point likelihood | none |
| E13 | Proportion dominance 2 | QID178 B | QID179 C | 6-point likelihood | none |
| E14 | Proportion dominance 2 | QID177 A | QID179 C | 6-point likelihood | none |
| E15 | WTA vs WTP (certainty) | QID189 WTP | QID190 WTA | 6 price bands, ordered | willingness to accept exceeds willingness to pay |
| E16 | Certainty (WTP) | QID191 WTP uncertain | QID189 WTP certain | 6 price bands, ordered | none |
| E17 | Anchoring, Africa (manipulation check only) | QID163 low anchor | QID165 high anchor | more / fewer | the share saying "more" is higher with the low anchor |
| E18 | Anchoring, redwood (manipulation check only) | QID167 low anchor | QID169 high anchor | more / less | the share saying "more" is higher with the low anchor |

E17 and E18 only check that the anchor was read; the anchoring effect proper is on the numeric
estimate, which is excluded below. Multi-arm experiments (E06 to E14) use all pairwise
contrasts; I propose Holm correction only for the false-positive test, not for effect sizes.

## Excluded, with reasons

| Experiment | Items | Reason |
| --- | --- | --- |
| Anchoring estimates | QID164, QID166, QID168, QID170 | free-text numbers, not mapped by the adapter |
| Sunk cost | QID181, QID182 | free-text numbers |
| Base rate | QID154, QID156 | 0-100 sliders, not mapped |
| Linda conjunction | QID159, QID160 | three-row matrix; contrast between arms needs a row pairing I would rather you chose |
| Probability matching | QID198, QID203 | within-person matrices, not an arm contrast |
| False consensus | QID287 | within-person matrix |

Supporting the free-text and slider outcomes would need a numeric head, which is not in the
spec (Noul, Choice and Score only). I suggest leaving them out of v1.

## Decisions I need

1. Human reference effect from all persons, or from the evaluated partition only (definition 3).
2. A new `heldout_experiment` split on the eval-change branch, or no model-level effect test in v1 (caveat 7).
3. Keep, drop or reword any pair; add priors you are sure of; decide the Linda row pairing.
4. Whether false-positive rate counts placebo contrasts and human-null pairs together or only pairs (definition 6).
