# 0004: Two-tower model with a text-derived item prior

**Date:** 2026-10-29 · **Status:** approved (Nitesh, 2026-10-29: "Okay do the two tower model"); acceptance criteria unchanged (see Consequences)

## Context

The first M4 run of the cross-encoder design (`research/reports/M4.md`) is worse than the item-mean baseline B1 on Twin-2K-500 held-out people, and costs about two days per epoch on the available hardware. Item marginals explain nearly all of what demographics can explain (B3 improves on B1 by about half a per cent), so the model has to be able to reproduce the item prior cheaply before person information can add anything. SPEC lists the individual-decision design as a working hypothesis that evidence may replace.

## Decision

Replace the single shared context encoder with two towers and an explicit prior-plus-residual output, as a second model class (`SchellingTwoTower`, `model_type = "schelling_tt"`) beside the existing one. The Predictor API, typed interface, evaluation harness and Hub packaging are unchanged.

- **Question tower:** the situation and question text (never the person) go through the backbone once into an item vector `u`. Option and level texts go through the same backbone into `e_i`.
- **Person tower:** each attribute becomes a short string `key: value`, encoded by the backbone and pooled by one small self-attention layer with no positional encoding (a set encoder), giving a person vector `p`. No person identifier is ever used.
- **Prior branch (item only):** Noul `MLP(u)`; Choice `s_i = MLP([u, e_i, mean(e)])`; Score thresholds `t_k = MLP([u, (e_k+e_{k+1})/2, mean(e)])`. The mean over options lets the prior tell apart items that share their question text but differ in options (the Bipolar rows). It is text-derived, so unseen items still get a prior; there is no item-id lookup.
- **Residual branch:** `r = MLP([p, u, p*u])`, then a residual logit per head, with the last layer of each residual zero-initialised so the model starts as its prior.
- **Output:** prior logit plus residual logit, then the same per-head temperature, softmax and CORN construction as before (Choice and Score remain order-equivariant and monotone by construction).
- **Training loss:** proper-scoring loss on the full output plus a weighted loss on the prior alone, so the prior branch learns the item marginal directly.
- **Stage A (this record):** backbone frozen. Question, option and attribute vectors are computed once and cached, and only the towers' small networks are trained, on all training rows, in minutes. Psych-101 is left out of Stage A: its states are per-record transcripts that cannot be cached per item, and its unit of prediction is still open (decision 0001, campaign C04). Stage B (unfreezing top layers) only if Stage A shows promise.
- At inference the model caches vectors by token sequence, so simulating a population costs one combiner pass per (person, question) after the first sight of each text.

## Options considered

1. Keep the cross-encoder and spend more compute (needs paid GPU; no sign that more of the same clears the bar).
2. Cross-encoder plus a per-item bias table (breaks the "text only" principle and unseen items).
3. This design.

## Consequences

- New files under `model/`, `infer/` and `train/`; the cross-encoder stays for comparison and the M4 run remains reproducible.
- **Acceptance criteria are not changed by this record.** The M4 done-criterion (beat B1 on Twin-2K-500 `heldout_person` validation NLL; Psych-101 excluded, decision 0001) still applies. I will additionally report held-out-item and leave-one-experiment-out results, as evidence, without redefining the bar; whether held-out-people NLL becomes a guard instead of the headline is still your call.
- Because training is now cheap, I will report five seeds and ablations (no person tower, no prior loss) in line with the research protocol, and run the eight experiment folds.
- A golden-metrics file for this model would live under `tests/golden/`, which is sealed; adding it needs an eval-change branch, so none is added here.
- Risks: separate towers cannot express fine person-by-question interactions beyond what the combiner learns (acceptable for 14 attributes; to be revisited for long histories in C01); frozen-backbone vectors may limit generalisation to unseen items (measured, not assumed).
