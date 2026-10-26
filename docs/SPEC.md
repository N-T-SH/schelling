# Schelling — Build Spec

This file is the source of truth for the build. It mirrors the "Schelling — Build Spec" doc (renamed from HS1 on 26 Oct 2026); if the two disagree, ask the human which is current.

## Overview

Schelling models how individual human decisions add up to **sociological outcomes**. The name refers to Thomas Schelling's *Micromotives and Macrobehavior*: small individual choices aggregate into collective outcomes, such as segregation, cooperation or norm formation, that no individual intended. Getting the full distribution of individual decisions right, not just the most likely choice, is the precondition for trusting any population-level simulation built on them.

**Goal:** given a person context, a situation and a typed question, return calibrated probabilities over the answer options, cheap enough to sample millions of decisions for agent-based social simulation.

**Working hypothesis at the individual-decision level:** each decision is made by a fast, intuitive "System 1"-style model, a small non-generative predictor with typed outputs. Its interface borrows from TypeSafe AI's Jev (typed Choice, Score and yes/no "Noul" outputs with probabilities), but it is trained and evaluated against what real people chose. This is a design bet about how to model one decision, not the project's goal; it is replaced through a decision record (`docs/decisions/`) if evidence favours another design.

**Non-goals:**

- Judging correctness or truth (that is what Jev does).
- Generating text or explanations.
- Claims about cognitive mechanism. Schelling is a predictor of behaviour, not a theory of mind.
- Deciding anything about real individuals (see Hugging Face release).

**How it differs from prior work:**

| System | Output | Trained on | Main known weakness |
| --- | --- | --- | --- |
| Jev (TypeSafe AI) | Typed probabilities | Correctness / oracle labels | Not built to predict people; binary probabilities poorly placed around 0.5 |
| Centaur | Generated tokens | Psych-101 trial data | 70B LLM, slow; contested generalisation |
| Persona-prompted LLMs | Text | Pretraining only | Variance collapse, stereotyping |
| **Schelling** | Typed probabilities | Individual human responses | To be measured: the point of this spec |

The success criterion is distributional: a synthetic population sampled from Schelling must match real human means, variance, subgroup differences and treatment effects, not just individual accuracy.

## Typed interface

Every request carries one person, one state and several independent typed questions; every answer is a probability distribution that can be sampled. Define all schemas with pydantic v2 in `src/schelling/schema.py` and export JSON Schema to `schemas/` in CI.

```python
class PersonContext(BaseModel):
    person_id: str | None = None          # opaque, never a real identifier
    attributes: dict[str, str | float | int] = {}   # demographics, trait scores
    history: list[HistoryItem] = []        # prior (question, options, answer) triples

class Noul(BaseModel):
    type: Literal["noul"] = "noul"
    proposition: str                       # "I would accept this offer"
    true_description: str | None = None
    false_description: str | None = None

class Choice(BaseModel):
    type: Literal["choice"] = "choice"
    prompt: str
    options: list[str]                     # 2..20, unique

class Score(BaseModel):
    type: Literal["score"] = "score"
    prompt: str
    levels: list[str]                      # 2..10, ordered low -> high

class DecisionRequest(BaseModel):
    person: PersonContext | None = None
    state: str | dict | list[str]
    questions: dict[str, Noul | Choice | Score]   # discriminated on `type`

class Answer(BaseModel):
    type: Literal["noul", "choice", "score"]
    probs: list[float]                     # noul: [p_no, p_yes]; sums to 1
    p_yes: float | None = None             # noul only
    expected_level: float | None = None    # score only, in [0, L-1]
    entropy: float
```

Public API (`src/schelling/infer/predictor.py`):

- `Predictor.from_pretrained(repo_or_path, revision=None)`
- `predictor.decide(request) -> dict[str, Answer]`
- `predictor.decide_batch(requests, batch_size=...)`
- `predictor.encode_person(person) -> PersonEmbedding` (cache once, reuse across questions)
- `simulate.sample_population(predictor, persons, request_template, n_draws, seed) -> DataFrame`

Semantics that must hold and be tested: questions are answered independently; option order does not change the probability attached to each option; `p_yes` is the predicted share of similar people answering yes, not the probability the proposition is true.

## Architecture

One shared text encoder feeds three typed heads; the person is encoded once and cached, so simulating a population costs one forward pass per (person, question) after the first.

Data flow: person context → person encoder (cached per person) → context vector *h*; state + question → shared text encoder → context vector *h*; option and level texts → shared text encoder → option embeddings. *h* feeds the Noul, Choice and Score heads; the Choice and Score heads score *h* against the option/level embeddings; per-head temperature scaling is applied to every output.

1. **Backbone** (configurable): a pretrained bidirectional encoder. Dev default `answerdotai/ModernBERT-base`, release candidate `ModernBERT-large`; tests use a tiny random BERT. Also support a small decoder LM used as an encoder (mean-pooled last layer) as an ablation.
2. **Person encoder:** v0 serialises `attributes` + the last *k* history items into the context text. v1 encodes each history item separately with the backbone, then attention-pools them into *m* person tokens (default m = 8) that are prepended to the state. v1 is what makes caching possible.
3. **Context vector** *h* = pooled representation of `[PERSON] [STATE] [QUESTION]`.
4. **Heads:**
    - **Noul:** MLP(h) → logit → sigmoid.
    - **Choice:** each option text is encoded by the backbone to *e_i*; logit_i = ⟨W·h, e_i⟩ / τ. Shared weights across options make it permutation-equivariant by construction; padded options are masked to −∞.
    - **Score:** cumulative-link ordinal head (CORN-style): L−1 threshold logits conditioned on h and on the level-text embeddings, converted to a distribution over L levels. Score is not treated as plain Choice, because ordinal structure matters for variance.
5. **Calibration:** one learned temperature per head type, fit post hoc on the validation split by minimising NLL and stored in the config.
6. **Epistemic uncertainty (v1, optional):** a 5-seed deep ensemble; report spread separately from the predicted human distribution. The predicted distribution is aleatoric by design: human disagreement is the target, not noise.

Implement as a Hugging Face `PreTrainedModel` subclass (`SchellingModel`) with an `SchellingConfig(PretrainedConfig)` so `save_pretrained` / `from_pretrained` and Hub loading work natively.

Known risk to watch: JEPA-style representation collapse is not the issue here; **behavioural mode collapse** is. The evaluation plan treats it as a first-class metric.

## Data

Start with two clean, openly licensed sources (Psych-101 and Twin-2K-500), and add survey microdata only after the pipeline and splits are proven. Raw data is never committed; each source has an adapter that downloads, checksums and converts it.

| Source | What it gives | Licence | Phase |
| --- | --- | --- | --- |
| [Psych-101](https://huggingface.co/datasets/marcelbinz/Psych-101) | Trial-level choices from 160 experiments, 60k+ participants, 10M+ choices; choices marked `<<` `>>` | Apache-2.0 | M1 |
| [Psych-101-test](https://huggingface.co/datasets/marcelbinz/Psych-101-test) | Official held-out test set | CC-BY-ND-4.0, gated, do not redistribute | M4 eval only |
| [Twin-2K-500](https://huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500) | 2,058 US people, ~500 questions each; wave 4 repeats tasks two weeks later (test-retest ceiling) | CC-BY-4.0 | M1 |
| GSS / ANES / WVS microdata | Survey Choice and Score items with demographics | Verify each; WVS needs registration | M5+ |
| choices13k (Peterson et al.) | Aggregate choice rates over many gambles | Verify | M5+, soft labels only |

**Unified record** (`src/schelling/data/schema.py`, stored as Parquet, one row per observed response):

```python
class DecisionRecord(BaseModel):
    record_id: str          # sha1(source, study_id, person_id, item_id, trial_idx)
    source: str; study_id: str; item_id: str; person_id: str
    trial_idx: int | None   # order within person, for sequential tasks
    qtype: Literal["noul", "choice", "score"]
    question: str; options: list[str] | None; levels: list[str] | None
    state: str
    person_ref: str         # key into persons.parquet
    response: int | None    # observed index (individual records)
    response_dist: list[float] | None   # aggregate records only
    condition: str | None   # experimental arm, for treatment effects
    wave: int | None        # for test-retest
```

**Splits** (deterministic, hash-based, written to `splits/<name>.json` with a manifest hash):

- `heldout_person`: 10% of persons per source; none of their records in train.
- `heldout_item`: 10% of items within seen studies.
- `heldout_study`: whole studies or experiments excluded (Psych-101: ~10% of experiments).
- `heldout_domain`: train on lab tasks, test on survey items (and the reverse).
- `temporal`: for sequential data, predict trials after index *t* from earlier ones.
- `retest`: Twin-2K-500 wave 4 used only to compute the test-retest ceiling.

Each split has `train`, `val` and `test` partitions; the `test` partition is sealed (see AGENTS.md rule 3).

**Leakage controls:** person, item and study disjointness asserted in tests; exact and near-duplicate (MinHash, Jaccard ≥ 0.8) question text between train and held-out sets reported and removed; for Psych-101, evaluate on the official test set where licence allows; a memorisation probe (paraphrased and option-shuffled items) flags contamination from the backbone's pretraining.

## Training

Train with a proper scoring rule on observed human responses and nothing that distorts probabilities; calibrate afterwards.

- **Loss:** cross-entropy (NLL) of the observed response for individual records; cross-entropy against `response_dist` for aggregate records. Score heads use the CORN loss. No label smoothing: it biases calibration.
- **Source balancing:** temperature sampling over studies (p ∝ n^0.5) so Psych-101's 10M trials do not drown the surveys; per-source loss logged separately.
- **Optional diversity regulariser** (off by default, ablated): within a batch grouped by item, penalise |SD_pred − SD_human| across persons. Ship it only if it improves held-out population metrics without hurting NLL.
- **Augmentation:** random option permutation (equivariance check doubles as augmentation) and, from M5, paraphrased question wording generated offline once and stored with a hash.
- **Optimiser:** AdamW, cosine schedule with warmup, bf16 mixed precision, gradient checkpointing for large backbones. Early stop on validation NLL of `heldout_person`.
- **Post-hoc calibration:** fit per-head temperature on validation; never on test.
- **Reproducibility:** every run writes `run.json` with resolved config, git SHA, dirty flag, split manifest hash, package versions and seeds. `schelling train --config configs/x.yaml` is the only entry point.
- **Config:** pydantic models loaded from YAML (no Hydra magic). Overrides via `--set key=value`.
- **Tracking:** TensorBoard by default; W&B optional behind an extra.

## Baselines

Build the baselines and the evaluation harness before the model: the item-mean baseline is the bar that richer-persona LLMs have repeatedly failed to clear. All baselines implement the same `Predictor` protocol so the harness scores them identically.

| ID | Baseline | Applies to | Needs |
| --- | --- | --- | --- |
| B0 | Uniform over options | All | Nothing |
| B1 | Item mean: marginal human distribution per item from train persons | Seen items | Train split |
| B2 | Demographic-cell mean with backoff to B1 when a cell has < 20 people | Seen items with attributes | Train split |
| B3 | Gradient-boosted classifier on person attributes + item id | Seen items | LightGBM |
| B4 | Jev zero-shot: person + situation as `state`, same typed question | All | `TYPESAFE_API_KEY`; responses cached |
| B5 | LLM persona prompt eliciting a probability per option | All | OpenAI-compatible endpoint; cached |
| B6 | Centaur, Psych-101 only | Psych-101 | 80 GB GPU; optional |

API baselines (B4, B5) cache every response on disk keyed by sha256 of (model version, request JSON), so reruns are free and reproducible. Implement B4 against TypeSafe's current API documentation; do not hard-code a guessed schema. Tests and CI never call external APIs: they replay cached fixtures.

## Evaluation plan

Schelling is judged at four levels, and population fidelity outranks individual accuracy. `schelling eval --model <path|baseline> --split <name>` writes `reports/<run>/metrics.json` (validated against a JSON Schema), reliability plots and a Markdown summary that the model card embeds.

**1. Individual level** (every split, per source and per question type)

- NLL (primary), Brier score, top-1 accuracy (reported but never used for selection).
- Expected calibration error with adaptive bins, calibration slope and intercept, reliability diagram.
- Fraction of test-retest ceiling on Twin-2K-500: agreement with a person's wave-4 answers divided by their own wave-1-to-wave-4 agreement.

**2. Population level** (for each held-out item, sample one draw per held-out person, repeat 20 seeds)

- Wasserstein-1 (ordinal) and total variation distance to the human distribution.
- SD ratio and entropy ratio (synthetic ÷ human): the variance-collapse metric.
- Mode-collapse rate: share of items where synthetic modal-option share exceeds human by > 0.2.
- Subgroup fidelity: the above per demographic subgroup, plus the worst-group gap.
- Correlation structure: Frobenius distance between item-by-item correlation matrices; sign agreement of regression coefficients for pre-registered models.
- Treatment effects (studies with `condition`): effect-size correlation, sign agreement, and the false-positive rate on effects that are null in humans.

**3. Robustness**

- Option-permutation invariance: max |Δp| for the same option across 10 random orders.
- Paraphrase invariance: mean TVD between original and paraphrased items.
- Memorisation probe: NLL gap between canonical and paraphrased wording of public survey items; a large gap flags contamination.

**4. Efficiency:** p50/p95 latency per request and decisions per second on CPU and one GPU, with and without cached person embeddings.

**Release acceptance criteria (v1.0; revise after M2 baselines land):**

| Check | Target |
| --- | --- |
| NLL vs B1 item mean, `heldout_person` | Better on every M1 source |
| NLL vs B4 Jev and B5 LLM, all splits | Better on average |
| ECE, `heldout_person` | ≤ 0.05 |
| Median SD ratio, population level | 0.85 to 1.15 |
| Option-permutation max \|Δp\| | ≤ 0.01 |
| False-positive rate on human-null effects | ≤ 0.10 |
| Twin-2K-500 fraction of test-retest ceiling | Reported; no target until baselines measured |
| p50 latency, base model, 1 GPU | ≤ 50 ms per request |

Any metric missing a target is reported honestly in the model card, including failures.

## Test plan

Every test below is required; unit and property tests run on CPU in under 5 minutes using a tiny random backbone and fixture data in `tests/fixtures/` (about 200 synthetic records per source, generated by a committed script). Markers: `unit`, `property`, `integration`, `slow`, `gpu`, `network`; default CI runs everything except `slow`, `gpu`, `network`.

**Schema** (`tests/unit/test_schema.py`)

- Rejects Choice with < 2 or > 20 options, duplicate options, Score with < 2 or > 10 levels, Noul without a proposition, unknown `type`.
- JSON round trip is lossless; exported JSON Schema matches the committed file (snapshot).

**Data adapters and splits** (`test_adapters.py`, `test_splits.py`)

- Each adapter on its fixture emits records that validate, with `response` within range and stable `record_id`s across two runs.
- Psych-101 parser extracts the `<<choice>>` correctly on hand-written trial strings, including multi-choice trials.
- Splits: person, item and study sets are disjoint where the split promises; identical output for identical seed; sizes within ±1 percentage point of target.
- Contamination: no held-out question text appears in train exactly; near-duplicates above threshold are reported.

**Model heads** (`test_heads.py`)

- Choice probabilities sum to 1 (atol 1e-5); padded options get probability 0 in mixed-size batches.
- Noul `p_yes` in (0, 1); `probs == [1 - p_yes, p_yes]`.
- Score distribution sums to 1; `expected_level` in [0, L−1]; monotone thresholds.
- No NaN or Inf with empty history, empty attributes or maximum-length input.

**Property tests** (`tests/property/`, Hypothesis)

- Permuting options permutes probabilities identically (atol 1e-5) for random requests.
- Answers to question A do not change when question B is added to the same request.
- Batch and single-request inference give identical outputs (atol 1e-5).

**Losses, calibration, metrics** (`test_losses.py`, `test_metrics.py`)

- Soft-label CE equals hard CE for one-hot targets; KL is 0 for identical distributions.
- Temperature fitting lowers NLL on synthetically over-confident logits and returns T > 0.
- Hand-computed values for NLL, Brier, ECE, Wasserstein, TVD; ECE ≈ 0 on synthetic perfectly calibrated data (n = 100k); SD ratio = 1 for identical distributions; TVD symmetric.
- Population sampler is deterministic given a seed; empirical frequencies match p within a binomial 99.9% interval at n = 10k.

**Baselines** (`test_baselines.py`)

- B1 reproduces the train marginal exactly; B2 backs off to B1 for small cells.
- B4 and B5 replay cached fixtures with no network; a cache miss with `network` marker absent raises a clear error.

**Integration** (`tests/integration/`)

- 50-step training run on fixtures: loss decreases; `run.json` written with all fields.
- `save_pretrained` → `from_pretrained` gives identical outputs (atol 1e-6); weights saved as safetensors only.
- `AutoModel.from_pretrained(path, trust_remote_code=True)` works in a fresh process.
- CLI smoke tests: `schelling data build`, `train`, `eval`, `simulate`, `push --dry-run` each exit 0 on fixtures.
- Eval report validates against its JSON Schema.
- Same seed, same config, CPU: bitwise-identical metrics across two runs.

**Regression** (`tests/regression/`, `slow`)

- Golden metrics on a frozen mini-benchmark (`tests/golden/metrics.json`); fail if NLL worsens by > 1% or ECE by > 0.01. Updating goldens requires a commit message tag `[golden-update]`.
- Latency budget on CPU for the tiny model.

**Release-only** (`release.yml`)

- Install the built wheel in a clean environment, download the just-pushed Hub revision, run one `decide` call and compare to the local reference outputs.

## Repo layout and engineering practices

A standard `src/` Python package with typed code, one CLI, and CI that never touches the network or a GPU.

```
schelling/
  pyproject.toml          # hatchling build, uv lockfile, extras: [train], [api-baselines], [dev]
  README.md  LICENSE  CITATION.cff  CHANGELOG.md  CONTRIBUTING.md  AGENTS.md
  src/schelling/
    schema.py             # request/response models
    cli.py                # typer: data | train | eval | simulate | push
    data/   schema.py registry.py splits.py contamination.py manifest.py
            adapters/ psych101.py twin2k.py gss.py wvs.py choices13k.py
    model/  configuration_schelling.py modeling_schelling.py heads.py person_encoder.py calibration.py
    train/  trainer.py losses.py sampler.py
    baselines/ base.py uniform.py item_mean.py demo_cell.py gbm.py jev_api.py llm_prompt.py centaur.py
    eval/   metrics.py population.py effects.py robustness.py efficiency.py report.py
    infer/  predictor.py simulate.py
    hub/    card.py push.py
  configs/                # dev-tiny.yaml, base.yaml, large.yaml
  schemas/                # exported JSON Schemas (request, response, metrics)
  data/                   # gitignored; manifests/ committed
  docs/   SPEC.md decisions/
  research/               # protocol, assumptions, state, campaigns, reports
  tests/  unit/ property/ integration/ regression/ fixtures/ golden/
  scripts/                # make_fixtures.py, paraphrase_items.py, audit_report.py, log_run.py
  .github/workflows/      # ci.yml, nightly.yml, release.yml
```

**Tooling**

- Python 3.11+; `uv` for environments and lockfile.
- `ruff` (lint + format), `mypy --strict` on `src/`, `pre-commit` running both plus a large-file guard.
- Typer CLI; pydantic v2 for every config and schema.
- Data manifests: `data/manifests/<source>.json` with upstream URL, revision, licence, sha256 of each file and record counts; `schelling data build` fails on checksum mismatch.

**CI** (GitHub Actions)

- `ci.yml` on every PR: lint, type-check, unit + property + integration on CPU, coverage ≥ 85% for `src/` (API baselines excluded), JSON Schema snapshot check.
- `nightly.yml`: `slow` regression suite and a small real-data run on Psych-101 subset; uploads the report as an artifact.
- `release.yml` on tag `v*`: build wheel and sdist, run release-only tests, push model to the Hub at a matching revision tag, create the GitHub release with the eval report.

**Conventions**

- Semantic versioning: model weights and package share the version; any change to schema or head semantics is a major bump.
- Conventional commits; `CHANGELOG.md` updated in every PR that changes behaviour.
- Secrets only via environment (`HF_TOKEN`, `TYPESAFE_API_KEY`, `OPENAI_API_KEY`); never in configs or logs.
- No notebooks in `src/`; exploratory notebooks live in `notebooks/` and are not imported.

## Hugging Face release

Ship Schelling as a standard Transformers custom architecture with safetensors weights, a generated model card carrying the eval results, and gated access with an intended-use agreement.

**Model packaging**

- `SchellingConfig(PretrainedConfig)` with `model_type = "schelling"`; `SchellingModel(PreTrainedModel)`; call `register_for_auto_class("AutoConfig")` and `("AutoModel")` so `configuration_schelling.py` and `modeling_schelling.py` are copied to the repo and `AutoModel.from_pretrained("<org>/schelling-base", trust_remote_code=True)` works.
- Save weights as safetensors only; save the tokenizer and calibration temperatures with the model.
- The pip package `schelling` provides the typed `Predictor` on top; the Hub repo also works without it via `trust_remote_code`.
- Repos: `schelling-tiny` (CI test artifact, private), `schelling-base`, later `schelling-large`. Hub revision tags match git tags (`v1.0.0`).

**Model card** (generated by `schelling/hub/card.py` from `metrics.json`; never hand-edited)

- YAML metadata: `license`, `library_name: transformers`, `base_model`, `datasets` (Psych-101, Twin-2K-500), `tags` (human-behavior, decision-modeling, calibration), and a `model-index` block with the headline metrics per dataset and split.
- Sections: what it predicts (human response distributions, not truth); intended use; out-of-scope use; training data and licences with attribution; full evaluation tables including failures; subgroup results; calibration plots; limitations; citation.
- **Out of scope, stated explicitly:** decisions about real individuals (hiring, credit, insurance, policing, eligibility); profiling or targeting people; replacing human-subjects research for confirmatory claims; any use with personal data without consent.

**Licensing**

- Code: Apache-2.0.
- Weights: Apache-2.0 is compatible with Psych-101 (Apache-2.0) and Twin-2K-500 (CC-BY-4.0, attribution in the card). Re-check before adding each new source; survey microdata terms may forbid redistribution of derived models or data.
- Psych-101-test is CC-BY-ND and gated: use it for evaluation only; publish scores, never the data.
- Do not publish a derived dataset unless every source permits it; otherwise publish the builder scripts and manifests.

**Privacy and safety**

- Never publish per-person embeddings or any participant IDs.
- Run a membership-inference sanity check before release (train vs held-out person NLL gap); report it in the card.
- Enable gated access on the Hub with a short form confirming the intended use.
- Optional: a Gradio Space demo with synthetic personas only.

## Milestones for the coding agent

Build in this order; each milestone ends with green CI and a short entry in `CHANGELOG.md`. The evaluation harness and baselines come before the model so every model change is scored from day one.

1. **M0 Scaffold.** Package skeleton, `pyproject.toml`, ruff, mypy, pre-commit, CI, Typer CLI stubs, `schema.py` with schema tests and JSON Schema export. *Done when:* `uv run pytest` and CI pass on an empty model.
2. **M1 Data.** `DecisionRecord`, Psych-101 and Twin-2K-500 adapters, manifests with checksums, all five splits, contamination report, fixtures script. *Done when:* adapter, split and contamination tests pass; `schelling data build` reproduces identical manifests twice.
3. **M2 Harness + simple baselines.** Metrics, population sampler, robustness and efficiency modules, report writer; B0 to B3. *Done when:* metric tests pass and `schelling eval` produces a full report for B0 to B3 on both sources.
4. **M3 Model v0.** `SchellingConfig`, `SchellingModel`, three heads, serialised person context, save/load, Auto-class registration. *Done when:* head, property and round-trip tests pass with the tiny backbone.
5. **M4 Real training.** Trainer, losses, balancing, calibration; ModernBERT-base on M1 data. *Done when:* Schelling beats B1 on NLL on `heldout_person` for both sources; goldens frozen.
6. **M5 Population fidelity.** Person encoder v1 with caching, optional diversity regulariser, paraphrase augmentation, survey sources. *Done when:* population metrics reported with an ablation table.
7. **M6 External baselines.** B4 Jev and B5 LLM with caching; B6 Centaur if compute allows. *Done when:* comparison table regenerated from cached responses with no network.
8. **M7 Release.** Model card generator, licence review, membership-inference check, gated Hub repo, release workflow. *Done when:* tagging `v1.0.0` publishes the model and the release-only test passes against the Hub revision.

## Risks and open questions

The biggest risk is that a small encoder cannot beat the item-mean baseline on unseen people; M4 is designed to find that out early and cheaply.

- **Data scale:** human decision data is small next to text; Schelling may need a larger backbone or LLM-distilled pretraining. Open question: does distilling a large LLM's predicted distributions (soft labels) before human fine-tuning help or import its collapse?
- **Psych-101 format:** trials are long natural-language transcripts; deciding the unit of prediction (single trial vs. trial with history) affects both leakage and cost.
- **Correlated errors at scale:** sampling many agents from one model does not give independent samples. Population metrics must be reported per seed and per subgroup, never only pooled.
- **Licence drift:** survey sources may forbid derived-model distribution; verify before M5.
- **Misuse:** a calibrated predictor of individual choices is a profiling tool. Gated access and explicit out-of-scope use are required, not optional.
- **Downstream research question:** whether Schelling populations make reputation and ostracism dynamics in agent societies more realistic than LLM-agent populations. That is a downstream experiment, not part of v1.

Sources: [Psych-101](https://huggingface.co/datasets/marcelbinz/Psych-101), [Psych-101-test](https://huggingface.co/datasets/marcelbinz/Psych-101-test), [Twin-2K-500](https://huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500), [Centaur adapter](https://huggingface.co/marcelbinz/Llama-3.1-Centaur-70B-adapter), [Evaluating Jev](https://arxiv.org/pdf/2609.37647), [Jev in DeepEval](https://deepeval.com/blog/introducing-jev-in-deepeval).
