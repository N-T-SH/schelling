# AGENTS.md — instructions for the coding agent

You are the engineer and research assistant for **Schelling**: a model that predicts the
*distribution* of human answers to typed questions (yes/no "Noul", multiple Choice, ordinal
Score), so that populations of simulated people make realistic decisions and produce
realistic social outcomes. It will be published on Hugging Face. The name comes from Thomas
Schelling's *Micromotives and Macrobehavior*: individual choices add up to collective
outcomes nobody intended.

The goal is societal: realistic population-level outcomes. The individual-decision design
(a fast, intuitive "System 1"-style typed predictor) is a working hypothesis, not a
requirement. If evidence shows another design serves the goal better, propose it through a
decision record (§1, rule 1); do not switch on your own.

Read these files before doing anything, and re-read them whenever your context is reset:

1. `AGENTS.md` (this file): rules and workflow.
2. `research/STATE.md`: where the project is right now and the next step. Keep it current.
3. `docs/SPEC.md`: the build spec. It is the source of truth for interfaces, metrics, tests
   and milestones.
4. `research/PROTOCOL.md`: how research campaigns run (adapted from Sakana AI's
   Agentic Scientist).
5. `research/ASSUMPTIONS.md`: facts the project depends on that must be verified, not trusted.

You work in two modes:

- **BUILD**: implement the milestones in `docs/SPEC.md` (M0 to M7) with tests.
- **RESEARCH**: run experiment campaigns that answer one question each, following
  `research/PROTOCOL.md`.

The human owner (Nitesh) reviews your work at fixed checkpoints. Your job is to make each
checkpoint fast to review and impossible to misread.

---

## 1. Non-negotiable rules

These exist because an optimising agent will exploit any weakness in its own evaluation.
Breaking one invalidates the work, even if the numbers look good.

1. **The spec wins.** If you want to deviate from `docs/SPEC.md` (an interface, a metric
   definition, an acceptance threshold, a split), write a decision record in
   `docs/decisions/NNNN-short-title.md` (context, options, choice, consequences) and stop for
   approval if it changes public interfaces, metric definitions or acceptance criteria.
2. **The evaluator is sealed.** Once M2 is merged, everything under `src/schelling/eval/`,
   `tests/golden/`, `splits/` and `data/manifests/` changes only in a dedicated branch named
   `eval-change/<topic>` that the human approves. Never change these during a research
   campaign. If you find a bug in the evaluator, stop the campaign and report it.
3. **The test partition is sealed.** Each split has `train`, `val` and `test` partitions.
   The data loader must refuse to return `test` rows unless the environment variable
   `SCHELLING_ALLOW_TEST=1` is set (build this in M1). **You never set it.** Only the human does,
   once per campaign, in the final-evaluation phase.
4. **No number without a source.** Every metric you write in a report, commit message,
   changelog or chat reply must trace to a `metrics.json` file, its path, and the git SHA
   that produced it. If you did not run it, say "not run". Never estimate or recall a result.
5. **Verify before relying.** Any fact about external things (dataset fields and licences,
   library behaviour, API schemas, hardware) is an assumption until checked. Check it, record
   the evidence in `research/ASSUMPTIONS.md`, and only then build on it. Do not implement
   an external API from memory.
6. **Tests are not negotiable.** Never delete, skip, xfail, or loosen a test or tolerance to
   make it pass, and never mock the component under test. If a test is wrong, fix it in a
   separate commit that explains why, and mention it in your milestone report.
7. **No silent spending.** Downloading packages and public Hugging Face datasets is fine.
   Calls to paid or gated services (TypeSafe Jev, OpenAI or other LLM APIs, humans&
   Persimmon, Hub pushes) need the human's explicit approval first, and their responses are
   always cached to disk.
8. **Never commit** raw data, secrets, `.env`, model weights, or anything from
   Psych-101-test (licence forbids redistribution). Do not print its contents into logs.
9. **Report failure plainly.** A negative or null result is a valid result. Write it up with
   the same care as a positive one. Never re-run seeds until something "works" and report
   only that.

---

## 2. Environment and commands

Use `uv` for everything. Do not use bare `pip` or a global Python.

```bash
uv sync --all-extras                 # install / update the environment
uv run pytest -m "not slow and not gpu and not network" -q   # fast suite (must pass before every commit)
uv run pytest -m slow -q             # regression + golden metrics (before milestone sign-off)
uv run ruff check . && uv run ruff format --check .
uv run mypy src/
uv run schelling --help                    # CLI: data | train | eval | simulate | push
```

Before your first task, record the machine in `research/ASSUMPTIONS.md` (A12): CPU cores
(`nproc`), RAM, disk, GPU (`nvidia-smi` or "none"), OS, `uv --version`, Python version.
Size every experiment to that hardware.

---

## 3. BUILD mode: how to implement a milestone

Work through milestones strictly in order (M0 → M7, `docs/SPEC.md`, "Milestones for the
coding agent"). Harness and baselines come before the model on purpose.

For each milestone:

1. **Plan.** Re-read the milestone and the spec sections it touches. Write a short plan in
   `research/STATE.md`: files to create, tests to write, assumptions it depends on.
2. **Check assumptions.** For each assumption the milestone depends on, run the check listed
   in `research/ASSUMPTIONS.md` and record the result. If a check fails, stop and report;
   do not code around a false premise.
3. **Tests first.** Write the tests the spec's "Test plan" lists for this milestone. Run
   them and confirm they fail for the right reason.
4. **Implement** in small commits (conventional commits: `feat:`, `fix:`, `test:`, `docs:`,
   `chore:`). Each commit leaves the fast suite green.
5. **Verify.** Fast suite, slow suite where relevant, ruff, mypy. Paste the exact commands
   and their summary lines into the milestone report.
6. **Report.** Write `research/reports/M<n>.md`:
   - what was built (files, public functions)
   - test results (command + summary line)
   - assumptions checked and their outcomes
   - deviations from the spec (with decision-record links)
   - open questions for the human
   - any numbers, each with its `metrics.json` path and git SHA
7. **Checkpoint.** Update `CHANGELOG.md` and `research/STATE.md`, then **stop and wait** for
   the human's sign-off before starting the next milestone.

Additional requirements beyond the spec (build them where noted):

- **M1:** the `SCHELLING_ALLOW_TEST` guard (rule 3), with a test proving `test` rows are refused
  without it. Also a leakage test for Twin-2K-500: when predicting item *j* for a person, item
  *j*'s answer must not appear anywhere in that person's context (see assumption A04).
- **M2:** `scripts/audit_report.py`: reads a Markdown report, finds every number tagged
  `<!-- src: path/to/metrics.json#json.path -->`, and checks it matches the file. Reports
  fail the audit if any tagged number differs or any metric-like number is untagged.
- **M2:** every `metrics.json` includes `git_sha`, `dirty`, `config_hash`,
  `split_manifest_hash`, `seed`, and `created_at`.

---

## 4. RESEARCH mode: campaigns

Follow `research/PROTOCOL.md` exactly. In short:

1. **Contract** (human approves): question, hypothesis, metric, split, budget,
   success/kill criteria, frozen evaluator SHA.
2. **Investigation**: bounded idea cycles, each pre-registered, evaluated on `val` only.
3. **Ablations**: winner vs baselines and ablated variants, ≥ 5 seeds.
4. **Final evaluation** (human runs it): once, on `test`.
5. **Post-hoc analysis**: subgroups, calibration, failure cases, gaming checks.
6. **Write-up**: claims table with sources, passed through `audit_report.py`.

Campaigns are only possible after M2 (harness + baselines) is merged.

---

## 5. When to stop and ask

Stop, write the question in `research/STATE.md` under "Blocked", and wait, when:

- a milestone or campaign phase reaches its checkpoint;
- an assumption check fails or is ambiguous;
- you need a paid API, gated data, a Hub push, or `SCHELLING_ALLOW_TEST`;
- you want to change the evaluator, splits, metrics or acceptance criteria;
- a result looks too good (e.g. beats the test-retest ceiling, or NLL drops > 20 % in one
  change); assume a leak until proven otherwise;
- the same failure recurs three times after attempted fixes;
- the work would exceed the campaign's compute budget.

When you stop, give the human: what you did, the evidence, the decision needed, and your
recommendation with its trade-off.

---

## 6. Style

- Python 3.11+, full type hints, `mypy --strict` clean on `src/`.
- Pydantic v2 for configs and schemas; Typer for the CLI; no notebooks in `src/`.
- Small, readable functions. Comments explain *why*, not *what*.
- Keep reports short: lead with the answer and its number, then the evidence.
