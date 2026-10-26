# Schelling

Schelling predicts how **people actually decide**: given a person, a situation and a typed
question (yes/no, multiple choice, or an ordinal scale), it returns calibrated probabilities
over the answers, matching the real spread of human responses rather than the single most
likely one. Populations of simulated people sampled from it are meant to make realistic
decisions, and so to produce realistic social outcomes.

The name comes from Thomas Schelling's *Micromotives and Macrobehavior*: individual choices
add up to collective outcomes, such as segregation, cooperation or norm formation, that
nobody intended. Getting the micro-level decisions and their diversity right is the
precondition for trusting any macro-level simulation built on them.

**Status:** pre-alpha. Nothing is implemented yet; see `research/STATE.md`.

## Repository guide

| Path | What it is |
| --- | --- |
| `docs/SPEC.md` | Build spec: interface, architecture, data, evaluation, tests, milestones |
| `AGENTS.md` | Rules and workflow for the coding agent that implements this repo |
| `research/STATE.md` | Current milestone or campaign, next step, open questions |
| `research/PROTOCOL.md` | How research campaigns run (contract → search → ablations → sealed test → write-up) |
| `research/ASSUMPTIONS.md` | External facts to verify before relying on them |
| `research/FINDINGS.md` | Results of finished campaigns, including negative ones |
| `docs/decisions/` | Decision records for any change to the spec |

## Working with the coding agent

Kickoff prompt for the first session:

> Read `AGENTS.md`, then `research/STATE.md`, `docs/SPEC.md`, `research/PROTOCOL.md` and
> `research/ASSUMPTIONS.md`. Follow AGENTS.md exactly. Start in BUILD mode: verify
> assumption A12 (hardware), then plan and implement milestone M0. Stop at the M0 checkpoint
> with a report in `research/reports/M0.md`, and wait for my sign-off.

Every later session:

> Read `AGENTS.md` and `research/STATE.md`, then continue from the next step. Stop at the
> next checkpoint.

Human checkpoints:

| When | Decision |
| --- | --- |
| End of each milestone (M0–M7) | Sign off, or send back with changes |
| Campaign contract | Approve question, metric, budget and kill criteria |
| After ablations | Whether the winner goes to the test set |
| Final evaluation | Set `SCHELLING_ALLOW_TEST=1` and run the prepared commands yourself |
| Paid API, gated dataset or Hub push | Approve spend and access |

## Intended use

Research on human decision-making and agent-based social simulation. **Not** for decisions
about real individuals (hiring, credit, insurance, policing, eligibility), profiling or
targeting people, or replacing human-subjects research for confirmatory claims.

## Licence

Code: Apache-2.0 (see `LICENSE`). Training data keeps its own licences; see `docs/SPEC.md`.
