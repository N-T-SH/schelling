# Fork D: distilled prior

An alternate approach tested beside the mainline: give the small typed model an LLM's breadth of
questions by distillation, then correct it on real human answers. See `DESIGN.md`.

`AGENTS.md` and `research/PROTOCOL.md` apply in full. The rules below are additions.

## Rules for not interfering with the mainline

| Thing | Fork uses | Never touches |
| --- | --- | --- |
| Git | branch `fork/distill`, experiment branches `fork/distill/exp/<campaign>/<idea>`, its own worktree | the main checkout's branch or working tree |
| Docs and state | `research/forks/distill/` | `research/STATE.md`, `research/PROTOCOL.md`, `research/ASSUMPTIONS.md`, `docs/SPEC.md`, `CHANGELOG.md` |
| Campaign ids | `D01` to `D05` | `C…` |
| Configs | `configs/fork-distill/` | existing configs |
| Outputs | `reports/FD-*`, `runs/fd-*`, `data/synth/` | `reports/M*`, `reports/C*`, `data/processed/` |
| Spend log | `cache/llm/usage.jsonl` inside the fork worktree (its own `cache/`) | the main checkout's `cache/` |
| Sealed | nothing | `src/schelling/eval/`, `tests/golden/`, `splits/`, `data/manifests/`, the test partition |

- **Shared code** (`cached_trainer.py`, `config.py`): additive, default-off, with a test proving existing
  configs train bit-identically.
- **Reference model:** M4b is re-trained inside the fork at the fork's SHA with the same seeds. Mainline
  numbers are context, never the paired baseline.
- **Compute:** the mainline has priority on the one machine (A12). Fork training runs at `nice 19`, one
  job, two threads, and is paused while a mainline job runs. API batches are network-only.
- **Data in the worktree:** `data/` and `cache/` are git-ignored, so the worktree has none. Point the CLI
  at the main checkout's `data/processed` with `--processed` (read-only), and keep the fork's own outputs
  in the worktree.
- **Merging:** by decision record and PR, only on the human's say. A failed fork leaves the mainline
  unchanged.

## Campaign order

```text
D01 external benchmark ──gate──► D02 distilled prior ──► D03 dispersion ──► D05 head to head (test, once)
                                          └────────────► D04 treatment effects
```

Each contract is approved separately. D03 and D04 can run in either order after D02.
