# Contributing to OpenDPD

Thanks for helping. This guide covers the development workflow for the core
library and OpenDPD Studio (the local browser workbench). Local agent instruction
files (`AGENTS.md` and `CLAUDE.md`) are ignored and must not be committed.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or your CUDA build
pip install -e ".[dev,gui]"
```

Frontend development additionally needs Node.js 20+:

```bash
cd frontend && npm ci && npm run dev
```

End users never need Node.js: the built frontend is shipped inside the wheel.

## Test layers

| Layer | What | Command | When |
|---|---|---|---|
| L0 static + unit | flake8 critical errors, schema/state-machine/metric unit tests, frontend `tsc`/vitest | `pytest tests -m "not extended and not integration"` | every PR |
| L1 integration | tiny CPU training through worker, dataset doctor, cancel, event replay, GUI/CLI consistency | `pytest tests -m integration` | every relevant PR |
| L2 packaged E2E | build wheel, install in a clean venv without Node.js, start `opendpd gui`, browser journey | `python -m build && pytest tests/packaging` | release candidates and packaging PRs |
| L3 platform / stress | multi-OS, stress datasets, long logs, crash recovery | scheduled `weekly.yml` | scheduled |
| L4 GPU / statistical | CUDA/MPS capability, multi-seed benchmark | manual, bound to an exact commit | after human approval |
| Performance report | plan §8.2 targets against a real `opendpd gui` server and a real browser | `python scripts/perf_report.py --workspace <scratch>/ws --minutes 30 --stress` (regenerates `docs/releases/performance-report.md`; run on an idle machine, never next to pytest) | release candidates |

Tests must include failure paths. Coverage numbers do not replace correctness.

## Pull request definition of done

A change is complete only when code, contracts (OpenAPI/schemas), tests,
documentation and evidence land together. The PR template asks for:

1. the user problem being solved,
2. scope and non-goals,
3. how it was verified (commands, environment, artifacts),
4. anything left unfinished,
5. whether a protected scientific path (listed below) is touched.

Keep unrelated formatting, dependency swaps and seed changes out of feature PRs.
Pure refactors must show unchanged behaviour on frozen inputs.

## Protected scientific paths

Metric definitions, data splits, golden references, acceptance thresholds and
published benchmark results require a separate PR with the
`science-review-approved` label. Protected paths are `opendpd/core/metrics/`,
`opendpd/core/splits.py`, `tests/golden/`, `benchmark/`, `docs/protocols/`,
`.github/CODEOWNERS` and `.github/workflows/protected-paths.yml`.
Never relax a tolerance, change a seed set or checkpoint-selection metric, or
replace expected values to make a failing test pass.

## Adding models, metric profiles and documentation

- New models: `docs/tutorials/adding-a-model.md` (one registry entry serves
  CLI, API and GUI; evidence before "supported").
- New metric profiles or thresholds: protected paths, separate science-reviewed
  change (`docs/protocols/metric-profiles.md`, `docs/protocols/acceptance-thresholds.md`).
- Tutorials: every `opendpd …` command in `docs/tutorials/*.md` is executed by
  `tests/integration/test_docs_commands.py` (or by the weekly workflow for the
  benchmark family) and every documented flag must exist in the parser, so
  update the docs and the CLI together.

## Commit messages

Use an imperative summary line, keep the body factual, and reference the plan
step (for example `S03`) when the change belongs to one.
