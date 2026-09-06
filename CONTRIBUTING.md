# Contributing to OpenDPD

Thanks for helping. This guide covers the development workflow for the core
library and OpenDPD Studio (the local browser workbench). Rules for coding
agents live in [AGENTS.md](AGENTS.md); both humans and agents follow them.

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

Tests must include failure paths. Coverage numbers do not replace correctness.

## Pull request definition of done

A change is complete only when code, contracts (OpenAPI/schemas), tests,
documentation and evidence land together. The PR template asks for:

1. the user problem being solved,
2. scope and non-goals,
3. how it was verified (commands, environment, artifacts),
4. anything left unfinished,
5. whether a protected path (see AGENTS.md §3) is touched.

Keep unrelated formatting, dependency swaps and seed changes out of feature PRs.
Pure refactors must show unchanged behaviour on frozen inputs.

## Protected scientific paths

Metric definitions, data splits, golden references, acceptance thresholds and
published benchmark results require a separate PR with the
`science-review-approved` label. See AGENTS.md §3.

## Commit messages

Use an imperative summary line, keep the body factual, and reference the plan
step (for example `S03`) when the change belongs to one.
