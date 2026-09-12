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

Frontend development additionally needs Node.js 22.22+:

```bash
cd frontend && npm ci && npm run dev
```

The current Studio preview is installed from source and needs a frontend build
(`npm run build` in `frontend/`). A wheel that already contains the built
frontend does not need Node.js at runtime.

## Testing and documentation

Use the [test guide](docs/testing.md) for commands and test layers, and the
[documentation guide](docs/documentation.md) for content ownership and preview.
The README is the short entry point; detailed guides live under `docs/` and are
also published by MkDocs. Edit shared content at its source instead of copying
it into a second page.

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
- Tutorials: `tests/integration/test_docs_commands.py` checks documented flags
  and executes the command families in its tutorial list (the benchmark family
  also has weekly/protocol checks). Keep that list and the CLI documentation
  aligned when adding a guide or command.

## Commit messages

Use an imperative summary line, keep the body factual, and reference the plan
step (for example `S03`) when the change belongs to one.
