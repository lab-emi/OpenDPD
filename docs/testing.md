# Testing and continuous integration

The [CI workflow](https://github.com/lab-emi/OpenDPD/actions/workflows/ci.yml) checks critical Python errors, the Python 3.10–3.13 test matrix, frontend types/lint/tests/build, browser journeys and distribution packaging. The [Weekly workflow](https://github.com/lab-emi/OpenDPD/actions/workflows/weekly.yml) covers broader backbone, dataset and platform checks. Read individual job results for the verified scope; a browser mock test alone is not computation evidence.

## Run locally

From an activated development environment in the source checkout:

```bash
uv venv --python 3.12
uv pip install --python .venv -e ".[dev]" --torch-backend=auto
npm --prefix frontend ci
npm --prefix frontend run build
python -m pytest tests/ -m "not extended"
```

The frontend build is required by packaging checks. For frontend-only changes:

```bash
npm --prefix frontend run typecheck
npm --prefix frontend run lint
npm --prefix frontend test
```

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

## Documentation checks

```bash
python -m pip install -r docs/requirements.txt
python -m mkdocs build --strict
python -m pytest tests/integration/test_docs_commands.py
```

The documentation integration test validates CLI flags and runs real CPU experiments for the tutorial command families it covers. It uses small temporary workspaces; it does not establish GPU or physical RF performance. See [Maintaining documentation](documentation.md) for content ownership, link conventions and local preview.

## Repository layout

```
.
├── backbones/       # Neural backbone implementations
├── bash_scripts/    # Batch experiment scripts (train_all_*.sh, quant_*.sh)
├── datasets/        # Built-in PA datasets (CSV + spec.json)
├── examples/        # API examples and tutorials
├── modules/         # Data pipeline, logging, and training utilities
├── opendpd/         # API, shared services, runtime, server and Studio launcher
├── frontend/        # Studio interface
├── docs/            # Guides, protocols and MkDocs pages
├── quant/           # Quantization-aware training components
├── steps/           # CLI entry points (train_pa, train_dpd, run_dpd)
├── tests/           # Pytest suite (unit + end-to-end smoke tests, run in CI)
├── utils/           # Miscellaneous helper functions
├── Makefile         # Convenience targets (install, clean, etc.)
├── main.py          # Legacy CLI entry mirrored by opendpd-cli
└── project.py       # Core configuration & training orchestration
```

## Default installation checks

CI installs the default package on Linux, macOS and Windows with uv automatic PyTorch backend selection. `scripts/check_install.py` verifies platform bindings and tensor arithmetic on each available CPU/CUDA/MPS backend. A subprocess regression checks that a failed browser opener leaves the service reachable. This does not claim a native window was manually exercised on all three platforms.

## Dependency advisory checks

CI audits the frontend lockfile with `npm audit --audit-level=moderate`. The Linux default-install job audits its resolved Python environment using `scripts/audit_requirements.py` and `pip-audit --strict --no-deps --disable-pip`. The helper normalizes official PyTorch build suffixes to the upstream release and respects environment import precedence; it omits only this source-reviewed application. Deployment review separately scans the actual GPU image, including OS packages. A clean advisory scan is a dated check, not a guarantee against undisclosed defects.
