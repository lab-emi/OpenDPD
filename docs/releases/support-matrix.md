# Support matrix (Studio alpha)

Status legend: **verified** = real test evidence linked below; **unverified** =
designed for but no test evidence yet; **unsupported** = out of scope.
A platform being able to open the page does not imply every model or
accelerator path is supported, so three tables are kept separately.

Last update: 2026-09-06 (S13; baseline in `docs/baseline/baseline-report.md`, hardening in `docs/releases/hardening-report.md`).

## Core library and CLI (CPU)

| OS | Python | Status | Evidence |
|---|---|---|---|
| Linux x86-64 (CachyOS, kernel 7.2) | 3.13.14 | verified | S00 baseline: 197 tests, legacy CPU train/dpd/run chain |
| Linux x86-64 (ubuntu-latest) | 3.10–3.13 | verified (CI) | `.github/workflows/ci.yml` on `main` |
| macOS Apple Silicon (macos-latest) | 3.13 | verified (weekly CI) | `.github/workflows/weekly.yml` |
| Windows x86-64 | 3.11–3.13 | unverified | no runner yet |

## GUI (`opendpd gui`, CPU)

| OS | Python | Status | Evidence |
|---|---|---|---|
| Linux x86-64 (CachyOS) | 3.13.14 | verified (automated) | `tests/packaging/test_wheel_install.py::test_wheel_ships_gui_and_one_command_serves_it`: wheel installed in a fresh venv, `opendpd gui --no-browser` healthy/ready/serves the page, trains a smoke run, exports a share package and exits clean (S13); real-browser journeys in `frontend/e2e` (below); performance report `docs/releases/performance-report.md` |
| Linux x86-64 (ubuntu-latest) | 3.10–3.13 | verified (CI) | `ci.yml` builds the frontend and runs the packaging layer; `weekly.yml` drives the real service with Chromium, Firefox and WebKit (`frontend/e2e/live.spec.ts`) |
| macOS Apple Silicon | 3.11–3.13 | **unverified** | needs a person to run `opendpd gui` on a real desktop (browser opening, Ctrl+C cleanup); weekly CI covers only the headless service on macOS |
| Windows x86-64 | 3.11–3.13 | **unverified** | needs a person: browser opening, process-group cleanup, paths with spaces/Unicode |

## Browsers (Studio frontend)

| Engine | Where | Status | Evidence |
|---|---|---|---|
| Chromium (Playwright build) | Linux, 1366×768 and 1920×1080 | verified (automated) | `frontend/e2e/journey.spec.ts` (J1–J3, gallery), `e2e/a11y.spec.ts` (axe-core WCAG 2.1 AA, keyboard-only journey, offline), `e2e/live.spec.ts` against the real service; timings in the performance report |
| Firefox (Playwright build) | Linux | verified (automated) | same journeys and accessibility audit, run locally and in CI |
| WebKit (Playwright Linux port) | Linux (CI only) | verified in CI only | `ci.yml` runs the mock-API journeys on WebKit; `weekly.yml` runs the live journey; the Linux port shares the engine but not Safari's UI or platform integration |
| Safari (macOS/iOS) | macOS | **unverified** | needs a person with a Mac; nothing in the frontend is Chromium-specific (no WebGL, no non-standard APIs) but that is not evidence |
| System default browser launch | any | **unverified** | automated tests mock `webbrowser`; a real launch is a human checklist item (plan S06/S13) |

Automated tests that mock `webbrowser` prove only the call logic; a real
browser launch on macOS/Windows is a human checklist item (plan S06).

## CUDA

| Platform | Torch | Status | Evidence |
|---|---|---|---|
| Linux x86-64, RTX 4090 Laptop (CUDA 13.2) | 2.13.0+cu132 | verified (smoke) | S00 `train_pa --accelerator cuda`, 2 epochs |
| Windows CUDA | — | unverified | |

## MPS (Apple)

| Platform | Status | Evidence |
|---|---|---|
| macOS Apple Silicon | unverified | weekly CI runs CPU only |

## Instrument adapters (S16)

| Adapter | Status | Evidence |
|---|---|---|
| `mock` (dry run, no RF) | verified (automated) | `tests/unit/test_instruments.py` (interlock: arming, limits, timeout, lost link, failure, abort, block exit), `tests/integration/test_cli_run.py` and `test_docs_commands.py` (dry-run → import → `dpd_measured` mock result) |
| any real generator/analyser chain | **pending human** | no laboratory chain in this environment; a real adapter needs `OPENDPD_ALLOW_RF_OUTPUT=1` in an approved session and a recorded trial (`docs/protocols/measured-dpd.md` §7) |

## Adaptation protocol (S17)

| Condition set | Status | Evidence |
|---|---|---|
| synthetic three-condition cards (`drive`) | verified (automated, rehearsal only) | `tests/integration/test_adaptation.py` (18 cells, refusals kept, hash binding, warm start, budget, transfer), `test_docs_commands.py` (tutorial) |
| `apa-200mhz-batches-v1` (built in, two capture batches) | audited, below the bar | `opendpd adaptation card apa-200mhz-batches-v1`; two conditions |
| a measured card that meets the bar (≥ 3 conditions, independent batches) | **pending human** | no such data in this repository (`docs/protocols/conditions-v1.md` §7) |

## Streaming variants (S18)

| Variant | Executes | Status | Evidence |
|---|---|---|---|
| `gru_stream` | `gru` weights, hidden state carried across chunks | verified (automated, CPU), **experimental** | `tests/unit/test_streaming.py`, `tests/integration/test_streaming_eval.py`, `docs/releases/streaming-semantics-report.md` |
| `gmp_stream` | `gmp` weights, 20-sample history carried across chunks | verified (automated, CPU), **experimental** | same |
| look-ahead models (`tres_gru`, `tres_deltagru`, `tcn`) | — | no streaming variant | look-ahead stated in the registry and in every result; buffering cost in the semantics report |

## Deployment export backends (S19)

| Backend | Model | Status | Evidence |
|---|---|---|---|
| C99 reference (`fixed-point-v1`) | one-layer `gru` as `gru_stream` | verified bit for bit on every package (Linux, gcc), **rules pending human approval** | `tests/unit/test_fixed_point.py`, `tests/integration/test_deploy.py`, `test_docs_commands.py` |
| ONNX, HLS, RTL | — | not provided | an implementer verifies against the package's golden vectors and state trace (`docs/protocols/fixed-point-v1.md` §7) |

## Leaderboard tooling (S20)

| Part | Status | Evidence |
|---|---|---|
| `opendpd leaderboard prepare / check / seed / add / review / amend` | verified on CPU (Linux) with share packages of the built-in data; recomputation imports into a fresh workspace and re-scores from the checkpoint | `tests/integration/test_leaderboard.py`, `test_docs_commands.py` |
| Boards `docs/leaderboard/v2026.09/*` | reference benchmark: the maintainers' `cpu_regression` entries, self-reported; **no external submission, no independent recomputation, no external protocol reviewer yet (pending human)** | `tests/unit/test_leaderboard_schema.py` (intact hashes, copied from the report) |
| Tracks `standard_evaluation`, `robustness`, `deployment` | closed until their stage gates pass (S15, S17, S19) | `TRACK_GATES` in `opendpd/schemas/leaderboard.py` |
