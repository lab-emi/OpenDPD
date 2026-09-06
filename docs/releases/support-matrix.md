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
