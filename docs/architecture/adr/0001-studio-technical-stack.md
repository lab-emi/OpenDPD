# ADR-0001: OpenDPD Studio technical stack and process model

- **Status:** accepted (agent proposal; maintainer may amend)
- **Date:** 2026-09-06
- **Plan step:** S01
- **Deciders:** OpenDPD maintainers

## Context

The development plan fixes FastAPI + React + TypeScript and a local-first,
single-user browser workbench that shares one compute core with the CLI and
the Python API. Several second-order choices still had to be made before
frontend and backend could be built in parallel: validation library, UI
component system, chart library, process model for training, persistence,
event transport, and how the legacy `sys.argv`/CWD-based trainer is wrapped
without a rewrite.

## Decision

| Concern | Decision | Notes |
|---|---|---|
| Contracts | **Pydantic v2** models in `opendpd/schemas`, added as a *core* dependency | One validator for CLI, API and GUI; OpenAPI and TS types are generated from it. Pydantic is small next to torch and ships wheels for every supported platform. |
| Web API | FastAPI + Uvicorn under `/api/v1`; `/healthz`, `/readyz` outside the prefix | Only in the `gui` extra. |
| Training execution | One in-process **supervisor** thread + one **worker subprocess per run** (`python -m opendpd.runtime.worker`) | Never `BackgroundTasks`; the worker runs with its run directory as CWD so the legacy path convention (`save/`, `log/`, `dpd_out/`) is preserved per run without touching `modules/paths.py`. |
| Legacy trainer | Wrapped by an adapter that builds an explicit `argparse.Namespace` from the resolved config (`opendpd/services/legacy_adapter.py`) | No new code reads `sys.argv`. The old `opendpd.api` functions stay as they are (documented single-threaded). |
| Persistence | SQLite (WAL) in `workspace/metadata.sqlite` for runs/events/datasets; files for arrays, logs, checkpoints | SQLite is the authority for run status; `events.jsonl` is an export. Single writer (the server process). |
| Worker ↔ server | Worker appends JSON-lines events to `<run>/events.jsonl`; the supervisor tails the file into SQLite and assigns sequence numbers. stdout/stderr go to `logs/worker.log`. Cancel = a `CANCEL` file polled at epoch boundaries | Amended in S03 (was: stdout pipe). A file survives stdout pollution by C-level prints, needs no signals on Windows, and the dead worker's events can be re-ingested at recovery. SQLite stays the single authority; the file is the worker's log. |
| Push to browser | Server-Sent Events (`GET /api/v1/runs/{id}/events?after=<seq>`) with replay from SQLite; polling fallback | No WebSocket layer in the first version. |
| Local security | Bootstrap token printed at start → session cookie (HttpOnly, SameSite=Strict) + CSRF header on writes; Host/Origin checks; loopback bind only | Details in `docs/architecture/threat-model.md` (S04). |
| Frontend | React 18 + TypeScript strict + Vite; **MUI** for components; **Plotly.js basic bundle** via `react-plotly.js` for scientific charts | Charts receive server-decimated data only. |
| Frontend tests | Vitest + React Testing Library; Playwright (Chromium first) for packaged E2E | Firefox/WebKit added in S13. |
| i18n | English default, all strings in `frontend/src/i18n/en.json` | Enables a later Chinese locale without touching components. |
| Distribution | Wheel/sdist carry the Vite build under `opendpd/studio/static/` | End users never need Node.js. |
| Process/OS helpers | `psutil` in the `gui` extra for process identity (pid + create time) and process-tree cleanup | |

## Alternatives considered

| Option | Why not |
|---|---|
| Dataclasses + hand-written validation to keep core dependency-free | Two validators (core vs API) would drift; the plan forbids duplicated semantics. |
| Celery/Redis/Kubernetes executors | Multi-machine is out of scope; a subprocess supervisor meets every first-version requirement. |
| Threads for training inside the server | Global torch state and the legacy `argparse` design are not thread-safe; a crash would take the UI down. |
| WebSockets | Bidirectional transport not needed; SSE replays from a durable sequence trivially. |
| Rewriting `project.py`/`steps/*` into a clean core now | Violates "no one-shot rewrite"; the adapter keeps checkpoints, logs and paper scripts working. |
| Ant Design / Chakra / hand-rolled CSS | MUI has the most complete accessible form/table set; a single component library is required. |
| Full Plotly bundle | ~3.5 MB; the basic bundle covers scatter/line needs. |

## Consequences

- `pydantic>=2.5` becomes a hard dependency of `opendpd`; CI installs it on every Python version.
- Agents may not swap any row of the decision table in a feature PR; a new ADR is required.
- The worker-CWD approach means a run directory contains the legacy layout; exporters must translate paths to artifact ids.
- Streaming/multi-machine executors, if ever needed, must implement the same supervisor interface.

## Verification

- `tests/unit/test_schemas.py` validates every contract example.
- S04 CI job diffs generated TypeScript types against the committed ones.
- S06 packaging test installs the wheel without Node.js and opens the GUI.
