# First execution tasks (issue-ready backlog)

Ordered as in the plan §10. Each entry follows the issue template of plan §9.4
and can be pasted into a GitHub issue verbatim. Maintainers create the issues;
agents do not open them on their own.

## 1. S00 — Freeze baseline and compatibility tests
- **Goal:** know what must be preserved before any change.
- **Acceptance:** `docs/baseline/baseline-report.md` with commands, env, results; golden metric fixture; legacy checkpoint fixture loads; governance files present; protected-path CI guard active.

## 2. S01 — Freeze minimal config/run/result contracts and user journeys
- **Goal:** frontend and backend build against one vocabulary.
- **Acceptance:** Pydantic v2 schemas for `DatasetManifest`, `DiagnosticReport`, `ExperimentConfig`, `RunRecord`, `MetricProfile`, `EvaluationResult`, `ArtifactManifest`, `RunEvent`; examples including missing metadata, N/A metrics, interrupted run, log disconnect, legacy import; UX spec; ADR for the tech stack; mock fixtures labelled as mock.
- **Non-goals:** any training code.

## 3A. S02 — Explicit configuration and workspace entry
- **Acceptance:** two experiments in sequence do not leak config; GUI/CLI/API normalise to the same resolved config; model list comes from a registry; legacy CLI untouched; `pip install opendpd` has no GUI deps.

## 3B. S05 — React design system and mock pages
- **Acceptance:** TypeScript strict; clickable full journey on mock; loading/empty/error/disconnected states; keyboard-only forms; 1366×768 and 1920×1080.

## 4. S03/S04 — One Tiny run submitted through the API and persisted
- **Acceptance:** real worker process, logs, cancel, event replay after reconnect, interrupted detection after restart, session/Origin/CSRF/path-boundary negative tests.

## 5. S06 — Release package and one-command launch
- **Acceptance:** wheel installed in a clean venv without Node.js, `opendpd gui` opens the real app; port/browser/no-desktop/duplicate-start cases recorded.

## 6. S07/S08 — Dataset doctor and explicit legacy metric profile
- **Acceptance:** import CSV/NumPy/legacy dataset dir; diagnostics with error/warning/info and evidence; `legacy-opendpd-v1` profile frozen; new general profiles with analytic reference tests.

## 7. S09–S11 — One constrained real PA/DPD recipe end to end
- **Acceptance:** GUI run + export; another workspace re-evaluates through CLI within tolerance.
