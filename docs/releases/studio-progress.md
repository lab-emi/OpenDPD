# OpenDPD Studio — stage progress

Per-stage status against the acceptance criteria of
`OpenDPD_Studio_Development_Plan.md`. "done" means evidence exists in this
repository; "pending human" means a maintainer decision is required;
"not verified" means the environment could not produce the evidence.

| Stage | Status | Commit | Notes |
|---|---|---|---|
| S00 baseline & governance | done except maintainer approvals | `6f026b1` | `docs/baseline/baseline-report.md` |
| S01 contracts & UX | done except prototype (delivered with S05) | `0f66ad0` | schemas, ADR-0001, UX spec, tokens, mock examples |
| S02 explicit config, registry, workspace | done | `9ea598c` | `opendpd run/validate/models/recipes/datasets`, registry, resolver, adapter, workspace |
| S03 task runtime | done (Windows/macOS cleanup not verified) | `fd63107` | SQLite store, supervisor, worker subprocess, cancel, recovery |
| S04 local service & API | done | `076de58` | FastAPI app, sessions/CSRF/Host checks, SSE replay, OpenAPI contract, threat model |
| S05 React workbench | done (visual baselines local only) | — | `frontend/`: pages, domain components, states, Vitest + Playwright journeys, generated API types |

## S00 acceptance items

| Item | Status |
|---|---|
| `baseline-report.md` with commands, env, results, failure causes | done |
| CLI/API compatibility cases, metric goldens, legacy checkpoint load case registered | done (`tests/golden`) |
| Maintainer approval of G0–G2 scope, support matrix, risks, thresholds | **pending human** |
| Protected paths cannot be changed by ordinary PRs | done (`protected-paths.yml`, CODEOWNERS) |
| Agent scope, budget, max pending PRs, stop conditions configured | done (`AGENTS.md`) |

## S01 acceptance items

| Item | Status |
|---|---|
| Three journeys with success / failure / exit paths | done (`docs/architecture/ux-spec.md`) |
| Page structure reviewed by maintainer; advanced params collapsed | spec written; **pending human review** |
| Every result expresses `pa_modeling` / `dpd_surrogate` / `dpd_measured` with source and profile version | done (`EvaluationResult` validators) |
| Schema examples: missing metadata, N/A metric, interrupted run, log disconnect, legacy import | done (`opendpd/schemas/examples.py`, 49 tests) |
| Frontend prototype on fixed, labelled mock | deferred to S05 (mock fixtures are generated from the same examples) |

## S02 acceptance items

| Item | Status |
|---|---|
| Two experiments in sequence do not leak configuration; parallel workers do not share output files | done (`tests/integration/test_cli_run.py::test_sequential_runs_do_not_leak_configuration`; each run has its own directory) |
| GUI / CLI / Python API normalise identically; defaults have one source | done (`opendpd.services.config.resolve`; `test_training_defaults_have_one_source`; legacy-CLI numeric equality test) |
| Model list from the registry; CLI choices and GUI options not hand-written twice; unknown models cannot be submitted | done for the new path (`opendpd models`, `test_registry_covers_every_working_legacy_backbone`); the legacy parser keeps its own list for compatibility |
| MP/GMP-style non-neural methods can express their own fitting logic | registry has `training_method`; least-squares MP/GMP integration is S12 work — **open** |
| Legacy commands and public API compatibility tests pass; deprecations announced | done (existing suite passes; nothing deprecated yet) |
| Plain `pip install opendpd` needs no GUI dependency; runs work from a read-only install into an external workspace | done (`tests/packaging/test_wheel_install.py`, `tests/unit/test_lazy_imports.py`) |

## S03 acceptance items

| Item | Status |
|---|---|
| Same idempotency key never creates a duplicate; each run has its own directory | done (`test_run_completes_through_worker_with_events`, DB unique index) |
| API stays responsive while a run trains; queued runs and cancel handled | done (`test_api_stays_responsive_and_queue_is_serial`) |
| Run visible after closing/reopening the browser; events resume from the last seq | store-level done (`events_after`); browser-level verified in S04/S05 |
| Killed worker, OOM, disk write failure, config error give explicit terminal states | done: SIGKILL → `failed/worker_died` with OOM hint; unwritable run dir → `failed`; invalid config → rejected before a run exists |
| Service restart marks unfinished runs `interrupted`; no ghost running runs | done (`test_restart_recovery_marks_interrupted`, orphan terminated) |
| Cancel/exit leave no child processes; identity is pid + creation time | done on Linux; **Windows/macOS not verified** |
| No pause/resume button; checkpoint resume not offered | done (not implemented, not shown) |

## S04 acceptance items

| Item | Status |
|---|---|
| Browser refresh / reopen shows running run and continues logs & metrics without loss | API level done: snapshot + `events/list?after=` + SSE `Last-Event-ID` (`test_studio_api.py::test_run_lifecycle_events_logs_artifacts_result`); browser level in S05 |
| Foreign web pages cannot call the local API; DNS rebinding refused | done (`test_cross_origin_and_csrf`, `test_host_header_enforced`; threat model) |
| Paths from clients cannot escape the workspace | done (artifact download by registered id only; `test_artifact_download_by_id_only`) |
| Error responses are structured, field-level, with hints | done (`error.code/message/details[]/hint`; `test_validate_reports_structured_errors_without_starting`) |
| OpenAPI is the single contract; TypeScript types generated and checked in CI | contract committed and checked (`scripts/export_openapi.py --check` in CI); TypeScript generation lands with the frontend in S05 |
| No request causes training inside the server process | done (all runs go through the S03 supervisor; `test_cancel_through_api` measures cancel latency < 1 s while a worker trains) |

## S05 acceptance items

| Item | Status |
|---|---|
| TypeScript strict, lint, unit and interaction tests pass without `any` / disabled rules | done (`npm run typecheck`, `npm run lint`, 21 Vitest tests; CI job `frontend`) |
| Main forms usable by keyboard; required fields, errors and focus explicit; colour never the only encoding | done (`NewExperimentPage.test.tsx` keyboard-only submit; server errors on the field with `aria-invalid`; `StatusChip.test.tsx` label + icon) |
| 1366×768 and 1920×1080 layouts without horizontal scroll; charts enlargeable | done (`e2e/journey.spec.ts` "fit" test on both viewports; enlarge dialog) |
| Unconfigured, running, failed and disconnected states have an actionable next step | done (`StateBlock`, `RunDetailPage` next-step banners, `RunDetailPage.test.tsx` disconnected and failed cases) |
| Run detail refreshable and addressable; refresh never resubmits | done (`/runs/:runId`; e2e reload keeps the run and the fake API records one submission) |
| One real-size visualisation prototype with performance evidence | done (component gallery: PSD 2560 bins × 3 traces + I/Q window of 20 000 samples; probe asserts < 2 s, measured ≈ 210 ms per chart in headless Chromium) |
| Frontend prototype on fixed, labelled mock (carried over from S01) | done (`/gallery` and `ResultView` on `frontend/mocks/*`, every mock result carries the MOCK badge) |
| Visual regression baseline | Playwright screenshots of the gallery captured on Linux/Chromium and committed; **compared locally only** (CI runs `--ignore-snapshots` because font rendering differs) |
