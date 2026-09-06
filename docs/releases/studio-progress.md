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
| S05 React workbench | done (visual baselines local only) | `c002ca2` + `c5f5b83` | `frontend/`: pages, domain components, states, Vitest + Playwright journeys, generated API types |
| S06 packaging & one-command launch | done on Linux; **macOS/Windows not verified** | `86ddd7a` | `opendpd gui`, `opendpd doctor`, wheel/sdist carry the built frontend, packaged L2 test |
| S07 data import, Dataset Doctor, traceable preprocessing | done | `4fea772` + `dd85377` | import roots + upload, `dataset-doctor-v1`, `preprocess-v1` versions, `contiguous-v1` split, datasets pages, J2 journey (mock) and headless CLI J2 on real data |
| S08 versioned metrics & evaluation evidence | done | `999f031` + `51891cc` | `opendpd/core/metrics` registry (`legacy-opendpd-v1` frozen, `general-spectral-v1`), analytic reference tests, results re-scored from the best checkpoint under every profile, profile switcher and definitions in the GUI |
| S09 PA training closure & experiment management | done except the human-gated items below | `370616e` + `a6f68c5` | registry-generated model parameters, config import/export/re-run, device refusal at validation, GUI-vs-CLI consistency test, real-browser PA training on imported data |
| S10 DPD training, signal generation & surrogate evaluation | done | — | explicit x/u/y chain in every DPD result, baselines under one reference, surrogate amplitude coverage, `run_dpd` results through any surrogate (never overwriting), export sidecar + reload test, lineage route/card, `opendpd apply`, Apply-DPD dialog |

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

## S06 acceptance items

| Item | Status |
|---|---|
| Wheel installed outside the repo without Node.js; one command opens a usable GUI | done on Linux (`test_wheel_ships_gui_and_one_command_serves_it`: fresh venv, `opendpd gui --no-browser`, healthy → ready → page served → clean exit) |
| sdist also carries usable static assets, or generates them in an explicit build stage | done: `MANIFEST.in` includes `opendpd/studio/static`; CI `build` job builds the frontend before `python -m build` and asserts both artifacts contain it; a source checkout without the build shows a diagnostic page and `opendpd doctor` names the fix |
| Offline after install: example, fonts, icons, charts, training and export work | fonts are system stacks, icons are bundled SVG, Plotly is bundled; `tests/unit/test_offline_assets.py` asserts no external loads in the built assets. Export is S11 |
| Port in use, no default browser, no desktop, duplicate start, paths with spaces/Chinese | done: explicit busy port → exit 2 with message; auto-port skips busy ports; `webbrowser` failure never aborts (URL printed); second start reuses the running instance via `.studio.lock`; tests use workspaces named with spaces and Chinese characters |
| Browser opens only after the health check; missing/mismatched static shows a diagnostic, never a blank page | done (`test_browser_opens_only_after_health_check_with_bootstrap_url`; `/readyz` reports frontend presence and version; SPA route returns the diagnostic page when assets are missing or mismatched) |
| Real browser launch on three platforms | Linux: verified 2026-09-06 in a real desktop browser against `opendpd gui` (bootstrap URL → register example → new experiment validated server-side → 3-epoch run with live metrics → result with evidence badge); the first real pass found and fixed two defects (`evaluation` was required in the contract; a port left in TIME_WAIT was reported busy). **macOS and Windows: pending human** |
| Access session and file boundary active; no "open all local files for the demo" | done (S04 boundary unchanged; artifacts by id only) |

## S07 acceptance items

| Item | Status |
|---|---|
| Built-in data and user CSV import; wrong column names, lengths, I/Q order and units can be explained and corrected | done: `POST /datasets/inspect` returns headers, a preview, problems and a suggested mapping (alias table incl. `tx_i`/`rx_q`); the GUI mapping selects and the CLI `--map LOGICAL=COLUMN` fix swapped I/Q; length/dtype/object-array problems are reported before anything is written (`tests/unit/test_datasets_service.py`) |
| Known delay, gain, outliers and clipping detected within the fixture protocol; natural PA non-linearity is not called "broken" | done: `tests/fixtures/manifest.json["doctor_protocol"]` (delay ±0.1 sample at 3/7/40, gain ±0.5 dB / ±3°, clipping at 60 % of peak, 5 spikes, NaN blocks); `tests/unit/test_doctor.py` (10 tests incl. clean memory-polynomial PA raising no defect) |
| error/warning/info with evidence and suggestions; insufficient metadata blocks unreliable evaluation | done: every item carries `evidence`; `metadata_missing`, NaN, length, silent signal are blocking and set `evaluation_blocked` (`docs/protocols/dataset-doctor.md`) |
| Preview before a new version is created on confirmation; raw hash unchanged; parameters and code version recorded | done: `POST …/preprocess/preview` then `POST …/preprocess` with a version name; `versions/<name>/version.json` stores params, `code_version=preprocess-v1`, `fit_range`, steps; `raw/` is hashed and never modified (`test_versions_keep_raw_untouched_and_fit_only_on_train`) |
| Split before framing; boundary isolation covers the context | done: `contiguous-v1` splits the continuous signal with a guard (default 256 samples) before any framing (`opendpd/core/splits.py`, protected path); validation warns when `training.frame_length` exceeds the guard (`test_validate_names_missing_versions_and_guard_shorter_than_the_frame`). Built-in/legacy directories keep their original split (guard 0, no warning) |
| Fit range of learned preprocessing recorded; never fitted on the test set | done: `normalize=peak_input` requires `fit_range` = the training split and refuses otherwise; measurement alignment (delay/gain) is a separate, declared correction |
| Large files read incrementally; browser upload and authorised-directory import are distinct; no arbitrary path browsing | done: CSV read in 200k-row chunks, NumPy via memory-mapped `np.load(mmap_mode="r", allow_pickle=False)`; upload streams into `imports/uploads/` (2 GB cap, other bodies 2 MB); import only by `(root_id, relative path)` with traversal refused (`test_roots_listing_and_traversal_refused`) |
| GUI, CLI and Python API return the same diagnostic report and manifest | done: one service (`opendpd.services.datasets`) behind `opendpd datasets import/doctor/preprocess` and the routes; `test_inspect_import_doctor_preprocess_flow` compares the CLI JSON report with the API one |
| Real-data end-to-end through GUI/CLI consistency (G1 gate, partial) | CLI: `test_cli_import_doctor_preprocess_and_train_on_a_version` imports a synthetic CSV, runs the doctor, creates `aligned-v1`, trains on it and gets a result. GUI: J2 journey runs against the mock API (`frontend/e2e`); a real-browser pass on real data is a G1 checklist item, not yet done |

## S08 acceptance items

| Item | Status |
|---|---|
| Historical profile keeps its old numbers on frozen fixtures; fixes need a new version or explicit compatibility mode | done: `legacy-opendpd-v1` is `frozen=True`; `tests/golden/test_legacy_metrics_golden.py` (rel/abs 1e-7) and `test_legacy_profile_through_the_registry_reproduces_the_frozen_goldens` pin it; `opendpd/core/metrics/` is a protected path |
| Analytic / independent reference tests: zero error, known complex gain error, known noise power, known band power | done (`tests/unit/test_metrics_general.py`): zero error → documented −300 dB floor; gain g → exactly 20·log10\|g−1\|; white noise → SNR ±0.1 dB pooled and SNR+10·log10(BW/fs) ±0.15 dB in band; adjacent tone → its power ratio ±0.05 dBc |
| Negative-dBc leakage vs positive-dB suppression kept apart; name, formula and better-direction consistent | done: `ACPR_L/R` are leakage (adjacent/main, dBc, lower is better); the suppression convention is documented as −ACPR and not reported; the legacy `ACLR_*` notes state their convention; `MetricDefinition` carries display name, formula, aggregation, unit and direction and the GUI renders them from the registry |
| Pooled NMSE vs mean of per-segment dB; EVM ratio / percent / dB not mixed | done: `general` NMSE is pooled, `legacy` NMSE is the mean of segment dB (`test_pooled_nmse_is_not_the_mean_of_per_segment_db`); `IBE` is a dB power ratio with the ratio and percent forms documented in its notes; the legacy EVM is labelled repo-specific and "not a demodulated EVM" |
| No valid reference, zero reference energy, band outside capture, empty band, non-finite → explicit statuses, never a fake 0 or silent NaN | done: `test_every_undefined_case_has_an_explicit_status` covers all six cases; `MetricValue` forbids a value without `ok` and `ok` without a finite value |
| GUI shows evidence label and metric profile by default; surrogate results never displayed as measured | done: `EvidenceBadge` + profile chip on every result, the profile switcher lists the stored profiles, `MetricCard` shows the registry display name and a definition tooltip (`ResultDetailPage.test.tsx`, e2e J1); evidence type is declared by the worker, never inferred in the GUI |
| Display decimation, zoom or chart filtering never change formal metrics | done by construction: metrics are computed once in the worker from the best checkpoint over the full valid range and stored (`results/<profile>.json`); charts (S11) come from separate decimated data. Documented in `docs/protocols/metric-profiles.md` |
| No automatic ranking across incompatible protocols; side-by-side allowed with the reason | done at the core level: `opendpd.core.metrics.incompatibilities(a, b)` names profile/version, evidence type, dataset, preprocessing version, split protocol, split, reference kind and mock status (`test_results_under_different_protocols_are_not_comparable`); the comparison page (S11) must use it |
| Results carry data, model, preprocessing, reference and evaluation source | done: `EvaluationResult` (dataset evidence incl. preprocessing/split versions and valid range, model evidence with weight hashes, reference kind and gain rule, evidence type, profile id/version); GUI / CLI (`opendpd evaluate`, `opendpd profiles`) / Python API (`opendpd.services.evaluation`) share one implementation |

## S09 acceptance items

| Item | Status |
|---|---|
| After importing data, a real PA training through the GUI produces a valid checkpoint, logs and an evaluation result | done: verified 2026-09-06 in the desktop browser against `opendpd gui` on a scratch workspace: `imports/lab/capture.csv` (40 000 synthetic samples, +6 samples delay, 3 spikes) → Import my data (mapping guessed, metadata typed) → Dataset Doctor found the delay (+6.04), the 3 spikes and the +0.31 dB gain → Preprocess with the doctor's estimates, preview (40 000 → 39 992, aligned +0.00) → `aligned-v1` → New experiment on `aligned-v1` → 3-epoch GRU run succeeded with live per-epoch metrics → result page: PA-model evidence badge, `legacy-opendpd-v1 v1 · frozen` chip, NMSE −23.61 dB, switchable to `general-spectral-v1` (pooled NMSE −24.49 dB, IBE, ACPR). On disk: `save/…/*.pt`, history/best logs, `result.json`, `results/{legacy,general}.json`, provenance naming `aligned-v1` |
| Data, device, model parameters and paths are checked before submission; unsupported devices are refused, never switched silently | done: `experiments.submission_issues` (dataset version, split guard vs frame context, device detected) runs in `POST /experiments/validate`, `POST /runs` and `opendpd run`; `test_undetected_device_is_refused_never_switched` (422 `invalid_config` on `execution.device` with the hint to choose cpu or a detected device); model parameters are validated by the registry and reported per field |
| Quick demos and formal training are clearly distinct; short runs are never dressed up as benchmarks | done: recipes carry `purpose` (smoke / research), the form shows the smoke banner, the resolver warns on `training.epochs`, and every result of a short run lists the limitation (`SMOKE_EPOCH_LIMIT`) |
| Same CPU deterministic fixture through GUI and CLI: identical resolved configuration, outputs within the frozen tolerance | done: `tests/integration/test_entry_consistency.py` runs one configuration (`reproducibility: hard`, 2 epochs) through the HTTP API + worker subprocess and through `opendpd run` in-process: same `config_sha256`, same resolved configuration, same selected epoch, every metric within 1e-3 dB (tolerance recorded in `docs/protocols/acceptance-thresholds.md`, **pending maintainer approval**) |
| Checkpoint selection uses the configured validation metric, never the test split | done: the resolver fixes `evaluation.checkpoint_selection_metric` per task protocol (`test_config_resolve.py`), the legacy trainer selects on validation, and `test_result_metrics_come_from_the_registry_and_agree_with_the_training_log` asserts `selected_epoch` = argmin `VAL_NMSE` of the history log |
| Failure, cancel, busy device, queue wait and missing checkpoint have real states; no invented progress | done for failure/cancel/queue/missing checkpoint (S03: `failed/worker_died`, `cancelled`, `queued`, `input_missing`; progress comes only from epoch events); an unavailable device is refused before a run exists (above). A *busy* accelerator (another process holding it) is not detected separately: it surfaces as a worker failure with the CUDA error message — **known gap, tracked for S13** |
| Logs paged / virtualised, errors located to a stage, no sensitive paths or credentials in outward diagnostic packages | logs: done (byte-offset paging, `LogViewer`); errors: done (`RunError.stage` = prepare / train / evaluate / finalize); outward diagnostic packages do not exist yet (S11 export, S13 hardening): **open** |
| Reference recipes, registry-generated parameters, experiment list, curves, logs, checkpoints, config comparison, config import/export | done: recipes (S02), advanced settings generated from `ModelInfo.params` (name, type, limits, description; `NewExperimentPage.test.tsx`), experiments list + run detail with per-epoch curves / logs / artifacts / resolved config (S05), `ConfigDiff` component, **Download configuration (JSON)** and **Re-run with this configuration** on the run page, **Import configuration…** on the form (resolution block dropped, submitted as is) |

## S10 acceptance items

| Item | Status |
|---|---|
| Without a compatible PA surrogate a DPD recipe cannot start, and the remedy is stated | done: `bind_references` refuses missing / failed / foreign-dataset PA runs and seed-frame mismatches with field-level issues whose hints name the fix (`test_dpd_rejects_incompatible_pa`); the GUI lists only succeeded PA runs of the dataset and explains when there is none |
| `run_dpd` output is marked as the pre-distorted PA input, never a linearised PA output | done: artifact description, sidecar `dpd-output-meta` (`signal_role: pa_input_predistorted`, statement, column meaning), the `u` stage of the signal chain, `opendpd apply` output text and the Apply dialog all say so (`test_run_dpd_exports_predistorted_input`) |
| The simulation chain records the PA checkpoint hash and the evaluation profile; another surrogate produces a new result without overwriting the old | done: `ModelEvidence.weights_sha256` for PA and DPD, `metric_profile_id/version` in every result; `run_dpd` with `pa_reference` writes its own `result.json` + `results/<profile>.json` (`test_swapping_the_surrogate_yields_a_separate_result` checks the DPD run's result file is byte-identical afterwards) |
| Input scaling, output-power-related metadata and the reference gain are recorded; no dBm / efficiency without physical calibration | done: `EvaluationResult.scaling` (amplitude units, input scaling from the data version, reference gain, `physical_calibration: false`), the validator rejects a calibration claim, the GUI shows the uncalibrated note; nothing derives absolute power |
| Out-of-range amplitude hints have an explainable basis and do not claim to prove surrogate validity | done: `surrogate_coverage` (fitted peak from the surrogate's training split, peak of u, fraction above) with a note that says so in both branches; a non-zero fraction is listed as an extrapolation limitation |
| Baseline and DPD use one consistent, traceable reference; no separate normalisation | done: `baselines` (`surrogate_without_dpd`, `measured_without_dpd`) are scored against the same `reference_gain · x` with the same profile and valid range; the validator requires the same metric set; the GUI table shows them side by side (`ResultDetailPage.test.tsx`) |
| Exported pre-distorted arrays reload with the same order, dtype, amplitude and metadata | done: `test_run_dpd_exports_predistorted_input` rebuilds `u` independently over the whole test split and compares the reloaded CSV (float32, atol 1e-6), the x columns (exact), the row count and the recorded peak |
| Deliverables: DPD workflow, linked experiment graph, signal export, simulation evidence page, integration tests | done: Apply-DPD dialog + `opendpd apply` + `run_dpd_config(pa_run_id=)`; `GET /runs/{id}/lineage` + Lineage card; sidecar export; result page chain / baselines / coverage / scaling; `tests/integration/test_cli_run.py` (15 tests incl. 5 new), API lineage test, schema tests, Vitest pages tests; `docs/protocols/simulation-chain.md` |
