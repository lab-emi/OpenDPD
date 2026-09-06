# Hardening report (S13)

What was checked before calling the Studio a release candidate, with the
evidence for each claim. Numbers live in `docs/releases/performance-report.md`;
this file covers security, platform, migration/recovery, accessibility and
the candidate build. Items that need a person or a machine that does not
exist here are listed as **pending**, not as done.

Date: 2026-09-06. Machine: the S00 baseline machine (`docs/baseline/baseline-report.md`).

## Security checks

### Negative tests (all through the real ASGI application)

`tests/integration/test_hardening.py` (16 tests) plus the earlier S04/S07/S11
tests. One row per threat in `docs/architecture/threat-model.md`:

| Check | Result |
|---|---|
| Malicious Origin / cross-site write / DNS rebinding | refused (403 `cross_origin_write`, 400 `host_not_allowed`), CSRF header required for every write, no CORS |
| Path traversal and symlink escape on artifact download | refused (404 `artifact_missing`); traversal ids never reach the file system |
| Import roots: `../`, symlink to a file outside, symlink to a directory outside | not listed, not readable (409 "escapes the import root") |
| Experiment packages: traversal member, symlink member, member larger than recorded (64 MB zero bomb in a 1 MB zip), 10 001 members, damaged hash, unsupported version | refused before any write (`unsafe_path`, `unsafe_member`, `hash_mismatch`, `too_many_members`, `unsupported_version`); the workspace is unchanged |
| Chunked request body without `Content-Length` over the 2 MB cap | 413 `payload_too_large` |
| Uploads over the cap (dataset file, package) | 413, partial file removed |
| XSS: HTML in run names, notes; no HTML sinks in the frontend; diagnostic pages | returned as JSON data and escaped by React; `dangerouslySetInnerHTML`/`innerHTML`/`eval` absent from `frontend/src`; CSP `script-src 'self'` on every response; the real-server journey fails on any browser console error |
| Malicious pickle in a checkpoint (`__reduce__` → `open(marker, "w")`) | refused by the single restricted loader (`load_checkpoint`, `weights_only=True`), marker never created, evaluation raises a workspace error; legacy state_dict checkpoints still load; no `torch.load` without `weights_only=True`, no `allow_pickle=True`, no `pickle.load` anywhere in `opendpd/`, `steps/`, `modules/`, `utils/` |
| Accelerator held by another process / out of memory | classified as `device_busy_or_out_of_memory` with a hint; never retried or moved silently |
| Security headers | CSP, `nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: same-origin`, `Cache-Control: no-store` on API responses |

### Dependency and code scanning (2026-09-06)

| Scan | Result | Action |
|---|---|---|
| `pip-audit` (development venv, Python 3.13) | pillow 12.2.0 (14 PYSEC advisories, fixed in 12.3.0) and setuptools 78.1.0 (PYSEC-2025-49, PYSEC-2026-3447) | both upgraded in the venv; re-run: **no known vulnerabilities**; torch/torchaudio/torchcodec/torchvision are CUDA nightlies not on PyPI and were skipped by the tool |
| `npm audit` (frontend, runtime and dev) | 0 vulnerabilities | — |
| flake8 critical selection (`E9,F63,F7,F82,F401,F841`) | 0 | — |
| Manual review of the local boundary (`opendpd/server/security.py`) | body cap now holds without `Content-Length`; every response gets the security headers; no CORS | — |

Neither tool pins the project's dependencies: `pyproject.toml` declares lower
bounds, so a user installing today gets the fixed versions. The release
checklist (S14) repeats both scans on the release runner.

### Accepted residual risks

Listed in the threat model: same-user local processes, plain HTTP on
loopback, in-memory sessions, inline styles in the CSP, Plotly's HTML subset
for server-generated trace names, absolute paths in provenance records
(private, full packages only).

## Platform evidence

| Item | Status | Evidence |
|---|---|---|
| Packaged GUI: install, train smoke, export, exit (Linux, fresh venv) | verified | `tests/packaging/test_wheel_install.py::test_wheel_ships_gui_and_one_command_serves_it` drives the packaged service over HTTP: bootstrap → built-in dataset → 2-epoch run → result → share package download → healthy → SIGTERM exits clean and releases the lock |
| Chromium journeys (1366×768, 1920×1080) | verified | `frontend/e2e/journey.spec.ts` J1–J3 + gallery |
| Firefox journeys | verified locally and in CI | same spec, `firefox-1366` project |
| WebKit journeys | CI only | `ci.yml` installs Chromium, Firefox and WebKit with `--with-deps`; this Arch host lacks the Ubuntu-built WebKit's libraries (`libicu74`, `libflite1`) so WebKit could not be launched here |
| Real service driven by a real browser | verified (Chromium, Firefox) | `frontend/e2e/live.spec.ts`: bootstrap URL → register example → train → logs → result charts → share export download, with console errors (including CSP violations) failing the test; `weekly.yml` repeats it on ubuntu with all three engines |
| Playwright projects | 4 | `chromium-1366`, `chromium-1920`, `firefox-1366`, `webkit-1366` (visual baselines Chromium-only) |
| macOS: real `opendpd gui`, default browser launch, Ctrl+C cleanup | **pending (human)** | no Mac here; weekly CI covers the headless service on macOS only |
| Windows: `opendpd gui`, process-group cleanup, paths with spaces/Unicode | **pending (human)** | no Windows machine or runner |
| Safari (real) | **pending (human)** | WebKit Linux port is not Safari |

## Migration and recovery

`tests/integration/test_workspace_recovery.py` (3 tests) and the S03 runtime tests:

| Scenario | Result |
|---|---|
| Workspace moved to another path (with spaces, `ω`, `工作区`, parentheses) | runs, results and checkpoints found; `run.json`, `config.resolved.json`, `artifacts.json`, `result.json` carry no absolute path; re-evaluation reproduces the stored metrics within the frozen tolerance; the service indexes the moved runs, serves artifact downloads and exports a package |
| Backup copy restored elsewhere while the live copy is damaged | the copy opens, exports a full package, which imports into a fresh workspace and re-evaluates |
| Port already in use | `opendpd gui --port N` exits non-zero with "port N is already in use; pick another with --port or omit it to auto-select"; no lock file left |
| Server restart with a live worker | S03: interrupted runs are marked `interrupted` with a reason (`test_restart_recovery_marks_interrupted`) |
| Killed worker | S03: `failed/worker_died` |
| Unwritable run directory | S03: explicit failure before start |
| Malicious import | see the package rows above |

Provenance files keep the absolute command that was executed (a record, not
an input); they travel only in full packages.

## Accessibility

- Automated: `frontend/e2e/a11y.spec.ts` runs axe-core (WCAG 2.1 A/AA rule
  tags) on Home, Datasets, Experiments, New experiment, Results, a result
  page, Settings, the component gallery and every tab of a live run detail.
  Serious/critical violations fail the test. Findings fixed in S13: primary
  colour 3.36:1 → `#3a6f99` (5.4:1 on white), warning tone 4.35:1 → `#9d5200`
  (5.4:1), unselected toggle buttons 4.49:1 → `action.active #5a5c5e`
  (6.3:1), chart containers `role="img"` with focusable children →
  `role="figure"`, malformed evidence definition lists.
- Keyboard: the same spec completes register example → New experiment →
  Start run → Logs tab with Tab/Enter/Arrow keys only (ARIA tabs pattern).
- Colour is never the only encoding: status chips carry a label and an icon
  (`StatusChip`), evidence badges carry text, spectrum traces differ by dash
  pattern, AM-AM/AM-PM traces by marker symbol, training curves by both
  (`seriesDash`/`seriesSymbol`, unit-tested).
- **Pending (human):** a manual screen-reader pass; the automated audit
  covers rules, not the experience.

## Performance

The numbers are in `docs/releases/performance-report.md`, generated by
`scripts/perf_report.py` against a real `opendpd gui` server, a real worker
and a real browser (`frontend/e2e/live.spec.ts`); the JSON behind the report
records machine, data hashes and software versions. Two of the first
measurements failed a frozen target, and the plan forbids relaxing a target
or skipping a measurement, so both were investigated on the same machine:

| Finding (first full run, commit `68953a1`) | Cause | What changed |
|---|---|---|
| GUI overhead on training: pairs 3.8 %, 15.4 %, 12.8 % (median 12.8 %, target ≤ 5 %) | Bookkeeping, not training. The supervisor stamped `started_at` when it *spawned* the worker and `finished_at` when it *noticed* the exit at a later poll tick, while `opendpd run` stamps both inside the executor. The gap was a constant ≈ 0.8 s per run: with an explicit budget of 8 threads on both legs the difference stayed at ≈ 0.7 s (11 %), and the worker's own status events showed the training itself within 0.2 % of the CLI. | The supervisor now adopts the executor's `started_at`/`finished_at` from the worker's record (`docs/architecture/runtime.md`, "Timestamps"; `test_run_completes_through_worker_with_events`), so the two paths report the same clock. Start-up and exit detection are reported next to the training wall as lifecycle time, not hidden. |
| Event latency max 2.408 s in one of 247 events (target ≤ 2 s; p95 0.311 s); 2.939 s in the second full run | A measurement artifact, not the service: the per-event diagnostics of the second run show the two slowest "events" are the store's own status events written at spawn (sequence 1 and 2) and the next three are the first epoch's events, all emitted while the harness was still busy with the API-latency phase and before it started polling. Their "latency" was the harness's own delay; every slow poll itself took 3 ms. Measured with the poll loop already open the maximum was 0.354 s (`--phases events` alone). | The harness now opens the poll loop first and counts only events emitted after that moment, and records the five slowest events with type, sequence number and the duration of the poll that returned them; the final report (max 0.359 s, p95 0.314 s over 245 events) was produced this way. |
| Stress import extra RSS 18.9 MB in the first run, then 1 788.9 MB (target ≤ 1 GB) in the second | Both numbers were wrong the same way: `ru_maxrss` counts the file-backed pages of the memory-mapped 1.6 GB source and of the two memory-mapped outputs, which are the OS file cache the plan excludes; the first run only looked good because its baseline process happened to be equally inflated. The import itself streams (chunked CSV and `.npy` writing). | The harness samples `RssAnon` of the import process every 20 ms and reports the peak anonymous memory (2.9 MB above the baseline) as the target figure, with the file-backed peak (2.3 GB) and the total RSS stated next to it. A failed import can no longer show as a pass, and the stress import runs in its own scratch workspace (the second run had failed on an existing dataset id and reported 0 MB). |
| `execution.num_threads` accepted by the schema but applied nowhere | A declared knob that did nothing. | `apply_thread_budget` in the shared executor applies it on every path (`test_thread_budget_is_applied_by_the_executor`); the harness measures overhead with torch's default budget (the target figure); `--overhead-threads default,8` adds the same pairs with an explicit equal budget as a diagnostic (included in this stage's report: median −0.5 %, so the budget was never the cause). |

Regenerate with `python scripts/perf_report.py --workspace <scratch>/ws --minutes 30 --stress`
on an idle machine; `--phases` re-runs a subset and merges it into the JSON.

## Release candidate

Built with `python -m build` from the S13 tree (checksums are recorded in the
progress table entry of the closing commit, `docs/releases/studio-progress.md`);
`twine check` passed for both files. The wheel carries the built frontend
(`opendpd/studio/static/build-info.json`), installs without Node.js and serves
the GUI with one command (packaging tests above). Version `2.2.0.dev0` is a
candidate label: tagging a release is a maintainer decision (S14).

## Open items carried to S14

- Human platform checks: macOS and Windows desktops, real Safari, default
  browser launch.
- External trial and onboarding measurements (S14).
- GPU-tier performance and the full benchmark matrix (L4, human-approved).
