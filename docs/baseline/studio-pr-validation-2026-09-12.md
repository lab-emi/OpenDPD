# Studio PR validation — 2026-09-12

PR: https://github.com/lab-emi/OpenDPD/pull/20

Base: `OpenDPD-Studio`; head: `codex/studio-experience`. The protected-path diff
against this base is empty. Environment: macOS 26.6.2, Apple Silicon, Python 3.13,
CPU computation, Chromium. Raw logs and experiment captures remain local under
the repository's ignore policy; this report records the reproducible commands
and their outcomes.

## Verification

- `npm --prefix frontend test -- --maxWorkers=2`: 178 passed in 37 files.
- From `frontend`, `npx playwright test --project=chromium-1366
  --ignore-snapshots --workers=1`: 12 passed, 3 skipped in 43.4 seconds. This includes
  the existing chart performance budget, nine-language menu, keyboard journeys,
  accessibility and loopback-only browser requests. The skipped cases require an
  explicitly configured live server; these mocked browser journeys do not replace
  the real API/CPU evidence in the existing experiment and appearance reports.
- `python scripts/export_openapi.py --check` and
  `npm --prefix frontend run types:check`: passed.
- Critical flake8 checks (`E9,F63,F7,F82`): passed.
- `python -m pytest tests/ -q -m 'not extended' --durations=10`:
  **658 passed, 17 skipped, 19 deselected, 2 failed**, in 452.71 seconds.
  One failure was the former blanket ban on outbound service calls, which conflicts
  with the requested public GitHub activity on About. The revised check permits
  only the launcher and About client calls; an added test verifies the two fixed
  public repository endpoints, GET method, headers, timeout and absence of a
  request body. `python -m pytest tests/unit/test_about.py
  tests/integration/test_hardening.py -q`: **19 passed** after that change.

## Remaining blocker

`tests/unit/test_offline_assets.py::test_built_assets_reference_no_external_hosts`
still fails. The Plotly strict bundle includes map attribution hyperlinks and a
Mapbox icon loader using jsDelivr. Studio's current scatter journeys did not make
external requests, but the static bundle check rejects those references. The
offline assertion is unchanged; an exploratory filter for inert anchors was
discarded after it still exposed the CDN loader. This PR remains draft with this
failure reported rather than exempting the dependency or weakening the check.

The first CI run on `a4a7904` is recorded at
https://github.com/lab-emi/OpenDPD/actions/runs/34704488168. Lint and distribution
build passed. Python jobs failed during collection because the `pytest` executable
could not import `tests.fixtures`; CI now uses `python -m pytest`, with the affected
16 tests successfully collected locally. The frontend job exceeded an existing
five-second interaction-test timeout under default concurrency; CI now uses two
workers, matching the passing local run. No timeout, scientific tolerance, seed or
golden reference was changed. Full CI has not been rerun while the known offline
bundle failure remains.

## Repository hygiene

- Expanded ignore rules for experiment workspaces, captures, model outputs,
  logs, database state, environment files, credentials, private keys and browser
  authentication state. Only the published PA CSVs and synthetic tutorial are
  excepted from the dataset capture rules.
- Removed 176 raw baseline captures from tracking and retained every local copy.
  Also removed the already tracked `.DS_Store` and IDE metadata from tracking.
- Verified 16 representative private/output paths are ignored and 12 representative
  source/example paths remain eligible for tracking. A second check covered all
  28 bundled CSVs, frozen checkpoints and the UI design-token file.
- Scanned 906 previously tracked files for common credential prefixes, private-key
  headers and populated Studio bootstrap links; no matches were found. This is a
  pattern scan, not proof that arbitrary sensitive content cannot exist.
- Existing commits remain in Git history. Ignore rules and index cleanup do not
  retroactively erase previously pushed data. No history rewrite was performed.

Windows/Linux native rendering and physical GPU/RF paths remain unverified here.
