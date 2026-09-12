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

## Initial blocker (resolved during the merge review)

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

## Follow-up review for merging into main

The offline bundle failure is now fixed in the implementation. The WebGL loader
uses Plotly 4.0.0's official custom strict build with only `scatter` and `scattergl`.
The unmodified, licensed runtime is vendored with its upstream commit, source
lockfile hash, output hash and rebuild instructions. The full strict npm package
is removed. Third-party generated code is excluded from application-source lint;
the offline assertion and every numerical/performance threshold remain unchanged.

Validation after the dependency change:

- Production build and application lint passed. The WebGL output chunk is
  1,612.53 kB (526.15 kB gzip).
- `python -m pytest tests/unit/test_offline_assets.py -q`: **1 passed**.
- `npm test -- --maxWorkers=2`: **178 passed** in 37 files.
- Real-server Chromium journey: **1 passed** in 12.4 seconds, including bootstrap,
  measured dataset analysis, dense WebGL scatter under the server CSP, a real CPU
  PA training run, result inspection and share-package export. A new assertion
  verifies that a WebGL-capable browser actually uses `scattergl` and draws its
  canvas; the SVG-only mocked gallery could not detect a broken strict GL build.
  Console/CSP errors were absent.

Full integration now follows the separate foundation PR #21, which preserves
main's quantization fix and documentation site. The foundation's 30 protected
paths require explicit maintainer scientific approval. The root workspace is
on `main`; fixes and verification use isolated worktrees. PR CI records the
complete CPU/Python/browser matrix for the final branch revisions.
