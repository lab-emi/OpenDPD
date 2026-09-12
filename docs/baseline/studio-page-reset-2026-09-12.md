# Studio page reset

Verified 2026-09-12 on macOS arm64, CPU, with the production frontend and a
separate loopback API/workspace. Existing uncommitted work was preserved;
[source-changes.diff](studio-page-reset-2026-09-12/source-changes.diff) records
only this task's changes relative to the starting working tree.

## Behavior and scope

The existing reset remounted only the route outlet, kept selected records in
detail URLs, left the independent Terminal mounted, and did not reset scroll
position or announce completion. Its trigger was disabled during mutations.

Reset now opens an explicit warning with **Continue / Cancel**, with Cancel
focused initially. Cancel and Escape leave progress untouched. Continue clears
the current form, setup completion markers, selections, plot view state and
Terminal controls, scrolls to the top, and announces completion. Dataset detail
returns to the dataset catalogue; run detail returns to the first setup step
for that run's PA/DPD training/testing task; result detail/comparison returns
to the results list. New experiment resets preserve only the task type in the
URL, removing dataset, imported configuration, model references and hash.

A pending save/submission still allows opening the prompt, which explains the
delay; Continue is disabled until that operation finishes. The CSV form uses
the same warning and clears its file, validation report, metadata and steps.
The existing global reset continues to open the dataset guide with focus inside.
All seven UI languages include the warning, busy state, destination and success
messages.

Scope is current-page UI progress. Saved datasets, results, language choice and
submitted jobs remain intact, as stated in the prompt. No deletion or cancellation
API was added. The user was asked whether saved records were also intended;
without a further answer, implementation retained the existing storage boundary.

## Evidence

- `npm --prefix frontend test -- --maxWorkers=2`: **154 tests, 33 files passed**,
  31.26 seconds. Includes new confirmation, busy-state, route, scroll and Terminal
  regressions and the existing real form tests. Selectors in two existing tests
  were updated for the requested Continue label; scientific expectations and
  tolerances were untouched.
- Targeted reset/form/catalogue tests: **40 tests, 5 files passed**.
- `npm --prefix frontend run build`, `npm --prefix frontend run lint`, and
  `git diff --check`: passed. Build retains the existing Plotly chunk-size warning.
- Production Chromium against actual API: dataset pan survives Cancel; reset
  returns to the catalogue; all four completed runs return to their task's first
  setup step; Terminal collapses; PA edits, result selections/comparisons and
  experiment filters reset; Home/Settings/About show feedback and scroll to top.
- Real CSV upload/whole-file validation of the bundled tutorial file: Cancel
  preserved edited metadata; Continue cleared the file, validation and steps.
- Four real CPU jobs on the bundled measured DPA_200MHz dataset succeeded via
  `/api/v1/runs`: PA training/testing and DPD training/testing. Training used the
  existing three-epoch smoke recipes. Real signal analysis processed 38,400 samples.
  [Run records and results](studio-page-reset-2026-09-12/real-runs.json) are retained.
- SHA-256 comparison: **all 79 saved dataset/run files unchanged** after resets;
  [storage check](studio-page-reset-2026-09-12/storage-check.json).
- Chinese confirmation: axe WCAG 2 A/AA and 2.1 A/AA reported **0 violations,
  20 passing checks**. English and Chinese screenshots were visually inspected.

See [checks](studio-page-reset-2026-09-12/checks.json),
[browser replay](studio-page-reset-2026-09-12/browser-check.js),
[CSV replay](studio-page-reset-2026-09-12/csv-check.js),
[Chinese warning](studio-page-reset-2026-09-12/page-reset-confirm-zh.png),
[English warning](studio-page-reset-2026-09-12/page-reset-confirm-en.png) and
[reset CSV form](studio-page-reset-2026-09-12/page-reset-csv-after.png).
Browser scripts assume an authenticated QA session and the bundled dataset/runs.

## Corrections and limits

The first lint pass rejected a test helper that assigned a completion callback
during render; it now uses Testing Library rerender. Browser harness corrections
were needed because its VM does not expose `URL`, and the dataset guide's
accessible title includes “Guided setup”. An initial screenshot preceded the
opening transition; the final screenshot was captured after it settled. These
were harness issues, not relaxed product assertions.

No compute core, metric, split, protocol, protected path or dependency changed.
No GPU, RF hardware, Windows or native WebKit UI verification is claimed. The
browser evidence uses real API/computation, not mocks. QA-only processes and
temporary workspaces were removed after preserving evidence. The built frontend
is available to the normal Studio launcher.
