# Studio: nine languages and generated explanations

Date: 2026-09-12. Environment: macOS 26.6.2, Apple Silicon, repository Python 3.13 `.venv`, Node/Vite production build, real loopback Studio API and Chromium. Browser experiments used an isolated temporary workspace, not the user's workspace.

## Change

The menu order is English, Nederlands, 中文, Français, Deutsch, Italiano, 日本語, 한국어, Español. The preferred three come first; the remainder are alphabetized by English language name. All nine languages have 750 interface entries and 365 shared presentation entries, with matching keys and interpolation variables. Dutch and Italian also have MUI locale packs, flag assets, settings schema support, and all 23 native shell strings.

The shared catalogues under `opendpd/studio/locales` translate application-generated diagnostics, parameter help and validation, progress phases, plot legends/toolbars, built-in dataset captions, and report explanations. Downloads accept an allowlisted locale, defaulting to the workspace choice. HTML/Markdown reports and package reports use the selected language. Numeric values, hashes, configuration JSON, commands, user names, third-party logs and GitHub commit text retain their source values. Logs and commit messages are labelled as original text in the UI. No metric implementation, split protocol, reference number or checkpoint-selection rule changed.

Language imports obey the latest selection, settings writes are serialized, and the startup language query runs once. This prevents a slow earlier save or catalogue import from undoing the most recent choice. Selecting the currently displayed language also cancels an older pending choice.

## Verification

| Check | Result |
| --- | --- |
| `npm --prefix frontend run lint` | Passed |
| `npm --prefix frontend test -- --maxWorkers=2` | 174 tests passed, 35 files |
| Final phase-label checks: `npm --prefix frontend test -- src/i18n src/components/PlotlyChart.test.tsx src/components/MetricHistoryChart.test.tsx src/components/LiveRunDashboard.test.tsx --maxWorkers=2` | 54 tests passed, 7 files |
| `npm --prefix frontend run build` | Passed; final entry `index-qZp-08SR.js` |
| `.venv/bin/python -m pytest tests/unit/test_localization.py tests/unit/test_window.py tests/integration/test_studio_api.py::test_settings_default_roundtrip_validation_and_csrf tests/integration/test_packages.py -q` | 50 tests passed |
| `uv build --wheel --offline --out-dir /tmp/opendpd-i18n-audit/wheels` | Built a wheel; ZIP inspection confirmed all nine shared locale files are packaged |
| `git diff --check` | Passed |

The backend checks run real CPU PA and DPD training, package export/import/re-evaluation, all nine HTML/Markdown report locales through the real API, explicit-locale precedence, rejection of an unsupported locale, and localized package reports. They verify stored NMSE values, checkpoint hashes and reproduction commands and confirm that `result.json` is byte-for-byte unchanged after report generation. The existing frozen numerical tolerances were not changed.

Browser checks used the real application and computation:

- 81 page checks: nine languages across Home, Datasets, dataset inspection, Experiments, experiment setup, run detail, result detail, Settings and About at 1366 × 900. No JavaScript errors or document-level horizontal overflow.
- 27 mobile page checks: Home, dataset inspection and experiment setup in every language at 390 × 844. Each document remained 390 pixels wide.
- Every language: navigate the wizard, enter an invalid epoch count, check its localized validation, restore the value, open the reset confirmation and cancel. Saving the selected language survives navigation/reload.
- Real built-in MyCustomPA PA/DPD smoke runs; an additional 100-epoch CPU PA run completed successfully. The language was changed from Italian to Dutch while its status was `running`; the live batch geometry and Terminal UI changed language without changing the run. Final metrics included NMSE −48.03 dB. All nine languages were then checked against its completed live dashboard and preview histories.
- The live check found raw `complete` and `test_probe` labels; both were localized and regression checks added. Other findings fixed during the audit included the smoke-run warning, built-in dataset caption, ACLR direction labels, scaling descriptions, report prose and composed inspection tooltips.

Machine-readable viewport/route results: [checks.json](studio-nine-languages-2026-09-12/checks.json).

## Visual evidence

- [Dutch homepage](studio-nine-languages-2026-09-12/i18n-nl.png)
- [Italian signal inspection](studio-nine-languages-2026-09-12/i18n-it.png)
- [Dutch mobile homepage](studio-nine-languages-2026-09-12/i18n-mobile-nl.png)
- [Dutch live result and localized preview curves](studio-nine-languages-2026-09-12/i18n-live-nl.png)

The first Italian inspection screenshot predates the subsequent translation of the generated built-in dataset caption. The final implementation and wizard checks include that correction.

## Limits and cleanup

This is macOS/Chromium evidence. Windows/Linux native menu rendering and other physical monitors were not verified. Native quit prompts reread the workspace language; toolkit menus are initialized with the chosen language when a desktop window opens. Mathematical expressions, source identifiers, user-authored text and third-party process output are preserved, not machine-translated.

The build retains the existing large Plotly chunk warning. Python tests emit existing Starlette/httpx deprecation warnings. The initial `python -m pip wheel` attempt could not run because this uv-managed environment has no pip; the offline uv wheel build succeeded. Early added test-harness issues were corrected by excluding original configuration blocks from prose assertions and keeping the isolated settings query alive; no scientific expected values were changed.

No dependency was added. The auxiliary translation scripts, temporary workspace, QA server, browser session and scratch screenshots were removed after recording evidence. The user's idle Studio window was restarted to load the new frontend and settings/report support, retaining its existing English preference and saved data/results. The API health check passed, and macOS window metadata confirmed a visible OpenDPD Studio window at 1366 × 856; the computer-use app selector could not attach to the Python host, so no separate native pixel review is claimed. No publication or external upload was performed.
