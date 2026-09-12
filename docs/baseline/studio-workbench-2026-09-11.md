# Studio signal workbench: implementation and verification

Work started 2026-09-11 (UTC); final checks crossed midnight in Amsterdam.
Base commit: `f43cd32d8385efc3bf44980e19689caa1ae66e64`, with uncommitted changes.
This is a local development verification, not a release or a benchmark result.

## Result

The navigation now highlights the active route with a filled background, a
leading marker and `aria-current`. Run pages select Experiments. Dataset views
and experiment steps use a visible active underline/background.

The dataset overview puts frequency, time, I/Q and AM/AM–AM/PM charts beside a
compact measurement table. Metadata, versions and doctor details are separate
tabs. Metadata and diagnostic completion checks reflect backend state. The
selected data version follows the user into experiment configuration.

Experiment setup has three steps: data/recipe, model/training, and review/run.
Green checks appear only after the corresponding configuration has been
validated and confirmed. Relevant edits clear completion; changing the data
version invalidates both completed configuration steps. Existing server
validation and recipe defaults remain authoritative.

Shared charts use horizontal legends above the axes. Enlarged I/Q plots retain
equal I/Q axis scales. Axis translations now update when the language changes,
without requiring a reload. The seven existing language catalogues include the
new workbench and workflow labels.

See the [metric audit and Sionna assessment](studio-signal-metric-audit-2026-09-11.md)
for measurement definitions and the explicit BLA/EVM/OBW blocking note.

## Real API and computation

A local server at `127.0.0.1:8876` served the built frontend and an isolated
workspace, `/tmp/opendpd-workbench-preview`. The registered built-in measured
`DPA_200MHz` capture contains 38,400 paired samples. The GUI rendered measurements
from the new analysis endpoint, which calls the same Python service exercised
through HTTP and CLI in the integration tests.

Using the actual wizard, a `pa-gru-smoke-v1` run was submitted on CPU and reached
`succeeded`, exit code 0, after all three epochs. Its result page displayed the
real generated spectrum, time and AM plots; all four result-chart legends were
verified at `y=1.04`, horizontal. Run ID:
`run-20260911-215124-b99989`. The worker interval was 3.77 seconds. This only
demonstrates the pipeline, not converged PA modeling quality.

The [run record](studio-workbench-2026-09-11/cpu-smoke.json) preserves the config
hash, dataset hash and software provenance. The
[inspection record](studio-workbench-2026-09-11/inspection-summary.json) records
the table's measurements and unavailable states.

## Visual and accessibility evidence

| Live dataset view | Document size | Workbench bottom | Axe WCAG 2.1 A/AA violations |
|---|---|---|---|
| English, 1366 × 768 | 1366 × 768 | 721.39 px | 0 |
| English, 1920 × 1080 | 1920 × 1080 | 925.39 px | 0 |
| Chinese, 1366 × 768, after live language switching | 1366 × 768 | 721.39 px | 0 |

These checks used the real API and capture. The complete overview and its
measurement table fit in the first viewport in both cases. Long metadata,
expanded settings, doctor reports, and smaller screens may scroll normally.
The chart bounds are for this fully specified dataset, not a promise that every
warning or unusually long dataset name fits the same viewport.

The live walkthrough also exercised AM/PM switching, I/Q enlargement, Escape to
close the dialog, and language changes. A focusable explanation icon initially
had MUI's default `aria-hidden`; it now has an accessible SVG title. The live
axe check subsequently reported zero violations. The existing neutral status
chip color was darkened after a browser contrast failure. No accessibility
checks were suppressed.

Changing only translated axis titles also exposed a Plotly range-cache problem:
the time plot could fall back to `[-1, 6]` despite containing samples 0–255.
The shared wrapper now advances `datarevision` on requested draws so extrema
are recalculated. The existing browser language journey verifies that changing
the title preserves the original sample range. The real capture was also
switched Chinese → English → Chinese and retained 0–255 throughout the final
check. This uses Plotly's documented
[revision mechanism](https://plotly.com/javascript/plotlyjs-function-reference/#plotlyreact)
without changing the underlying signal arrays.

Axe was evaluated through the browser's automation evaluation context using
the installed local bundle. Attempting an inline script element was correctly
blocked by the server's CSP; no CSP, permissions or networking policy changed.

- [1366 × 768 overview](studio-workbench-2026-09-11/dataset-final-1366.png)
- [1920 × 1080 overview with AM/PM](studio-workbench-2026-09-11/dataset-final-1920.png)
- [Chinese 1366 × 768 overview](studio-workbench-2026-09-11/dataset-final-zh-1366.png)
- [Validated review step with green checks](studio-workbench-2026-09-11/wizard-review.png)
- [Enlarged I/Q cloud](studio-workbench-2026-09-11/iq-enlarged.png)
- [1366 layout/axe record](studio-workbench-2026-09-11/layout-a11y-1366.json)
- [1920 layout/axe record](studio-workbench-2026-09-11/layout-a11y-1920.json)
- [Chinese layout/axe and axis-range record](studio-workbench-2026-09-11/layout-a11y-zh-1366.json)

## Test commands and outcomes

Environment: macOS 26.6.2, arm64; Python 3.13.12; PyTorch 2.14.0;
NumPy 2.5.3; SciPy 1.18.1; Node 26.8.1. Frontend checks use the locked repository
dependencies. Browser verification here is Chromium on macOS. Windows, Linux,
Firefox, WebKit, GPU/MPS and RF instruments were not verified by this task.

| Command | Outcome |
|---|---|
| `npm --prefix frontend run build` | Passed, including TypeScript checking and bundled production assets. |
| `npm --prefix frontend run lint` | Passed with no warnings. |
| `npm --prefix frontend run types:check` | Passed; generated TypeScript matches the exported OpenAPI contract. |
| `.venv/bin/python -m pytest tests/integration/test_dataset_analysis_api.py -q` | 5 passed: real HTTP/Python/CLI agreement, analytic adjacent tone, version scaling, missing metadata, bounded windows, NaN and short captures. |
| `.venv/bin/python -m pytest tests/integration/test_datasets_api.py tests/integration/test_studio_api.py -q` | 29 passed. |
| Plot/general/OFDM/registry unit tests and `tests/golden/test_legacy_metrics_golden.py` | 32 passed in the initial combined invocation below. |

The initial numerical invocation was:

```sh
.venv/bin/python -m pytest tests/integration/test_dataset_analysis_api.py tests/unit/test_plots.py tests/unit/test_metrics_general.py tests/unit/test_metrics_ofdm_evm.py tests/unit/test_metrics_registry.py tests/golden/test_legacy_metrics_golden.py -q
```

That invocation had five fixture setup errors because the new test fixture had
not created its imports directory. Creating the fixture directory resolved all
five errors on rerun. No scientific assertion, tolerance or expected number
changed. FastAPI/Starlette emitted existing dependency deprecation warnings.

A prior isolated frontend pass had 86 passing tests, and Chromium journeys at
both sizes had 22 passes. A later simultaneous Vitest/Playwright invocation
encountered test and element-wait timeouts plus a 2,326 ms chart render against
the unchanged 2,000 ms limit. The
[failure record](studio-workbench-2026-09-11/parallel-run-failures.json)
preserves those outcomes. Final isolated rerun results are recorded below.

| Final command (browser commands run in `frontend/`) | Outcome |
|---|---|
| `npm --prefix frontend test -- --maxWorkers=2` | 86 passed, zero failures. |
| `npx playwright test --project=chromium-1366 --project=chromium-1920 --workers=2 --ignore-snapshots` | 22 passed in 37.2 s, 6 skipped because optional live-test environment variables were unset. |
| `npx playwright test e2e/journey.spec.ts --project=chromium-1366 --project=chromium-1920 --workers=2 --grep 'language selector' --ignore-snapshots` | Both passed after using Plotly's public `layout` property in the range check. |

Running the suites separately with two workers resolved the timing failures,
consistent with contention during the concurrent run. The original timeouts
and `<2000 ms` chart threshold remain unchanged. Browser pixel-baseline
assertions were not run (`--ignore-snapshots`); the new layout was checked using
the real screenshots linked above. Mock browser journeys are UI regression
coverage; the actual CPU run and HTTP integration tests provide computation
evidence. Long-log/performance soak tests were not run.

The [compact test summary](studio-workbench-2026-09-11/frontend-tests-summary.json)
and [browser log](studio-workbench-2026-09-11/browser-journeys.log) preserve the
final results.

## Scope and ablation

The service composes existing core calculations; no metric is implemented in a
route, CLI handler or frontend. Only bounded chart payloads are sent to the
browser. No new package dependency, external-module adapter, telemetry,
instrument control, published artifact or protected-path change was added.
The server remains loopback-only by default. Screenshot/inspection scratch
files are removed after the reviewed evidence is copied here.
The owned preview browser/server were stopped and the temporary preview
workspace was removed after archiving its inspection and run records.

Existing uncommitted native-window/launcher work and release documentation were
left untouched. No commit, PR, push or release was made by this task.

## Follow-up: dataset-specific constellation

The raw I/Q panel documented above was superseded by the 2026-09-12
[constellation correction](studio-constellation-2026-09-12.md): known packaged
captures now default to their existing dataset-specific demodulated symbols,
with a separate raw-I/Q toggle. The evidence and source hashes in this original
report describe the earlier UI stage.
