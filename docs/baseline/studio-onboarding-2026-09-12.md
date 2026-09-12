# Studio onboarding, task selection and reset

Verified on 2026-09-12, macOS arm64, CPU. This extends the
[dataset creation work](studio-datasets-2026-09-12.md).
The implementation is an uncommitted working tree; see
[environment.json](studio-onboarding-2026-09-12/environment.json) and
[source-sha256.json](studio-onboarding-2026-09-12/source-sha256.json).
Existing native-window and dataset changes in the working tree are separate
work; this task changes frontend behavior and its tests.

## User-visible behavior

Home has one **Get Started** action. It opens Datasets with a floating guide,
a blurred backdrop, and a **Skip tutorial** button. Users can choose their own
CSV or a packaged dataset. The guide accompanies the actual three-step CSV
form: upload and validate columns, set split and metadata, then review and
create. It opens the resulting dataset's signal inspection page, explains
readiness checks, and points to **Configure experiment**. Skipping during
creation removes the guide while preserving the current file and form.
Datasets also has **Show tutorial** to reopen it.

Experiments offers four task cards:

- **PA Model Training**: dataset, model architecture, starting settings,
  training hyperparameters, and model-specific parameters.
- **PA Model Testing**: successful PA training run, test dataset, device,
  metric profile, CPU threads, and compatible execution options.
- **DPD Model Training**: dataset, architecture, a trained PA reference,
  training hyperparameters, and model-specific parameters.
- **DPD Model Testing**: successful DPD training run, its dataset, and the PA
  model used for the test; execution and scoring options follow.

Training shows epochs, batch size, learning rate, frame length and stride,
optimizer, loss, seed, and registry-defined model parameters without requiring
an advanced accordion. Least-squares models do not show gradient-only controls.
Model presets and numerical defaults still come from the existing server
recipes and registry. Testing inherits the trained weights, architecture and
frame settings through the shared reference-binding service. Compatible
streaming options are exposed without changing checkpoint semantics.

JSON is absent from the normal setup view. **Advanced JSON** opens a separate
editable/importable dialog. Invalid JSON stays in that dialog; applying a
configuration invokes the same server validation used by the form. All seven
existing languages include the new controls and guidance.

The top bar offers **Reset page** and **Reset Studio**, each with a concise
confirmation. The CSV form also has its own reset button. Page reset restores
unsaved controls and filters; an experiment keeps its selected task but resets
its draft. Global reset clears transient setup state and reopens the dataset
guide. Both preserve saved datasets, runs, results and running jobs; the global
reset also preserves the language preference. They do not call deletion,
cancellation or settings-write endpoints. Reset is disabled while a mutation
is pending. Keyboard focus returns to the original control after a normal
reset dialog, and stays inside the new guide after global reset.

See [Chinese onboarding](studio-onboarding-2026-09-12/onboarding-guide-zh-1366.png),
[CSV review](studio-onboarding-2026-09-12/guided-csv-review-1366.png),
[four task entries and successful runs](studio-onboarding-2026-09-12/four-tasks-succeeded-1366.png),
[training settings](studio-onboarding-2026-09-12/pa-training-settings-zh-1920.png),
[test settings](studio-onboarding-2026-09-12/dpd-test-settings-1366.png), and
[global reset](studio-onboarding-2026-09-12/reset-studio-zh-1366.png).

## Real computation

A separate local API served the production frontend at `127.0.0.1:8880`, with
an isolated workspace at `/tmp/opendpd-onboarding-qa`. Every run below was
configured and submitted through the browser UI, with actual DPA_200MHz data.
Both training runs used the existing three-epoch CPU quick-trial defaults.

| Task | Run ID | Result |
|---|---|---|
| PA Model Training | `run-20260912-092350-d2dcc8` | succeeded, 3 epochs |
| PA Model Testing | `run-20260912-092900-5a4c4b` | succeeded, reused PA weights |
| DPD Model Training | `run-20260912-093024-0fd539` | succeeded, 3 epochs, trained PA reference |
| DPD Model Testing | `run-20260912-093152-ae00f1` | succeeded, reused DPD and PA weights |

All four records have exit code 0 and `is_mock: false`. The standalone PA test
matches the training run's test scores; the standalone DPD test matches the
DPD training run's test scores. PA weight SHA-256 begins `c2abd553a1c9`; DPD
weight SHA-256 begins `f7b3e62591db`. Full configurations, provenance, results,
weight hashes and worker logs are retained in the four task subdirectories.
The combined [real-runs.json](studio-onboarding-2026-09-12/real-runs.json)
records the evidence. These short runs validate the workflow, not benchmark
quality. DPD results are PA-surrogate evidence, not physical RF measurements.

## Guide, reset and browser evidence

The own-CSV guide imported the explicitly synthetic MyCustomPA tutorial CSV,
labeled **dummy dataset for tutorial purpose**. All 102,400 rows passed the
real whole-file validation. At 60/20/20 and the shared 256-sample boundary
guards, the review showed 61,132 / 20,377 / 20,379 samples. Creation produced
`onboarding-tutorial-csv` with `origin: synthetic` and the source's known
80 MHz sample rate and 2 MHz channel metadata. It then opened real signal
inspection. See its
[manifest](studio-onboarding-2026-09-12/onboarding-tutorial-csv-manifest.json).

Live browser checks confirmed that skipping preserved the uploaded CSV and
its validation report, while form reset removed them and returned to step 1.
Page reset cleared a search filter and restored all four successful rows.
Changing epochs to 7 and resetting restored the server default of 3.
Global reset reopened the guide with Chinese preserved and both datasets
present. All 87 saved dataset/run files had identical SHA-256 hashes before
and after resets, with no added or removed files; see the
[storage check](studio-onboarding-2026-09-12/reset-storage-check.json).

Real-browser axe scans found zero WCAG 2.1 A/AA violations on the guided CSV
review, reset confirmation, PA training settings and guide at 1366×768 and
1920×1080. The measured guide backdrop is `blur(6px)`, all scanned dialogs fit
the viewport, and no scanned page overflowed horizontally. Representative
screenshots were visually inspected. The
[browser checks](studio-onboarding-2026-09-12/browser-checks.json) also record
JSON visibility/error handling and the final focus-in-guide check.

## Commands and results

Run from the repository root unless noted:

| Command | Result |
|---|---|
| `npm --prefix frontend run build` | TypeScript and production build passed |
| `npm --prefix frontend run lint` | passed |
| `npm --prefix frontend run types:check` | generated API types match the contract |
| `npm --prefix frontend run test -- --reporter=default --reporter=json --outputFile=/tmp/opendpd-onboarding-frontend-final.json` | 100 tests, 25 files passed |
| `npm run e2e -- --project=chromium-1366 --project=chromium-1920 --ignore-snapshots` (in `frontend/`) | 24 passed, 6 opt-in live/performance checks skipped |
| `git diff --check` | passed |

The browser suite covers both dimensions, the four routes, the keyboard-only
dataset-to-training journey, and focus during page/global reset. Its API is
mocked; the separate four-run computation and CSV creation above supply real
API/computation evidence. Pixel-baseline matching was explicitly disabled;
the screenshots are manual layout evidence. Browser, build and unit-test logs
are stored alongside this report.

An initial native-server test launch accidentally resolved the virtualenv's
Python symlink to the system interpreter and could not import FastAPI. Using
the virtualenv executable directly fixed the QA setup. Initial browser checks
also identified an empty default select label, hidden numeric placeholders,
and reset focus restoration; these were corrected without changing defaults.
A targeted reset unit test exceeded its existing 5-second timeout while the
full browser suite was running concurrently. The subsequent isolated full
Vitest run passed all 100 tests; no timeout or expectation was relaxed.

## Scope and ablation

No dependencies, APIs, metrics, split definitions, scientific defaults or
protected paths changed in this task. No GPU, Windows or additional browser
engine claim is made. The four measured packaged datasets remain unchanged.
Obsolete Home example-registration copy and the direct recipe picker were
removed. The UI reuses existing dataset, validation, reference-binding and
run-submission services. QA-only server/browser processes and scratch files
were removed after retaining this evidence; the user's workspace is separate.
