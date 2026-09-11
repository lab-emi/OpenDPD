# OpenDPD Studio — UX specification (S01)

Scope: single-user, local, browser workbench. English by default; every
string lives in `frontend/src/i18n/en.json`.

## 1. User journeys

Each journey lists its success, failure and exit paths so a page can be
judged without a chat explanation.

### J1 — Reproduce the built-in example

| Step | Page | Success | Failure path | Exit |
|---|---|---|---|---|
| 1 | Home | sees workspace path, "Example: DPA_200MHz" card labelled **built-in measured data** | no workspace → "Choose or create a workspace" | — |
| 2 | Datasets | example dataset registered by one click; Doctor report shows info/warnings | Doctor error (should not happen for built-ins) → item explains and links docs | back to Home |
| 3 | Experiments → New | recipe `pa-gru-smoke-v1` selected, advanced settings collapsed, "smoke, not a benchmark" banner | validation errors shown per field before submit | cancel returns to list |
| 4 | Run detail | live progress, metrics per epoch, log tail; refresh keeps the same run | worker fails → `failed` with error stage + hint; cancel → `cancel_requested` then `cancelled` | list |
| 5 | Results | evidence badge **pa_modeling**, metric profile shown, values with units and better-direction | metric N/A shown as "not applicable: <reason>", never 0 | export |

### J2 — Use my own data

| Step | Page | Success | Failure path | Exit |
|---|---|---|---|---|
| 1 | Datasets → Import | choose CSV/NumPy from an authorised import root, map columns, confirm units and sample rate | wrong columns → mapping dialog with detected headers; NaN/length mismatch → blocking error with counts | cancel discards nothing on disk |
| 2 | Doctor | report with evidence per item; delay/gain suggestions with confidence | missing metadata blocks evaluation with a clear list of what to fill | edit manifest |
| 3 | Preprocess | preview → confirm → new `processed/<version>`; raw hash unchanged | preprocessing fails → nothing written, error shown | keep raw |
| 4–5 | as J1 | PA then DPD run using `pa_reference` | incompatible PA surrogate → submit blocked with fix instructions | — |
| 6 | Results → Export | private package (with data) or share package (no raw data, paths stripped) | missing artifact → export refused with the missing list | — |

### J3 — Add a new method (developer)

Runs outside the GUI: implement a backbone, register it in the model
registry (S02), run `opendpd run --config` on the tiny fixture, and open the
GUI to compare. Success = the model appears in the GUI model list *without*
frontend changes. Failure = registry validation error naming the missing
capability field. Exit = the CLI.

## 2. Navigation and page hierarchy

```
Home        workspace card, recent runs, "what is running now"
Datasets    list → detail (manifest, Doctor report, preprocessing versions)
Experiments list → new (recipe + advanced) → run detail (progress, metrics, logs, artifacts)
Results     list → detail (evidence, metrics, charts) → compare (same-protocol only)
Settings    workspace, devices (detected vs tested), about, diagnostics bundle
```

- Advanced parameters are collapsed by default; the recipe explains purpose
  and limits (smoke/demo vs research).
- Run detail is addressable (`/runs/<id>`); refreshing never resubmits.
- Every list has loading, empty (with next action), error (with retry) and
  **disconnected** (SSE lost; "reconnecting… last update hh:mm") states.

- The top bar carries the language selector on every page: the flag and the
  native name of the current language; a choice applies at once and is stored
  in the workspace (`settings.json`). Seven languages: English, Français,
  Deutsch, Español, 中文, 日本語, 한국어. Server text stays English.

## 3. State labels

| Status | Label | Colour token | Icon (never colour alone) |
|---|---|---|---|
| queued | Queued | neutral | hourglass |
| running | Running | info | play |
| cancel_requested | Stopping… | warning | stop outline |
| cancelled | Cancelled | neutral | stop |
| succeeded | Succeeded | success | check |
| failed | Failed | error | error |
| interrupted | Interrupted | warning | bolt |

Evidence badges: **PA model** (`pa_modeling`), **DPD · surrogate**
(`dpd_surrogate`), **DPD · measured** (`dpd_measured`), **MOCK** (shown with
a stripe; export disabled).

## 4. Design tokens

Single source: `frontend/src/theme/tokens.json` (colours, spacing, radii,
typography, chart palette). MUI theme and Plotly layouts derive from it.
Contrast ≥ 4.5:1 for text; status colours are paired with icons and text.

## 5. Component catalogue (domain components, S05)

| Component | Purpose |
|---|---|
| `MetricCard` | one `MetricValue` with unit, better-direction arrow and status reason |
| `EvidenceBadge` | evidence type / mock marker |
| `SpectrumPlot` | PSD traces from server-decimated data, dB axis, ACLR bands |
| `IQPreview` | short time-domain window, input vs output |
| `DiagnosticItem` | severity, evidence numbers, suggestion |
| `RunTimeline` | status transitions and heartbeats |
| `ConfigDiff` | resolved-config diff between two runs |
| `LogViewer` | virtualised, searchable, paged |
| `LanguageMenu` | flag-and-name language switcher (top bar and Settings); applies at once, stored per workspace |

## 6. Accessibility and layout

Keyboard-only completion of all forms; visible focus; required fields and
errors announced; 1366×768 and 1920×1080 without horizontal scroll for the
main tasks; charts can be enlarged in a dialog.
