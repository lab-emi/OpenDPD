# Studio branding, custom dataset pause and responsive layout — 2026-09-12

Environment: macOS 26.6.2 (25G83), Apple Silicon, Python 3.13.12, PyTorch 2.14.0, pywebview 6.2.1/Cocoa. The repository was already dirty. This change leaves the legacy training implementation and protected scientific paths untouched.

The generated OpenDPD Studio identity follows the EMI website's navy, blue and cyan palette. An open circular O encloses a radio waveform ending in three circuit terminals. The full wordmark appears in the sidebar and About; below 900 CSS pixels the sidebar shows the emblem. EMI/TU Delft identities and the existing leader portraits remain alongside it. The [exact prompts and source references](../design/opendpd-studio-logo-prompt.md) include the rejected checkerboard draft and the white-background correction. Generation used the built-in image tool, whose interface does not expose a model selector or a verifiable model version; **GPT Image 2.5 cannot be verified**. The selected original is bundled unchanged, 2172×724 pixels, 885,094 bytes.

Custom CSV creation, advanced import, onboarding CSV selection, experiment package import and measured-capture import are disabled and grey, with “Coming soon” / “即将上线”. Their implementation remains. The default server capability is false, and a shared ASGI boundary returns 403 for uploads, CSV preview/creation, source inspection/import, package imports and import-root browsing **before reading the body or parsing multipart data**. Existing import tests explicitly enable the retained application-factory capability; there is no user-facing switch. Built-in data, preprocessing, training and results remain available. This is a local Studio restriction; no GitHub Pages deployment or public training backend was created or certified.

The layout now uses a compact toolbar and sidebar on narrow windows, wrapped experiment filters and pagination, locally scrollable keyboard-accessible tables, one-column cards/metadata where needed, and a maximum content width of 1800 CSS pixels. Plot layouts continue to resize through the existing shared ResizeObserver. Native startup uses logical display dimensions, with a default size fitted to the screen and a 640×480 minimum when the screen supports it. No whole-page CSS zoom or doubled Retina scale is introduced.

| Check / command | Result |
| --- | --- |
| `npm --prefix frontend test -- --maxWorkers=2` | 156 tests in 34 files passed; 30.85 seconds on the final functional changes. |
| Targeted frontend reruns after subsequent layout corrections | Result/terminal/log viewer: 11 passed; experiment setup/list/run detail: 27 passed. |
| `npm --prefix frontend run lint` | Passed. |
| `npm --prefix frontend run build` | Passed. Existing lazy Plotly chunk size warning remains. Final entry asset: `index-DERLwA4U.js`. |
| `npm --prefix frontend run types:check` | Passed; generated OpenAPI client matches the contract. |
| `.venv/bin/pytest tests/integration/test_custom_dataset_gate.py tests/integration/test_datasets_api.py tests/integration/test_dataset_csv_creation.py -q` | 30 passed. Includes body-not-read assertion, nine closed entry checks, retained import paths and real built-in signal analysis. |
| `.venv/bin/pytest tests/unit/test_window.py -q` | 19 passed, including small, Retina and 4K logical display geometry cases. |
| `.venv/bin/pytest tests/integration/test_studio_api.py tests/integration/test_hardening.py -q` | 40 passed, one existing contract conflict described below. |
| `git diff --check`; Python compile check of changed server/window modules | Passed. |

The remaining hardening failure is `test_the_service_and_the_page_never_call_out`: it rejects `urlopen` in the pre-existing `opendpd/services/about.py`. That service implements the user's earlier request for live GitHub contributions. This task did not edit that service, remove the live contributor feature, or relax the test expectation. The expanded backend suite is therefore **not fully green**.

Real acceptance used an isolated workspace and the real loopback API on port 8765. Built-in `DPA_200MHz` analysis contained 38,400 samples at 800 MS/s. CPU GRU training completed three epochs as `run-20260912-134217-fe8a79`, with real test results and live/terminal records. Test NMSE was −21.38387014248225 dB. This smoke run is evidence of an operational path, not a benchmark. The user's workspace and experiment records were not used for these checks.

Browser evidence is retained in [the evidence directory](studio-branding-access-responsive-2026-09-12/):

- Chromium: ten real pages × seven viewports (320×700, 390×844, 640×480, 960×600, 1366×768, 1920×1080, 3840×2160), DPR 1. All 70 settled states fit the page width. Across 105 chart measurements the largest difference between Plotly and its container was 0.344 CSS pixels.
- WebKit: the same ten pages at 390×844, 640×480, 1366×768 and 1920×1080, DPR 2. All 40 states fit. The [audit script](studio-branding-access-responsive-2026-09-12/responsive-audit.js) is run through the Playwright CLI `run-code --filename=… --raw` against an authenticated local Studio with the built-in dataset and completed run.
- Expanded metric definitions and Terminal: eight additional states at widths 320/390/640/1366, no page overflow after plot resizing settled. Wide scientific tables and logs retain their own horizontal scrolling.
- Chinese upload labels and an inert CSV guide choice were verified in the browser. German home/data/experiment/settings pages fit widths 320/640/960/1366. The Get Started link still opens the built-in-data guide.
- axe WCAG 2 A/AA and 2.1 A/AA checks: no violations on Home, Datasets, Experiments and About. This is targeted automated coverage, not an accessibility certification.

Native evidence used the production `run_window` function and pywebview's resize API. A temporary observer captured the app's own WKWebView through Cocoa `takeSnapshotWithConfiguration`, without enabling a JS bridge or changing CSP. The current display reports 1470×956 logical pixels at 2×. The initial window measured 1366×856 (content 1366×828); shrinking it to 640×480 produced content 640×452 with the compact navigation and reachable Get Started action. Native About and real dataset plots were also visually inspected. Raw native measurements retain an intermediate About capture whose actual window was 1470×872; its filename is not evidence of a 640-pixel window. Windows/Linux and moving between multiple physical monitors were not verified here. Simulated 4K browser viewports are not a claim of a physical 4K monitor test.

Ablation: no new runtime dependency, alternate compute core, placeholder metric or global scaling layer was added. Test fixtures retain import functionality through one explicit capability. Scratch browser sessions, temporary bootstrap credentials, the isolated compute workspace and excess screenshots are removed after selected evidence is retained. The generated source image and exact prompts remain available for future brand revisions.
