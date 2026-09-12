# Studio plot interaction evidence — 2026-09-12

Implemented shared pan/zoom controls for inline and enlarged plots. The changes are frontend only: 21 changed/new source files, with 658 preexisting files unchanged from the start of this request. Protected-path diff is empty. Numerical services, data, scientific expectations, tolerances and training defaults were not changed.

## Controls

| Input | Action |
| --- | --- |
| Two-finger scroll / wheel | Pan in both directions |
| Pinch / Ctrl or Cmd + wheel | Zoom around the pointer |
| Mouse drag | Pan; select the zoom tool for box zoom and unmodified wheel zoom |
| Toolbar | Zoom in, zoom out, autoscale and reset |
| Focused plot: arrows, + / − | Pan and zoom; Shift increases arrow movement |
| Home / 0, double-click | Fit the data |
| P / Z | Select pan / zoom |
| Alt + wheel, or wheel outside the axes | Scroll the page |

The same controller handles both sizes, normalizes pixel/line/page wheel deltas, supports WebKit gesture events, and permits only one draw plus one pending target. Source updates, cleanup and resize operations wait for active gesture draws. Camera state survives opening/closing the enlarged view and language changes. Changing datasets, coordinate modes, runs or metrics resets the appropriate view. Enlarged charts retain the IQ and AM/PM selectors. Control help is available in all seven UI languages.

Dense marker plots use WebGL without dropping points. Ordinary curves load the small Plotly basic bundle; dense scatter loads the strict bundle only when WebGL is available. Both are pinned to 4.0.0. The strict bundle supports WebGL under the existing same-origin CSP. Context loss or initialization failure falls back to SVG. The additional lazy chunk is 4.39 MB minified / 1.34 MB gzip; the basic chunk is approximately 1.14 MB. No CSP or test size/time limit was relaxed. See [Plotly bundle documentation](https://github.com/plotly/plotly.js/blob/main/dist/README.md), [wheel events](https://developer.mozilla.org/en-US/docs/Web/API/Element/wheel_event), and [Plotly relayout](https://plotly.com/javascript/plotlyjs-function-reference/).

## Validation

Full records, probes, screenshots and source hashes are in [the evidence directory](studio-plot-interactions-2026-09-12/).

- `npm test -- --reporter=default --reporter=json`: **118 passed**, 27 files, 21.56 seconds. This includes input normalization, cursor anchoring, bounded queues, draw/cleanup serialization, camera persistence, and WebGL fallback.
- `npm run build`, `npm run lint`, `npm run types:check`, `git diff --check`: passed. Vite retains its warning about the large optional strict bundle.
- `npx playwright test --project=chromium-1366 --project=chromium-1920 --project=webkit-1366 --ignore-snapshots --workers=4`: **36 passed**, 9 opt-in live/performance cases skipped, 50.8 seconds. Existing two-second gallery acceptance passed. Pixel baselines were not asserted or updated.
- Real-server dataset chart probes: **28 checks each** in Chromium 152, WebKit 26.5 and Firefox 155. Frequency, time, constellation and PA-response plots pass inline/enlarged pan, cursor zoom, view transfer and keyboard fit.
- Mouse/keyboard/gesture/resize probes: 24 checks each in Chromium and WebKit. Additional native WebGL mouse pan and box zoom checks passed at both sizes in both engines.
- Live axe scans of plot help and the enlarged plot: **zero WCAG 2.1 AA violations**. Representative Chinese UI screenshots at 1366×768 and 1920×1080 were visually inspected.
- Actual WebGL context loss on the final implementation recovered all 6,100 constellation points and exactly the saved axis ranges. Before/after data SHA-256: `cbb3ce7a9da3c617615070922524fbcd82bf61f60cca2e85da4009578c296ce4`.

Real computation used a separate loopback server and temporary workspace. Built-in DPA_200MHz was registered through the GUI and analyzed by the existing service. Real CPU training `run-20260912-102757-83f03e` used the server's three-epoch PA GRU trial recipe and succeeded with exit code 0, `is_mock=false`. Its five history charts passed 35 interaction checks and four result charts passed 28. Configurations, worker log, result, provenance and artifact hashes are preserved. This short run is workflow evidence, not a model-quality benchmark.

## Measured rendering improvement

The probe sends 240 fractional wheel events in 60 animation-frame batches and measures Plotly draw promises. It verifies accumulated displacement and maximum concurrency of one. These are local automated measurements, not physical touchpad measurements or a guarantee of 60 fps.

| Engine / 6,100-point constellation | SVG median, inline / enlarged | Final WebGL median, inline / enlarged |
| --- | --- | --- |
| Chromium | 63.6 / 65.0 ms | 5.4 / 6.0 ms |
| WebKit | 68 / 68 ms | 11 / 12 ms |

Each final probe coalesced 240 events into 60 draws. WebKit final p95 was 21 / 18 ms, so occasional slower frames remain. The final probes use Chinese labels; these comparisons are representative local runs, not a controlled benchmark. An animation API experiment was slower and was not included in production.

## Blocking notes and limits

Physical Mac/Windows touchpad feel and Windows hardware are **not verified**. macOS browser engine tests exercise wheel/pinch event paths; they do not substitute for a human using the hardware. SVG fallback preserves functionality but does not promise WebGL-level speed.

The broader Firefox keyboard journey at `frontend/e2e/a11y.spec.ts:91` still fails while seeking the second Continue button within 80 Tabs. The inspected trace repeatedly reports focus on Cancel. Its cause is not established; the experiment form source is unchanged from this request's starting state. No assertion or tab limit was changed. Full Firefox application keyboard acceptance is therefore **blocked**, despite its 28 real chart checks passing. The initial Firefox gallery result was 2,114 ms against the unchanged 2,000 ms threshold; separating bundle loading fixed that performance failure and its targeted rerun passed. Both failures and rerun outcomes are retained.

No release, upload, RF action, GPU training job or CI permission change was performed. A fresh native preview window was opened on the existing user server/workspace; the original server was preserved. Temporary QA processes and files are removed after evidence collection.
