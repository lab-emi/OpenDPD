# Elastic plot recovery — 2026-09-12

The shared Plotly wrapper now fits a displaced view after 800 ms without input when no visible sample or connecting line segment intersects the view, or more than 85% of the view lies outside the visible data's bounding rectangle. Each chart's Plot controls popover has an on/off switch and 65%, 75%, 85%, and 95% thresholds. The settings apply to that chart and its enlarged view for the lifetime of the component; they are not workspace preferences.

Ordinary background between thin curves and sparse points is not counted as empty area. Hidden traces, non-finite pairs, line gaps, reversed axes, flat lines, and single-point histories are handled explicitly. Constant dimensions rely on actual visibility rather than a zero-area bounding rectangle. Band annotations are not signal data.

One scan runs after the gesture settles, not per wheel event or animation frame. Pointer holds and active pinch gestures suspend recovery. Fit uses the existing serialized Plotly redraw queue. Pending checks survive controller replacement during data/layout updates and inline/enlarged transitions; disposal cancels the timer and listeners. Native mouse pan/zoom is observed through [Plotly events](https://plotly.com/javascript/plotlyjs-events/) and fits use [Plotly.relayout](https://plotly.com/javascript/plotlyjs-function-reference/).

Real-browser verification found that Plotly could restore the inline plot's old manual autorange flags after a fit in its enlarged view. The wrapper now reconciles saved autorange flags as well as manual ranges. This behavior has both a regression test and a passing real-browser check.

## Verification

Environment: macOS arm64, repository Python virtual environment, local headless Chromium, Vite 8.2.2. The actual Studio server ran on loopback port 8891 with a disposable workspace. Its real API imported `DPA_200MHz` and computed dataset analysis over 38,400 samples, returning 12 measurement rows. Browser checks used those results, with no mocked API or substituted plot data.

- `npm --prefix frontend test -- --maxWorkers=2 src/components/plotRecovery.test.ts src/components/plotInteractions.test.ts src/components/PlotlyChart.test.tsx src/i18n/catalogues.test.ts` — **53 passed**.
- `npm --prefix frontend run build` — passed, with the existing Plotly chunk-size warning.
- `npm --prefix frontend run lint` and `git diff --check` — passed.
- Four actual plots retained small pans after the idle delay, recovered large displacements, and retained their displayed sample counts: frequency domain (5,120), time domain (1,024), constellation (6,100, WebGL), and PA response (3,840, WebGL). [Results](studio-plot-recovery-2026-09-12/four-chart-results.json), [CLI browser check](studio-plot-recovery-2026-09-12/four-chart-check.js).
- Native mouse dragging left 174 of 5,120 frequency-domain samples visible at the edge; the chart fitted after release. This exercises the threshold, not only a completely empty view. [Results](studio-plot-recovery-2026-09-12/native-pan-results.json).
- A roughly 90% displaced view remained at the 95% threshold and recovered at 75% after further input. Disabling protection retained an empty view. The enlarged chart shared that setting, recovered when re-enabled, and returned the fitted view to the inline chart. [Results](studio-plot-recovery-2026-09-12/settings-results.json), [CLI browser check](studio-plot-recovery-2026-09-12/settings-check.js).
- axe-core WCAG 2 A/AA and 2.1 AA scan with the new controls open: **zero violations**, 23 passing rules. [Result](studio-plot-recovery-2026-09-12/a11y.json).
- Geometry-only local Node timing, 20 warmups and 100 checks per size: 10,000 points p95 **0.154 ms**, 100,000 points p95 **0.480 ms** (maximum **0.772 ms**). This excludes Plotly drawing and is not a physical touchpad or frame-rate measurement. [Runtime and measurements](studio-plot-recovery-2026-09-12/geometry-performance.json).

The browser settings check initially used a non-exact dialog-name selector and matched the enlarged dialog as well as its help popover; using the exact help name fixed the harness. A subsequent real failure exposed the inline autorange restoration bug described above; the implementation was fixed and the check passed without relaxing an expectation or timeout.

Physical trackpad behavior and other browser/platform combinations are not verified here; synthetic WebKit-style gesture events are covered by unit tests. No scientific metric, data split, expected scientific result, backend, or dependency changed. Temporary browser/server/workspace and scratch files were removed after validation. [Changes relative to the starting working tree](studio-plot-recovery-2026-09-12/source-changes.diff).

## Previews

- [Displaced frequency plot](studio-plot-recovery-2026-09-12/plot-recovery-before.png)
- [Automatically fitted plot](studio-plot-recovery-2026-09-12/plot-recovery-after.png)
- [Per-chart recovery settings](studio-plot-recovery-2026-09-12/plot-recovery-settings.png)
