# Studio system appearance and two-background SVG identity

Date: 2026-09-12. Environment: macOS 26.6.2, Apple Silicon, repository Python 3.13 environment, Chromium, native pywebview/Cocoa WKWebView, and Retina device pixel ratio 2. Browser and native QA used a separate temporary workspace and the real loopback API. The user's experiment workspace was not used for test computation.

## Result

Studio follows `prefers-color-scheme` on startup and while open. Light and dark semantic tokens drive MUI surfaces, controls, focus indicators, navigation, status/evidence badges, dataset diagnostics, configuration blocks, Terminal, and Plotly backgrounds, grids, labels, toolbars and trace colors. The welcome illustration keeps its intentional dark brand surface in both themes. Official EMI and TU Delft assets retain their colors on light brand plates.

The full Studio wordmark and compact navigation emblem each have transparent SVG versions for light and dark surfaces; the favicon follows system appearance too. The light waveform was darkened for clarity on white. Both full logos share identical geometry, with outlined text and no raster image, background rectangle, external font or filter. See the [vector identity specification](../design/opendpd-studio-vector-logo.md).

The React Query client now remains stable across appearance and language changes. Switching appearance retains the active route, form values, query cache, Terminal selection and expansion, and Plotly's manual viewport. Color changes do not restart training or alter numerical computation. No dependency, backend source, protected scientific file, benchmark tolerance or reference value changed in this task.

## Verification

| Command / check | Result |
| --- | --- |
| `npm --prefix frontend run lint` | Passed |
| `npm --prefix frontend test -- --maxWorkers=2` | 178 tests passed, 37 files |
| `npm --prefix frontend run build` | Passed; final entry `index-BtD517td.js` |
| `git diff --check` | Passed |
| SVG XML inspection | All four identity assets contain only SVG metadata, groups, paths and circles; no raster or external resource references |
| Semantic palette contrast regression | Both themes: text/status/evidence colors meet 4.5:1 on their tested surfaces; primary button text meets 4.5:1; input outlines and full-opacity trace colors meet 3:1 |

The initial palette checks caught two light-theme contrast issues (mock evidence text and success text on selected surfaces). Their colors were darkened; the contrast targets were retained. The build retains the existing large Plotly chunk warning.

Real browser checks used the Playwright CLI `studio-theme` session and `run-code --filename` scripts:

- 36 route/viewport checks: both themes on Home, About, Datasets, dataset inspection, Experiments, setup, PA run, DPD run, result and Settings at 1366 × 900; Home, About, inspection and setup also at 390 × 844 and 1920 × 1080. No JavaScript errors or document-level horizontal overflow. All inspected charts used the corresponding palette and retained their full supplied point arrays.
- Two additional comparison-page checks at 1470 pixels wide: table swatches match the overlaid run traces in both themes; the reference trace uses the theme's secondary text color.
- Twelve axe `color-contrast` scans across six representative pages in both themes: 700 node checks passed, no reported violations. Axe flagged 250 node instances as incomplete, primarily cases requiring visual assessment such as gradients and plot content; these are not reported as automated passes. Screenshots were also inspected.
- Real wizard state: enter an experiment name and 100 epochs, change appearance both ways, open the reset warning, and cancel. The inputs and their DOM identity remained unchanged. There were zero new session/settings/model/recipe requests caused by the appearance switches.
- Real shared-service CPU computation: `pa-gru-smoke-v1` and `dpd-gru-smoke-v1` on the built-in MyCustomPA dataset both succeeded. A further 100-epoch PA experiment was submitted through the GUI and succeeded. During its `running` state, both appearance changes retained the same spectrum element and manual frequency range, kept the Terminal expanded with process output, and displayed the running highlight. Selecting the DPD terminal tab did not navigate away from the active PA experiment.
- Chromium 2× captures checked both logo versions at a 480-CSS-pixel width.

Native verification used an auxiliary pywebview window on the isolated QA API, with `NSWindow.setAppearance_` switched between Aqua and Dark Aqua. This changed only that QA window's appearance, not the user's system preference. WKWebView reported the corresponding media query/theme, selected the right full SVG in navigation and About, and had no horizontal overflow at 1366 CSS pixels and device pixel ratio 2. Native snapshots were captured with WKWebView's snapshot API and visually reviewed.

Machine-readable results: [checks.json](studio-system-theme-2026-09-12/checks.json).

## Visual evidence

- Home: [light](studio-system-theme-2026-09-12/home-light.png), [dark](studio-system-theme-2026-09-12/home-dark.png), [mobile dark](studio-system-theme-2026-09-12/home-dark-390.png).
- About and full logos: [light](studio-system-theme-2026-09-12/about-light.png), [dark](studio-system-theme-2026-09-12/about-dark.png).
- Signal inspection: [light](studio-system-theme-2026-09-12/inspection-light.png), [dark](studio-system-theme-2026-09-12/inspection-dark.png).
- Running Terminal: [light](studio-system-theme-2026-09-12/terminal-running-light.png), [dark](studio-system-theme-2026-09-12/terminal-running-dark.png).
- Native Retina WKWebView: [light](studio-system-theme-2026-09-12/native-light.png), [dark](studio-system-theme-2026-09-12/native-dark.png).

## Limits and cleanup

Windows/Linux native rendering and other physical displays were not verified. Theme tests do not assert that translucent scientific markers or every exported Plotly image meet a text contrast threshold. No unsupported platform claim is made.

Unused color definitions and the separate comparison palette were removed or consolidated. Temporary scripts, workspace, test server, browser session and redundant screenshots were removed after preserving this evidence. The user's idle native Studio was reopened against the same workspace to load the final frontend: API health returned 200, and macOS window metadata confirmed a visible 1366 × 856 OpenDPD Studio window. No experiment was active when reopening it. No publication or external upload was performed.
