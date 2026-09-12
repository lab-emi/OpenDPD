# Studio home and navigation — 2026-09-12

Removed Robustness and Component Gallery from the sidebar. Settings now starts its own navigation group using its route rather than an item index. Existing direct routes remain available; this task removes their tabs, not the underlying scientific services or chart test fixtures.

Home now includes the bundled EMI Lab logo, TU Delft attribution, a short introduction, and a prominent Get Started action leading to `/datasets?guide=start`. A three-step strip explains I/Q data → PA modeling → DPD training and testing. Workspace information occupies a compact header, and recent runs remain below the introduction in a horizontally scrollable table container.

The decorative signal ribbon is a static 22.52 kB SVG (7.52 kB gzip), hidden from assistive technology. It is artwork, not a measurement. No animation loop, chart computation, network-dependent artwork, new dependency, or backend change was added. All seven language catalogues include the new copy; existing translations were preserved.

## Verification

Environment: macOS arm64, repository virtual environment, Vite 8.2.2, Chromium. A temporary workspace was served by the actual `opendpd gui --no-browser --workspace <temporary workspace> --port 8891` application. No mocked API was installed in the browser.

- `npm --prefix frontend run build` — passed; existing Plotly chunk-size warning remains.
- `npm --prefix frontend run lint` — passed.
- `npm --prefix frontend test -- --maxWorkers=2 src/pages/Onboarding.test.tsx src/i18n/catalogues.test.ts` — 21 tests passed. This exercises the existing Get Started guide and reset behavior, plus catalogue parity.
- `git diff --check` — passed.
- Actual sidebar contains only `/`, `/datasets`, `/experiments`, `/results`, `/settings`, `/about`.
- Chinese at 1024 × 768, 1366 × 768, and 1920 × 1080: no horizontal page overflow; logo loaded; Get Started fully inside the viewport. CTA lower edge was 436.53 px at 1024 and 453.5 px at the two larger widths.
- English, German, and French at both 1024 × 768 and 1366 × 768: no horizontal page overflow; localized CTA fully inside the viewport. Checks wait for the lazy language catalogue to finish loading.
- Clicking the Chinese Get Started action opened the actual dataset guide at `/datasets?guide=start`; the real workspace dataset listing was visible behind it. Skipping returned to `/datasets`.
- axe-core WCAG 2 A/AA + 2.1 AA scan of the actual home page: zero violations, 22 passing rules.
- Visually reviewed Chinese and English home pages and the narrower Chinese layout.

An initial headed browser session was interrupted by file chooser dialogs; the checks were rerun in a fresh headless session. An early locale sweep read the previous language before its lazy import completed; an explicit language readiness wait fixed the verification script, and the sweep passed on rerun. No application workaround or test tolerance change was needed.

The temporary browser, server, workspace, and scratch scripts were removed after verification. No protected scientific paths were edited.

## Previews

- [Chinese, 1366 × 768](studio-home-2026-09-12/home-zh-1366.png)
- [English, 1366 × 768](studio-home-2026-09-12/home-en-1366.png)
- [Chinese, 1024 × 768](studio-home-2026-09-12/home-zh-1024.png)
- [Chinese, 1920 × 1080](studio-home-2026-09-12/home-zh-1920.png)
- [Get Started opens the dataset guide](studio-home-2026-09-12/home-get-started-guide.png)
