# Transparent vector logo — 2026-09-12

The sidebar and About page now load true SVG geometry with outlined lettering. The old application PNG and white backing panels have been removed. Light and dark wordmarks share the same curves; compact navigation uses a separate square emblem. The original generated PNG remains in the image-generation output directory.

Environment: macOS 26.6.2 (25G83), Apple Silicon; repository `.venv`; local production frontend served by the real Studio API in an isolated temporary workspace. No user's experiment or dataset was changed during this check.

## Checks

- `npm --prefix frontend run build` — passed; existing Plotly chunk-size advisory only.
- `npm --prefix frontend run lint` — passed.
- `npm --prefix frontend test -- src/pages/AboutPage.test.tsx src/layout/AppShell.test.tsx --maxWorkers=2` — 14 tests passed across two files.
- XML inspection of all three SVG assets — only SVG metadata, groups, paths and circles; no raster payload, external resource, background rectangle, font text or script.
- Playwright Chromium at device pixel ratio 1 and WebKit at ratio 2, with Home and About at 1366×900 and 390×900 CSS pixels — all logos loaded, no horizontal page overflow, and logo containers had transparent backgrounds.
- The production About SVG was rendered into a 2400×672 transparent canvas. All four corners, all three circuit terminal holes and the first letter's inner counter had alpha zero. Chromium measured 16,756 partially transparent edge pixels; WebKit measured 17,538, confirming antialiased edges. See the retained diagnostic script and browser results.
- Visual review covered the large wordmark, dark sidebar, compact navigation and 24/32/48-pixel emblem previews. Ordinary and Retina-density browser rendering were checked; no claim is made about every physical monitor or operating system.

The wordmarks are 9,417 bytes each; the compact SVG is 897 bytes (19,731 bytes combined). The former raster application asset was 885,094 bytes. Vite inlines the compact mark as vector SVG; the two wordmarks remain SVG files. No runtime dependency was added. No compute core or scientific metric changed, so training was not rerun for this visual revision.

Evidence: [preview](studio-vector-logo-2026-09-12/preview.png), [Chromium About](studio-vector-logo-2026-09-12/about-chromium.png), [WebKit at Retina density](studio-vector-logo-2026-09-12/about-webkit-retina.png), [compact navigation](studio-vector-logo-2026-09-12/compact-navigation.png), [Chromium measurements](studio-vector-logo-2026-09-12/chromium.json), [WebKit measurements](studio-vector-logo-2026-09-12/webkit-results.json), and [inspection script](studio-vector-logo-2026-09-12/inspect.js). The script requires a bootstrapped local Studio browser session and writes screenshots to `output/playwright/`.

Design geometry, font provenance and licensing are recorded in [the vector identity notes](../design/opendpd-studio-vector-logo.md). The obsolete raster app asset and temporary generation/verification files were removed; the original image and relevant evidence were retained.
