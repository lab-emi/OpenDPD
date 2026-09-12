# About leadership portraits — 2026-09-12

Added the two original WebP portraits linked by name on the [EMI Lab homepage](https://www.tudemi.com/), as requested by the project leader:

- Chang Gao: [official portrait](https://www.tudemi.com/images/thumbs/24f427b3eb9474.webp), saved unchanged as `frontend/src/assets/chang-gao.webp` (800 × 800, 45,504 bytes).
- Yizhuo Wu: [official portrait](https://www.tudemi.com/images/thumbs/b5334890e6bfd8.webp), saved unchanged as `frontend/src/assets/yizhuo-wu.webp` (764 × 764, 68,530 bytes).

About renders each portrait at 64 × 64 with the person's name as alternative text and retains initials as a load-failure fallback. Vite bundles both assets locally; displaying the portraits does not require a request to the lab website. No compute or scientific code changed.

Verification on macOS, using the local Studio service at `127.0.0.1:8765` and Chromium at 1366 × 900:

- `npm --prefix frontend run build` — passed (existing Plotly chunk-size warning).
- `npm --prefix frontend test -- --maxWorkers=2 src/pages/AboutPage.test.tsx` — 1 test passed.
- `git diff --check` — passed.
- Playwright opened the actual `/about` page: both images completed loading from local `/assets/` URLs, had the expected natural dimensions, and rendered at 64 × 64. No horizontal overflow. Visually checked both circular portraits and their name/role placement.
- No new dependencies, tests, or runtime abstractions; temporary browser session and scratch files removed.

[Verified page screenshot](studio-about-portraits-2026-09-12.png)
