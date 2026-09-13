# Studio 2.2.4 validation

Validation date: 2026-09-13. Synthetic runs validate software behavior, not physical RF performance.

## Local checks

- Python unit and integration regression: **667 passed** with two numerical threads (355.46 s). The follow-up shadow-forward/RNG and figure replay checks passed **17 tests**.
- Frontend: **53 files, 223 tests passed**. Strict types, lint, generated API types and production build checked.
- Local Chromium browser journeys and accessibility checks: **24 passed**, including both desktop sizes and the gallery's existing two-second draw budget. Six optional live cases were skipped in this mocked suite; the separate actual-worker flow below completed all eight jobs.
- Actual Chromium UI: generation → input CSV/JSON → explicit Virtual PA simulation → exact paired CSV → dataset → PA training → DPD training → PA testing → DPD testing, at **1366×768 and 1920×1080**. All **eight CPU runs succeeded**; 32,768 paired samples and 6,452 test samples per dataset. No JavaScript errors or dataset publication submissions.
- PSD review at **1366×768, 1920×1080 and 390×844**: exactly one x trace, one u trace and four output references/baselines; all drawn PSD bins exactly match the saved artifact. Initial dB ranges match. Zoom and legend changes remain independent. Saved figures carry all three positions and export successfully.
- Figure integration tests verify byte-identical standalone PNG replay, exact exported numerical values and refusal of changed sources.
- Refreshed Home, generator, PA Library and PSD screenshots use the actual current GUI. Home and input/model screenshots use a separate empty workspace with synthetic input.

Machine-readable records: [local workflow](studio-2.2.4/local-workflow.json), [signal positions](studio-2.2.4/signal-chain.json), [screenshot sources](studio-2.2.4/screenshots.json).

## Scope and release checks

The unchanged metric/profile and split protocols retain their existing acceptance status. Windows/native platform limitations, independent acquisitions, hardware measurement and external EVM cross-validation remain as documented in the [support matrix](../releases/support-matrix.md).

PR CI, final release and public deployment are checked against the merged revision. Production evidence is recorded with the GitHub release after the API, GPU worker and hosted frontend have been updated together.
