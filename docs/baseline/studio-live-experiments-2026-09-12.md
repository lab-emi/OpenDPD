# Experiment live visualization and About — 2026-09-12

The Experiment workspace now shows batch geometry, live PA/DPD metrics, time-domain and spectral comparisons, and a collapsed terminal with independent step tabs. About includes the EMI Lab and TU Delft logos, the requested project leadership, and current public GitHub activity. Source changes and validation artifacts are in [the evidence directory](studio-live-experiments-2026-09-12/).

## Implementation

- The default signal preview runs after the first training batch, then only when **both 25 training batches and 2 seconds** have elapsed. The interval adapts to measured preview cost with a 5% overhead target. Initial import/model-copy warm-up is excluded from that adaptation. This is a target, not a measured universal overhead guarantee.
- Progress is sampled every 0.5 seconds, with first/last batch and stage boundaries retained. The browser coalesces SSE events into 500 ms updates. Plot snapshots contain bounded arrays and overwrite a single `live.json`; arrays are not sent in every progress event. Structural sharing avoids redraws of unchanged snapshots. Existing pan/zoom controls and view preservation remain shared.
- PA training displays NMSE, fixed-validation-segment predictions versus measured PA output, and the original per-epoch validation/test metrics. DPD training displays ACLR and NMSE for the DPD → PA-surrogate chain. The preview identifies its segment and metric profile and does not select checkpoints.
- PA/DPD tests show actual evaluation progress and the stored final test metrics and plots. They do not display an invented epoch history. The final cards respect the selected metric profile, including ACPR/IBE for the general spectral profile. Metric units come from the shared metric definitions/results; EVM remains dB under the legacy profile.
- Geometry distinguishes configured capacity from the current/last batch: sequences × complex I/Q samples per sequence @ sample rate. Training stride is shown explicitly. Evaluation padding, when present in the existing manifest boundaries, is identified separately. No split boundary is recomputed.
- Terminal is collapsed by default, with a neutral icon/bar when idle and a blue `Running` highlight while a run is active. It follows navigation among PA training/testing and DPD training/testing. Its own tabs and run selector only select logs. The latest log suffix is bounded to 256 KiB before selecting lines; the existing virtualized viewer supports follow/filter and byte cursors. Collapsed terminals do not poll logs.

The service adapter wraps a Project instance and its existing loaders; `models.py`, `project.py`, `backbones/`, `modules/`, `steps/`, and `utils/` are unchanged. Training still uses the original iterator, optimizer, metrics, validation passes and checkpoint selection. Preview inference uses a separate frozen copy with RNG isolation and a fixed dataset item, avoiding an extra shuffled-loader iterator. Test previews observe an already-computed forward output. Final results/plots come from the existing evaluation service.

All **18 native trainable backbones** are exposed through registry-derived recipes in both PA and DPD training: GRU, TRes-GRU, TRes-DeltaGRU, gradient GMP, LSTM, VDLSTM, DGRU, TCN, RVTDCNN, PGJANET, DVRJANET, DeltaGRU, DeltaJANET, QGRU, QGRU amp1, BOJANET, APNRRU and MCLDNN. Existing MP/GMP least-squares entries bring the GUI menu to **20 choices**. Compatible streaming executions remain test options for trained weights. `neuraltx` is not advertised: its referenced implementation is absent from this checkout. No duplicate backbone implementation was introduced.

About credits **Chang Gao — Project Leader** and **Yizhuo Wu — Leading Developer**, followed by the public contributor list. The two fixed public GitHub API requests are cached for five minutes, only run when About is requested, and send no workspace data or credentials. Network failure preserves the last successful in-memory snapshot with a stale notice and repository links. The initial list is capped at 100, with a link to all contributors. Public commit/contribution totals can also be cached by GitHub.

Logo provenance: [EMI Lab original SVG](https://www.tudemi.com/images/emi-logo.svg), [TU Delft original SVG](https://www.tudelft.nl/_assets/2f383d4a929ad3d42eff11e81cdd4068/img/logo.svg), discovered from the official university homepage. Both assets are served locally. The About page links to [EMI Lab](https://www.tudemi.com/), [TU Delft](https://www.tudelft.nl/en/), and [OpenDPD on GitHub](https://github.com/lab-emi/OpenDPD). UI text is included in all seven existing languages.

## Real computation and measured cadence

Environment: macOS 26.6.2 arm64, Python 3.13.12, PyTorch 2.14.0, Node 26.8.1. The separate QA workspace used the actual loopback API/server and the original built-in DPA_200MHz data. Runs were submitted through the GUI, used CPU, and succeeded with exit code 0:

| Task | Run | Evidence |
| --- | --- | --- |
| PA training, final implementation | `run-20260912-120526-ea97e7` | 120 epochs, 24.894 s, 11 training probes; intervals **2.019–2.077 s**, at least 239 training steps between previews |
| PA test | `run-20260912-115014-35f242` | Real checkpoint evaluation, NMSE/EVM, time/spectrum and complete terminal output |
| DPD training | `run-20260912-115431-efe610` | 120 epochs, 22 training probes; intervals **2.033–2.122 s**; ACLR/NMSE and surrogate signal chain |
| DPD test | `run-20260912-115618-c6ce8c` | Real DPD application and evaluation; final ACLR_AVG −41.5136 dBc, NMSE −32.6961 dB |

PA testing and DPD training reference the first real PA run, `run-20260912-114532-b11de6`, which also completed successfully. That initial observer implementation included warm-up in its cost estimate and delayed its second preview by 14.671 seconds. The final implementation corrects that scheduling issue. The two 120-epoch PA runs have **bitwise-equal checkpoint tensors and exactly equal official metric values**. A separate automated three-epoch ablation also compares the observer against no observer and requires identical checkpoint tensors and complete metric results.

The final PA run records training capacity **64 × 50**, stride 16, at **800 MSa/s**: 3,200 complex samples for a full training batch. Its evaluation capacity is 256 × 2,560, but its actual last batch has **3 × 2,560 = 7,680** complex samples. The final UI was checked against these recorded values.

During real DPD training, a 12.228-second browser observation counted **6 redraws each** for the time and spectrum plots and 17 for the main metric curve. Selecting the PA terminal tab kept the DPD run URL and heading unchanged. Navigating to DPD testing selected that terminal step automatically. Completed logs contain original trainer tables, stage/batch messages and final metrics. Final snapshot files measured approximately 381 KiB for PA and 703 KiB for DPD. These are local workflow/performance observations, not model-quality benchmarks or a frame-rate guarantee.

## Automated checks

Commands are run from the repository root unless noted. Logs are retained in the evidence directory.

| Command | Result |
| --- | --- |
| `npm --prefix frontend test -- --maxWorkers=2` | **122 passed**, 30 files, 26.29 s |
| `.venv/bin/python -m pytest tests/integration/test_live_experiment.py tests/unit/test_live_monitor.py tests/unit/test_about.py -q` | **24 passed**, 31.18 s; includes real training/preview/final evaluation for every native backbone, all four tasks, recipe coverage and numerical ablation |
| `.venv/bin/python -m pytest tests/unit/test_log_tail.py -q` | **1 passed**, 1.22 s; 50,000-line UTF-8 log suffix, byte cursor and unfinished final line |
| `.venv/bin/python -m pytest tests/test_backbones.py tests/unit/test_live_monitor.py tests/integration/test_live_experiment.py tests/integration/test_studio_api.py -q` | Earlier regression pass: **64 passed**, 52.19 s; before adding the broader model parameterization and final log suffix bound |
| `npm --prefix frontend run e2e -- --project=chromium-1366 --project=chromium-1920 --project=webkit-1366 --ignore-snapshots --workers=1` | **36 passed**, 9 opt-in live/performance cases skipped, 2.3 min; includes About/terminal accessibility, keyboard journeys and the existing 2 s gallery limit |
| `npm --prefix frontend run build`, `npm --prefix frontend run lint`, `npm --prefix frontend run types:check`, `git diff --check` | Passed |

The browser suite above uses a mock API for UI contracts; it is not used as real-computation evidence. The actual GUI runs and model integration tests provide that evidence. Pixel baselines were neither asserted nor updated. Separate real-page axe scans of About (1366), PA (1366/1920), DPD test (1920), and expanded Terminal (1366) found **zero WCAG 2.1 AA violations and zero horizontal overflow**. Representative screenshots were visually inspected.

Development failures are retained rather than hidden:

- New routes initially disagreed with the generated OpenAPI snapshot. The canonical exporter and TypeScript generator were run; the later API regression passed. Frozen scientific references were not edited.
- Live batch progress initially interfered with the established completed-epoch event contract. Live events now carry a separate scope; original event expectations remain intact.
- Running 30 Vitest workers concurrently with browser tests caused timeout failures. Final test suites were run separately with bounded workers. No timeout, numerical tolerance or gallery performance limit was raised.
- The added independent-terminal check initially started from Configuration while expecting the Overview URL. Its setup now returns to Overview before exercising the terminal; the original URL assertion remains unchanged.
- Removing epoch charts from test pages also removed an existing empty-history explanation. The explanation was restored in the implementation; its existing assertion was unchanged.
- Real layout inspection found an existing screen-reader-only chart description used MUI fractional `width: 1`. It now uses explicit `1px` dimensions, eliminating horizontal overflow while retaining accessibility.
- The final build retains the existing large optional Plotly chunk warning. The browser suite logged one non-failing loopback proxy request during test teardown.

## Scope, ablation and limits

The starting worktree was already dirty. The task changed 25 preexisting files and added 14 source/test/assets files; earlier user changes were preserved. The stored starting-file hash audit confirms no changes in the original compute modules or any protected metrics/splits/goldens/benchmark/protocol/guard path. No seed set, scientific expected value, tolerance or checkpoint selection metric was changed. No runtime dependency was added.

Unused observer paths were removed, event redraws were coalesced, test-only epoch charts were removed, and temporary QA files/processes are cleaned after evidence capture. Training remains shared between CLI/API/GUI through the same services. The new native GUI command was launched on the existing user workspace at loopback port 8765; its process and `/healthz` were verified. The native Python window could not be enumerated by the desktop UI inspection connector, so visual verification here is of the real browser UI and WebKit engine, not a separate native-window interaction claim.

GPU/MPS execution, Windows, physical RF hardware and physical touchpad feel are **not verified**. Each native backbone received a short CPU workflow check, not a convergence study. Preview segments over the 16,384-sample budget show an explicit preview-unavailable message; the original full evaluation remains available. Least-squares fitting and the original full-sequence DPD export expose stages rather than fabricated gradient-batch progress. Streaming variants retain their existing execution path and final results instead of presenting their offline bootstrap as a live streaming prediction.

No release, external data/model upload, telemetry, RF output action or CI permission change was performed.
