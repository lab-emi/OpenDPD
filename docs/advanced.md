# Advanced workflows

Start with a successful [PA → DPD experiment](tutorials/gui-quickstart.md). These guides add a specific research or deployment workflow; they are not prerequisites for a first run.

| Goal | Guide | What it covers |
| --- | --- | --- |
| Fit classical MP/GMP baselines | [Classical baselines](training.md#classical-baselines) | Least squares, ILA and fit stability. |
| Add a backbone | [Adding a model](tutorials/adding-a-model.md) | Register a model once for CLI, API and GUI. |
| Evaluate a known waveform | [Waveform evaluation](tutorials/waveform-evaluation.md) | Reference binding, data-aided metrics and their limitations. |
| Validate on a physical PA | [Measured DPD](tutorials/measured-dpd.md) | Capture provenance, alignment, with/without-DPD comparisons. |
| Compare adaptation strategies | [Adaptation experiments](tutorials/adaptation-benchmark.md) | Zero update, few-shot and full retraining, with costs and failures. |
| Process consecutive chunks | [Streaming](tutorials/streaming.md) | State, look-ahead, latency and chunk consistency. |
| Export a fixed-point model | [Deployment export](tutorials/deployment-export.md) | Golden vectors and a verified C99 reference package. |
| Share benchmark evidence | [Leaderboard submission](tutorials/leaderboard-submission.md) | Packages, recomputation, review and versioned boards. |

For comparable results, read the [benchmark protocol](protocols/benchmark-protocol.md) and [metric profiles](protocols/metric-profiles.md). Current verification evidence is recorded in the [support matrix](releases/support-matrix.md), [performance report](releases/performance-report.md) and [hardening report](releases/hardening-report.md).
