# Acceptance thresholds (proposed, pending maintainer approval)

These numbers are copied from the development plan §8 as the *initial*
acceptance targets. They are **not** measured results. Changing a threshold
requires a performance report, a reason and human approval (protected path).

Approval status: **proposed** — recorded at S00, awaiting maintainer sign-off.

## Reference machine class

4+ physical CPU cores, 16 GB RAM, local SSD, 1920×1080. Every report records
CPU, OS, Python, browser and package versions. GPU numbers come from a fixed
test machine, never from cloud CI variance.

## Measurement rules

- Startup: ≥ 20 runs, cold and warm reported separately.
- Interaction / API latency: ≥ 100 samples, p50 and p95.
- Training overhead: ≥ 3 paired runs (same config/device/limits), median and spread.

## G2 must-pass targets

| Metric | Target | Boundary |
|---|---|---|
| Command → service ready | p95 ≤ 10 s | installed env; excludes installs and large workspace migrations |
| Service ready → first page interactive | p95 ≤ 2 s | static assets local, reference browser |
| Ordinary UI feedback | p95 ≤ 200 ms | buttons, tabs, expanders; not full FFT |
| Status / metadata API | p95 ≤ 250 ms | excludes large files and compute; measured during training too |
| Cached chart zoom / switch | p95 ≤ 300 ms | viewport / pixel-budget data only |
| Training status visibility | ≤ 2 s | at the agreed log/metric cadence |
| GUI overhead on training | median ≤ 5 % (paired) | same numerics, thread budget and metric cadence |
| Control-plane memory growth | ≤ 100 MB net in last 20 of 30 min | worker model memory recorded separately |
| Stress import extra memory | peak extra RSS ≤ 1 GB | OS file cache excluded |
| Cancel feedback | UI ≤ 1 s to `cancel_requested`; cooperative stop ≤ 30 s | then forced termination + cleanup |
| Logs / history | 50 000 lines and 1 000 runs scroll/search/page | never mounted fully in the DOM |
| Offline use | main journeys need no network | no external fonts/CDN/telemetry |

## Numerical tolerances (deterministic layer)

| Check | Tolerance | Basis |
|---|---|---|
| Legacy metric goldens (`tests/golden/legacy_metrics_v1.json`) | rel 1e-7 / abs 1e-7 | float64 NumPy/SciPy on CPU |
| Frozen checkpoint re-evaluation, CPU float32 | rel 1e-4 / abs 1e-5 | cross-CPU / torch-version float32 drift |
| Frozen checkpoint re-evaluation, CUDA/MPS | recorded per device | never assumed bit-identical across devices |
| GUI (API + worker subprocess) vs CLI (in-process), same config, CPU, `reproducibility: hard` | abs 1e-3 dB on every metric; identical resolved config hash and selected epoch | `tests/integration/test_entry_consistency.py` (S09); proposed, pending maintainer approval |
| `run_dpd` through the DPD's training surrogate vs the DPD run's own result (same weights, same reference gain), CPU | abs 1e-3 dB on every metric and on both no-DPD baselines | `test_apply_through_the_training_surrogate_reproduces_the_dpd_result` (S10); proposed, pending maintainer approval |
| Packaged checkpoint imported into a new workspace and re-evaluated under the stored profile, CPU | rel 1e-4 / abs 1e-5 on every metric (the frozen-checkpoint row above) | `test_full_package_round_trips_into_a_new_workspace_and_re_evaluates` (S11); proposed, pending maintainer approval. Says nothing about re-training (`docs/protocols/experiment-packages.md`) |

| Least-squares MP/GMP fit of a known polynomial (analytic) | coefficients rel 1e-9 / abs 1e-11; ILA of a linear PA gives the identity within abs 1e-9 | `tests/unit/test_polynomial.py` (S12) |
| Least-squares baseline re-evaluated from its stored coefficients | identical metrics (abs 1e-6): no seed, no epochs | `test_mp_fit_identifies_the_synthetic_pa_and_records_the_fit` (S12) |

## Statistical tolerances (re-training layer, benchmark-v1)

Re-training regressions are *statistical*: a pre-registered plan (fixed
recipe, budget, metric profile, device) with at least three seeds. The
regression band per entry and metric is the seed mean ± max(0.5 dB, 2 ×
sample std) measured on a named machine and recorded in a
`RegressionBaseline`; it blocks only once a maintainer approved it, in either
direction (`docs/protocols/benchmark-protocol.md`). No baseline is approved
yet: `benchmark/regression/` holds drafts for review.
