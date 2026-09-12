# Studio integration into main: blocking review evidence

Review date: 2026-09-12. Intended target: `main` (`1d28fb5`). The integration is
split into [foundation PR #21](https://github.com/lab-emi/OpenDPD/pull/21) and
[UI PR #20](https://github.com/lab-emi/OpenDPD/pull/20). Neither is merged.
The UI diff against the foundation contains no protected paths.

## Foundation numerical regression

The [CPU CI run](https://github.com/lab-emi/OpenDPD/actions/runs/34705497410)
tested foundation commit `395728991ace3d6b897adfd4054879cf327802d3` on Linux.
Every Python job reported **1 failed, 547 passed, 17 skipped, 19 deselected**.
The failure is `tests/integration/test_cli_run.py::test_matches_legacy_cli_numerically`.
The unchanged assertion compares the stored Studio result with the legacy CLI's
best-checkpoint CSV, at an absolute tolerance of `1e-6` dB.

| Python | Studio NMSE (dB) | Legacy CSV NMSE (dB) | Absolute difference (dB) |
| --- | ---: | ---: | ---: |
| 3.10.21 | -21.38386835954562 | -21.383871 | 0.00000264045438 |
| 3.11.16 | -21.38386835954562 | -21.383871 | 0.00000264045438 |
| 3.12.14 | -21.38386835954562 | -21.383871 | 0.00000264045438 |
| 3.13.15 | -21.38386972277282 | -21.383871 | 0.00000127722718 |

The foundation's frontend job, critical lint, distribution build and strict
documentation build passed. Its protected-path guard remains blocked because
maintainer scientific approval has not been recorded.

### Read-only diagnosis

`modules.train_funcs.net_eval` returns float32 prediction/reference arrays.
The original `calculate_metrics` passes those directly to `utils.metrics`.
In contrast, `opendpd/core/metrics/legacy_v1.py:64` converts the arrays to float64
before calling the same function; its padding buffer also uses float64.

Re-scoring the existing local PA checkpoint from the real-server journey, without
training again or editing any metric, produced the following on macOS / Apple
Silicon / Python 3.13 / NumPy 2.5.3 / PyTorch 2.14.0, on CPU:

- Prediction and reference shape: `(3, 2560, 2)`, both float32.
- Direct original `utils.metrics.NMSE(prediction, reference)`: `-21.38387107849121`.
- Studio legacy profile: `-21.38387014248225`.
- Original function on those same arrays converted to float64:
  `-21.38387014248225`, exactly the Studio result.
- Absolute local difference: `0.0000009360089592291843` dB. This local difference
  is below the existing tolerance; it does not invalidate the Linux failures.

This demonstrates a precision difference introduced by the wrapper. Establishing
whether it fully explains the Linux failure requires checking identical Linux
predictions/checkpoints as part of the correction. A dtype-preserving correction
would touch a protected metric file and must receive human scientific review.
No tolerance, seed, checkpoint-selection metric, golden reference, expected value
or metric implementation was changed during this diagnosis.

## UI regression evidence

The [UI CI run](https://github.com/lab-emi/OpenDPD/actions/runs/34705657886) tested
`ebdf0624632e86b93d15f371b2f41d57ec84e6a5`. The frontend job passed types, lint,
178 unit/interaction tests and the production build. Its browser matrix reported
**45 passed, 12 skipped, 3 failed**:

- Firefox's keyboard driver pressed Tab even when Continue already had focus,
  eventually moving focus into browser chrome. Commit `bb9e491` corrects the
  driver to check existing focus before moving. The unchanged keyboard assertions
  then passed in all four local browser projects: **8 passed in 13.3 seconds**.
- WebKit's dataset guide accessibility check reported serious `color-contrast`
  violations on the overline, another guide text element and the dialog action.
  This remote failure is unresolved; the passing local accessibility checks do
  not replace it.
- WebKit's component-gallery chart render measured **2,209 ms** against the
  unchanged **less than 2,000 ms** budget. This remote performance failure is
  unresolved. No budget, retry policy or failing assertion was relaxed.

The complete local browser run before the keyboard correction had **47 passed,
12 skipped, 1 failed**. The skipped journeys require an explicit live server.
The separate real-server Chromium journey passed with measured data inspection,
actual dense WebGL rendering under the server CSP, CPU PA training, result
inspection and package export. Mock browser journeys are not computation evidence.

The separate queued full CI run for `bb9e491` was cancelled after the foundation
blocker was established; it has no acceptance result. A complete corrected-head
CI run is still required before merging. Windows native and physical GPU/RF
behavior remain unverified.

## Completed review fixes

- Replace the unused full strict Plotly runtime with the official, unmodified
  scatter/scattergl strict custom build. Source/output hashes, reproducible build
  instructions and the MIT license are checked in. The existing offline-assets
  assertion passes; no CSP exception was added.
- Preserve main's quantization/API fixes and connect Studio documentation to the
  existing MkDocs site. Both strict documentation builds pass. Raw validation
  captures remain outside the published site and Git tracking.
- Stop printing the weekly real-server launcher log, which contains a bootstrap
  URL. Create its temporary files with a private umask, record the server PID and
  stop the server in an always-run cleanup step. Use module-based pytest there
  as in ordinary CI. YAML and all seven shell blocks pass syntax checks; workflow
  permissions remain `contents: read`. The weekly workload was not dispatched.

## Merge gate

AGENTS.md requires: “If a test disagrees with the code, stop and report; do not
\"fix\" the expectation.” The numerical failure is therefore reported for a human
decision before modifying the protected legacy profile. PR #21 remains draft
without `science-review-approved`, and PR #20 remains draft pending that foundation
and the outstanding WebKit checks. Local `main` was only fast-forwarded to the
existing remote main; no Studio integration has been pushed to main.
