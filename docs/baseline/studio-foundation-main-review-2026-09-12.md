# Studio foundation integration review — 2026-09-12

This integration brings the existing `OpenDPD-Studio` foundation (`d3e7c5e`) onto
`main` (`1d28fb5`). It precedes the live experiment, native window, branding and
localization work in PR #20. The latter will target `main` after this foundation
is merged.

## Integration review

The only merge conflicts were in `.gitignore` and `README.md`. Both the Studio
instructions and the documentation site's snippet boundaries were preserved,
along with the tutorial notebook exception and MkDocs build exclusions. Main's
float-checkpoint quantization fix, API update and new tests remain present.
CI uses `python -m pytest` so `tests.fixtures` imports work in a clean checkout;
frontend unit execution is bounded to two workers. Permissions are unchanged.

There are 30 protected-path additions relative to main: the metric profiles and
comparison rules, split protocol, frozen metric/checkpoint references, benchmark
regression drafts, scientific protocol documents, CODEOWNERS and its guard.
Their content is identical to `d3e7c5e`; this integration does not edit them.
They require a separate scientific-review approval before merging.

## Scientific scope for maintainer review

- `legacy-opendpd-v1` remains the default and delegates to existing
  `utils.metrics`; segment aggregation and zero padding remain frozen by golden
  tests. `general-spectral-v1` is separately named and uses pooled sample powers,
  explicit valid ranges and documented Welch band integration.
- The OFDM profile is distinct and retains its pending external-validation
  status. Its existence is not a claim of standard conformance.
- `contiguous-v1` splits raw samples before framing, with 60/20/20 default ratios
  and a default 256-sample guard. The published built-in split conventions remain
  explicit; this protocol is not silently substituted into legacy datasets.
- Result comparability requires matching profiles, evidence types, datasets,
  preprocessing and split versions, reference gain, PA surrogate, operating
  point and execution semantics. Package imports reject a reused dataset ID with
  different raw content.
- Benchmark baselines and acceptance thresholds remain proposed. Hardware,
  independent waveform validation, measured multi-condition data, fixed-point
  rule approval and external leaderboard reviews retain their recorded human
  gates. Only the mock instrument adapter is enabled; no RF output is unlocked.

## Local evidence

Environment: macOS 26.6.2 / Apple Silicon, Python 3.13, CPU. No tolerance, seed,
checkpoint-selection metric or expected numerical value was changed.

```sh
python -m pytest tests/golden tests/unit/test_metrics_general.py \
  tests/unit/test_metrics_registry.py tests/unit/test_metrics_ofdm_evm.py \
  tests/unit/test_splits.py tests/unit/test_waveform_ofdm.py \
  tests/unit/test_fixed_point.py tests/unit/test_polynomial.py \
  tests/unit/test_streaming.py tests/test_quant_pretrained.py tests/test_api.py -q
```

Result: **81 passed in 8.05 seconds**. This covers the frozen checkpoint and
metric values, analytic signal cases, guards, streaming, fixed-point/polynomial
software references, public API compatibility and main's quantization regression.
The frontend production build passed. Full CPU regression and the Python version
matrix are recorded in the PR CI runs; these focused checks do not replace them.
Physical GPU/RF paths and Windows native behavior are not verified here.
