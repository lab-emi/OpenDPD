# Studio signal inspection: metric audit and Sionna assessment

Date: 2026-09-11. Scope: the dataset inspection workbench, its shared service,
and the existing measurement implementations. This is an audit and a blocking
note for scientific work; it does not approve a new measurement protocol.

**2026-09-12 correction:** the first inspection implementation missed the
existing per-dataset receivers in `datasets/`. They are now connected for
descriptive constellation plots; see the [demodulation follow-up](studio-constellation-2026-09-12.md).
This corrects the visualization gap without approving a new EVM metric.

## Existing calculations and reuse decisions

| Implementation | What the number means | Decision for dataset inspection |
|---|---|---|
| `core/metrics/legacy_v1.py` → `utils/metrics.py`, NMSE | Mean of segment-level dB errors, including the historical final-segment padding. It compares signals at their supplied gain and phase. | Preserve frozen behavior. Do not compare raw PA output with raw input and label the result “nonlinearity.” |
| Same legacy profile, EVM | Repository-specific spectral error based on FFT subchannels and mean amplitude ratios. It is not RMS error of demodulated symbols. | Keep its existing descriptor in result evaluation; do not present it as symbol EVM in the dataset table. |
| Same legacy profile, ACLR | Adjacent-channel power relative to the strongest main subchannel, negative dBc. `ACLR_AVG` is the mean of the two dB values. | Preserve for historical reproducibility. It is not interchangeable with a ratio to total main-channel power. |
| `core/metrics/general_v1.py`, NMSE and IBE | Pooled sample-error power and in-band error power, respectively, against an explicit reference. Neither removes arbitrary gain, phase or memory automatically. | Continue using the existing evaluation service for model results. A raw dataset inspection has no approved linear reference for these error metrics. |
| Same general profile, ACPR left/right | Hann Welch PSD, density scaling, 50% overlap, no detrending. Main band is `[-B/2, B/2)`; adjacent bands are `[-3B/2, -B/2)` and `[B/2, 3B/2)`. Bin-center integration, adjacent/main, negative dBc. | Reuse `evaluate("general-spectral-v1", signal, None, spec)` separately for input and output. Missing metadata, insufficient samples or uncaptured adjacent bands remain unavailable. No standards pass/fail is inferred. |
| `core/metrics/ofdm_evm_v1.py` and `core/waveforms/ofdm.py` | Known-reference OFDM demodulation and pooled RMS symbol EVM; the profile also specifies 18 MHz measurement bands at ±20 MHz for ACLR. It explicitly remains `pending_cross_validation`. | Show unavailable/review-required states. A `64QAM` or `LTE` metadata label is insufficient to bind a reference waveform or recover its transmitted symbols. |
| `datasets/demodulator.py` and `datasets/<name>/demod.py` | Existing dataset-specific IFFT-frame or CP-OFDM symbol extraction, also used by `datasets/plot_utils.py`. Output visualization uses input-referenced, smoothed per-subcarrier equalization. | Reuse for known packaged captures with the correct frame coordinates; label equalization and retain raw I/Q as a separate view. This is not a standards EVM receiver. |
| `core/doctor.py`, PAPR/RMS/peak | Capture amplitude statistics. Output statistics exclude isolated spikes selected by the existing doctor; input statistics use the input samples. | Reuse the doctor evidence, with the filtering convention in the measurement tooltip. These are not calibrated dBm or PA output-power measurements. |
| Same doctor, delay and gain/phase | Correlation-based delay with sub-sample refinement. Complex LS gain is fitted to the lowest 40% of input amplitudes after integer delay alignment. | Reuse as estimates, clearly labeled capture gain/phase. This includes receiver/attenuator scale and is not a whole-window BLA. |
| Same doctor, `occupied_bandwidth` | Counts the strongest FFT bins needed to contain 99% of power, then multiplies their count by bin spacing. Those bins need not be contiguous. | Do not add this value to the professional measurement table as standard 99% OBW. Its existing diagnostic behavior remains unchanged pending a protocol decision. |

The general spectral implementation's analytic tests, the OFDM tests, registry
tests and frozen legacy golden test pass. This establishes consistency with the
repository's definitions, not independent instrument or standards conformance.
In particular, the analytic adjacent-tone inspection gives −46.0206 dBc as
expected, and multiplying a capture by a constant does not change ACPR.

## What changed without changing scientific definitions

`opendpd/services/dataset_analysis.py::analyze_dataset` reads the selected
version, uses the existing central `analysis_window` bounded by the doctor's
131,072-sample limit, and composes existing measurements and plot functions.
It does not change data, metadata, preprocessing, splits or saved diagnostics.
The GUI's authenticated GET endpoint and the CLI call this same service:

```sh
opendpd datasets analyze DATASET_ID --workspace /path/to/workspace --version raw-v1
```

Python callers can import `analyze_dataset` from
`opendpd.services.dataset_analysis`; its typed result is
`opendpd.schemas.analysis.DatasetAnalysis`.

Frequency measurements use the complete analysis window, while the time chart
shows 256 samples and the I/Q and AM/AM–AM/PM plots are bounded to 4,000 points.
All charts identify their capture window. No automatic gain or delay correction
is applied to the displayed AM/AM or AM/PM data. The I/Q cloud is explicitly
waveform samples, not a demodulated symbol constellation. The subsequent
constellation view has its own frame-aligned sample range and receiver metadata;
it uses complete contiguous samples before reducing the displayed symbol count.

A display bug was fixed in `core/plots.py`: rounding normalized FFT frequencies
to three decimals could collapse distinct bins (for example with `nperseg=2560`).
Nine decimals retain the bin ordering. This changes chart coordinates only;
metrics are calculated from the original arrays. The physical-frequency plot
now labels density as dB/Hz; normalized-frequency plots do not claim Hz units.

## Blocking note: BLA, symbol EVM and OBW

[AGENTS.md](../../AGENTS.md) requires changes under `opendpd/core/metrics/**`
and `docs/protocols/**` to be a separate PR carrying `science-review-approved`.
It also requires a blocking note when a protocol is ambiguous. No protected
path, frozen expectation, tolerance, seed or checkpoint-selection rule was
changed in this work.

The following decisions remain before implementing a BLA residual profile:

1. Define integer/fractional delay treatment, valid overlap and how linear
   memory is handled. A scalar gain fit leaves frequency-selective linear
   distortion in the residual; an FIR reference introduces filter length,
   regularization and boundary decisions.
2. Define the fitting and reporting windows. For aligned scalar LS on the same
   chosen window, `alpha = (xᴴy)/(xᴴx)` and `e = y − alpha*x`. A training-fitted
   reference scored on another window is a different protocol.
3. Name the denominator explicitly: `10 log10(||e||² / ||y||²)` and
   `10 log10(||e||² / ||alpha*x||²)` are different ratios. Both can be invariant
   to a common output scaling when the reference is fitted consistently; neither
   is raw-output/raw-input NMSE. Zero energy and numerical floors need defined
   statuses.
4. Review evidence for linear gain/phase, known delay, linear memory, noise,
   nonlinear distortion and output scaling before exposing a number. Residuals
   include noise and unmodelled memory; no universal “nonlinearity grade” follows.

For symbol EVM, complete the independent validation already required by the
existing waveform profile, including synchronization, rate conversion,
equalizer fitting and the known symbol grid. Only then expose its numerical EVM. The existing per-dataset descriptive
constellation can be shown independently; it does not require or establish
that EVM profile’s waveform binding. The current dataset table returns
`missing_reference` for unbound waveforms and `review_required` for the pending
profile; it never substitutes the legacy spectral EVM.

For OBW, choose a reviewed contiguous-band convention (including tail-power
allocation, estimator/window and edge interpolation). A separated two-tone
capture illustrates why counting strongest bins is not the same quantity.
Changing this definition in the ordinary UI task would conceal a scientific
semantics change, so the existing doctor is not silently rewritten.

## Sionna: useful optional validator, no direct metric substitution

The upstream source inspected is commit
`6498239a72267ee25edb600e5b826b0531971d7f`, whose package metadata identifies
version 2.1.0. It requires Python ≥3.11, PyTorch ≥2.9.1, NumPy ≥2.2.6,
SciPy ≥1.15.3 and `sionna-rt`; OpenDPD currently declares Python ≥3.10 and
PyTorch ≥2.4.0. An unconditional dependency would therefore raise the supported
environment floor and install an unrelated ray-tracing dependency.
[Upstream package metadata](https://github.com/NVlabs/sionna/blob/6498239a72267ee25edb600e5b826b0531971d7f/pyproject.toml).

| Candidate | Assessment |
|---|---|
| `sionna.phy.signal.empirical_psd` | Uses the mean squared magnitude of a normalized FFT. It does not implement our Hann/overlap/Welch density convention, and its endpoint-inclusive frequency grid also differs. It cannot replace the PSD estimator while retaining current numerical semantics. |
| `sionna.phy.signal.empirical_aclr` | Returns total out-of-band divided by in-band power as a linear ratio. It does not separately integrate the left/right adjacent measurement channels, and uses different boundary handling. Converting its result to dB would still not make it our ACPR or profile-specific ACLR. |

These distinctions were checked against the actual
[signal utility source](https://github.com/NVlabs/sionna/blob/6498239a72267ee25edb600e5b826b0531971d7f/src/sionna/phy/signal/utils.py),
not inferred from function names.

Sionna's [OFDM resource grids and modulator/demodulator](https://nvlabs.github.io/sionna/phy/api/ofdm/index.html)
and [mapping/constellation components](https://nvlabs.github.io/sionna/phy/api/mapping/index.html)
are promising for an independent known-waveform reference chain. The proposed
use is an optional, version-pinned validation environment with explicit
numerology, mapping, precision and synchronization settings. Compare equivalent
symbol grids and residual definitions before adapting any runtime call through
the shared service layer.

No Sionna package or unused adapter was added. This is a source/compatibility
assessment; Sionna numerical equivalence, installation on the support matrix,
GPU execution and RF instrument conformance are **not verified**.
