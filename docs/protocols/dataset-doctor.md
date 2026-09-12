# Dataset Doctor `dataset-doctor-v1` and split protocol `contiguous-v1`

Both live under `opendpd/core/` (`doctor.py`, `splits.py`); `splits.py` is a
protected path. Changing a detector threshold, a suggestion rule or the
split arithmetic is a protocol change: bump the version string and keep the
old behaviour reachable.

## What the doctor reports

| Code | Severity | Evidence | Suggestion |
|---|---|---|---|
| `length_mismatch` | error, blocks | `n_input`, `n_output` | fix mapping / trim |
| `too_few_samples` | error, blocks | `n_samples`, `minimum` (1024) | — |
| `non_finite_samples` | error, blocks | counts, `first_indices` | `interpolate_non_finite` |
| `silent_signal` | error, blocks | RMS values | — |
| `metadata_missing` | error, blocks | `missing` field list | fill the manifest |
| `output_outliers` | warning | count, fraction, threshold (median + 25 robust sigmas), indices | `remove_outliers` |
| `possible_output_clipping` / `possible_input_clipping` | warning | `fraction_at_peak`, `count_at_peak`, `peak_magnitude` | check ADC range; keep if intentional |
| `time_misalignment` / `alignment_ok` | warning / info | `delay_samples` (sub-sample), `correlation` | `delay_samples` correction |
| `linear_gain_phase` | warning if > 3 dB or > 10°, else info | `gain_db`, `phase_deg`, `fit_residual`, `fitted_on` | gain/phase correction (optional) |
| `input_exceeds_unit_range` | warning | peaks, RMS, PAPR | `normalize=peak_input` |
| `amplitude_units_unconfirmed` / `amplitude_range` | info | peaks, RMS, PAPR | confirm units |
| `insufficient_oversampling` | warning | occupied bandwidth, fs | capture with more oversampling |
| `bandwidth_metadata_mismatch` / `spectrum_summary` | warning / info | occupied bandwidth vs declared | check `bandwidth_hz` |

Design rules:

- **Physics is not a defect.** Smooth compression is never reported; only
  a hard plateau (≥ 0.1 % of samples within 0.1 % of the peak, at least 20
  samples) is called *possible* clipping, and the message says it may be
  intentional saturation.
- **Estimates carry confidence.** Delay confidence is the normalised
  cross-correlation peak; gain confidence is `1 − relative residual` of a
  least-squares fit on the small-signal 40 % of samples after integer
  alignment. No estimate is applied automatically.
- **Blocking is explicit.** `evaluation_blocked` is true only for error
  items whose presence would make formal metrics unreliable.
- **Analysis windows.** Correlation and spectra use the first 2¹⁷ samples;
  everything else uses the full capture (float64, streaming import).

## Fixture protocol (tests/fixtures/manifest.json → `doctor_protocol`)

| Impairment (synthetic PA, 20 000 samples) | Requirement |
|---|---|
| delay 3, 7, 40 samples | estimate within ±0.1 sample, confidence ≥ 0.9 |
| complex gain 1.4·e^{j25°} | ±0.5 dB, ±3° |
| clip level at 60 % of the unclipped peak | detected; clean PA never flagged |
| 5 spikes of magnitude 50 | all found, none on clean data |
| NaN at two positions | blocks evaluation, indices reported |
| missing sample rate / bandwidth / n_sub_ch / nperseg | blocks evaluation |

Verified by `tests/unit/test_doctor.py`.

## Split protocol `contiguous-v1`

The capture is cut in time order into train | guard | val | guard | test
**before** framing (`contiguous_boundaries`). Ratios apply to the usable
length (total − 2·guard); rounding remainders go to the test split. The
default guard is 256 samples, above the longest recipe frame (200), so no
frame in one split overlaps in time with the memory context of a frame in
another split (`context_is_isolated`). The trainer only ever sees the three
split files, so frames cannot straddle a boundary by construction.

Built-in datasets keep the split their authors made (guard 0, boundaries as
found); the manifest records it, and results on them are comparable with
the published benchmark.

## Preprocessing `preprocess-v1`

Raw files are copied once and hashed; every preprocessing result is a new
**version** (`datasets/<id>/versions/<name>/`) with a `version.json` that
records the parameters, the code version, the steps actually applied and
the **fit range** of anything fitted (normalisation uses the training split
of the base version only, never validation or test samples). Versions can
be previewed (nothing written; the doctor re-runs on the result) and are
created only on confirmation. Measurement alignment done here is part of
data preparation and is recorded as such; it is not an evaluation-protocol
step.
