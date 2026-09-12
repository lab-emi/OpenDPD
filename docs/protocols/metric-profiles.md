# Metric profiles (`opendpd/core/metrics/`, protected path)

Every score in an `EvaluationResult` names the profile and version that
produced it. A profile fixes the algorithm, the band layout, the PSD
estimator, the aggregation, the unit and the "better" direction. Changing any
of these is a scientific-semantics change: bump the profile version (or add a
profile) and keep the old one reachable. `opendpd profiles` prints the
registry; `GET /metrics/profiles` serves it to the GUI.

## `legacy-opendpd-v1` (frozen)

The historical OpenDPD numbers, computed by `utils/metrics.py` exactly as the
trainer calls it. Values are pinned by `tests/golden/legacy_metrics_v1.json`
(rel/abs 1e-7) and the registry reproduces them
(`tests/unit/test_metrics_registry.py`).

| Metric | Unit | Definition | Aggregation |
|---|---|---|---|
| `NMSE` | dB | mean over segments of 10·log10(Σ\|e\|² / Σ\|y\|²) | **mean of per-segment dB**, not pooled |
| `EVM` | dB | 20·log10 of the mean sub-channel spectral error ratio | repo-specific; **not** a demodulated EVM |
| `ACLR_L` / `ACLR_R` | dBc | 10·log10(P_adjacent / max sub-channel power) | leakage convention, negative, lower is better |
| `ACLR_AVG` | dBc | (ACLR_L + ACLR_R)/2 | arithmetic mean of dB values |

Segments are `nperseg` samples; the last one is zero-padded and included.

## `general-spectral-v1`

Explicit conventions, analytic reference tests in
`tests/unit/test_metrics_general.py`.

| Metric | Unit | Definition | Reference check |
|---|---|---|---|
| `NMSE` | dB | 10·log10(Σ\|y−r\|² / Σ\|r\|²) **pooled** over the valid range | complex gain g: exactly 20·log10\|g−1\|; white noise: its SNR ±0.1 dB |
| `IBE` in-band error | dB | 10·log10(P_main(y−r) / P_main(r)) from the Welch PSD | gain error exact; noise: SNR + 10·log10(BW/fs) ±0.15 dB |
| `ACPR_L` / `ACPR_R` | dBc | 10·log10(P_adjacent(y) / P_main(y)) of the evaluated signal | a tone of known power in the adjacent channel ±0.05 dB |

Parameters (all recorded in the profile): Hann window, `nperseg` from the
dataset, 50 % overlap, two-sided density PSD, band power = Σ bins·(fs/nperseg);
main channel [−BW/2, +BW/2), adjacent channels of the same width immediately
next to it; no normalisation; pooled aggregation; floor −300 dB instead of
−∞ when the error is exactly zero.

**Conventions kept apart.** `ACPR` is *leakage* (adjacent/main, negative dBc,
lower is better). The positive *suppression* convention (main/adjacent,
higher is better) equals −ACPR and is deliberately not reported so that name,
formula and direction never disagree. `IBE` is a power ratio in dB; the
amplitude ratio is 10^(IBE/20) and the percentage 100·10^(IBE/20); the three
forms are documented, never mixed in one field. Pooled NMSE and the legacy
mean-of-segment-dB NMSE are different numbers (`test_pooled_nmse_is_not_the_mean_of_per_segment_db`).

**Explicit statuses, never fake numbers.** `MetricValue.status` is `ok` only
with a finite value; otherwise the reason is stored:

| Situation | Status |
|---|---|
| no reference signal | `missing_reference` (NMSE, IBE) |
| reference has zero energy / no in-band power | `invalid` |
| NaN/Inf in the signals | `invalid` |
| `sample_rate_hz`, `bandwidth_hz` or `nperseg` missing | `not_applicable` (spectral metrics) |
| a band lies outside the captured ±fs/2 | `not_applicable`, reason names the band |
| fewer valid samples than one PSD segment / no bin in a band | `not_applicable` |
| computation raised | `failed` with the exception |

## `ofdm-lte20-evm-v1` (pending cross-validation, hidden in the GUI)

Data-aided RMS EVM and E-UTRA-style ACLR of a capture bound to the
`ofdm-lte20-v1` reference waveform (CP-OFDM with the LTE 20 MHz numerology
and known 64QAM symbols). Procedure, deviations from a standard's EVM
definition, error budget and the cross-validation protocol are in
`docs/protocols/waveform-profiles.md`.

| Metric | Unit | Definition |
|---|---|---|
| `EVM_RMS` | % | 100·sqrt(Σ\|Ŝ − S\|² / Σ\|S\|²) over every occupied subcarrier of every complete OFDM symbol, after timing, frequency-offset and per-subcarrier least-squares equalisation from the known symbols |
| `EVM_DB` | dB | 20·log10(EVM_RMS / 100) |
| `ACLR_L` / `ACLR_R` | dBc | 10·log10(P[±20 MHz ± 9 MHz] / P[−9, +9 MHz]) from the Welch PSD at the capture rate (leakage, negative) |

Statuses: `missing_reference` without a waveform binding or when the signal
does not correlate with the waveform; `not_applicable` when the capture rate
cannot be converted exactly to 30.72 MS/s, is below it, or (ACLR) does not
contain the first adjacent channel (< 58 MS/s). The built-in ten-carrier
captures are `missing_reference` under this profile by design.

## Validation status of a profile

`MetricProfile.validation` says how a profile's numbers have been checked:
`golden` (frozen references of the historical code: `legacy-opendpd-v1`),
`analytic` (closed-form reference tests: `general-spectral-v1`),
`pending_cross_validation` (implemented and tested, the independent-backend
comparison has not run: `ofdm-lte20-evm-v1`) or `cross_validated`. The
service computes and stores every profile; the GUI offers only profiles past
`pending_cross_validation`; `opendpd profiles` prints the status and
`opendpd evaluate` says so before scoring under a pending profile. Every
result of a pending profile carries the limitation "pending cross-validation".

## Where the numbers come from

The worker scores the **best checkpoint over the test split** once at the end
of a run (`opendpd.services.evaluation.evaluate_all`): predictions are computed
once and every registered profile is stored under `runs/<id>/results/<profile>.json`;
the configured profile (`evaluation.profile_id`, default legacy) is the
primary `result.json`. Under the legacy profile this reproduces the trainer's
own `TEST_*` log values (`test_result_metrics_come_from_the_registry_and_agree_with_the_training_log`).
`opendpd evaluate RUN --profile P` re-scores later from the same checkpoint.
Display decimation, zoom or chart filtering never touch these files: charts
are produced from separate, decimated data and formal metrics are only ever
computed here.

## Comparability

`opendpd.core.metrics.incompatibilities(a, b)` lists why two results must not
be ranked against each other: different profile/version, evidence type
(`pa_modeling` vs `dpd_surrogate` vs `dpd_measured`), dataset, preprocessing
version, split protocol, evaluated split, reference kind, reference gain
(the operating point), PA surrogate (for simulated DPD results: the weight
hash the DPD was scored through), execution semantics or mock status.
Results with an empty list may be ranked; others may only be shown side by
side with the reasons (`GET /results/compare`, the Compare page).
