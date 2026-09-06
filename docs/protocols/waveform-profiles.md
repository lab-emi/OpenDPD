# Waveform evaluation profiles (plan S15) — `ofdm-lte20-v1` and `ofdm-lte20-evm-v1`

Status: **proposed, pending science review and cross-validation**. The
waveform, the measurement procedure, the reference-vector package and the
tests below exist in the repository; the numbers of this profile become
citable only after the cross-validation in §6 has been run by a maintainer
and the error budget in §5 has been approved. Until then the profile is
computed and stored like every other profile but is **hidden in the GUI**
(`MetricProfile.validation = pending_cross_validation`) and every result it
produces carries the limitation "profile pending cross-validation".

## 1. Scope

One waveform configuration, chosen because the lab's own captures
(`DPA_200MHz`, `DPA_160MHz`) are LTE 20 MHz carriers:

| Item | Value | Source |
|---|---|---|
| Waveform id | `ofdm-lte20-v1` | this document |
| Numerology | CP-OFDM, 15 kHz subcarrier spacing, 2048-point FFT at 30.72 MS/s, normal cyclic prefix (160, 144 × 6 samples per 0.5 ms slot), 14 symbols per 1 ms subframe | 3GPP TS 36.211 §6.12 (OFDM baseband signal generation); **standard version to be fixed at cross-validation** |
| Occupied subcarriers | 1200 (100 resource blocks × 12) at k = −600 … −1, +1 … +600; the DC subcarrier is empty | same |
| Modulation | 64QAM on every occupied subcarrier of every symbol, unit average power, points (±1, ±3, ±5, ±7)/√42 in I and Q, drawn uniformly from a seeded generator (`numpy.random.default_rng(seed)`) | `opendpd/core/waveforms/ofdm.py` |
| Length | `n_subframes` × 30 720 samples (default 10 = one 10 ms frame); instruments play it in a loop | spec |
| What it is | a **test waveform with known symbols**: no synchronisation signals, reference signals, control or shared channels, no resource mapping, no filtering or windowing | — |
| What it is not | a conformance test signal (E-TM), a 3GPP-compliant transmitter, or evidence of standard compliance; the name "LTE 20 MHz" refers to the numerology only | — |

Everything is regenerated from the specification (`WaveformSpec`: id,
version, seed, `n_subframes`); the reference-vector package (§4) is a
convenience for instruments and for cross-validation, never the source of
truth.

## 2. Measurement procedure of `ofdm-lte20-evm-v1`

Input: the evaluated I/Q signal (a PA model output, a simulated or measured
PA output with or without DPD) at its capture rate `fs`, and the dataset's
binding to the waveform (§3). Every step is deterministic and recorded in
`MetricProfile.parameters`.

1. **Rate conversion.** If `fs ≠ 30.72 MS/s`, the signal is converted with a
   polyphase resampler (`scipy.signal.resample_poly`, Kaiser-windowed FIR)
   using the exact rational ratio; `fs` below 30.72 MS/s or without a small
   rational ratio (denominator ≤ 10 000) gives `not_applicable`.
2. **Timing.** Circular cross-correlation with the regenerated waveform
   (one period) gives the integer offset and the normalised peak; a peak
   below 0.3 means the signal is not (a distorted copy of) the bound waveform
   → `missing_reference`.
3. **Coarse frequency offset** from the cyclic-prefix correlation of every
   complete symbol, skipping the first 32 prefix samples (channel memory of
   the previous symbol).
4. **Equaliser.** One complex gain per subcarrier, least squares over all
   complete symbols in the capture, from the known symbols (data aided).
5. **Fine frequency offset** from the slope of the per-symbol residual phase
   (known symbols) against time; steps 4–5 iterate at most three times until
   the update is below 1 mHz. No per-symbol phase tracking: whatever remains
   counts as error.
6. **EVM.** RMS over every occupied subcarrier of every complete symbol,
   `EVM_RMS = 100 · sqrt(Σ|Ŝ − S|² / Σ|S|²)` in percent, also reported in dB
   (`20·log10`). Symbols are demodulated at the nominal FFT position (end of
   the cyclic prefix).
7. **ACLR** at the capture rate: Welch PSD (Hann, `nperseg` of the dataset,
   50 % overlap), main channel = [−9, +9] MHz (the 18 MHz transmission
   bandwidth configuration), first adjacent channels centred at ±20 MHz
   with the same 18 MHz rectangular measurement bandwidth,
   `ACLR = 10·log10(P_adjacent / P_main)` (leakage, negative dBc, lower is
   better, the repository's convention). Requires `fs ≥ 58 MS/s`; otherwise
   `not_applicable` with the reason. Definition after 3GPP TS 36.104 §6.6.2
   (E-UTRA ACLR); version to be fixed at cross-validation.

### Deviations from a standard EVM procedure (stated, to be quantified)

| Standard procedure (TS 36.104 Annex E principles) | This profile | Why |
|---|---|---|
| Channel estimate from reference signals with time/frequency averaging | least-squares gain per subcarrier from *all* known symbols | data-aided test waveform; no reference signals exist |
| EVM evaluated at two FFT timing positions (window edges), maximum reported | one position: end of the cyclic prefix | first version; the two-position rule is a candidate for v2 after cross-validation |
| Frequency offset estimated per standard procedure | cyclic-prefix coarse estimate + data-aided fine estimate | same numbers expected for a static offset; drift within the window is not tracked |
| Measurement over specific subframes/slots and physical channels | every complete symbol in the capture | no physical channels |

The data-aided equaliser over `L` symbols removes `1/L` of the noise power
from the reported EVM (bias `sqrt(1 − 1/L)`); with `L ≥ 28` this is below
2 % relative and is part of the error budget.

## 3. Binding a dataset to the waveform

`opendpd datasets import … --waveform <package>/waveform.json` regenerates
the waveform and cross-correlates it with the imported **input** column:
the offset and the normalised peak are stored as `signal.waveform`
(`WaveformBinding`) next to the package hash. The binding says which symbols
were sent; the captured data is untouched. Without a binding, or when the
input does not correlate with the waveform, the profile reports
`missing_reference` for the EVM metrics and never falls back to another
definition under the same name. Multi-carrier captures (the built-in
`DPA_200MHz`, ten carriers at 800 MS/s) are therefore reported as not
applicable to this profile, which is the correct answer.

## 4. Reference-vector package

`opendpd waveforms generate --seed S --subframes N --out DIR` writes

| File | Content |
|---|---|
| `waveform.json` | the `WaveformSpec`, the package hash (spec + symbols), sample count, formats |
| `x.npy` | float32 `(n, 2)` I/Q at 30.72 MS/s, unit average power, to be played in a loop |
| `symbols.npy` | complex64 `(n_symbols, 1200)` reference symbols in subcarrier order |

Expected results on the ideal signal (self-evaluation, recorded by the tests
in `tests/unit/test_waveform_ofdm.py`): EVM below 1e-9 % at 30.72 MS/s;
EVM floor 0.037 % after rate conversion from 122.88 MS/s or 800 MS/s
(polyphase filter ripple); a frequency offset of ±350 Hz and a linear channel
shorter than the cyclic prefix are removed exactly; white noise of SNR 30 dB
gives EVM = 100/√SNR · √(1200/2048) · √(1 − 1/L) within 0.03 percentage
points.

## 5. Error budget (proposed, pending approval)

| Contribution | Bound in this implementation | Evidence |
|---|---|---|
| Numerical floor (no rate conversion) | < 1e-9 % EVM | `test_ideal_signal_demodulates_to_zero_evm_with_exact_timing` |
| Rate conversion floor | 0.037 % EVM (122.88 and 800 MS/s) | `test_resampling_from_a_capture_rate_has_a_small_recorded_floor` |
| Equaliser noise bias | factor √(1 − 1/L), L = complete symbols in the capture | `test_white_noise_of_known_power_gives_the_predicted_evm` |
| Frequency offset residual | estimate within 0.01 Hz of an injected static offset; drift not tracked | `test_frequency_offset_and_phase_are_estimated_and_removed` |
| Timing | integer sample; sub-sample delay absorbed by the equaliser as linear phase | `test_timing_offset_and_looped_playback_are_recovered`, `test_linear_channel_inside_the_cyclic_prefix_is_equalised_exactly` |
| ACLR PSD estimate | Welch with the dataset's `nperseg`; bias of a tone of known power below 0.05 dB (as for `general-spectral-v1`) | `tests/unit/test_metrics_general.py` conventions |

Comparison targets against the cross-validation backend (plan S15, fixed
before any model comparison, never relaxed afterwards): adjacent-channel
ratio within **0.1 dB**, RMS EVM within **0.05 percentage points** on the
same digital reference signal. Values below the measurement floor are
reported with the floor stated, not scored.

## 6. Cross-validation protocol (maintainer step; MATLAB not required for users)

1. Generate a package (`--seed 1 --subframes 10`) and, in MATLAB LTE
   Toolbox, build the same resource grid from `symbols.npy` and modulate it
   (`lteOFDMModulate` with a 20 MHz, normal-CP configuration); the
   time-domain signals must agree to numerical precision after scaling.
2. Pass the same distorted signals through both chains: the package played
   through a PA capture (or the synthetic PA of `tests/fixtures/synthetic.py`),
   scored here and by the toolbox's EVM/ACLR functions with their default
   procedure; record both numbers, the differences and the toolbox version.
3. Record the outcome in this document (§7), set
   `validation = cross_validated` in `opendpd/core/metrics/ofdm_evm_v1.py`
   (protected path, science review) and fix the standard versions cited above.

Users never need MATLAB: the reference vectors and the evaluation run in
Python; MATLAB is the independent backend that checks them.

## 7. Validation record

| Date | Backend and version | Signals | ACLR difference | EVM difference | Outcome |
|---|---|---|---|---|---|
| — | — | — | — | — | **not run yet** |

## 8. Effect on existing results

Nothing changes for `legacy-opendpd-v1` (frozen) or `general-spectral-v1`:
the new profile is an additional file under `runs/<id>/results/` and reports
explicit statuses on every existing dataset (`tests/unit/test_metrics_ofdm_evm.py`).
