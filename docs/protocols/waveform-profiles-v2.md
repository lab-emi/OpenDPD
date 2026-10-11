# Version 2 LTE-numerology measurement and cross-validation registration

Registered 2026-10-11, before implementation or any version 2 comparison result.
Status: proposed, awaiting implementation, independent validation and science review.

## Reason for a new version

The original `ofdm-lte20-evm-v1` and its registered results remain unchanged and
pending cross-validation. A fresh run of that implementation on 2026-10-11 again
produced 102/115 scored items within budget, seven outside and six unevaluable.
Its comparison asks a single Hann/segmented/half-open-band estimate to agree with
both a full-record Hamming estimate and a segmented Hann estimate. In the existing
record, S2 left ACLR at 2048 samples has MATLAB references -41.334805433548354 and
-41.05397753150406 dBc: a separation of 0.28082790204429386 dB. No single value can
be within 0.1 dB of both. Raising the tolerance or fitting a correction to those
observations would not validate the measurement algorithm.

V2 therefore names both measurement configurations explicitly and compares each
production path to the corresponding independent MATLAB configuration. The old
115 comparisons are not retroactively declared passed, removed or rewritten.

## Production profiles

Both profiles retain the waveform, rate conversion, equalizer, EVM formula and
non-conformance scope of [v1](waveform-profiles.md). They are additional profiles;
`legacy-opendpd-v1`, `general-spectral-v1`, `opendpd-spectral-v2` and the v1 LTE
profile retain their numerical behavior and validation states.

| Profile | Spectral estimator | Independent MATLAB comparator |
| --- | --- | --- |
| `ofdm-lte20-evm-v2` | Welch, periodic Hann, dataset `nperseg`, floor(nperseg/2) overlap, FFT length equal to nperseg, detrend off | `comm.ACPR` with Specify window parameters, Hann, 50% overlap, same segment/FFT length |
| `ofdm-lte20-evm-auto-v2` | One full-record segment, symmetric Hamming, no overlap, next-power-of-two FFT at least 256, detrend off | `comm.ACPR` in Auto mode with its existing defaults |

Both use the actual FFT frequency grid and include the enclosing bin range, from
the last bin at or below the lower band edge through the first at or above the
upper edge, inclusive. Integrate PSD times fs/FFT length. The main band is
[-9,+9] MHz; adjacent bands are [-29,-11] and [+11,+29] MHz. Report
10 log10(adjacent/main), negative leakage dBc. Bands must be fully captured; a
missing band, invalid input or zero main power cannot become a passing number.
The full-record profile explicitly records that it does not use dataset nperseg.
The definitions follow the installed R2026a `comm.ACPR` behavior; see also the
[public estimator documentation](https://www.mathworks.com/help/comm/ref/comm.acpr-system-object.html).

## Frequency-tolerant acquisition

The v1 timing check coherently correlates a whole 10 ms frame before estimating
frequency offset. At 350 Hz it cancels much of the valid reference and rejects the
signal before its cyclic-prefix frequency estimator can run.

V2 first tries the existing complex correlation. If its normalized peak is below
0.3, obtain a **candidate** timing from circular correlation of signal magnitudes
with reference magnitudes. Estimate coarse frequency offset from complete cyclic
prefixes at that candidate (skip the first 32 prefix samples, as v1), compensate
that constant offset, and repeat complex correlation against the regenerated
reference. Only the final complex peak >=0.3 establishes the binding; magnitude
correlation alone never accepts a waveform. With final timing, evaluate the
original input through the original CP coarse/data-aided fine estimator and
least-squares per-subcarrier equalizer. Maximum fine iterations remains three;
no per-symbol phase tracking or lower correlation threshold is introduced.

This option belongs to v2. Calling the v1 demodulator or binding a v1 waveform
retains its previous behavior. False waveform/seed, noise and insufficient-symbol
inputs must still be refused. Unit tests must cover both signs of 350 Hz, 1/10 ms
frames, delayed captures, no offset, and a wrong reference.

## Frozen comparison matrix and budgets

Use exactly S0–S9, S6b and S9b from the original registration, including seed 1,
10 subframes, PA drive levels, noise seed/power, rates and segment-length sets.
Preserve their input hashes. Do not replace +350 Hz with the easier +30 Hz cases.
Every originally scored P1/P2/P3 item has a v2 counterpart:

- P1: three waveform checks, same independent LTE Toolbox generation, same scale
  fit, exact count/alignment and relative RMS <=1e-6.
- P2: all 84 signal/side/resolution/configuration comparisons remain scored. For
  each Auto comparator use the Auto production profile; for each aligned
  comparator use the dataset-segment production profile. Both pass only at an
  absolute difference <=0.1 dB. The reference MATLAB object settings are unchanged.
- P3: all 28 existing EVM/timing/clean-frequency items remain scored, including
  S6 and S9 at +350 Hz. EVM difference <=0.05 percentage points, exact integer
  timing, and each clean S9/S9b frequency estimate within 0.01 Hz of injection.
  Implement v2 acquisition independently in MATLAB from this written procedure,
  then retain the existing MathWorks demodulation/EVM building blocks.
- The two v2 production profiles must give the same EVM; changing the PSD mode or
  dataset nperseg must not change EVM. Analytic band-power/white-noise/zero-input
  tests additionally test product behavior beyond the registered capture fixtures.

Thus the original-shaped v2 matrix has 115 scored items. Diagnostics remain
explicit diagnostics. A refused signal, missing value, non-finite result, missing
case or result from the wrong profile/configuration is a failure, not agreement.
No threshold, seed, drive, sample count or existing scored case is relaxed.

## Reproducibility and acceptance

Keep the original harness and JSON records. The v2 harness must call the registry
production paths, write separate JSON/Markdown results, record source commit,
dirty state, script hashes, MATLAB/products and Python/library versions, and
verify that case identities, budgets and signal hashes match this registration.
Use a separate MATLAB preference directory. Run the complete applicable MATLAB
toolbox tests and Python regression suites; changes to the legacy/general/Arena
metric outputs are release blockers.

Only after all 115 v2 scored items and the additional regression tests pass may
science review promote the two v2 profiles to `cross_validated`. That designation
means agreement for this registered digital measurement, not compliance with
3GPP TS 36.104 Annex E, a real-instrument qualification, or approval of the old v1
comparison. Append the actual results and any limitations after the run; do not
edit this registration to fit them. Any procedural change requires a new dated
amendment committed before its comparison result is read.

## Implementation clarification 1 — 2026-10-11, before the first v2 comparison

Reading the installed R2026a `comm.internal.PowerMeasurements` implementation
confirmed that its Hann and Hamming windows are **symmetric**. Therefore the v2
segmented profile uses a symmetric Hann window; this supersedes "periodic Hann"
in the initial table. The old profiles retain their existing periodic Hann.
This is a configuration correction from primary implementation evidence, made
before any Python-v2/MATLAB-v2 comparison result was read; no tolerance, signal or
scored comparison is changed. A two-sample symmetric Hann has zero window power,
so the segmented v2 profile explicitly refuses segment lengths below three.

At a captured Nyquist endpoint the enclosing range is clipped to the available
FFT grid, as `comm.ACPR` does. A requested band outside the physical Nyquist range
is still refused. None of the registered 122.88/245.76 MS/s bands needs clipping;
analytic tests cover that boundary separately.
