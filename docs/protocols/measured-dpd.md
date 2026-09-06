# Measured DPD evidence (`dpd_measured`): protocol, S16

Status: protocol implemented and tested on synthetic and mock captures;
**no measurement of a physical PA has been made with it yet** (§7). Every
result under this protocol says so in its attestation until an operator's
capture is imported, and even then it says "user-provided, not independently
verified".

## 1. What the evidence is

A `run_dpd` run exports `u = DPD(x)` for the test split (the `dpd-output`
artifact, with a sidecar naming the dataset, the DPD weights and the
scaling). The operator plays `u` through a physical PA and captures the output
`y`; under the same conditions they play `x` itself and capture `y0`. An
`evaluate_measured` run scores `y` as captured and stores everything that
makes the number traceable:

| Recorded | Where |
|---|---|
| the played export (copy) and its hash | `captures/played.csv`, `measurement.played_sha256`, artifact `played-signal` |
| the raw capture files (copies, as uploaded) and their hashes | `captures/with_dpd.*`, `captures/without_dpd.*`, artifacts `capture-with-dpd`, `capture-without-dpd` |
| the operator's declaration: PA, capture chain, capture rate, drive, gain, calibration, time, temperature, operator, notes | `MeasurementConditions` in the configuration and in `measurement.json` |
| the alignment of every capture: delay, wrap, correlation, least-squares gain, rms, peak, declared output power | `CaptureAlignment` in `measurement.json` and in every result's `measurement` block |
| the DPD weights (hash), the dataset, version and split | `dpd_reference`, `dataset` in the resolved configuration |

A screenshot of a spectrum is not evidence under this protocol: without the
capture file and the played file nothing can be recomputed.

## 2. Alignment (`opendpd/core/measurement.py`)

1. If the capture rate differs from the dataset rate, the capture is converted
   with an exact rational polyphase resampler (denominator ≤ 10 000); other
   ratios are refused at submission.
2. The integer delay is the peak of the circular cross-correlation between the
   first period of the capture and the played signal (`u` with DPD, `x`
   without). With looped playback the window may wrap around the file; with
   single-shot playback the capture must hold the full period after the delay.
3. The correlation coefficient between the played signal and the window must
   be ≥ 0.5, otherwise the run fails with `capture_rejected` (wrong file, no
   signal, a different waveform). Nothing is scored.
4. One complex least-squares gain `g` of the window onto the **target input**
   `x` (not onto `u`) is recorded. It is the gain a linear PA would show for
   this capture chain, in capture units. The window itself is never rescaled.

Not corrected, and therefore part of the reported distortion: fractional
timing error, IQ imbalance, DC offset, phase noise, receiver non-linearity.

## 3. Scoring

- Prediction: the aligned window `y` (capture units). Reference:
  `g · x`. Every registered metric profile is computed by the same registry
  as simulated results; the configured profile is the primary result.
- Baseline `measured_without_dpd`: `y0` aligned with its own delay and gain
  `g0`, scored against `g0 · x`. Two acquisitions cannot share one scale, so
  each carries its own alignment, both are recorded, and the **level
  difference** `20 log10(rms y / rms y0)` in capture units is stored. Beyond
  0.5 dB it is a limitation on the result: *the difference between the two
  captures is not attributable to the DPD alone*. The declared output powers
  and their difference are stored next to it. Lowering the drive for the
  with-DPD capture therefore shows up as a level difference and a declared
  power difference, never as an improvement.
- Limitations every result carries: the attestation, "no physical
  calibration" with the declared powers, the alignment caveat, "one capture
  per condition: no repeatability statistics", and (when they apply) the
  level difference, the wrapped window and the resampling ratio.

## 4. Comparability (`opendpd/core/metrics/compare.py`)

Measured and simulated results are shown side by side and never ranked: the
comparison key differs in evidence type. Between measured results the
reference gain is not part of the key (it is an alignment of each capture);
the **operating point** is: PA, drive, declared output power of the with-DPD
capture, capture chain and capture rate, as declared. Two measured results
rank only when every one of those agrees; otherwise the report lists what
differs.

## 5. Mock and manual sources

`MeasurementConfig.source` is `manual` (operator-provided files) or
`mock_adapter` (files from `opendpd instruments dry-run`). Mock results are
`source = "mock"`, `is_mock = true`, carry the mock attestation and are
never evidence about a PA; they exist so the whole path runs in CI
(`tests/integration/test_cli_run.py`, `test_docs_commands.py`).

## 6. Operating procedure for a real chain (human)

1. `opendpd apply <dpd-run>` and download `dpd-output` (`u`) and `x` from the
   same file; note the sidecar's scaling and the dataset sample rate.
2. Set the generator to the recorded drive; keep every attenuator, cable and
   analyser setting identical for both captures; note the PA, temperature and
   time.
3. Play `x`, capture ≥ one full period (looped) or the full period after the
   trigger (single shot). Play `u`, capture the same way. Do not touch the
   drive between the two captures.
4. Import with `opendpd measurements import --apply-run <run> --with-dpd ...
   --without-dpd ... --conditions conditions.json --power-with ... --power-without ...`
   (or the run page's "Import measured captures…"), read the alignment
   (delay, correlation, gain) and the level difference before reading a
   metric.
5. With an instrument adapter: `opendpd instruments dry-run` first (mock),
   then the real adapter with `OPENDPD_ALLOW_RF_OUTPUT=1 … --arm "<name>"`
   in an approved laboratory session only; the interlock rules of
   `docs/architecture/instruments.md` §2 apply.

## 7. Hardware trial record

| Item | Status |
|---|---|
| Chain (generator, PA, analyser), operator, date | **pending human**: no laboratory chain is available in this environment |
| Export → play → capture → import → evaluate completed under supervision | pending human |
| Level difference between the two captures, declared powers | pending human |
| Adapter used (mock only until a real one exists) | mock verified in CI; real adapter pending human (`docs/architecture/instruments.md` §4) |

One successful chain is evidence for that chain only.
