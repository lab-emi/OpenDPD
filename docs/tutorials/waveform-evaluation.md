# Evaluating a PA or DPD with a reference waveform (EVM and ACLR)

Plan S15 adds one reference waveform, `ofdm-lte20-v1` (CP-OFDM with the LTE
20 MHz numerology and known 64QAM symbols), and one evaluation profile,
`ofdm-lte20-evm-v1`, that measures the RMS error vector magnitude and the
adjacent-channel leakage of a capture made while that waveform was played.
Everything runs in Python; MATLAB is only the independent backend maintainers
use to cross-validate it (`docs/protocols/waveform-profiles.md`).

The profile is **pending cross-validation**: the GUI does not offer it yet,
`opendpd profiles` lists it with that status, and every result it produces
says so. Nothing about the existing profiles changes.

## 1. Generate the waveform and play it

```bash
opendpd waveforms generate --seed 1 --subframes 10 --out ./waveforms/lte20-seed1
opendpd waveforms show ./waveforms/lte20-seed1/waveform.json
```

`x.npy` (float32 I/Q at 30.72 MS/s, unit average power) is the signal to
play in a loop; `symbols.npy` holds the reference symbols; `waveform.json`
holds the specification and the package hash. The package is regenerated
from the specification (id, seed, length) whenever it is needed, so a copy
cannot drift from what was played.

## 2. Capture and import with the binding

Capture the PA input and output as usual (any rate that is an exact rational
multiple of 30.72 MS/s, for example 122.88 MS/s; 800 MS/s also works). Then
import the capture and bind it to the waveform:

```bash
opendpd datasets import capture.csv --id pa-lte20 --fs 122.88e6 --bandwidth 18e6 --n-sub-ch 1 --nperseg 4096 --units normalized --waveform ./waveforms/lte20-seed1/waveform.json --workspace WS
```

The import regenerates the waveform, cross-correlates it with the imported
**input** column and records the offset and the correlation peak in the
dataset (`signal.waveform`). An input that does not correlate with the
waveform is refused with the peak value; the captured data itself is never
modified.

## 3. Train and read the result

```bash
opendpd run --recipe pa-gru-smoke-v1 --dataset pa-lte20 --workspace WS
opendpd evaluate <run_id> --workspace WS --profile ofdm-lte20-evm-v1
```

Every run stores the profile next to the others under `runs/<id>/results/`.
`EVM_RMS` (%) and `EVM_DB` come from the data-aided demodulation of the
evaluated signal (the PA model output for a PA run, the linearised output for
a DPD run); `ACLR_L` / `ACLR_R` (dBc, leakage convention) come from the PSD
at the capture rate and need at least 58 MS/s so that the first adjacent
channel is inside the capture.

## What the statuses mean

| Status | Why |
|---|---|
| `missing_reference` | the dataset is not bound to the waveform, or the evaluated signal does not correlate with it (peak below 0.3) |
| `not_applicable` | the capture rate cannot be converted exactly, is below 30.72 MS/s, or (ACLR) does not contain the adjacent channel; `nperseg` or the sample rate is unknown |
| `invalid` | non-finite samples |

The built-in ten-carrier captures are not bound to any waveform and report
`missing_reference` under this profile: that is the correct answer, not a
gap to fill by re-using the profile's name for another definition.
