# OpenDPD 2.2.4

Studio now separates PSD charts by physical signal-chain position and provides an
explicit workflow for generating PA input, simulating a Virtual PA output and
creating a paired synthetic dataset.

## Changes

- Independent DPD Input, DPD Output / PA Input and PA Output PSD charts throughout
  inspection, live training/testing, result review, comparison and publication.
  Output targets and with/without-DPD baselines remain together at PA Output.
- Compact per-chart legends, full run-ID keys for comparisons, matching initial
  dB scales, independent zoom and restored per-position figure ranges. Legacy
  mixed figures remain readable and export as separate positions.
- Live DPD previews include the actual predistorted drive captured in the same
  bounded shadow forward pass as the PA output. PSD values and scoring protocols
  are unchanged.
- Signal Generator creates a PA Input Dataset only. Separate input CSV and JSON
  metadata downloads clearly state that no PA output exists yet.
- PA Library provides nine mathematical Virtual PAs: linear, Rapp, Rapp with
  AM/PM, Saleh, memory polynomial, lagging-envelope GMP, GaN trap/thermal,
  illustrative Doherty and envelope tracking. Linked sliders, numeric inputs
  and formulas explain parameter effects.
- Explicit output simulation, separate output/paired CSV and metadata exports,
  frozen provenance and paired dataset creation before PA model training.
- A compact expandable workflow diagram separates dataset making from PA/DPD
  training. Existing paired datasets go straight to PA Training.
- Updated README, workflow/API/review/visualization guides and real GUI screenshots.

## Scope

Virtual PAs are normalized behavioral simulations, not calibrated device models.
Generated pairs remain synthetic throughout training and export. Signal presets
remain uncoded engineering stimuli; experimental Wi-Fi 8 presets make no standards
conformance claim. Independent RF acquisitions, hardware reports and external EVM
cross-validation remain separate research acceptance items.

Public sessions retain tenant isolation, sample/size quotas and expiry. Dataset
contribution requires an operator-configured identity, an explicit submission and
human PR merge review. No email is sent automatically.

## Install and use

```bash
python -m pip install --upgrade "opendpd[gui]==2.2.4"
opendpd gui
```

[Hosted Studio](https://opendpd.com/studio/) · [Signal Generator](../guides/signal-generator.md) ·
[PA Library](../guides/virtual-pa-library.md) · [PSD guide](../guides/signal-chain-spectra.md)

Verification and deployment evidence is recorded in the [2.2.4 validation record](../performance/studio-2.2.4.md).
