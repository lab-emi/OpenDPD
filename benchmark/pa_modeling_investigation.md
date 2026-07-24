# APA_200MHz PA-Modeling NMSE Investigation

Question: why does PA modeling on `APA_200MHz` saturate at ~-31.5 dB NMSE, and is
the `benchmark_report.md` figure of -43.52 dB attainable?

Answer: **-31.5 dB is an undertrained-recipe artifact, not a data defect.** The
dataset's measurement noise floor is ~-38.6 dB, every architecture reaches it once
trained with the OpenDPDv2 recipe, and -43.52 dB lies *below* the noise floor, so
it is not physically attainable on this dataset.

## 1. Backbone comparison on APA_200MHz (PA modeling, val/test NMSE in dB)

All models sized to ~1900 parameters for a fair comparison.

| Backbone | Params | Report recipe<br>(lr 5e-4, bs 256, 100 ep, no sched) | OpenDPDv2 recipe<br>(lr 5e-3, bs 64, 240 ep, ReduceLROnPlateau) |
|---|---|---|---|
| GRU (h23) | 1,911 | -31.48 / -31.40 | **-38.79 / -38.87** |
| DGRU (h23) | 2,751 | -31.67 / -31.61 | **-39.13 / -39.21** |
| TCN (h66) | 1,914 | -37.44 / -37.30 | **-38.38 / -38.25** |
| TRes-GRU (h22) | 1,916 | -35.62 / -35.50 | **-38.94 / -39.01** |

Two observations:

1. **The recipe, not the architecture, causes the -31.5 dB plateau.** The same GRU
   improves by 7.4 dB purely from the training recipe. Recurrent models are the
   most sensitive: with lr=5e-4 and no schedule they are badly undertrained at
   100 epochs, while the convolutional TCN already reaches -37.4 dB.
2. **With a proper recipe all four architectures land within 0.9 dB of each other**
   (-38.25 to -39.21), because they are all at the data's noise floor. Ranking
   there is not architecturally meaningful.

## 2. Data ceiling: closed-form least-squares fits

Closed-form identification is immune to optimizer/learning-rate issues, so it
measures what the *data* supports. Basis built on the contiguous signal, fit on
the train split, evaluated on the test split (64-sample guard).

| Model | Real params | APA_200MHz | APA_200MHz_b | DPA_160MHz |
|---|---|---|---|---|
| MP K=5 Q=10 | 100 | -31.94 | -32.81 | -38.14 |
| MP K=7 Q=20 | 280 | -37.13 | -33.24 | -38.48 |
| MP K=9 Q=40 | 720 | -37.22 | -33.23 | -38.50 |
| GMP rich | 1,180 | -38.54 | -38.42 | -39.48 |
| GMP extra-rich | 3,520 | **-38.83** | **-38.48** | **-39.51** |

All three datasets saturate in the same place, -38.5 to -39.5 dB. APA_200MHz is
**not** a worse dataset than DPA_160MHz.

Note the damning comparison: a 100-real-parameter memory polynomial (-31.94 dB)
matches the 1,911-parameter GRU trained with the report recipe (-31.40 dB).

## 3. The residual is measurement noise, not unmodeled distortion

Residual of the GMP-rich fit, analyzed spectrally and against input amplitude:

| Quantity | APA_200MHz | DPA_160MHz |
|---|---|---|
| Residual NMSE | -38.64 dB | -39.48 dB |
| Residual PSD in-band vs out-of-band | +3.9 dB | +4.8 dB |
| *Signal* PSD in-band vs out-of-band | +33.4 dB | +37.8 dB |
| Residual power at \|x\|→0 | -48.0 dB | -41.4 dB |
| Residual power at \|x\|→max | -44.5 dB | -37.3 dB |

The residual is nearly white (+3.9 dB in/out-of-band tilt, against +33.4 dB for
the signal) and **its power barely depends on input amplitude — it stays at
-48 dB even where the input goes to zero.** Unmodeled nonlinear distortion must
vanish as the drive vanishes; additive measurement noise does not. So ~-38.6 dB
is the noise floor of this capture.

**Consequence: -43.52 dB is ~5 dB below the measurement noise floor and cannot be
achieved on APA_200MHz by any model.** This independently confirms it is a
misrecorded figure.

## 4. Dataset integrity: no defects found

- **Git history**: only 4 commits ever touched `datasets/APA_200MHz/`. The CSVs
  have been bit-identical since 2025-07-09; every later commit changed only
  `spec.json` metadata and plots. `fs`, `nperseg` and split ratios never changed.
- **Provenance**: R&S SMW200A generator -> 3.5 GHz Ampleon GaN Doherty PA ->
  Keysight N9042B IQ capture (`Matlab/` scripts).
- **A vs B**: `APA_200MHz` and `APA_200MHz_b` have **byte-identical inputs** and
  different output captures. Commit `2c7f91a` swapped the two output sets four
  hours after the initial release and renamed `APA_200MHz_backup` to
  `APA_200MHz_b`, so today's primary dataset is the capture the OpenDPDv2 paper
  numbers were reported on.
- **Signal QA** (all three datasets): no NaN/Inf, no duplicate rows, no clipping
  (only the 1-2 samples defining the normalization peak), DC offset < 1e-3,
  AM/AM relative spread 0.045-0.049 (APA is in line with DPA).
- **Splits**: power and complex gain are consistent across train/val/test
  (APA gain 1.1626 / 1.1628 / 1.1636); no discontinuity at split boundaries.
- **Alignment**: input/output are already time-aligned (peak lag 0, normalized
  cross-correlation 0.9945).
- **Measurement A vs B**: NMSE between the two output captures of the same
  stimulus is -23.75 dB (gain 0.98987, phase +0.309 deg). Since each capture is
  individually modelable to -38.5 dB, this gap is a systematic operating-point
  difference between the two measurements, not random capture noise.

## 5. Downstream impact: the benchmark rests on an undertrained surrogate

Re-running the QR benchmarks on APA_200MHz with the *same* code but a properly
trained PA surrogate (GRU h23 at -38.87 dB instead of -31.40 dB):

| Model | Metric | Surrogate at -31.4 dB<br>(as in benchmark_report) | Surrogate at -38.9 dB | Change |
|---|---|---|---|---|
| MP (QR) | ACLR_AVG | -41.02 | -41.81 | -0.8 |
| MP (QR) | EVM | -32.70 | **-47.41** | **-14.7** |
| GMP (QR) | ACLR_AVG | -38.75 | -39.58 | -0.8 |
| GMP (QR) | EVM | -38.50 | **-34.89** | **+3.6** |

EVM moves by up to 14.7 dB and the MP-vs-GMP ranking flips (MP now wins on both
metrics). `benchmark_report.md` Section 8 concludes that AI beats the best
traditional method by "10.6 dB in EVM" on APA_200MHz; with a converged surrogate
MP alone reaches -47.41 dB EVM, so that conclusion does not survive. The
DPA_160MHz surrogate was converged (-38.43 dB, at its noise floor), which is why
that half of the report reproduces cleanly.

## 6. New backbone: TRes-GRU

`backbones/tres_gru.py` is TRes-DeltaGRU with the DeltaGRU cell replaced by a
dense `nn.GRU` (bias-free, so parameter counts match at equal hidden size: h=10
gives 524 params, identical to TRes-DeltaGRU). Registered as `tres_gru` in
`models.py` and in both backbone choice lists in `arguments.py`.

Validated: with `thx=thh=0`, TRes-DeltaGRU is mathematically a dense GRU, and the
two implementations agree numerically (max abs difference 8.2e-7, relative L2
1.3e-6) after weight transfer.

## Reproducing

```bash
# Report recipe
python main.py --step train_pa --dataset_name APA_200MHz --PA_backbone tcn --PA_hidden_size 66 --n_epochs 100

# OpenDPDv2 recipe
python main.py --step train_pa --dataset_name APA_200MHz --PA_backbone gru --PA_hidden_size 23 \
  --n_epochs 240 --lr 5e-3 --lr_schedule 1 --lr_end 1e-4 --decay_factor 0.5 --patience 10 --batch_size 64
```

Caution: the checkpoint/log path (`model_id`) does not encode the learning rate,
batch size or epoch count, so re-running the same architecture with a different
recipe silently overwrites `save/<dataset>/train_pa/<model_id>.pt` and its logs.
The checkpoint currently on disk for `PA_S_0_M_GRU_H_23_F_200_P_1911` is the
OpenDPDv2-recipe model (-38.87 dB), not the report-recipe one.
