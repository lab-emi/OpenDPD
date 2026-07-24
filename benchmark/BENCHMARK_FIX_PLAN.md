# Benchmark Verification & Fix Plan

Status: **verification complete, fixes partially applied, re-benchmark pending.**

This document records a full experimental re-run of `benchmark/benchmark_report.md`,
the defects that re-run exposed, and the concrete work needed to correct them.
Everything below was measured on this machine (RTX 4090 Laptop, `.venv`,
PyTorch 2.13.0+cu132, seed 0) unless stated otherwise.

---

## 1. What was verified

Every number in `benchmark_report.md` was re-run from the commands the report
itself gives.

### 1.1 Reproduces correctly

| Item | Reported | Reproduced | Delta |
|---|---|---|---|
| PA NMSE, DPA_160MHz | -38.43 | -38.44 / -38.80 (val/test) | exact |
| APA TRes-DeltaGRU DPD | -53.35 / -49.08 | -53.68 / -48.96 | ≤0.33 |
| APA GRU DPD | -52.61 / -47.50 | -52.10 / -47.33 | ≤0.51 |
| APA MP (QR) | -41.01 / -32.68 | -41.02 / -32.70 | ≤0.02 |
| APA GMP (QR) | -38.80 / -38.53 | -38.75 / -38.50 | ≤0.05 |
| DPA TRes-DeltaGRU DPD | -56.81 / -54.00 | -56.82 / -53.96 | ≤0.04 |
| DPA GMP (QR) | -54.02 / -51.08 | -54.02 / -51.07 | ≤0.01 |
| DPA GRU DPD | -51.93 / -49.97 | -51.92 / -50.02 | ≤0.05 |
| DPA MP (QR) | -51.29 / -50.26 | -51.29 / -50.26 | exact |
| Section 8 GMP-SGD-100ep | -29.67 / -44.46 | -29.63 / -44.46 | ≤0.04 |

(ACLR_AVG / EVM in dB. QR rows required the GMP fix in §2.2. The 0.3-0.5 dB
spread on the two APA SGD rows is normal run-to-run variation: training runs
under soft determinism with `cudnn.benchmark=True`.)

Also verified static: all parameter counts (1,911 / 2,066 / 519 / 524 / 500 /
510), dataset sizes, split ratios, sampling rates, `nperseg`, APA PAPR, and the
TRes-DeltaGRU architecture description.

### 1.2 Does not reproduce

| Item | Reported | Measured | Status |
|---|---|---|---|
| PA NMSE, APA_200MHz | **-43.52 dB** | **-31.48 dB** with the report's own recipe | wrong |
| DPA_160MHz PAPR | 10.38 dB | 10.13 dB (CCDF 0.001% and absolute peak) | wrong |
| §8 "10.6 dB EVM gap vs MP" | 10.6 | 16.4 vs MP; 10.6 is the gap vs GMP | misattributed |

---

## 2. Defects found

### 2.1 CRITICAL — `benchmark_report.md` §4 APA PA NMSE is unattainable

The report claims -43.52 dB. Three independent lines of evidence say the true
figure for that configuration is ≈ -31.5 dB, and that -43.52 dB cannot be reached
by any model on this dataset:

1. **Direct re-run.** GRU h23, 100 epochs, lr 5e-4 → -31.48 (val) / -31.40
   (test), fully converged (slope over the last 20 epochs: +0.0002 dB/epoch).
2. **Closed-form ceiling.** Least-squares GMP with 3,520 real parameters reaches
   only -38.83 dB on the APA test split. Closed-form identification is immune to
   optimizer problems, so this bounds what the data supports.
3. **The residual is measurement noise.** The GMP residual is near-white
   (in-band vs out-of-band PSD tilt +3.9 dB, against +33.4 dB for the signal)
   and its power is essentially independent of drive level — it stays at -48 dB
   even where `|x| → 0`. Unmodeled distortion vanishes with the drive; additive
   noise does not. **The measurement noise floor of APA_200MHz is ≈ -38.6 dB.**

-43.52 dB is ~5 dB below that floor. Alternative sources were checked and
excluded: `APA_200MHz_b` (-36.25), the OpenDPDv2.sh recipe with DGRU h23
(-39.13), the `DPA_200MHz` DGRU fixture (-31.0), and the OpenDPDv2 paper tables.

### 2.2 CRITICAL — `benchmark_volterra_qr.py` never ran GMP

`main()` called `build_mp_basis` unconditionally and never branched on
`args.model`, so `--model gmp` silently produced Memory Polynomial results. The
bug has been present since the script was first committed (`4924129`), meaning
the report's GMP rows cannot have come from the script as committed.

**Fixed in this branch**: `main()` now selects the basis from `args.model`, and
the printed coefficient count follows the GMP hyperparameters. With the fix, both
GMP rows reproduce to ≤0.05 dB.

### 2.3 HIGH — the whole APA benchmark rests on an undertrained surrogate

The report's recipe (lr 5e-4, batch 256, 100 epochs, no LR schedule) leaves
recurrent PA models ~7 dB short of the noise floor. Re-running the QR benchmarks
with a converged surrogate (GRU h23 at -38.87 dB) changes the downstream numbers
substantially:

| Model | Metric | Surrogate -31.4 dB | Surrogate -38.9 dB | Change |
|---|---|---|---|---|
| MP | ACLR_AVG | -41.02 | -41.81 | -0.8 |
| MP | EVM | -32.70 | **-47.41** | **-14.7** |
| GMP | ACLR_AVG | -38.75 | -39.58 | -0.8 |
| GMP | EVM | -38.50 | -34.89 | +3.6 |

EVM moves by up to 14.7 dB and the MP-vs-GMP ranking flips. §8's claim that AI
beats the best traditional method by 10.6 dB in EVM on APA_200MHz does not
survive: MP alone reaches -47.41 dB EVM against a converged surrogate. The
DPA_160MHz half of the report is unaffected — its surrogate (-38.43 dB) was
already at its noise floor, which is exactly why that half reproduces cleanly.

### 2.4 MEDIUM — reproduction commands are wrong or incomplete

- §9 QR commands omit the `benchmark/` path prefix; as written they fail.
- §5.1 advertises "aggressive temporal sparsity" but §9's commands leave
  `--thx/--thh` at their 0.0 defaults, i.e. a dense DeltaGRU is trained.
- `frame_length=200` / `frame_stride=1` are silent defaults, never stated.

### 2.5 LOW — metric semantics are non-standard and undocumented

- `EVM` is a spectral magnitude MAE (per-bin `mean|S_pred - S_gt| / mean|S_gt|`),
  not a 3GPP constellation EVM: no demodulation, equalization, or symbol
  decisions. Not comparable to instrument EVM.
- `ACLR` normalizes by the **maximum single sub-channel** power, not total
  main-channel power.
- `NMSE` averages per-segment dB values rather than taking dB of the pooled
  ratio.
- All Section 7 numbers are **simulated through the frozen PA surrogate**, not
  measured on hardware, and therefore inherit the surrogate's error.

These are self-consistent within the benchmark (all models scored identically),
but the report should say so.

### 2.6 LOW — checkpoints collide across recipes

`model_id` encodes only backbone / hidden size / frame length / parameter count.
Re-running the same architecture under a different recipe silently overwrites
`save/<dataset>/train_pa/<model_id>.pt` and its logs. The GRU h23 checkpoint on
disk is now the OpenDPDv2-recipe model (-38.87 dB), not the report-recipe one.

---

## 3. Dataset audit: APA_200MHz is clean

The low NMSE is **not** a data defect.

- **Git history**: only 4 commits ever touched `datasets/APA_200MHz/`; the CSVs
  have been bit-identical since 2025-07-09 and every later commit changed only
  `spec.json` metadata and plots. `fs`, `nperseg` and split ratios never changed.
- **Signal QA**: no NaN/Inf, no duplicate rows, no clipping (only the 1-2 samples
  defining the normalization peak), DC offset < 1e-3, AM/AM relative spread
  0.045 (DPA_160MHz: 0.048).
- **Splits**: consistent power and complex gain across train/val/test (1.1626 /
  1.1628 / 1.1636); no discontinuity at the boundaries.
- **Alignment**: already time-aligned (peak lag 0, normalized cross-correlation
  0.9945).
- **Provenance**: R&S SMW200A → 3.5 GHz Ampleon GaN Doherty PA → Keysight N9042B
  IQ capture (`Matlab/` scripts).
- **A vs B**: `APA_200MHz` and `APA_200MHz_b` have **byte-identical inputs** and
  different output captures. Commit `2c7f91a` swapped the two output sets four
  hours after release and renamed `APA_200MHz_backup` → `APA_200MHz_b`, so the
  current primary dataset is the capture the OpenDPDv2 paper reported on. NMSE
  between the two captures is -23.75 dB; since each is individually modelable to
  -38.5 dB, that gap is a systematic operating-point difference, not noise.

The closed-form ceiling is essentially the same for all three datasets
(APA -38.83, APA_b -38.48, DPA -39.51), so APA_200MHz is not a worse dataset.

---

## 4. Root cause of the -31.5 dB plateau: the training recipe

Backbone comparison at ~1,900 parameters on APA_200MHz (val / test NMSE, dB):

| Backbone | Params | Report recipe<br>lr 5e-4, bs 256, 100 ep | OpenDPDv2 recipe<br>lr 5e-3, bs 64, 240 ep, ReduceLROnPlateau |
|---|---|---|---|
| GRU h23 | 1,911 | -31.48 / -31.40 | **-38.79 / -38.87** |
| DGRU h23 | 2,751 | -31.67 / -31.61 | **-39.13 / -39.21** |
| TCN h66 | 1,914 | -37.44 / -37.30 | -38.38 / -38.25 |
| TRes-GRU h22 | 1,916 | -35.62 / -35.50 | **-38.94 / -39.01** |

The same GRU gains 7.4 dB from the recipe alone. Recurrent cells need the higher
learning rate and the schedule; the convolutional TCN is far less sensitive,
which is why it already reaches -37.4 dB under the weak recipe. With a proper
recipe all four architectures land within 0.9 dB of each other, because they are
all at the -38.6 dB noise floor — architecture ranking at that point is not
meaningful.

---

## 5. Fix plan

### 5.1 Applied in this branch

- [x] `benchmark/benchmark_volterra_qr.py`: branch on `--model`, so GMP actually
      builds the GMP basis (§2.2).
- [x] `backbones/tres_gru.py`: new TRes-GRU backbone (TRes-DeltaGRU with the
      DeltaGRU cell replaced by a dense `nn.GRU`, bias-free so parameter counts
      match at equal hidden size). Registered in `models.py` and in both backbone
      choice lists in `arguments.py`. Validated numerically equivalent to
      `tres_deltagru` at `thx=thh=0` (max abs diff 8.2e-7, relative L2 1.3e-6).
- [x] `benchmark/pa_modeling_investigation.md`: full write-up of the PA-modeling
      investigation.
- [x] This plan.

### 5.2 Pending — re-benchmark (the substantive work)

1. **Retrain the APA_200MHz PA surrogate with the OpenDPDv2 recipe.**
   ```bash
   python main.py --step train_pa --dataset_name APA_200MHz --PA_backbone gru --PA_hidden_size 23 \
     --n_epochs 240 --lr 5e-3 --lr_schedule 1 --lr_end 1e-4 --decay_factor 0.5 --patience 10 --batch_size 64
   ```
   Expected ≈ -38.8 dB. Do the same for DPA_160MHz for consistency (it is already
   at its floor, so little change is expected).

2. **Re-run all four APA DPD rows** (GRU h11, TRes-DeltaGRU h10, MP/QR, GMP/QR)
   against the retrained surrogate. Section 7's APA table and every Section 8
   conclusion derived from it must be recomputed — the EVM column in particular.

3. **Correct the report text**: §4 APA PA NMSE, §9 command paths, DPA PAPR
   (10.13 dB), the §8 EVM attribution, and add a note that Section 7 metrics are
   surrogate-simulated with the non-standard definitions listed in §2.5.

4. **Guard against regression**: make `model_id` encode the recipe (or at least
   warn on overwrite), and add a smoke test that `--model gmp` produces a
   different coefficient count than `--model mp`.

### 5.3 Pending — PA backbone comparison at a fixed budget (started, then stopped)

A sweep of every viable PA backbone at ~2,700 parameters under the OpenDPDv2
recipe was launched and **deliberately stopped before completion** at the user's
request. Hidden sizes were already calibrated:

| Backbone | h | Params | | Backbone | h | Params |
|---|---|---|---|---|---|---|
| GRU | 28 | 2,746 | | vDLSTM | 22 | 2,666 |
| DGRU | 23 | 2,751 | | DeltaGRU | 26 | 2,706 |
| TCN | 93 | 2,697 | | DeltaJANET | 32 | 2,626 |
| TRes-GRU | 27 | 2,751 | | PG-JANET | 19 | 2,719 |
| LSTM | 24 | 2,738 | | DVR-JANET | 19 | 2,665 |
| RVTDCNN | 68 | 2,684 | | QGRU | 27 | 2,729 |
| MCLDNN | 11 | 2,679 | | APNRRU | 34 | 2,723 |

Excluded: `gmp` (fixed at 495 params), `bojanet` (caps at 1,346; breaks above
h=18), `apnrnn` (not implemented in `models.py`). `tres_deltagru` is redundant
with `tres_gru` at zero thresholds.

Partial results before the stop (best val NMSE at the epoch reached, not final):

| Backbone | Params | Epochs run | Best val NMSE |
|---|---|---|---|
| LSTM h24 | 2,738 | 126/240 | -38.88 |
| GRU h28 | 2,746 | 126/240 | -38.85 |
| TCN h93 | 2,697 | 213/240 | -38.54 |
| MCLDNN h11 | 2,679 | 76/240 | -31.77 |
| RVTDCNN h68 | 2,684 | 240/240 | -31.13 |
| TRes-GRU h27 | 2,751 | 1/240 | -30.72 |

The runner script used is `benchmark/train_all_pa_2700.sh`. Resume by re-running
it; models already finished can be dropped from the `JOBS` list.

Note that at ~2,700 parameters the well-behaved architectures are already pinned
at the -38.6 dB noise floor, so this sweep characterizes *which architectures
reach the floor and how fast*, not which is most accurate. RVTDCNN and MCLDNN
stalling near -31 dB is the interesting signal and worth a separate look.
