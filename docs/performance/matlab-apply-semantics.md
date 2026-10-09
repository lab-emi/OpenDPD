# What `apply` does at a segment boundary: offline_segmented against streaming, measured after a PA

Status: **pre-registered on 2026-10-09; run the same day; the rule gave R2 (the default of `apply` becomes `auto`), see section 10.** This page
was committed before any segment-reset output was computed after a PA. Everything in sections 1 to 9 is fixed; a change after the results are known is an amendment, dated and appended
to section 10, never an edit of those sections. The maintainer authorised the GPU time for Part B on 2026-10-09.

## 1. Question

`opendpd.apply` can run a trained DPD in two ways (`docs/architecture/streaming.md`, `opendpd/services/inference.py`):

* `offline_segmented`, how every Studio run is scored: the waveform is cut into segments of `SegmentSamples` (the run's frozen
  `nperseg`), every segment starts from a zero state, the last segment is zero padded and the padding is trimmed;
* `streaming`, one state for the whole waveform and no reset anywhere, which is what a deployed DPD does. It exists only for
  models with a registered streaming variant (`gru` as `gru_stream`, `gmp` as `gmp_stream`).

A waveform that is going to hardware is produced by the first by default today. After a PA that was linearised to an EVM and
an ACLR near -50 dB, a state reset every L samples could matter, and `SegmentSamples` is a number the user chooses (Studio's
generated signals use 512 to 4096; the first toolbox default was 256). The 2.4 review (finding A3) therefore asked, before any
default is changed: how much do EVM and ACLR after the PA differ between the two, as a function of L, and which one should
`apply` use by default for a waveform that will be played?

## 2. The executions compared

| Name | Definition | Code |
| --- | --- | --- |
| `S(L)` | `offline_segmented` with `SegmentSamples = L`: disjoint segments, zero state at the start of each, last segment zero padded and trimmed | `opendpd/services/inference.py::_offline` |
| `T` | `streaming`: one state from the first sample, chunks of 200 samples (the output does not depend on the chunk size, `docs/architecture/streaming.md` section 4) | `opendpd/services/streaming.py::stream_outputs` |
| `O` | the Arena's offline semantics: overlapping 200-sample windows, the middle 100 samples kept. Context only; not a candidate | `opendpd/core/arena_engine.py::offline_output` |

`S(L)` equals `T` when L is at least the length of the waveform (one segment, one zero start), which is checked (section 7).

## 3. Material

### Part A: reference models, independent PA, the Arena's metrics

* Condition `apa-200mhz-b` of DPD Arena v6 (`docs/protocols/dpd-arena-v6.md`): the frozen TRes-GRU PA
  (`opendpd/arena_assets/apa-200mhz-b-pa-v6.npz`, 4,871 parameters) run with continuous state over the 66,270-sample test
  record, the transmitter peak limiter applied to u as in the Arena, the Arena's metric interval, symbol EVM and worse-side
  ACLR (`evm-aclr-v3`, `opendpd-spectral-v2`), and the quality
  `q = 0.5 (EVM_baseline - EVM_dpd) + 0.5 (ACLR_worse_baseline - ACLR_worse_dpd)` against the common linear baseline.
* DPDs: **every sealed GRU checkpoint** of the archived v6 APA B matrix (hidden size 7, 10, 16 and 23, seeds 0, 1, 2: 12
  checkpoints) and **every sealed GMP checkpoint** (seeds 0, 1, 2: 3). Nothing is trained, selected or tuned; the SHA-256 of
  each weight file is recorded in the run log.
* Segment lengths L: 64, 128, 256, 512, 1024, 2048, 4096, 8192 and 19,662 (the dataset's own `nperseg`).

### Part B: the Studio path, fresh models on two more datasets

* Datasets: the built-in `DPA_200MHz` (10 carriers of LTE 20 MHz at 800 MS/s, `nperseg` 2,560, test split 7,680 samples) and
  `APA_200MHz` (5 carriers of LTE 20 MHz at 983.04 MS/s, `nperseg` 19,662, test split 19,662 samples), imported with the
  built-in import of a new workspace, original 60/20/20 splits.
* PA: one `gru` PA surrogate per dataset (hidden size 23), trained with the SDK defaults (`TrainingConfig`: 150 epochs, batch 16,
  frames of 200 samples at stride 1, AdamW 5e-3 to 5e-5, seed 0). DPD: `gru` (hidden size 23, one layer) trained through it with
  the SDK defaults and seeds 0, 1 and 2. No setting is changed from the defaults, so the models are what a MATLAB user gets from
  `trainPA` and `trainDPD` without options. `gmp` is not run in Part B: its gradient-trained form takes 45 s to 2 min per epoch here and
  linearises by only 0.1 dB in the Arena (Part A has it).
* Input: the continuous test split exactly as the run was scored on (dataset units). The PA surrogate is applied with continuous
  state (primary, as the Arena does) and, as a sensitivity, with `offline_segmented` at the dataset's `nperseg`.
* Metrics: `opendpd-spectral-v2` at the dataset's own `nperseg` for every L (so that the PSD resolution does not change with L):
  in-band error (IBE), worse-side ACLR, NMSE. There is no demodulated EVM in Part B (no symbol grid is registered for these
  splits), and the quality is `q_spec = 0.5 (IBE gain) + 0.5 (ACLR_worse gain)`, the objective the Arena uses to select
  checkpoints.
* Segment lengths L: DPA 64, 128, 256, 512, 1024, 2560 (native) and 7680 (the whole record); APA 64, 128, 256, 512, 1024, 2048,
  4096 and 19,662 (native, the whole record).

## 4. Procedure

1. The harness is written and the validity checks of section 7 pass. Those checks use `O`, `T` and the identity `S(N) = T`; no
   `S(L)` with L below the length of the record is scored before this page is committed.
2. Part A: for every checkpoint and every execution (`O`, `T`, `S(L)`): DPD output, limiter, frozen PA, metrics, quality.
3. Part B: import the datasets, train the PAs and DPDs once, and for every DPD run every execution as above.
4. One run per part. If a run fails or a check of section 7 fails, it is repeated whole and the failure is logged in section 10;
   nothing in sections 1 to 9 changes after the first result is read.
5. The record (`docs/performance/matlab-apply-semantics.json`) holds every number below; this page holds the tables.

## 5. What is measured and how it is summarised

For a model m and a segment length L, `dq(m, L) = q_T(m) - q_S(m, L)` in dB: positive means that streaming is better. The
components are reported separately (`dEVM` and `dACLR` in Part A, `dIBE` and `dACLR` in Part B, each as the segmented value minus
the streaming value, so positive again means that streaming is better), together with the difference in output power between
`S(L)` and `T`. For each backbone, dataset and L the summary is the mean, minimum and maximum of `dq` over the models of that
backbone: the 12 GRU or 3 GMP checkpoints of Part A, the 3 seeds of Part B. Also reported: `q_T` itself, and `q_T - q_O` (what the
Arena already published for these weights), and for Part B the fidelity of the PA surrogate (NMSE of its continuous and its
segmented output against the measured output of the test split).

## 6. Decision rule

* `|dq| < 0.1 dB` is negligible and `|dq| >= 0.5 dB` is material (a sixth of the 3 dB that halving both costs is worth in the
  Arena's figure of merit). L is *typical* when it is one of 512, 1024, 2048, 4096 (the range Studio's generator uses).
* **R1.** If the mean `dq` of any (backbone, dataset) is `<= -0.5 dB` at any typical L (segmenting is materially better),
  the default stays `offline_segmented`.
* **R2.** Otherwise, if the mean `dq` of at least one (backbone, dataset) is `>= +0.5 dB` at some typical L, the default of
  `apply` (the `Execution` of `Job.apply` and `opendpd.apply`, both the job and the model-package form) becomes `auto`: streaming
  for a model with a registered streaming variant, `offline_segmented` for every other model, the choice and its reason
  recorded in the metadata. Both explicit values stay available and unchanged.
* **R3.** Otherwise (`|dq| < 0.5 dB` everywhere at typical L) the default stays `offline_segmented`.
* Whatever the outcome, the user-facing documentation gets the measured table, including L = 256 (the old default) and the
  native L, and says in which range of L the choice matters.

## 7. Validity checks (the run is void if one fails)

* **V1 (Part A).** `O` and `T` reproduce the published case metrics of `reference-results-v6-apa-b.json` for all 15
  checkpoints within 0.01 dB (every metric and q).
* **V2.** `S(L)` with L at least the length of the record equals `T` at waveform level within 1e-5, for every model.
* **V3 (Part B).** `Job.apply(x_test, execution="offline_segmented")` equals the harness's `S(nperseg)` and
  `Job.apply(x_test, execution="streaming")` equals `T`, within 1e-5, for every DPD run.
* **V4 (Part B).** The harness's `S(nperseg)` followed by the PA surrogate at `nperseg` reproduces the metrics stored in the run's
  own result (`NMSE`, `IBE`, `ACLR_L`, `ACLR_R`) within 0.01 dB.

## 8. Expectations (not criteria)

Written down so that the result can contradict them. The reset costs the first samples of every segment: the GRU's warm-up is 32
samples and the GMP's 20 (`docs/releases/streaming-semantics-report.md`), so `dq` should fall roughly as 1/L, be several dB at
L = 256 for the better-linearised GRUs, and be below 0.5 dB from L = 2048. The GMP, which linearises by 0.1 dB, should show
nothing. `q_T - q_O` should stay within 0.01 dB (published: 6e-5 dB for the GRU). A PA surrogate that was fitted on frames may be
less faithful with continuous state than with segments; that is reported, not corrected.

## 9. What this cannot show

It is model inference through PA models, not a hardware measurement, and says nothing about a PA that is not the surrogate.
It covers two backbones (`gru`, `gmp`: the only ones with a streaming variant; `tres_gru`, `mp_ls` and `gmp_ls` can only be
`offline_segmented` and do not enter the decision), three datasets (two measured captures of APA, one of DPA), and the training
recipes of section 3. The streaming variants stay `experimental` in the registry; promoting them to `supported` is a maintainer
decision that this page does not take, and an `auto` default would change only what the MATLAB toolbox and the SDK ask for.

## 10. Run log and amendments

Entries are appended in the order they happened. Nothing above this heading is edited after the first result is read.

### 2026-10-09: Part A scored; Part B training stopped by a platform rule (amendment 1, procedure only)

Part A was run as registered (commit `03a0c3a`) and scored. Part B training then stopped at the second DPD with the platform's own
error: *the legacy checkpoint convention requires the DPD run to use the same seed and frame_length as its PA surrogate (PA: seed=0,
frame_length=200)*. A DPD of seed 1 or 2 cannot be trained through the seed-0 PA that section 3 prescribes.

**Amendment 1.** Part B trains **three PA/DPD pairs per dataset: PA seed s with DPD seed s, s = 0, 1, 2**, everything else as
registered (SDK defaults, `gru`, hidden size 23). The seed-0 pair of `DPA_200MHz` is the one already trained before this
amendment (PA `run-20261009-082222-67f908`, 66 s; DPD `run-20261009-082328-e56c2d`, 101 s) and is kept; no S(L) output of Part B
had been computed. Section 3 ("one PA per dataset") and section 5 ("the 3 seeds") are read as "the 3 pairs"; the PA fidelity
diagnostic is reported per pair. The cause is the platform error above and not any Part A number, which did not enter the
decision to amend; the thresholds, lengths, decision rule and checks are unchanged.

### 2026-10-09: validity check V4 could not have passed on a GPU-trained run (amendment 2, check only)

Before the final Part B analysis, the analyser was debugged on the two DPA pairs finished at that point, printing the validity
checks only (no quality was printed; the debug record was never opened). V2 (1.2e-7) and V3 (1.2e-7 for both executions)
passed. **V4 did not: 0.058 dB and 0.049 dB** against the metrics stored in the runs' results, where 0.01 dB was registered.
A read-only probe on pair 0 (no SDK service) found the cause. The array shape (flat or segmented) and the metric code make no
difference (identical to four decimals). The evaluator's own `predict_test_split` run on the CPU reproduces the harness
**exactly** (largest difference 0.0 in both u and the PA output, identical metrics). What differs is the stored result: the
evaluator ran it on the GPU, where cuDNN uses TF32 (`torch.backends.cudnn.allow_tf32` is True by default), and its PA output
differs from the CPU computation of the same weights by up to 1.25e-3 in amplitude (3.3e-4 in u), which moves the stored NMSE by
0.05 dB (-44.40 stored, -44.45 on the CPU) and IBE by 0.06 dB. No CPU computation can reproduce a GPU-stored number to 0.01 dB,
so V4 as registered could not pass for a run trained on the GPU, which is what the SDK defaults give.

**Amendment 2.** V4 compares the harness with the evaluator's own pipeline (`predict_test_split`, the run's resolved
configuration with `device` set to `cpu`): NMSE, IBE, ACLR_L and ACLR_R within 0.01 dB. The difference to the metrics stored by
the GPU evaluation is reported for every run as information and is not a criterion. Nothing else changes: the Part B quality
is a difference between two executions of the same weights on the same device, so the GPU floor cancels, and both executions
are computed on the CPU throughout, as `Job.apply` does.

Also logged: the same debug run closed the project with `stop_service=True` while the first APA PA (`run-20261009-083154-885383`)
was training in that workspace, which cancelled that run. It is not part of the registered set; the training was restarted from
its manifest and trains that PA again.


### 2026-10-09: results and decision

**What was run.** Part A exactly as registered (commit `03a0c3a`): the 12 sealed GRU and 3 sealed GMP checkpoints of the archived v6
APA B matrix (the SHA-256 of every weight file is in the record), each as `O`, `T` and nine `S(L)`, judged with the Arena's own
functions on the CUDA device (torch 2.13.0+cu132, RTX 4090 Laptop); 35 s. Part B as registered and amended: three PA/DPD pairs
on each of `DPA_200MHz` and `APA_200MHz`, trained with the SDK defaults on the GPU (DPA: PA 65 to 66 s, DPD about 100 s each; APA: PA 164 to
216 s, DPD 253 to 317 s; early stopping ended the runs at the epochs listed in the record), then analysed on the CPU; 43 s. The
record `matlab-apply-semantics.json` holds every number: the metrics of every model under every execution, run IDs, checkpoint
and dataset hashes, the resolved training configurations and the stored GPU metrics. The harness (four short Python scripts) is
not in the repository: Part A reads a local archive of the sealed checkpoints that is not part of it, and the record is the
evidence. The Part B analysis ran on the working tree of `5148260` with the `auto` change below not yet committed; it calls only
explicit executions, which that change does not touch.

**Validity checks.** All pass.

| Check | Limit | Measured |
| --- | --- | --- |
| V1: `O` and `T` reproduce the published Arena cases (15 checkpoints) | 0.01 dB | `O` 3.0e-4 dB at most, `T` 0.0 |
| V2: `S(L)` with L at least the record equals `T` | 1e-5 | 1.8e-7 (Part A at L = 66,270; Part B) |
| V3: `Job.apply` equals the harness, both executions (6 runs) | 1e-5 | 1.5e-7 offline, 1.8e-7 streaming |
| V4 (amended): harness equals the evaluator on the CPU (6 runs) | 0.01 dB | 1.1e-5 dB at most |
| (information) harness against the metrics the GPU evaluation stored | | 0.058, 0.049, 0.019 dB on DPA; 0.001 dB or less on APA |

#### Part A: the 12 sealed GRU and 3 sealed GMP checkpoints of Arena v6 APA B

`dq` is `q_T - q_S(L)` in dB, so positive means that streaming is better; the columns after it are segmented minus streaming for
EVM (`dEVM`) and worse-side ACLR (`dACLR`); "u against streaming u" is the rms of `u_S - u_T` relative to the rms of `u_T`.

**Part A, apa-200mhz-b, gru (12 checkpoints)** — streaming `q_T` 21.94 dB (models: 14.75 to 27.39)

| L | dq mean (min to max) | dEVM | dACLR | output power, dB | u against streaming u, dB rms |
|---|---|---|---|---|---|
| 64 | 20.68 (18.04 to 23.66) | 20.83 | 20.54 | +0.059 | -21.6 |
| 128 | 17.67 (15.01 to 20.53) | 18.25 | 17.08 | +0.030 | -24.5 |
| 256 | 14.76 (12.19 to 17.82) | 15.35 | 14.18 | +0.014 | -27.6 |
| 512 | 12.06 (9.51 to 15.10) | 12.67 | 11.46 | +0.007 | -30.5 |
| 1024 | 9.50 (7.13 to 12.43) | 10.02 | 8.99 | +0.003 | -33.5 |
| 2048 | 7.01 (5.14 to 9.64) | 7.50 | 6.51 | +0.002 | -36.3 |
| 4096 | 4.84 (3.33 to 6.86) | 5.28 | 4.40 | +0.001 | -39.3 |
| 8192 | 2.94 (1.80 to 4.60) | 3.29 | 2.59 | +0.000 | -42.4 |
| 19662 | 1.23 (0.54 to 2.23) | 0.96 | 1.50 | +0.000 | -48.4 |

`q_T - q_O` over these checkpoints: largest absolute value 0.00090 dB.

**Part A, apa-200mhz-b, gmp (3 checkpoints)** — streaming `q_T` 0.14 dB (models: 0.07 to 0.24)

| L | dq mean (min to max) | dEVM | dACLR | output power, dB | u against streaming u, dB rms |
|---|---|---|---|---|---|
| 64 | 0.73 (0.70 to 0.78) | 0.14 | 1.32 | -0.003 | -28.4 |
| 128 | 0.41 (0.39 to 0.44) | 0.08 | 0.75 | -0.002 | -31.2 |
| 256 | 0.20 (0.19 to 0.22) | 0.04 | 0.36 | -0.001 | -34.6 |
| 512 | 0.12 (0.11 to 0.13) | 0.02 | 0.22 | -0.001 | -37.4 |
| 1024 | 0.07 (0.07 to 0.07) | 0.01 | 0.13 | -0.000 | -40.1 |
| 2048 | 0.04 (0.04 to 0.04) | 0.01 | 0.08 | -0.000 | -42.4 |
| 4096 | 0.02 (0.02 to 0.02) | 0.00 | 0.04 | -0.000 | -45.5 |
| 8192 | 0.01 (0.01 to 0.01) | 0.00 | 0.02 | -0.000 | -49.0 |
| 19662 | -0.00 (-0.01 to -0.00) | 0.00 | -0.01 | +0.000 | -56.8 |

`q_T - q_O` over these checkpoints: largest absolute value 0.00000 dB.

`dq` hardly depends on the size of the GRU: at L = 1024 the mean over the three seeds is 8.5 dB for hidden size 7, 10.1 for 10,
9.7 for 16 and 9.7 for 23.

#### Part B: fresh GRU pairs from the SDK defaults (spectral quality `q_spec`, no EVM)

**Part B, dpa-200mhz, gru, PA with continuous state (3 pairs)** — streaming `q_T` 39.16 dB (models: 38.31 to 40.47)

| L | dq mean (min to max) | dIBE | dACLR | output power, dB | u against streaming u, dB rms |
|---|---|---|---|---|---|
| 64 | 5.89 (5.09 to 7.11) | 11.27 | 0.51 | -0.003 | -35.1 |
| 128 | 4.47 (3.72 to 5.64) | 8.55 | 0.40 | -0.000 | -38.2 |
| 256 | 3.47 (2.79 to 4.54) | 6.66 | 0.27 | -0.000 | -40.5 |
| 512 | 2.17 (1.59 to 3.06) | 4.18 | 0.17 | -0.001 | -43.8 |
| 1024 | 0.76 (0.48 to 1.21) | 1.55 | -0.02 | +0.000 | -49.7 |
| 2560 | 0.25 (0.17 to 0.37) | 0.38 | 0.11 | -0.000 | -54.9 |
| 7680 | -0.00 (-0.00 to 0.00) | -0.00 | -0.00 | -0.000 | exact |

**Part B, apa-200mhz, gru, PA with continuous state (3 pairs)** — streaming `q_T` 57.78 dB (models: 57.66 to 57.91)

| L | dq mean (min to max) | dIBE | dACLR | output power, dB | u against streaming u, dB rms |
|---|---|---|---|---|---|
| 64 | 18.77 (18.62 to 18.95) | 20.10 | 17.45 | +0.000 | -35.2 |
| 128 | 16.15 (16.06 to 16.22) | 17.30 | 14.99 | -0.001 | -38.0 |
| 256 | 12.83 (12.81 to 12.85) | 13.96 | 11.70 | -0.000 | -41.0 |
| 512 | 11.04 (11.00 to 11.08) | 11.91 | 10.17 | +0.000 | -43.3 |
| 1024 | 7.07 (6.96 to 7.16) | 7.66 | 6.47 | -0.000 | -47.7 |
| 2048 | 4.59 (4.53 to 4.67) | 5.32 | 3.86 | -0.000 | -51.1 |
| 4096 | 3.55 (3.35 to 3.92) | 4.45 | 2.65 | -0.000 | -54.1 |
| 19662 | 0.00 (0.00 to 0.00) | 0.00 | 0.00 | -0.000 | exact |

Sensitivity, the PA surrogate run with `offline_segmented` at the dataset's `nperseg` instead of with continuous state: on
`APA_200MHz` the test split is one segment, so the two are identical; on `DPA_200MHz` mean `dq` changes by 0.12 to 0.39 dB at
each L below the whole record (L = 2560: +0.25 dB with a continuous PA, -0.14 dB with a segmented one; L = 512: 2.17 against 1.90), and the pattern is the
same. The full table is in the record.

Fidelity of the PA surrogates against the measured output of the test split (NMSE, dB): continuous state -35.13, -35.15, -35.15 on
DPA and -38.85, -38.87, -38.89 on APA; segmented at `nperseg` -35.18, -35.20, -35.20 and -38.85, -38.87, -38.89. A surrogate run
with continuous state is not less faithful (at most 0.05 dB different).

#### Decision

The rule of section 6 was applied to the records by a script, without judgement:

| Group | Mean `dq` at the typical lengths present in its length set |
| --- | --- |
| Part A, GMP | 0.12 (512), 0.07 (1024), 0.04 (2048), 0.02 (4096) |
| Part A, GRU | 12.06 (512), 9.50 (1024), 7.01 (2048), 4.84 (4096) |
| Part B, `DPA_200MHz`, GRU | 2.17 (512), 0.76 (1024) |
| Part B, `APA_200MHz`, GRU | 11.04 (512), 7.07 (1024), 4.59 (2048), 3.55 (4096) |

**R1 is not triggered**: no group has a mean `dq` of -0.5 dB or less at any typical length (the smallest value is +0.02 dB, GMP at
4096). **R2 is triggered**: the GRU groups of Part A and of `APA_200MHz` exceed +0.5 dB at every typical length, and `DPA_200MHz`
at 512 and 1024. The default of `apply` therefore becomes **`auto`**: `streaming_stateful` for a model with a registered streaming
variant (`gru`, `gmp`), `offline_segmented` for every other model, in `Job.apply` of the SDK and in `opendpd.apply` of the MATLAB
toolbox for jobs and for model packages, with the choice and its reason in the metadata. The explicit values are unchanged.

What the data say beyond the rule: the effect depends on the segment length *relative to the waveform*, not on the dataset. The
Studio test splits are one segment long (`APA_200MHz`, `dq` exactly 0 at its native `nperseg` of 19,662) or three (`DPA_200MHz`,
0.25 dB at 2,560), so the scores Studio stores for a run do not show it; a waveform of many segments does. For the GMP the
difference is at most 0.12 dB at the typical lengths, on DPDs that improve the PA by only 0.14 dB in total.

#### What did not match the expectations of section 8

* Expected `dq` below 0.5 dB from L = 2048. Measured: 7.0 dB (Arena GRUs) and 4.6 dB (`APA_200MHz`) at 2048; only
  `DPA_200MHz` is below 0.5 dB, and only from its native 2560. Even at the dataset's own `nperseg` of 19,662 the Arena GRUs lose
  1.2 dB on average (0.5 to 2.2) over the 66,270-sample record (four segments).
* Expected `dq` to fall roughly as 1/L. It falls by about 3 dB per doubling of L at short L (20.7, 17.7, 14.8, 12.1 dB for
  64 to 512 in Part A), as a 1/L law in power would give, but from a far larger starting point than I assumed. The direct
  measure is the DPD output itself: at L = 2048 the Arena GRUs' `u` differs from the streaming `u` by -36 dB rms, 10 dB above
  the -45.5 dB EVM they reach when streaming (with resets every 2048 samples their mean EVM is -38.0 dB and their worse-side ACLR
  -40.5 dBc instead of -47.0 dBc); the Studio-trained GRUs differ by -51 dB (APA at 2048) and -55 dB (DPA at 2560) against
  streaming IBE of -56.5 and -47.5 dB. This is a reading of the table, not a separate test.
* Expected the GMP to show nothing. Confirmed: at most 0.12 dB at the typical lengths.
* Expected `q_T - q_O` within 0.01 dB. Confirmed: 0.0009 dB at most (GRU), 0.0000 (GMP).
* Cautioned that a PA surrogate fitted on frames may be less faithful with continuous state. It is not (at most 0.05 dB).

#### Implemented, and what is not changed

`Job.apply` (SDK) and `opendpd.apply` (MATLAB: job, `opendpd.Model`) default to `auto`; `apply_waveform` accepts `auto`, resolves
it from the registered streaming variant of the run's model, and returns `execution` (what ran), `execution_requested` and, for
`auto`, `execution_reason`. A `FixedModel` accepts `auto` and runs its one execution. Tests that depend on the scored form name it
explicitly, and new tests pin `auto` for every supported model, both roles, in Python and in MATLAB.

Not changed, deliberately: the registry and the `experimental` status of the streaming variants (promoting them is a maintainer
decision, and `auto` is a default of two clients, not a claim about them); `apply_waveform`'s own default of
`offline_segmented`, which the exported golden vectors rely on; the default of `opendpd.generateCode`, which is how a block is fed;
Studio's API, scoring and screens; model causality; every threshold and metric.

#### Limits

Two backbones, three datasets, one PA model family (a GRU surrogate fitted to the capture), SDK-default training, three seeds
per Studio dataset and the sealed Arena checkpoints: it shows the size of the effect for GRU DPDs of this kind and says
nothing about other architectures, other training recipes or a real PA. The `auto` choice is for a waveform that will be played;
reproducing the score stored with a run needs `Execution="offline_segmented"`.
