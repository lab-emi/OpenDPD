# What `apply` does at a segment boundary: offline_segmented against streaming, measured after a PA

Status: **pre-registered on 2026-10-09; no result yet.** This page is committed before any segment-reset output is computed
after a PA. Everything in sections 1 to 9 is fixed; a change after the results are known is an amendment, dated and appended
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
