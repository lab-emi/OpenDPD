# DPD Arena v6 APA B: full-window training on APA_200MHz_b

Arena compares DPD architectures and sizes through one fixed TRes-GRU PA per
condition. The PA used for training is also the PA used for validation and test.
GRU and GMP remain DPD contestants; there are no additional learned PA judges.
The published Arena uses only **APA_200MHz_b** (200 MHz, 256-QAM), with one measured-source PA condition. Its 23 model entries and 233 test cases retain the original completed v6 observations and scores.
Synthetic PA datasets and ILC-based entries do not enter Arena rankings. PA and DPD train,
validation and test inputs are all unchanged samples from the original measured
captures. Cascade outputs are predictions through the frozen PA, not new
predistorted hardware measurements.

## PA identification and selection

PA and DPD use the existing disjoint measured partitions. Train three freshly
initialized H37 TRes-GRU candidates (4,871 real parameters), with seeds 0, 1
and 2. Each candidate receives 240 full passes over every 200-sample window at
stride 1, shuffled without replacement, batch 64 including the final partial
batch. AdamW uses initial learning rate 0.005 and weight decay 0.01. Minimum
whole-validation-record NMSE selects the checkpoint. A plateau scheduler
halves the rate after 10 unimproved epochs (0.01 dB absolute threshold), to a
floor of 0.0001. Record the validation improvement over the last 30 epochs.

Older APA PAs trained on a different partition remain ineligible because they saw samples belonging to the current test set. All PA
candidates and seed selections are frozen before loading any test arrays.
Test NMSE is reported after selection and never changes the chosen PA. The
validation residual spectrum is reported as an additional diagnostic.

Freeze one selected PA for all DPD architectures on each dataset. Rank displays
its parameters, validation/test NMSE and checkpoint hash. PA cost is excluded
from DPD FoM. See [PA qualification](../performance/arena-pa-qualification.md).

## Original measured inputs and partitions

Every saved `x_train`, `x_val` and `x_test` is an unchanged float32 slice of
the original capture; `y` retains its paired measured PA output. No generated
waveform, duplicated data, padding or resampling supplies missing samples.
The independent audit compares every split with the original capture arrays
and verifies nonoverlap, source bounds and file hashes. The protocol metadata
is in `calibration-v6-apa-b.json`.

APA_200MHz_b contains independently timed LTE carriers. Its original 19,662-sample validation/test records are
shorter than the 32,768-sample useful symbol, so they cannot support complete
symbol EVM. Repartition the original capture before PA or DPD fitting.

Use only the original input to find CP correlations independently per carrier,
with the existing fixed kurtosis timing refinement. Reserve the last complete
symbol per carrier that leaves the fixed 200-sample end context. Test begins
200 samples before the earliest of those symbols. Split the preceding samples
80/20 into training and validation, with 200 unused samples at each boundary.
No PA output or DPD outcome determines timing or boundaries.

| Dataset | Train samples | Validation samples | Test samples | Complete scored test symbols |
|---|---:|---:|---:|---|
| APA_200MHz_b | 25,307 | 6,327 | 66,270 | 1 independently timed symbol per carrier, 32,768 samples each |

APA bounds are `[0,25307)`, `[25507,31834)` and `[32034,98304)`.
The fixed metric context is 200 original samples at both ends of the test
record. Spectral metrics and the power gate use the middle interval; EVM
uses complete symbols within it. Receiver channel isolation uses the full
held-out record including its original context. No test data are borrowed
from training or validation to complete an EVM symbol.

PA inference carries continuous recurrent state through each record, including
internal chunks needed by cuDNN. PA qualification compares the full held-out
measured output from zero initial state; DPD metrics use the predefined context.


## Training and the common transmitter envelope

The sweep samples at most 250, 500, 1,000 and 2,000 real parameters. Each
distinct configuration is trained once, stored at its smallest fitting budget,
and can enter every larger budget ranking. Hidden-size models use their largest
registered size within each sampling budget; polynomial orders remain fixed
before scores are observed. Templates retain their 4,096-parameter admission
limit. Missing configurations are unavailable, not zero-valued observations.

Each neural configuration receives seeds 0, 1 and 2 and **240 complete training
passes**, using every valid 200-sample window at stride 1. A seeded NumPy
permutation chooses the order each epoch; all backbones share the same orders
for a dataset and seed. Batch size is 64, and the final partial batch is never
dropped. The order hash is SHA-256 of the concatenated epoch permutations as
little-endian signed 64-bit integers.

| Dataset | Windows per epoch | Updates per seed | Window exposures per seed |
|---|---:|---:|---:|
| APA_200MHz_b | 25,108 | 94,320 | 6,025,920 |

The cascade training loss remains MSE on the central 100 samples of each frame.
AdamW uses learning rate 0.005, weight decay 0.01 and gradient clipping at 200.
Validation occurs every epoch, on the original held-out validation input. Its
spectral estimator uses 4,096-sample Hann/Welch segments, 50% overlap and a
200-sample context excluded at each end. Select the checkpoint with feasible
validation output power first, then minimum
`0.5 * (in-band error dB + max(ACLR_L, ACLR_R))`. A tie keeps the earlier
checkpoint. The scheduler uses this same spectral objective, factor 0.5,
patience 10 epochs, 0.01 dB absolute threshold and minimum learning rate 0.0001.
This validation in-band error is explicitly not demodulated EVM; only complete
held-out test symbols supply the EVM entering final FoM.

Training workers load only train/validation arrays. All checkpoints in a
submission are frozen before test inputs are loaded; the official matrix
prefetches every configuration before beginning its test phase. Validation
metrics never enter final FoM or rank. Training state is saved each epoch so
interrupted local work can resume the same optimizer, scheduler, window order
and RNG state, with no extra or omitted updates. Resume state is bound to the
exact configuration, data, PA, training sources, seed and execution device.

Reviewed CUDA models replay forward/backward and clipping while AdamW remains
eager. Zero-threshold Delta cells and BOJANET use algebraically equivalent dense
GRU training operations; PGJANET, DVRJANET and APNRRU may compile their existing
forward on CUDA. These paths retain FP32 parameters, gradients and optimizer
state, with numerical parity tests. TF32 is disabled for DPD training and test
on both CUDA hosts. Validation, exported weights, test inference
and operation counts retain the original backbone. Optional seed concurrency
uses private models, PAs, optimizers, permutations and CUDA streams inside one
GPU process; all initialization and compilation finish before updates start.

The longer fixed budget addresses the v5 undertraining finding. A 240-epoch
budget does not by itself prove every architecture is fully converged. These
remain reproducible fixed-recipe results, rather than architecture optima or
a fresh blind holdout after protocol development.

The gain target and radial PA-input peak limit come from the measured
training partition. A common linear-gain baseline is calibrated using that
training input only. Every seed and condition must keep output power within
±0.5 dB of the fixed target; output is never renormalized after prediction.

MP/GMP least-squares fits use indirect learning on the training capture only:
regression inputs are `PA(x_train)/G`, and targets are `x_train`. They do not
consume ILC-generated data. Their orders and regularization are fixed presets;
validation/test samples do not fit coefficients. The gradient-trained GMP uses
the neural train/validation workflow. Deterministic methods run once and do not
receive invented seed variability.

ILC, including the former offline ILC→MP entry, is excluded from the Arena
catalogue, submissions, rankings and Pareto fronts. The historical ILC→MP path
used only the first 16,384 training samples and then deployed a fixed MP; it did
not iterate ILC on test data. That distinction does not make it an Arena entry
under this protocol. Studio's separate ILC workflow remains available. See the
[data-flow audit](../performance/arena-data-isolation.md).

Offline DPD uses overlapping 200-sample windows and retains their middle
100 samples. Stateful variants reuse the base weights but carry their state
and are evaluated and ranked separately. Noncausal context in offline models
does not imply a causal deployment.

## EVM and ACLR

For EVM, take each complete useful OFDM symbol, remove its CP, FFT it on the
declared grid and keep only the published occupied bins. For APA, first
frequency-shift and isolate each carrier independently, then use its frozen
input-synchronized FFT start and all 1,200 occupied subcarriers (±600, excluding
DC). This covers the full LTE 20 MHz allocation before the capture-rate doubling,
not the older constellation preview’s central-600 subset. No per-subcarrier fit
or output-dependent timing is used. Compare the PA output
`Y` with the known reference `R = Gx`, removing one common complex gain:

```text
c = sum(conj(R) Y) / sum(abs(R)^2)
EVM = sqrt(sum(abs(Y-cR)^2) / sum(abs(cR)^2))
EVM_dB = 20 log10(EVM); EVM_percent = 100 EVM
```

This is a reference-aided engineering EVM, not a complete LTE conformance receiver.
The standard receiver reference is after CP removal and FFT ([ETSI TS 136 104,
Annex E](https://www.etsi.org/deliver/etsi_ts/136100_136199/136104/13.04.00_60/ts_136104v130400p.pdf)).
No full-record band-error fallback is allowed. Missing symbol metadata or a
record without a complete symbol raises an error. In particular, Parseval's
identity does not make the previous APA band-error proxy equivalent to
symbol-demodulated EVM.

ACLR is measured from the cascade output itself, using the fixed
`opendpd-spectral-v2` Welch estimator and carrier bands:

```text
ACLR_L/R = 10 log10(P_adjacent_L/R(y) / strongest in-band carrier power(y))
ACLR_worse = max(ACLR_L, ACLR_R)
```

The convention is negative dBc, lower is better. This is the OpenDPD carrier
normalization, not a standards-conformance certification. Raw left/right,
baseline and reference ACLR remain available. The adjacent-error ratio (AER)
uses `y-Gx`; it remains a separately named diagnostic and never substitutes
for output ACLR in selection, scoring or plots. NMSE is also diagnostic.

## Figure of merit

For each configuration with actual DPD parameter count `P` and arithmetic
count `A`, define the following relative to its common no-DPD baseline:

```text
q_seed,condition = 0.5 (EVM_baseline_dB − EVM_DPD_dB)
                 + 0.5 (ACLR_worse_baseline − ACLR_worse_DPD)
q_seed = q_seed,condition  # one measured dataset per board
Q = mean_seed(q_seed)
FoM = Q − 5 log10(P/1000) − 5 log10(A/2000)
```

Every size uses fixed references of 1,000 parameters and 2,000 operations.
Halving both costs with unchanged EVM and ACLR increases FoM by 3.0103 dB.
The equal quality/cost weights are an explicit policy choice. FoM is not a
physical RF efficiency, latency, energy or silicon-area measurement.

All final performance metrics, quality gates, FoM and ranking observations come
from the held-out test split. Parameters and operation counts are properties
of the frozen DPD; validation diagnostics never contribute to the final score.

Report sample standard deviation across seeds separately. The mean-minus-SD
quantity remains a diagnostic, not the main score. Positive mean quality and
the output-power envelope are required to rank; no 3 dB threshold applies.
Retain negative FoM for otherwise valid configurations. Never turn missing
sizes or invalid configurations into zero-score measurements.

The main table ranks individual configurations. Optional backbone summaries
show their best *observed* configuration and are explicitly descriptive test
summaries, not an unbiased estimate of a model selected without test results.
Budget rankings admit every configuration whose actual size fits the cap.

## Hardware-agnostic cost model

Only DPD arithmetic and real parameter count enter FoM. OPs = real MUL + ADD
per output IQ sample in algorithmic steady-state execution, batch 1, dense
FP32. This excludes PA cost, training cost, host timing and duplicated work
from the overlap-window evaluation harness. Complex coefficients count as
two real parameters and their conventional product as four real MULs.

The existing itemized `arena_ops` ledger and nonlinear reference table apply:
most scalar nonlinear functions cost 1 MUL + 1 ADD using at most 512 uniform
first-order segments with 2^-12 reference-domain absolute error; Hardswish
costs 2 + 2 and atan2 3 + 4. Delays, indexing, signs, exact powers of two and
table reads are free. No sparsity, quantization, pruning or fusion is credited.
These assumptions are published so another implementation can use a different
cost model. QGRU runs FP32; Delta thresholds are zero.

## Pareto plots

Rank contains EVM vs parameters, ACLR vs parameters, EVM vs operations and
ACLR vs operations. Horizontal cost axes are logarithmic; lower and left is
better. Each point is an actual configuration, with per-seed mean and sample
standard deviation. Each measured dataset has its own board and front. Never
mix protocols, PA conditions, or offline/stateful execution when computing
a front. Shipped and workspace results retain separate cohorts.

A point is dominated if another eligible point is no worse in both plotted
coordinates and strictly better in at least one. Exact duplicate points remain
nondominated. Each projection has its own front; a two-dimensional EVM front
is not a joint four-dimensional EVM/ACLR/parameters/operations front. Grey
crosses preserve unranked observations. Connecting lines are visual guides,
not evaluated intermediate models.

## Reproduction and integrity

`benchmark/retrain_arena_pa.py` refits PAs on the unchanged v4 measured
partitions and writes `calibration-v6-apa-b.json`. The packaged original capture
and measured partitions retain their hashes and boundary metadata. The historical
`benchmark/prepare_measured_arena.py` utility requires its archived prior
calibration; it is not needed to reproduce the published APA B matrix. The runner
binds training sources, data, PA weights, metrics and frame draws to its
training fingerprint. The protocol additionally binds scoring sources,
presets and the arithmetic ledger. All DPD configurations are retrained under
v6. Generated-input fits and earlier fixed-budget fits are not reused. Prior
protocol score bundles do not enter v6 rankings.

Seeded fits may run on CPU or CUDA, including different devices across seeds.
Every fit retains the prescribed budget and frame draws and records its actual
training device. Completed checkpoints can be reused across devices after
source and weight-hash verification. CPU and CUDA kernels can round differently;
numerical replay therefore retains the recorded PA evaluation device while
changing the DPD chunk size. Arithmetic costs follow the same published ledger
on every device.

The independent auditor recomputes quality, costs and ranks without the
production aggregation; validates complete cases, budgets and checkpoint
selection; verifies operation counts; and replays stateful DPD with different
chunk sizes plus a second EVM implementation. See the
[reproduction guide](../guides/dpd-arena.md) and
[current results](../performance/arena-reference-results.md).

## Release scope and provenance

The `dpd-arena-v6-apa-b` release contains only APA_200MHz_b results. All 218 independent fits were completed and frozen before the original test evaluation. `benchmark.focus_arena_reference` verifies the archived matrix, copies its sealed APA B checkpoints, requires every raw observation and score to remain identical, and reruns the independent audit. There is no retraining or test-based checkpoint selection during projection. The `calibration-v6-apa-b.json` file contains only APA B. Narrowing calibration changes the training identity, so the projection records both the original and release identities while retaining identical weights, training history and RF observations. Other datasets are outside the public Arena.
