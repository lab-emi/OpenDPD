# DPD Arena in Studio 2.3

Open **DPD Arena** in the navigation. **Rank**, **Submit** and **Rules** share
one measured-source benchmark: **APA_200MHz_b** (200 MHz, 256-QAM). The result table and all four Pareto plots use this dataset.

Rank compares actual **backbone × parameter configuration** points. The main
score combines complete-symbol EVM, PA-output ACLR and DPD complexity:

```text
Q = mean over seeds and conditions of 0.5 (ΔEVM + ΔACLR)
FoM = Q − 5 log10(parameters / 1000) − 5 log10(operations / 2000)
```

The references are fixed for every size. Halving both costs at the same EVM
and ACLR adds 3.01 dB. Seed standard deviation is shown separately. Each
configuration must preserve output power within ±0.5 dB and improve mean
quality to rank. Missing sizes are unavailable, not zero scores. Negative FoM
is retained for otherwise valid configurations. See the
[versioned protocol](../protocols/dpd-arena-v6.md).

The current protocol uses 240 complete training passes with batch 64 and
stride 1. Every valid training window, including the final partial batch, is
used each epoch. Each neural configuration uses 94,320 optimizer updates per seed. Fixed epoch counts still do not establish architecture optima.

The benchmark displays its frozen TRes-GRU PA's parameter count and original
capture validation/test NMSE. The same PA is used for training and evaluation;
GRU/GMP DPD contestants do not act as additional PA judges. All DPD results
use original measured input samples for training, validation and test.
Cascade outputs are predictions through the frozen PA. APA is repartitioned
to retain complete test symbols on every carrier; its PA is retrained on the
new training split. DPD checkpoints minimize an equal-weight validation in-band error and
worse-side ACLR objective, subject to output power. Validation uses a declared
4,096-sample PSD; complete test symbols supply the EVM entering FoM. PA model NMSE does not impose a hard EVM/ACLR score ceiling.

## Compare quality and complexity

Four plots show EVM vs parameters, ACLR vs parameters, EVM vs operations and
ACLR vs operations. The cost axes are logarithmic, with lower and left better.
Diamonds highlight each plot's own Pareto front; error bars show seed standard
deviation. All rankings and Pareto fronts use APA_200MHz_b. Different
execution modes and workspace/shipped results remain separate.

Hover over a point for configuration costs and metrics. Plot controls support
zoom, enlargement and PNG export. Connecting lines are visual guides; no
intermediate configuration is implied. The EVM and ACLR projections can have
different frontiers.

The budget selector admits **all** fitting sizes: a 250-parameter model can
also participate in the ≤1,000-parameter ranking. Expand **Backbone summaries
and evidence** for the best observed configuration of each family, per-point
scores, raw cases, frozen-weight hashes and arithmetic breakdowns. These
optional family summaries are descriptive test summaries, not validation-only
model selections.

## Submit a backbone

Select a bundled DPD backbone or a validated custom template, name the entry,
and acknowledge the displayed protocol. Studio determines the data, sizes,
training and evaluation; clients cannot submit scores. Each distinct neural
configuration receives three seeds, 240 full epochs and 94,320 updates per seed. Deterministic
MP/GMP fits use one seed and training-only PA feedback, without ILC-generated
data. ILC and ILC→MP are excluded from Arena. All checkpoints are frozen before
loading test inputs; only test observations determine final metrics and ranks.

Results remain in the workspace. Hosted installations require the opt-in
[isolated GPU adapter](arena-hosted.md) and a worker advertising the exact
protocol. Existing submissions from a different protocol remain evidence only.

## Reproduce the matrix

From a source checkout with the training dependencies installed:

```bash
# Optional: retrain and select PAs on the existing measured partitions.
python -m benchmark.retrain_arena_pa --workspace /path/to/pa-preparation --device cuda

# Train all sizes, evaluate APA_200MHz_b and write the bundled reference JSON.
python -m benchmark.run_arena_baselines --workspace /path/to/arena-benchmark \
  --cpu-workers 8 --gpu-workers 1 --workers 2

# Independent scores, operation counts, checkpoint checks and streaming replay.
python -m benchmark.audit_arena_baselines --workspace /path/to/arena-benchmark \
  --out final-audit.json

python -m benchmark.report_arena_results
```

Use the shipped calibration assets to reproduce DPD without reselecting the
PA. Re-running PA preparation can produce different float32 checkpoint bytes
across hardware; changed assets intentionally change the protocol fingerprint.
The prefetch cache is keyed by configuration, condition, seed and training
fingerprint. `--prefetch-only` prepares weights without publishing. Keep
training sources and calibration assets fixed during a run.

For a long run shared with an explicitly configured SSH host, synchronize the
same source checkout and calibration assets first. The distributed launcher
rejects different training fingerprints and limits the remote host to one GPU
training process:

```bash
python -m benchmark.run_arena_distributed --workspace /path/to/arena-benchmark \
  --remote user@gpu-host --remote-root /path/to/remote-run \
  --remote-python /path/to/remote-run/venv/bin/python --remote-gpu-workers 1

# All 218 distinct fits must exist and verify before test evaluation starts.
# The candidate bundle is published locally only after the complete audit.
python -m benchmark.finalize_arena_reference --workspace /path/to/arena-benchmark
```

The remote root contains `repo/` and `work/`. Training uses only train and
validation arrays; verified checkpoints are copied to the local cache. The
final test phase runs locally and refuses to train missing checkpoints.
`distributed-progress.json` and `finalization-progress.json` record progress.
Add `--finalize` to the distributed command to run that evaluation/audit step
automatically after all fitting workers finish successfully.

All 23 Arena keys remain represented on APA_200MHz_b, including
failed or unranked entries. The independent auditor does not treat a partial
matrix as a complete reference. [Current reference results](../performance/arena-reference-results.md)
include configuration metrics and cost-weight sensitivity.
