# Benchmark protocol `benchmark-v1`

Status: implemented (S12). Schema: `opendpd/schemas/benchmark.py`; code:
`opendpd/services/benchmark.py`; CLI: `opendpd benchmark plan|run|report|check|baseline`.
Changing this protocol, a plan's matrix or an approved baseline is a
protected-path change (science review), never an agent's call.

## Two kinds of regression, two kinds of threshold

| Layer | What is compared | Threshold | Where |
|---|---|---|---|
| Deterministic | a stored checkpoint re-evaluated; a metric implementation against goldens; a package round trip | frozen numeric tolerances (rel 1e-4 / abs 1e-5 for float32 checkpoints, 1e-7 for goldens) | `acceptance-thresholds.md`, `tests/golden`, `tests/integration/test_packages.py` |
| Statistical | re-training under a pre-registered plan: fixed recipe, budget, ≥ 3 seeds | an approved band per entry and metric: mean over seeds ± max(0.5 dB, 2 × sample std), derived from a report on a named machine | `RegressionBaseline`, `opendpd benchmark check` |

A successful re-evaluation never implies that re-training reproduces the
numbers (`docs/protocols/experiment-packages.md`); a re-training result within
its band never implies bit-identity.

## Plans are pre-registered

`opendpd benchmark plan --dataset ID --tier cpu_regression|gpu_full --out plan.json`
writes the tier's fixed model matrix, the budget of every entry (epochs,
frames, optimiser), the metric profile, the device and the seeds. The hash of
that content (`plan_sha256`) identifies the plan; an edited plan file is
refused. At least **three** seeds are required (`MIN_SEEDS`); seeds are part
of the plan, never chosen after seeing results.

| Tier | Matrix | Budget | Purpose |
|---|---|---|---|
| `cpu_regression` | `pa-gru`, `pa-mp-ls` (K=5, Q=20), `dpd-gru` and `dpd-mp-ila` (K=5, Q=20) through `pa-gru` of the same seed | the smoke budgets (3 epochs, frame 50) | regression reference for ordinary PRs and the weekly run on CPU; its numbers are never research results |
| `gpu_full` | `pa-gru` (300 epochs), `pa-mp-ls` (K=9, Q=150), `pa-gmp-ls` (rcond 1e-4), `dpd-tres-deltagru` (300 epochs), `dpd-mp-ila` (K=5, Q=100), `dpd-gmp-ila`, all DPDs through `pa-gru` of the same seed | the research recipes and the benchmark's polynomial configurations | controlled full validation on a named GPU machine after human approval |

Selection rule (recorded in every plan and report): the checkpoint is
selected on the **validation** split by the task's protocol metric (PA:
NMSE, DPD: ACLR average); the **test** split is scored once per run by the
evaluation stage and never used for selection or tuning. Re-evaluating a
stored checkpoint under another profile re-reads the same checkpoint and is
a deterministic regression, not a new attempt.

## Execution

`opendpd benchmark run plan.json --workspace WS` executes every (entry, seed)
in this process, PA entries first. Each run's idempotency key is derived from
the plan hash, so re-running a plan reuses finished runs and only fills the
gaps; nothing is overwritten. A failed surrogate leaves its DPD entries
without a run for that seed, which the report lists as missing.

## Reports are bound, not typed

`opendpd benchmark report plan.json --workspace WS --out report.json [--markdown report.md]`
reads every number from a run's stored result under the plan's profile.
For every entry and seed it records the run id, the resolved-configuration
hash, the checkpoint hash, the surrogate run id (DPD), the selected epoch,
the metrics and the wall clock; for the entry the training path, parameter
count, look-ahead, execution semantics and, for least-squares fits, the fit
diagnostics (rank, condition number, cutoff, residual). The aggregate is
mean, sample std, min and max over the seeds; the per-seed rows always stand
next to it. The dataset audit states which split trained the surrogate,
which one the DPD was optimised on, which selected and which is reported, and
that agreement between a simulated and a measured outcome is not established
by the benchmark. `report_sha256` seals the content: `load_report` refuses a
report whose numbers were edited. Every report carries the notes that the
spread of N seeds is not evidence of generality and that equal parameter
counts are not equal compute cost (families differ in operations per sample,
memory and look-ahead; ILA and DLA are different training paths).

## Baselines, bands and release blocking

`opendpd benchmark baseline report.json --out baseline.json [--tolerance-db 0.5]`
drafts a baseline from a report: for every entry and metric the reference is
the seed mean and the tolerance is max(0.5 dB, 2 × sample std); `basis`
records the report, machine and rule. A draft has no `approved_by` /
`approved_on`: **an unapproved baseline never blocks** (`check` reports and
exits 0). Once a maintainer fills both fields, `opendpd benchmark check`
exits 1 whenever any metric leaves its band **in either direction**: a
degradation blocks the release until a reason and a new approval are
recorded; an unexpected improvement blocks too, because a protocol or metric
change must be ruled out before the baseline is re-approved. A baseline
applies to one plan hash only.

Accepting a degradation therefore means: a new baseline file with the reason
in `notes`, `approved_by`, `approved_on` — under `benchmark/regression/`, a
protected path.

## Baseline stability is recorded, not assumed

The classical baselines (`mp_ls`, `gmp_ls`) are fitted by column-normalised
least squares with an explicit singular-value cutoff fixed by the recipe
(`pa-gmp-ls-v1`: rcond 1e-4, as in `benchmark_report.md`). Rank, condition
number, cutoff and train residual are stored with the run (`fit.json`),
repeated in the result's limitations and in the benchmark report, so a
"gain" over a broken polynomial fit is visible as such. The cutoff is not
tuned per result; changing it is a protocol change.

## What the report does not claim

- Three seeds bound run-to-run variation on one dataset and machine; they
  do not show generality across PAs, signals or operating points.
- Simulated DPD results are evidence through a learned surrogate
  (`docs/protocols/simulation-chain.md`); measured evidence is a separate type.
- The GPU benchmark report (`benchmark/benchmark_report.md`) keeps its own
  reproduction script and evidence bundle; this protocol does not replace it
  and does not restate its numbers.

## Leaderboard submissions (policy)

Public submissions are prepared and executed on a machine that is **not**
connected to laboratory data stores, credentials or instrument networks; the
share package (no user data, no machine paths) is the only thing that leaves
that machine. This is an organisational rule the software cannot verify; it
is recorded here and in the S12 acceptance table as pending human process.
