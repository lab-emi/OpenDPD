# The DPD simulation chain: x, u = DPD(x), y

Status: implemented (S10). Semantics are versioned through the result
schema (`EvaluationResult.signal_chain`, `baselines`, `surrogate_coverage`,
`scaling`) and the metric profile that scores them.

## What a DPD result is

A `train_dpd` or `run_dpd` result with evidence type `dpd_surrogate` scores
the cascade **y = PA_surrogate(DPD(x))** against the linear target
**reference_gain · x** on the test split. Nothing in it is a measurement of a
linearised PA. Every result names the three stages explicitly:

| Stage | Meaning | Source recorded |
|---|---|---|
| `x` | target input: what the linearised PA should reproduce (times the reference gain) | dataset id, data version, split |
| `u = DPD(x)` | the pre-distorted **PA input** | DPD model key, weight hash, run id; for `run_dpd` the exported file (`dpd-output`) |
| `y = PA(u)` | the PA output | PA surrogate model key, weight hash, run id; always `simulated: true` |

The validator refuses a `dpd_surrogate` result whose `y` stage is not marked
simulated, and a `dpd_measured` result whose `y` stage is.

## Generating a signal is not verifying linearisation

`run_dpd` exports `u` for the test split as `dpd_out/<model id>.csv` with the
columns `I, Q, I_dpd, Q_dpd` (x then u) in the original sample order, float32
values, plus a sidecar `<model id>.meta.json` (artifact `dpd-output-meta`)
stating: the signal role (`pa_input_predistorted`), the column meaning, the
dtype, the sample order and split, the execution semantics (one pass over
the whole split with state carried), the amplitude units and input scaling,
the reference gain and its rule, the peak amplitudes, and the exact dataset,
DPD and PA-surrogate checkpoints (run ids and SHA-256) that produced it.
`tests/integration/test_cli_run.py::test_run_dpd_exports_predistorted_input`
rebuilds `u` independently, reloads the file and checks order, dtype,
amplitude and metadata.

The file is a PA *input*. Feeding it to a real PA and measuring the output
is the only way to obtain `dpd_measured` evidence (S16); the Studio never
labels a surrogate score as measured.

## Baselines under one reference

Each DPD result also scores two comparison signals against the **same**
`reference_gain · x`, with the same metric profile and valid sample range:

- `surrogate_without_dpd`: the PA surrogate driven by `x` directly;
- `measured_without_dpd`: the measured PA output of the test split.

The reference gain is `max|y_train| / max|x_train|` (the legacy
`set_target_gain`), recorded in `reference.gain_value` and `scaling.reference_gain`.
Because nothing is normalised separately, a DPD that merely changes the
output power cannot look better than the baseline; the validator requires
every baseline to score exactly the metrics of the result.

The two baselines also give an explainable sanity check of the surrogate:
if the surrogate driven by `x` scores far from the measured PA on the same
split, the surrogate is not reproducing the PA's distortion in that region.
This is evidence, not proof; nothing in the result claims the surrogate is
accurate.

## Amplitude coverage of the surrogate

`surrogate_coverage` records the largest input amplitude the PA surrogate
was fitted on (over its training split), the peak of `u`, and the fraction of
pre-distorted samples above the fitted peak. A non-zero fraction is added to
the limitations ("extrapolation"); the note states explicitly that staying
inside the range does not prove the surrogate accurate. The check is an
explainable warning about extrapolation, nothing more.

## Swapping the surrogate

`run_dpd` can be evaluated through another succeeded PA run of the same
dataset (`opendpd apply DPD_RUN --pa PA_RUN`, the **Apply DPD to the test
split** dialog on a DPD run page, or a `run_dpd` configuration with
`pa_reference`). Each application is a new run with its own result under
every registered profile; the DPD run's own result is never rewritten.
`test_swapping_the_surrogate_yields_a_separate_result` checks that the two
results name different PA weight hashes and that the original result file is
byte-identical afterwards. Applying through the training surrogate
reproduces the DPD run's metrics within 1e-3 dB
(`test_apply_through_the_training_surrogate_reproduces_the_dpd_result`).

`GET /runs/{id}/lineage` (Python: `opendpd.services.experiments.lineage`)
exposes the graph: which PA checkpoint a DPD used, which DPD and PA a
`run_dpd` used, which runs used a given checkpoint, and retry parents. It is
read from resolved configurations, never inferred from names.

## Scaling and what is not derived

`scaling` records the amplitude units of the dataset, how the input was
scaled before training (the preprocessing version's normalisation and fit
range, or "none"), the reference gain and `physical_calibration: false`.
There is no calibration path yet, so no absolute output power (dBm), PAE or
efficiency is derived from any number in a result; the validator rejects a
result claiming physical calibration.

## Compatibility rules enforced before a run exists

- A DPD recipe needs a **succeeded** `train_pa` run on the same dataset. A
  missing or failed run is refused with the remedy in the hint (train a PA
  model first with the same seed and frame length).
- The legacy checkpoint convention requires `train_dpd` to use the seed and
  frame length of its surrogate; the hint names the values to set.
- `run_dpd` binds model, training and quantisation from the DPD run and must
  target the dataset the DPD was trained on.
- Referenced checkpoints are hash-verified when copied into the run; a
  changed or deleted file fails the run with `input_missing`, never with a
  substitute.
