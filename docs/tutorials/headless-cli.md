# Headless experiments with `opendpd run`

The `opendpd` command runs the same experiment service the Studio GUI uses,
without a browser. Install the [source preview](../install.md) first; the
PyPI 2.1.0 release only provides the legacy CLI. Everything lands in a **workspace** directory you choose;
nothing is written into the installed package.

## 1. Create a workspace and register data

```bash
opendpd datasets import-builtin DPA_200MHz --workspace ./my-workspace
opendpd datasets list --workspace ./my-workspace
```

`import-builtin` copies the packaged dataset into
`my-workspace/datasets/dpa-200mhz/raw/` and writes a `manifest.json` with
file hashes, split boundaries and signal metadata. Raw files are never
modified afterwards.

## 2. Run a reference recipe

```bash
opendpd recipes                                   # smoke vs research recipes
opendpd run --recipe pa-gru-smoke-v1 --dataset dpa-200mhz --workspace ./my-workspace
```

The command prints the run id, the evidence type, the metric profile and the
test-split metrics with their limitations (a 3-epoch smoke run says so).
Results live in `my-workspace/runs/<run_id>/`:

| File | Content |
|---|---|
| `config.user.json` | what you submitted |
| `config.resolved.json` | every default filled in, hashed (`resolution.config_sha256`) |
| `provenance.json` | software versions, dataset hash, the equivalent legacy `python main.py` command |
| `run.json` | status, timestamps, worker identity, error (if any) |
| `artifacts.json` | registered files with SHA-256 (checkpoint, logs, outputs) |
| `result.json` | the formal `EvaluationResult` (metrics, evidence, reference, limitations) |
| `save/`, `log/`, `dpd_out/` | the unchanged legacy layout, confined to this run |

Train a DPD model through that PA surrogate, then generate the pre-distorted
signal:

```bash
opendpd run --recipe dpd-gru-smoke-v1 --dataset dpa-200mhz --pa-run <pa_run_id> --workspace ./my-workspace
opendpd apply <dpd_run_id> --workspace ./my-workspace
```

Replace `<pa_run_id>` and `<dpd_run_id>` with the successful run IDs printed
by the preceding commands. `apply` exports the PA **input** `u = DPD(x)` and
scores the cascade through the PA surrogate. The exported I/Q file itself is
not a measured linearized PA output.

## 3. Write your own configuration

```bash
opendpd validate --config experiment.json     # errors name the field, exit code 2
opendpd run --config experiment.json --workspace ./my-workspace
```

```json
{
  "task": "train_pa",
  "dataset": {"id": "dpa-200mhz"},
  "model": {"key": "gru", "parameters": {"hidden_size": 23}},
  "training": {"epochs": 3, "frame_length": 50, "frame_stride": 16, "batch_size_eval": 256},
  "evaluation": {"evidence_type": "pa_modeling"},
  "execution": {"device": "cpu"}
}
```

Model keys and their parameters come from the registry (`opendpd models`).
Checkpoint selection is fixed by protocol (validation NMSE for PA models,
validation ACLR for DPD models) and cannot be changed in a config.

`execution.num_threads` is the CPU thread budget torch trains with. Unset, torch
uses its own default (the physical-core count). The budget is applied by the one
executor every path shares, so a configuration trains with the same budget from
the CLI, the Python API and the GUI; on a many-core machine a smaller explicit
budget often trains a small model faster and leaves the Studio service its own
core (see the [performance report](../releases/performance-report.md)).

## Use your own data

Studio upload is currently disabled. The local CLI still supports importing your
own captures; the dataset format is documented in [Datasets](../datasets.md).
This example assumes four columns `I_in`, `Q_in`, `I_out`, `Q_out` and the
signal metadata shown below. Replace those values with your actual capture
metadata before running:

```bash
opendpd datasets import capture.csv --id mine --fs 800e6 --bandwidth 200e6 --n-sub-ch 10 --nperseg 2560 --units normalized --workspace ./my-workspace
opendpd datasets doctor mine --json --workspace ./my-workspace
```

Import records the raw data, hashes, metadata and split boundaries. Inspect the
Doctor findings before training. If alignment is needed, use its estimate to
preview a new preprocessing version; `6` below is only an example delay:

```bash
opendpd datasets preprocess mine --version aligned-v1 --delay 6 --preview --workspace ./my-workspace
opendpd datasets preprocess mine --version aligned-v1 --delay 6 --workspace ./my-workspace
```

Set `dataset.id` to `mine` and `dataset.preprocessing_version` to `aligned-v1`
in the experiment configuration. Raw data remain unchanged and preprocessing
versions record their fitted range. See the [Dataset Doctor and split
protocol](../protocols/dataset-doctor.md) for the exact rules.

## Compatibility

`python main.py ...` and `opendpd-cli` are unchanged. The `provenance.json`
of every run contains the equivalent legacy command line, and the
integration test `tests/integration/test_cli_run.py` checks that both paths
produce the same numbers for the same resolved configuration.
