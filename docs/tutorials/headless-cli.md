# Headless experiments with `opendpd run`

The `opendpd` command runs the same experiment service the Studio GUI uses,
without a browser. Everything lands in a **workspace** directory you choose;
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
```

`run_dpd` outputs are the PA **input** `u = DPD(x)`; they are not a linearised
PA output and carry no evaluation result.

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

## Compatibility

`python main.py ...` and `opendpd-cli` are unchanged. The `provenance.json`
of every run contains the equivalent legacy command line, and the
integration test `tests/integration/test_cli_run.py` checks that both paths
produce the same numbers for the same resolved configuration.
