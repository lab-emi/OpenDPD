# benchmark-v1 report (cpu_regression)

Plan `d79883def16e` · dataset `dpa-200mhz` (raw sha256 9464a16d9842, version raw-v1, split contiguous-v1, guard 0) · profile legacy-opendpd-v1 v1 · device cpu · seeds 0, 1, 2

Software: opendpd 2.2.0.dev0, python 3.13.14, torch 2.13.0+cu132, git 5fa202682726e2382cfa865f7ca5a2984cc009d8 (dirty) · machine: x86_64, 32 cores, Linux 7.2.2-1-cachyos (x86_64)

## pa-gru (train_pa, gru {'hidden_size': 23, 'num_layers': 1})

training path gradient · 1911 real parameters · look-ahead 0 samples · offline_segmented

| seed | run | selected epoch | ACLR_AVG | ACLR_L | ACLR_R | EVM | NMSE | wall clock (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | `run-20260906-114016-39ad61` | 2 | -32.59 | -33.13 | -32.05 | -22.37 | -21.38 | 2.25 |
| 1 | `run-20260906-114020-8c19f1` | 2 | -32.89 | -34.36 | -31.42 | -23.81 | -22.58 | 0.84 |
| 2 | `run-20260906-114023-d5594d` | 2 | -31.62 | -33.05 | -30.18 | -21.96 | -20.84 | 0.59 |
| **mean ± std (min … max)** | | | -32.36 ± 0.66 (-32.89 … -31.62) | -33.51 ± 0.73 (-34.36 … -33.05) | -31.22 ± 0.95 (-32.05 … -30.18) | -22.71 ± 0.97 (-23.81 … -21.96) | -21.60 ± 0.89 (-22.58 … -20.84) | |

## pa-mp-ls (train_pa, mp_ls {'K': 5, 'Q': 20, 'rcond': 0.0})

training path least_squares · 200 real parameters · look-ahead 0 samples · offline_segmented
fit: rank 100 of 100, condition number 1.21e+04, rcond 0.0

| seed | run | selected epoch | ACLR_AVG | ACLR_L | ACLR_R | EVM | NMSE | wall clock (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | `run-20260906-114018-790002` | n/a | -31.16 | -31.88 | -30.45 | -39.36 | -34.91 | 0.32 |
| 1 | `run-20260906-114021-88b110` | n/a | -31.16 | -31.88 | -30.45 | -39.36 | -34.91 | 0.30 |
| 2 | `run-20260906-114023-6a3b3b` | n/a | -31.16 | -31.88 | -30.45 | -39.36 | -34.91 | 0.27 |
| **mean ± std (min … max)** | | | -31.16 ± 0.00 (-31.16 … -31.16) | -31.88 ± 0.00 (-31.88 … -31.88) | -30.45 ± 0.00 (-30.45 … -30.45) | -39.36 ± 0.00 (-39.36 … -39.36) | -34.91 ± 0.00 (-34.91 … -34.91) | |

## dpd-gru (train_dpd, gru {'hidden_size': 15, 'num_layers': 1})

training path gradient_dla · 887 real parameters · look-ahead 0 samples · offline_segmented

| seed | run | selected epoch | ACLR_AVG | ACLR_L | ACLR_R | EVM | NMSE | wall clock (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | `run-20260906-114018-6448fc` | 2 | -29.88 | -28.24 | -31.53 | -21.22 | -20.02 | 1.13 |
| 1 | `run-20260906-114021-f34df4` | 2 | -31.83 | -32.67 | -30.99 | -21.99 | -20.88 | 1.22 |
| 2 | `run-20260906-114024-1a01e2` | 2 | -29.44 | -29.53 | -29.36 | -20.28 | -19.02 | 1.16 |
| **mean ± std (min … max)** | | | -30.39 ± 1.27 (-31.83 … -29.44) | -30.15 ± 2.28 (-32.67 … -28.24) | -30.63 ± 1.13 (-31.53 … -29.36) | -21.16 ± 0.86 (-21.99 … -20.28) | -19.97 ± 0.93 (-20.88 … -19.02) | |

## dpd-mp-ila (train_dpd, mp_ls {'K': 5, 'Q': 20, 'rcond': 0.0})

training path ila_least_squares · 200 real parameters · look-ahead 0 samples · offline_segmented
fit: rank 100 of 100, condition number 3.16e+03, rcond 0.0

| seed | run | selected epoch | ACLR_AVG | ACLR_L | ACLR_R | EVM | NMSE | wall clock (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | `run-20260906-114019-94885b` | n/a | -36.28 | -37.63 | -34.92 | -22.55 | -21.48 | 0.62 |
| 1 | `run-20260906-114022-c66858` | n/a | -34.53 | -36.04 | -33.03 | -23.95 | -22.97 | 0.56 |
| 2 | `run-20260906-114025-3876a4` | n/a | -33.30 | -34.85 | -31.75 | -22.10 | -21.17 | 0.45 |
| **mean ± std (min … max)** | | | -34.70 ± 1.50 (-36.28 … -33.30) | -36.17 ± 1.40 (-37.63 … -34.85) | -33.23 ± 1.60 (-34.92 … -31.75) | -22.87 ± 0.96 (-23.95 … -22.10) | -21.87 ± 0.96 (-22.97 … -21.17) | |

## Data audit

- dataset_id: dpa-200mhz
- raw_sha256: 9464a16d98426c749dec06bf65c93e4f3811b1b6a0d55c9a9d1385bebc94d9b4
- preprocessing_version: raw-v1
- split_version: contiguous-v1
- guard_samples: 0
- surrogate_training_split: train
- dpd_optimisation: train split: through the frozen PA surrogate of the same seed (gradient, DLA) or on the measured data (least squares, ILA)
- selection_split: val
- reported_split: test
- statement: the surrogate is a model fitted to measured data; agreement between a DPD's simulated and measured outcome is not established by this benchmark (cross-validation of the surrogate is not physical validation)

## Notes

- seeds are pre-registered in the plan; per-seed values are reported next to the aggregate; the spread of 3 seeds estimates run-to-run variation on this data and hardware and is not evidence of generality across devices, signals or operating points
- equal parameter counts are not equal compute cost: families differ in operations per sample, memory and look-ahead; a least-squares fit and gradient training are different procedures with different budgets, and ILA and DLA are different training paths
- the checkpoint is selected on the validation split by the protocol metric of the task (PA: NMSE, DPD: ACLR average); the test split is scored once, by the evaluation stage, and never used for selection or tuning
- every entry and seed is scored on the test split exactly once from its selected checkpoint; re-evaluation under another metric profile re-reads the same checkpoint and is a deterministic regression, not a new attempt
- cpu_regression uses the smoke budgets: its numbers are regression references for this data and hardware, never research results

report sha256 `be10351c3fea11be9aeed4837e844a98338cc27204f6e8cdab00b7ff1c697d81`
