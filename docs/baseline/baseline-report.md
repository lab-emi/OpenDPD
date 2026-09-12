# S00 baseline report

Fixed baseline for the OpenDPD Studio work. Every claim below was executed on
the machine described here; nothing was inferred from CI badges or memory.
Items that could not be run are listed as **not verified**, not as passed.

- Baseline commit: `7426bbf8a47624b59bd7f045a86641b403023f3c` (`main`, "Rebuild the benchmark with reproducible PA and DPD results (#19)")
- Working branch: `OpenDPD-Studio`
- Date: 2026-09-06

## Environment

| Item | Value |
|---|---|
| OS | Linux 7.2.2-1-cachyos x86_64 |
| CPU | 13th Gen Intel Core i9-13900HX (32 threads) |
| RAM | 94 GB |
| GPU | NVIDIA GeForce RTX 4090 Laptop GPU, driver 610.57.04, CUDA 13.2 |
| Python | 3.13.14 (`~/.venv`) |
| torch / numpy / scipy / pandas | 2.13.0+cu132 / 2.4.4 / 1.18.0 / 3.0.5 |
| matplotlib / pillow / tqdm / rich | 3.11.1 / 12.2.0 / 4.70.0 / 15.0.0 |
| Node.js / npm (frontend build only) | v26.8.1 / 12.0.2 |
| Install | `pip install -e ".[dev]"` |

Not a clean environment in the strict sense: the venv pre-existed. All
package imports used were verified to resolve to this checkout.

## Commands and results

| # | Command | Result | Notes |
|---|---|---|---|
| 1 | `pytest tests/ -m "not extended" -q --durations=15` | **197 passed, 19 deselected in 26.5 s** (exit 0) | includes the chained CPU E2E smoke (train_pa → train_dpd → quantized train_dpd → run_dpd → plot) and CUDA-specific tests, which ran on the GPU |
| 2 | `flake8 . --select=E9,F63,F7,F82` | **0 critical errors** | same selection as CI |
| 3 | `python -m build && twine check dist/*` | **PASSED** for wheel and sdist | wheel 31.0 MB (128 files, 25 CSV data files), sdist 43.6 MB |
| 4 | Legacy CPU chain, DPA_200MHz, 2 epochs, frame 50 / stride 16 (see below) | train_pa 2.79 s, train_dpd 2.98 s, run_dpd 1.80 s (exit 0 each) | artifacts in `save/`, `log/`, `dpd_out/` of the working directory |
| 5 | Same `train_pa` with `--accelerator cuda` | 2.66 s, exit 0, "Using GPU 0: NVIDIA GeForce RTX 4090 Laptop GPU" | val NMSE −16.09 dB (2 epochs; smoke only) |

Legacy chain command (run from an empty working directory):

```bash
python main.py --step train_pa --n_epochs 2 --dataset_name DPA_200MHz --accelerator cpu --frame_length 50 --frame_stride 16 --batch_size 64 --batch_size_eval 256
python main.py --step train_dpd --n_epochs 2 --dataset_name DPA_200MHz --accelerator cpu --frame_length 50 --frame_stride 16 --batch_size 64 --batch_size_eval 256
python main.py --step run_dpd   --n_epochs 2 --dataset_name DPA_200MHz --accelerator cpu --frame_length 50 --frame_stride 16 --batch_size 64 --batch_size_eval 256
```

Best-epoch rows (2-epoch smoke, not a benchmark):

| Run | VAL_NMSE (dB) | VAL_ACLR_AVG (dB) | TEST_NMSE (dB) | Params |
|---|---|---|---|---|
| PA GRU-H23 CPU | −16.092 | −28.068 | −15.755 | 1911 |
| PA GRU-H23 CUDA | −16.095 | −28.068 | −15.757 | 1911 |
| DPD GRU-H15 CPU | −14.990 | −26.820 | −14.872 | 887 (DPD) / 2798 (cascade) |

## Not verified in S00

| Item | Why | What is needed |
|---|---|---|
| Full benchmark matrix (`benchmark/reproduce_benchmark_report.sh`) | multi-hour GPU job; the published report was produced on an RTX PRO 6000 at commit `3df35e0` | a human-approved L4 run bound to an exact commit |
| Windows and macOS behaviour | no such machines here | platform CI runners / manual smoke (S06, S13) |
| MPS acceleration | no Apple hardware | same |
| Python 3.10–3.12 locally | only 3.13 in the venv; CI covers 3.10–3.13 on `main` | rely on CI matrix |
| Extended (`-m extended`) tests | run weekly by CI; skipped here to keep the baseline under a minute | `pytest tests -m extended` |

## Behaviour that must be preserved (compatibility register)

| Behaviour | Guarded by |
|---|---|
| `python main.py --step ...` and `opendpd-cli` argument surface and defaults (`tests/test_cli.py::test_training_defaults_match_opendpdv2_recipe`) | existing CLI tests |
| Public API: `train_pa`, `train_dpd`, `run_dpd`, `plot_dpd`, `load_dataset`, `create_dataset`, `OpenDPDTrainer` | `tests/test_api.py` |
| Checkpoint naming `PA_S_<seed>_M_<BACKBONE>_H_<hidden>_F_<frame>_P_<params>.pt` and `state_dict` format | `tests/golden/test_legacy_checkpoint_loads.py` |
| Metric semantics of `utils.metrics` (profile `legacy-opendpd-v1`) | `tests/golden/test_legacy_metrics_golden.py` |
| Split ratios 0.6/0.2/0.2 and `spec.json` fields of built-in datasets | `tests/test_datasets.py` |
| Benchmark evidence schema and collector | `tests/test_benchmark_regressions.py` |

## Known problems recorded at baseline (not fixed in S00)

1. `opendpd/api.py` passes parameters by rewriting `sys.argv` and runs training synchronously; it is not safe to call from concurrent threads (risk R12).
2. `modules/paths.py`, `steps/*.py` write `save/`, `log/`, `dpd_out/`, `plots/` relative to the current working directory (risk R13).
3. `steps/run_dpd.py` and `steps/plot.py` load data by `dataset_name` only; the `--dataset_path` option of `train_pa` is not honoured there.
4. `Project.load_spec` reads specs from the installed package's `datasets/` directory, so user datasets must be copied into the package unless `--dataset_path` is used.
5. `steps/run_dpd.py` calls `torch.load` without `weights_only=True`.
6. `.gitignore` ignored `docs/` (fixed in S00 so the plan's `docs/` tree can exist).

## Governance and fixtures added in S00

- `AGENTS.md`, `CONTRIBUTING.md`, `.github/PULL_REQUEST_TEMPLATE.md`, `.github/CODEOWNERS`, `docs/architecture/adr/0000-adr-template.md`
- `.github/workflows/protected-paths.yml`: PRs touching goldens / metrics / splits / benchmark / protocols fail without the `science-review-approved` label
- `tests/golden/`: frozen legacy metric values and a real legacy checkpoint fixture
- `tests/fixtures/`: synthetic PA generator with known impairments and the tier manifest
- `docs/protocols/acceptance-thresholds.md`, `docs/releases/support-matrix.md`, `docs/architecture/risk-register.md`, `docs/baseline/first-tasks.md`

## Pending maintainer decisions

The following S00 acceptance items require a human and are recorded as
**pending**, not done: approval of the G0–G2 scope, the support matrix, the
risk catalogue and the acceptance thresholds; creation of the GitHub issues
from `first-tasks.md`; configuration of the `science-review-approved` label
and the `@lab-emi/opendpd-maintainers` team referenced by CODEOWNERS.
