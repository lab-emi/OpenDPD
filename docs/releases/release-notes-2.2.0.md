# OpenDPD 2.2.0 — Studio alpha (release-candidate notes)

Status: **release candidate**. Tagging, publishing to PyPI and any announcement
are maintainer decisions (plan S14): nothing here is pushed to an external
platform automatically. Version string in the candidate: `2.2.0.dev0`.

## What is new

- **OpenDPD Studio**: `pip install "opendpd[gui]"` then `opendpd gui` starts a
  local, loopback-only workbench in your browser (no Node.js, no second
  terminal). Datasets, runs, results, comparison, packages and reports; live
  progress from a worker subprocess; cancel that stops at the epoch boundary.
- **One compute core, three entry points**: the GUI, `opendpd …` commands and
  the Python API resolve the same configuration to the same hash and the same
  numbers on CPU (`tests/integration/test_entry_consistency.py`).
- **Explicit science**: every result carries its evidence type (PA model, DPD
  on a surrogate, DPD measured), its metric profile and version
  (`legacy-opendpd-v1` frozen, `general-spectral-v1`), its data version and
  split protocol; comparisons list incompatibilities instead of ranking.
- **Data preparation**: import CSV/NumPy/legacy split directories, Dataset
  Doctor (delay, gain, clipping, outliers, bandwidth), versioned preprocessing
  that never touches raw files.
- **Reproducible packages**: full (private) and share (redacted) packages with
  hash-verified import, reports bound to stored results, re-evaluation from
  stored checkpoints.
- **Classical baselines**: memory-polynomial and generalised memory-polynomial
  models fitted by least squares (PA: direct; DPD: indirect learning), with
  recorded fit stability and stated training paths.
- **Benchmark protocol `benchmark-v1`**: pre-registered plans, ≥ 3 seeds,
  hash-bound per-seed reports with a data audit, regression baselines that
  block only once a maintainer approves them.
- **Reference waveform and data-aided evaluation (pending cross-validation)**:
  `opendpd waveforms generate` writes the `ofdm-lte20-v1` package (CP-OFDM,
  LTE 20 MHz numerology, known 64QAM symbols) to play in a loop;
  `opendpd datasets import --waveform` binds a capture to it by correlation;
  profile `ofdm-lte20-evm-v1` reports data-aided RMS EVM and E-UTRA-style
  ACLR with explicit statuses for everything else. Every profile now carries a
  `validation` status; this one is `pending_cross_validation`, so the GUI does
  not offer it and every result under it says so
  (`docs/protocols/waveform-profiles.md`).
- **Thread budget**: `execution.num_threads` is applied by the shared executor
  (torch intra-op threads) on every path; unset keeps torch's default. Run
  records stamp `started_at`/`finished_at` with the executor's own clock on
  every path, so GUI and CLI durations of one configuration compare directly.
- **Hardening**: content-security policy and security headers, body and
  upload caps, bounded package extraction, one restricted checkpoint loader,
  accessibility audit (WCAG 2.1 AA rules) and keyboard-only journeys, Firefox
  and WebKit coverage, performance measured against frozen targets
  (`docs/releases/performance-report.md`, `docs/releases/hardening-report.md`).

## Compatibility

| Surface | Status |
|---|---|
| `python main.py --step …` and `opendpd-cli` | unchanged (`tests/test_cli.py`) |
| Public API `train_pa`, `train_dpd`, `run_dpd`, `plot_dpd`, `load_dataset`, `create_dataset`, `OpenDPDTrainer` | unchanged (`tests/test_api.py`); still single-threaded (`sys.argv` based) |
| Checkpoint naming and `state_dict` format | unchanged; legacy checkpoints load through the restricted loader (`tests/golden/test_legacy_checkpoint_loads.py`) |
| Metric semantics of `utils.metrics` | frozen as profile `legacy-opendpd-v1` (`tests/golden/test_legacy_metrics_golden.py`) |
| Built-in datasets, split ratios and `spec.json` | unchanged (`tests/test_datasets.py`) |
| Benchmark evidence schema | unchanged; `benchmark-v1` adds its own files under `benchmark/regression/` |
| Python | 3.10–3.13 (CI); 3.13 verified locally |
| Platforms | see `docs/releases/support-matrix.md` (Linux verified; macOS/Windows desktops pending human checks) |

## Install

```bash
pip install "opendpd[gui]"
opendpd gui
```

Tutorials: `docs/tutorials/gui-quickstart.md`, `docs/tutorials/headless-cli.md`,
`docs/tutorials/adding-a-model.md`.

## Known gaps in this candidate

- macOS and Windows desktop launches, real Safari and the system default
  browser are not verified by a person yet.
- External trial and onboarding measurements (S14) have not happened; this is
  an alpha for that purpose.
- Regression baselines are drafts until a maintainer approves them; the
  leaderboard policy (S20) does not exist yet.
- CSV sources of stress size are parsed into RAM; use `.npy`/`.npz` for
  captures beyond about ten million samples.

## Artifacts

Wheel and sdist are built with `python -m build`; their SHA-256 checksums for
the candidate commit are recorded in `docs/releases/studio-progress.md` (S13
entry). Verify with `sha256sum -c` before installing a downloaded file.
