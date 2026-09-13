# OpenDPD 2.2.0 — OpenDPD Studio

Try the [hosted Studio](https://opendpd.com/studio/) or install `opendpd[gui]==2.2.0` from PyPI. The wheel contains the built frontend.

## What is new

- **Hosted Studio**: GitHub Pages frontend, Cloudflare Tunnel and isolated compute with a shared CUDA queue. Session workspaces and their files expire within 24 hours.
- **Validated CSV uploads**: UTF-8 CSV, two complex or four real I/Q columns, at most 25 MiB and 1,000,000 sample pairs. Quarantine validation rejects malformed, nonnumeric, binary and formula content before preview and deletes rejected uploads.
- **Clear training progress and model downloads**: separate completed-epoch and current-batch bars; download the best checkpoint during training and the selected final model afterward. Checkpoints are synchronized from the isolated GPU worker without exposing its filesystem.
- **Defaults and phone controls**: English initially, CUDA selected when detected, stable two-finger plot interaction and large touch controls.
- **OpenDPD Studio**: `pip install "opendpd[gui]"` then `opendpd gui` starts a
  local, loopback-only workbench in your browser (no Node.js, no second
  terminal). Datasets, runs, results, comparison, packages and reports; live
  progress from a worker subprocess; Stop that terminates the experiment process, including its isolated GPU container.
- **Native window**: `pip install "opendpd[desktop]"` makes `opendpd gui` open
  the workbench in an application window (WKWebView / WebView2 / WebKit2GTK)
  instead of a browser tab; the browser stays the fallback and `--browser` /
  `--window` / `--no-browser` choose explicitly.
- **Nine UI languages**: English, Dutch, Chinese, French, German, Italian, Japanese, Korean and Spanish, chosen from the flag menu in the top bar and stored per
  workspace; server text stays English.
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
- **Measured DPD evidence (`dpd_measured`)**: play a `run_dpd` export through a
  physical PA, capture the output, and import the captures with the conditions
  you declare (`opendpd measurements import`, or the run page's "Import
  measured captures…"). The service aligns each capture to what was played
  (integer delay, one complex gain), scores it as captured under every
  profile, stores the files with their hashes and labels the result
  "user-provided, not independently verified"; the output level difference
  between the with/without captures is reported, never normalised. Measured
  and simulated results are shown side by side and never ranked
  (`docs/protocols/measured-dpd.md`).
- **Instrument adapters with a fail-closed interlock**: RF output off by
  default, arming by a named person, limits checked before anything is sent,
  timeouts, loss of link, adapter failures and aborts all end with the output
  off. Only the mock adapter ships (`opendpd instruments list`, `dry-run`);
  a real adapter needs `OPENDPD_ALLOW_RF_OUTPUT=1` in an approved laboratory
  session (`docs/architecture/instruments.md`).
- **Adaptation across conditions (`conditions-v1`)**: a sealed condition
  card (one device, one varied dimension, one registered dataset per
  condition from its own capture batch, roles fixed before any run), a
  pre-registered plan whose hash keys every run, and a report that keeps every
  cell: zero update, few-shot under a sample budget and full retrain, each with
  new samples, wall clock, device, seed spread and failures with their reasons
  (`opendpd adaptation card/plan/run/report`, the Studio's Robustness page).
  Executor additions behind it: `evaluate_pa`, `training.train_samples`,
  `initialization` (warm start from a run's checkpoint), `run_dpd` transfer
  through another condition's PA. Reports below the evidence bar (three
  measured conditions from independent batches) call themselves a rehearsal
  (`docs/protocols/conditions-v1.md`).
- **Streaming execution contract (`streaming-v1`)**: `reset / state / chunk /
  flush` over I/Q streams with declared look-ahead (samples and seconds, an
  information bound that is not a latency), history, measured warm-up and
  tail policy; `gru_stream` and `gmp_stream` execute the weights of `gru` and
  `gmp` with the state carried across chunks (`opendpd stream <run>`, the run
  page's "Score under streaming semantics"). Every streaming result records
  its chunk-consistency check against the full-sequence run of the same
  variant and is never ranked against, or inherited from, the offline
  segment scores (`docs/architecture/streaming.md`,
  `docs/releases/streaming-semantics-report.md`).
- **Bit-exact fixed-point deployment (`fixed-point-v1`)**: one model, one
  precision scheme (the one-layer GRU as `gru_stream`): every operator and
  every stored quantity specified (16-bit words, 32-bit pre-activations at a
  fixed fraction, exact accumulation, one rounding rule, saturation
  everywhere, sigmoid and tanh as tables); `opendpd deploy <run>` and the
  result page's Deployment panel write a package with the quantised weights,
  six golden vectors (normal, extreme, saturation, all-zero, state reset,
  long sequence) carrying the state after every sample, and a C99 reference
  that is compiled and verified bit for bit against the software reference
  (a mismatch is located at a sample and a signal). The report labels the
  float-to-fixed loss, the theoretical resources, the measured execution
  time of the C reference, and says that nothing was synthesised or
  measured for power (`docs/protocols/fixed-point-v1.md`).
- **Leaderboard tooling (`leaderboard-v1`)**: `opendpd leaderboard prepare`
  drafts a submission card and one share package per seed from finished
  runs; `check` runs the same review checklist for everyone and, with
  `--recompute`, reproduces every number from the packages in a fresh
  workspace; boards are versioned files seeded from the hash-bound benchmark
  report (`seed`), with `add`, `review` and `amend` keeping every review,
  correction and retraction in the history. Entries rank only inside one
  comparability group and are shown with uncertainty, resources, failure
  conditions and evidence grade. Every board calls itself a **reference
  benchmark** until three external submissions are accepted and two are
  independently recomputed (`docs/leaderboard/README.md`,
  `docs/community/governance.md`, `CITATION.cff`).
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

## Known limitations

- macOS and Windows desktop launches, real Safari and the system default
  browser are not verified by a person yet.
- The hosted service is a temporary trial with shared compute quotas; it is not permanent storage.
- No physical PA has been measured with the S16 path yet: the protocol is
  verified on synthetic and mock captures only, and no real instrument adapter
  exists (`docs/protocols/measured-dpd.md` §7).
- The fixed-point rules of `fixed-point-v1` are pending a maintainer's
  approval; only the C99 reference is verified, no ONNX / HLS / RTL target
  exists, and no synthesis or power number was measured.
- Streaming variants exist for `gru` and `gmp` only and are experimental
  until a maintainer approves them; models with a look-ahead (`tres_gru`,
  `tres_deltagru`, `tcn`) state their buffering cost but have no streaming
  variant (`docs/releases/streaming-semantics-report.md`).
- No condition set meets the `conditions-v1` evidence bar: the built-in APA
  card has two capture batches, the three-condition cards are synthetic, and
  nobody outside the implementation has recomputed a report
  (`docs/protocols/conditions-v1.md` §7–8).
- Regression baselines are drafts until a maintainer approves them. The
  leaderboard is a reference benchmark with the maintainers' own smoke-tier
  entries: no external submission has been accepted, nobody outside the
  maintainers has recomputed an entry, and no external researcher has
  reviewed the protocol yet (`docs/leaderboard/README.md`).
- CSV sources of stress size are parsed into RAM; use `.npy`/`.npz` for
  captures beyond about ten million samples.

## Artifacts

The GitHub release includes the wheel, source distribution and SHA-256 checksums.
PyPI publishing uses GitHub OIDC Trusted Publishing. The wheel bundles Studio,
so the installed app requires no Node.js build.
