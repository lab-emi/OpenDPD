# OpenDPD Studio: one command to the workbench

```bash
pip install "opendpd[gui]"
opendpd gui
```

`opendpd gui` starts a local service on `127.0.0.1`, waits until it answers,
prints a one-time URL such as `http://127.0.0.1:8765/bootstrap?token=…` and
opens it in your default browser. No Node.js, no second terminal, no
copying of addresses. Press Ctrl+C to stop; running workers are terminated
and the workspace lock is released.

| Option | Effect |
|---|---|
| `--workspace DIR` | where datasets, runs and results live (default `$OPENDPD_WORKSPACE` or `~/opendpd-workspace`) |
| `--port N` | use exactly this port (fails if busy); without it the first free port from 8765 is used |
| `--no-browser` | print the URL only (SSH sessions, servers without a desktop) |

Starting `opendpd gui` again for the same workspace while one is running
opens the existing instance instead of a second server.

## What the browser shows

1. **Home**: the workspace path and the built-in example (measured DPA 200 MHz
   data). Register it with one click.
2. **Experiments → New**: pick a recipe (smoke recipes prove the pipeline in a
   minute and say so; research recipes are the paper settings), a dataset and
   a device. The server validates every change; errors are shown on the field.
3. **Run detail**: live progress, per-epoch metrics, logs, artifacts and the
   resolved configuration. Refreshing the page never resubmits anything.
4. **Results**: metrics with units and their evidence type (PA model, DPD on
   the surrogate, DPD measured). Mock data used for interface work is always
   badged **MOCK** and cannot be exported.

## Your own data

1. **Datasets → Import my data.** Files are read only from authorised import
   roots: `<workspace>/imports` (also the target of the upload button) and
   any directory added with `opendpd datasets add-root NAME PATH`. The browser
   never sends absolute paths; nothing outside these roots can be reached.
2. **Inspect.** Pick a CSV, `.npy` or `.npz` (or an existing OpenDPD split
   directory). The server reads the header/shape, shows the first rows, guesses
   the column mapping (`tx_i`, `rx_q`, … are understood) and lists problems.
   Fix swapped I/Q or wrong columns in the mapping selects; enter the sample
   rate, bandwidth, sub-channel count and `nperseg`, and confirm the
   amplitude units. Without this metadata formal metrics are blocked, not
   guessed.
3. **Import.** The original file is copied under `datasets/<id>/raw/`,
   hashed and never modified. The split (`contiguous-v1`: train | guard | val |
   guard | test, before any framing) and `raw-v1` are written.
4. **Run Dataset Doctor.** NaN/Inf, length, outliers, clipping plateaus,
   delay, gain/phase, amplitude range, bandwidth coverage and metadata are
   checked; each finding has evidence, a severity and a suggestion. Blocking
   findings stop evaluation until resolved.
5. **Preprocess → new version.** Accept the doctor's estimates (delay, gain,
   phase, outlier and NaN handling) or type your own, **Preview** the
   result (sample counts and the doctor's verdict after processing), then
   **Create version**. Versions are immutable; normalisation is fitted on the
   training split only and the fit range is recorded.
6. **New experiment → Data version** lets you train on any version; the
   result records which one was used.

The same flow headless:

```bash
opendpd datasets import capture.csv --id mine --fs 800e6 --bandwidth 200e6 --n-sub-ch 10 --nperseg 2560 --units normalized
opendpd datasets doctor mine --json
opendpd datasets preprocess mine --version aligned-v1 --delay 6 --preview
opendpd datasets preprocess mine --version aligned-v1 --delay 6
opendpd run --config exp.json   # with "dataset": {"id": "mine", "preprocessing_version": "aligned-v1"}
```

## DPD through a PA surrogate

1. **Train a PA model first** (a `pa-…` recipe on your dataset). A DPD recipe
   lists only succeeded PA runs of the same dataset as surrogate; without one
   it cannot start, and the validation error says what to train (same seed
   and frame length).
2. **Train the DPD** (a `dpd-…` recipe). The result page shows the chain
   **x → u = DPD(x) → y = PA(u)** with the source of every stage; `y` is
   marked *simulated* because it comes from the learned surrogate. Under the
   metrics, **Baselines under the same reference** scores the surrogate
   without DPD and the measured PA without DPD against the same linear
   target, and **Surrogate amplitude coverage** says how much of `u` leaves
   the amplitude range the surrogate was fitted on.
3. **Apply DPD to the test split…** on the DPD run page exports `u` (a PA
   *input*, downloadable from the result's chain table and the artifacts
   tab, with a `.meta.json` sidecar describing columns, order, dtype, scaling
   and the checkpoints used) and scores it through the chosen surrogate.
   Picking another PA run gives a new result; the DPD's own result is never
   overwritten. The run page's **Lineage** card shows which checkpoints a run
   used and which runs used it.
4. No absolute power is shown anywhere: the dataset carries no physical
   calibration, so dBm and efficiency are not derived.

The same headless:

```bash
opendpd run --config dpd.json --workspace WS        # "task": "train_dpd", "pa_reference": {"run_id": "run-…"}
opendpd apply run-DPD --workspace WS                 # export u = DPD(x), score through the training surrogate
opendpd apply run-DPD --workspace WS --pa run-PA2    # score the same DPD through another surrogate
```

## Classical baselines (MP / GMP by least squares)

- The recipes **pa-mp-ls-v1**, **pa-gmp-ls-v1**, **dpd-mp-ila-v1** and
  **dpd-gmp-ila-v1** fit the memory-polynomial baselines of the benchmark
  report in seconds on CPU. PA fits use direct least squares on the train
  split; DPD fits use indirect learning (ILA) on the measured train split and
  are scored through a gradient-trained PA surrogate like any other DPD.
- A fit has no seed and no epochs. Its stability is recorded instead: the
  retained rank, the condition number, the singular-value cutoff (`rcond`) and
  the train residual appear in the result's limitations and in `fit.json`.
- Every result names its **training path** (gradient, gradient through the
  surrogate, least squares, ILA least squares), so a comparison between a
  polynomial and a neural DPD shows that the two were obtained differently
  even when they are ranked under one protocol.
- A least-squares PA is a PA-modeling reference only; it is refused as a
  surrogate for DPD simulation.

## Benchmark protocol from the terminal

```bash
opendpd benchmark plan --dataset dpa-200mhz --tier cpu_regression --out plan.json
opendpd benchmark run plan.json --workspace WS
opendpd benchmark report plan.json --workspace WS --out report.json --markdown report.md
opendpd benchmark baseline report.json --out baseline.json
opendpd benchmark check report.json --baseline baseline.json
```

A plan fixes the model matrix, the budget and at least three seeds before
anything runs; the report lists every seed next to the aggregate with the
run ids and hashes behind each number; a baseline blocks a release only after
a maintainer approved it (`docs/protocols/benchmark-protocol.md`).

## Reproduce, export and import a configuration

- Every run's **Configuration** tab shows the resolved configuration (what
  actually ran, with its hash), a **Download configuration (JSON)** button and
  **Re-run with this configuration**, which opens the experiment form with
  that configuration imported.
- The experiment form's **Import configuration…** button accepts such a JSON
  file (the `resolution` block is dropped). An imported configuration is
  validated by the server and submitted exactly as is; the form fields are
  not applied to it.
- Model parameters in **Advanced settings** are generated from the model
  registry (`opendpd models`): the same names, types and limits the CLI and
  the Python API accept, so nothing is hand-written twice.
- The same configuration through the GUI and `opendpd run --config` resolves
  to the same hash and, on CPU with `reproducibility: hard`, to the same
  numbers (`tests/integration/test_entry_consistency.py`).

## Compare, charts, packages and reports

- A result page draws the **spectrum** (with the main and adjacent bands of
  the profile), a **time excerpt** and **AM-AM / AM-PM** from plot data the
  worker stored at run end (`plots/*.json`, plots-v1, fixed point budgets).
  Hiding a trace or zooming changes nothing in the stored metrics; the
  browser never recomputes a score. A run's **Overview** tab shows the
  per-epoch curves from the same source. Constellation/EVM plots are not
  drawn: there is no demodulation reference in the data.
- **Results → select two or more → Compare selected** puts the metrics side by
  side under one profile, marks the best value per metric, overlays the
  spectra with one colour per result and shows the configuration diff for a
  pair. Results from different data, data versions, splits, references,
  profile versions, evidence types or execution semantics are still shown,
  but the page says exactly which pair differs in what and does not rank
  them. **Download CSV** gives the same table.
- **Export share package** on a result page writes a zip with the resolved
  configuration, results under every profile, plot data, the checkpoints of
  every referenced run, the reports and the reproduction commands, and lists
  what it left out (worker logs, machine paths, your PA data) and what a
  recipient needs to re-evaluate. **Export full package (private)** adds the
  raw data and the used data version. Both stay in `<workspace>/exports/`
  until you move them.
- **Experiments → Import package…** verifies every file hash before anything
  is written, refuses damaged or conflicting packages with the reason, and
  reports whether the dataset arrived, already existed, was registered
  (built-in) or is missing. Re-evaluating an imported checkpoint reproduces
  the packaged numbers within the frozen tolerance; re-training is a new
  experiment and is never implied by that.
- **Report (HTML)** / **Report (Markdown)** render a document bound to the
  stored result and plot data. Same from the terminal:

```bash
opendpd export run-… --workspace WS --kind share     # or --kind full
opendpd import run-…-share-….zip --workspace WS2      # --inspect verifies only
opendpd evaluate run-… --workspace WS2 --profile legacy-opendpd-v1
opendpd report run-… --workspace WS --format html
```

## When something is wrong

- `opendpd doctor` prints versions, whether the frontend assets are present,
  the workspace state and a free port, and lists every blocking problem.
- If the page says the frontend is not available, the installed wheel was
  built without the frontend (source checkout without `npm run build`);
  install a release wheel or run `cd frontend && npm ci && npm run build`.
- Lost session (server restarted, cookie cleared): open the URL printed by
  `opendpd gui`, or paste its token into the "Session required" page.
- The service only ever listens on loopback. For remote machines use an SSH
  tunnel (`ssh -L 8765:127.0.0.1:8765 host`) and `--no-browser` on the host.

## Platform status

See `docs/releases/support-matrix.md`. Linux is verified by the packaged
test (`tests/packaging/test_wheel_install.py`); macOS and Windows launches
must be verified by a person on a real desktop before they are listed as
supported.
