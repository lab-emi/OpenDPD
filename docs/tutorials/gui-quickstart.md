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
