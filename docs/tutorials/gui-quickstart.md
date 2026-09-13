# Your first experiment in Studio

Open [Studio on the web](https://opendpd.com/studio/), or install the packaged local app with `pip install "opendpd[gui]==2.2.1"`.

```bash
opendpd gui
```

Studio opens locally in your browser, or in a native window if you installed the desktop extra. The interface and experiment services are the same in both.

## 1. Open and inspect a dataset

On **Home**, click **Get Started → Try a built-in dataset**. Choose **Add & inspect** beside **DPA_200MHz**, then **Inspect my dataset**.

The dataset page shows input/output I/Q signals, sample rate, bandwidth and signal quality. Use the **Dataset Doctor** tab to inspect findings, then **Configure experiment** to continue. Built-in measured datasets keep their supplied splits; **MyCustomPA** is explicitly labeled synthetic tutorial data.

Use **Get Started → Use my CSV file** to upload your own I/Q data. Studio validates the complete CSV in quarantine before preview. Invalid files are deleted; the hosted app removes all session files within 24 hours. Download your results before they expire. See [import and preprocessing](headless-cli.md#use-your-own-data) for local workflows.

## 2. Train and test a PA model

In **PA Model Training**, choose the dataset and model architecture. Under **Starting settings**, choose **Quick trial** for a short pipeline check (a smoke recipe) or **Full training** for a longer budget. Pick an available device and review the configuration. Continue through the review steps, optionally name the run, then click **Start run**.

The experiment form offers models from the shared registry. Advanced parameters come from that registry too; the server validates them before a run can start. Use the PA test step after a successful PA run to inspect predictions on held-out data.

During training, watch the **NMSE curve**, progress and signal comparisons. The view reports the configured and actual batch sizes, I/Q samples per sequence and sample rate. Live probe scores are previews; the final test result is identified separately.

## 3. Train and test DPD

Choose a DPD training recipe and select a **successful PA run on the same dataset** as its surrogate. The service checks checkpoint compatibility, including seed and frame length, before starting.

The chain is **x → u = DPD(x) → y = PA(u)**. The last stage uses the learned PA model and is labeled *simulated*. Inspect ACLR/NMSE or the metrics available under the selected profile, plus without/with-DPD signal comparisons. The result also records how much of the predistorted input lies outside the surrogate's fitted amplitude range.

The DPD test/apply step exports **u**, the predistorted PA input, with metadata identifying the signals and checkpoints used. It can score the cascade through the selected surrogate. This is not a measured linearized PA output; follow the [measured DPD guide](measured-dpd.md) for physical captures.

## 4. Follow progress in the terminal

The **Terminal** bar at the bottom starts collapsed. It highlights with a **Running** status while an experiment is active. Expand it to read the worker's actual output.

Its tabs follow the experiment step when you move from PA to DPD. You can also select an earlier step's terminal or another run without navigating away from the current experiment. The run-detail **Logs** tab provides the saved log and filtering controls.

Quick trial defaults to **10 epochs** and full training to **150 epochs**. Plots update **once per epoch**, reusing validation results. Under **Advanced settings → Plot updates**, you can choose a batch interval. This option and its warning are red: extra previews can severely slow training. The chosen cadence is saved with the run and shown above its plots. See [Visualization](../visualization.md) for live and saved plot behavior.

## 5. Keep and compare the results

- **Run detail:** inspect metrics, logs, artifacts and the resolved configuration. Reloading the page does not submit another run.
- **Configuration:** download the JSON or use **Re-run with this configuration**. The same configuration can run through `opendpd run --config`.
- **Results:** select results to compare metrics, spectra and configuration differences. Incompatible datasets, profiles, references or evidence types are explained and are not ranked together.
- **Export:** create a share package or a full private package, and generate HTML/Markdown reports. Share packages omit raw PA data and worker logs; full packages include the data. See the [package protocol](../protocols/experiment-packages.md).

For classical MP/GMP fits, streaming and deployment, continue to [Advanced guides](../advanced.md).

## Language, theme and reset

The language menu offers **English, Dutch and Chinese first**, followed by French, German, Italian, Japanese, Korean and Spanish. Your choice is stored in the workspace. Interface text and generated report prose are localized; raw worker logs, commands, configuration keys, user text and GitHub messages retain their original text for provenance.

Select a light, dark or system theme in **Settings**. The layout adapts to the window size. Interactive plots can be panned and zoomed; double-click to fit, and an automatic fit recovers views that have moved too far from the data.

**Reset page** restarts the current page's workflow after a confirmation explaining what progress will be removed. **Cancel** keeps it; continue only when you want to start that page again. **Reset Studio** has a broader scope and its own confirmation.

## Launch options

| Option | Effect |
| --- | --- |
| `--workspace DIR` | Choose the data/results directory; default is `$OPENDPD_WORKSPACE` or `~/opendpd-workspace`. |
| `--port N` | Use exactly this port; otherwise use the first free port starting at 8765. |
| `--browser` | Open the system browser even if the native backend is available. |
| `--window` | Require a native window and report why it cannot start. |
| `--no-browser` | Print the local session URL for a headless or SSH session. |

A second launch for the same active workspace opens the existing instance. Close Studio or press Ctrl+C to stop the launcher; active workers trigger a confirmation before shutdown.

The native window uses the operating system's web view. Run `opendpd doctor` for backend diagnostics; on Linux a WebKit2GTK installation may need `python3-gi gir1.2-webkit2-4.1`, or install a Qt backend with `python -m pip install "pywebview[qt]"`. Consult the [support matrix](../releases/support-matrix.md) for verified platforms.

For remote use, keep the server on loopback and use an SSH tunnel, for example `ssh -L 8765:127.0.0.1:8765 host`, with `--no-browser` on the host. If the session expires, reopen the current launcher's URL. Other setup problems are covered in [Installation](../install.md#troubleshooting).
