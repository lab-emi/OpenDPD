# FAQ

## Why did pip install not give me Studio?

Install the current package in a fresh Python environment using the [uv quick start](install.md). Version 2.2.5 includes the built Studio frontend, PyTorch and pywebview by default; no Node.js build or extra is needed. Older 2.1 packages do not contain Studio.

## Which command should I use?

- **`opendpd gui`** opens Studio.
- **`opendpd run`** executes workspace experiments using the same service as Studio.
- **`opendpd-cli`** mirrors the original **`python main.py`** pipeline.
- The high-level **Python API** wraps the original training and dataset functions.

Use the [workspace CLI guide](tutorials/headless-cli.md), [training guide](training.md) or [API reference](api.md) for the corresponding path.

## Can I upload my own dataset?

Yes. Choose **Upload CSV** in Get Started or Datasets and supply paired PA input/output with signal metadata. Generated signals are input-only until PA Library creates a synthetic output. Contributions to the shared collection require explicit submission and human PR review; private datasets stay in the current workspace.

## Why is a DPD result labeled simulated?

DPD training uses a learned PA surrogate. Its output is a model prediction; the exported `u = DPD(x)` is the input to be sent to a PA. A physical capture and its provenance are needed for measured evidence. See [Measured DPD](tutorials/measured-dpd.md).

## Why do live scores differ from the final result?

A live signal preview uses a bounded probe and is labeled as such. The final test result follows the configured evaluation profile and split. Changing plot refresh, pan or zoom does not change the stored score. See [Visualization](visualization.md).

## Why do APA_200MHz metadata and paper descriptions differ?

The original paper describes a TM3.1a, 5 × 40 MHz (200 MHz) 256-QAM test signal captured at 983.04 MS/s. The legacy evaluation metadata models it as a single 200 MHz channel to avoid inappropriate channel segmentation in that metric implementation.

![Structure of the test signal](../pics/5GNR.png)

The legacy FFT-based metric calculation and the dataset-specific constellation demodulator serve different purposes. A constellation figure alone does not establish a standards-compliant EVM value. The original MATLAB reference `Matlab/calculate_200MHz_256QAM_evm.m` is specific to this signal.

For the current signal and demodulator details, read [Datasets](datasets.md). For score definitions, limitations and reference-bound evaluation, read [Metric profiles](protocols/metric-profiles.md) and [Waveform evaluation](tutorials/waveform-evaluation.md). These guides document the conventions; they do not change the frozen legacy metric.

## Where does the generated PA output come from?

Signal Generator produces x only. In [PA Library](guides/virtual-pa-library.md), choose a mathematical Virtual PA and explicitly simulate y, then create the paired dataset. Both signals are synthetic. To assess a physical device, upload measured pairs or import actual captures through the measured-DPD workflow. PSD charts distinguish DPD output / PA input from PA output; see [signal positions](guides/signal-chain-spectra.md).

## Why does a localhost Studio link refuse the connection?

The launcher must still be running, and `127.0.0.1` refers to the browser's own computer. When a Mac browser connects to Studio launched on a Linux server, use an SSH port forward. A failed automatic browser opener does not stop the service. See [installation troubleshooting](install.md#troubleshooting-connection-refused).

## What did Metric profile mean?

It is the versioned method used to calculate a score: averaging, spectral bands, normalization and reference conventions. The GUI now calls it **Metric calculation** and uses descriptive labels. It remains in Metric definitions because comparing results with different calculation methods is misleading; selecting it does not retrain the model or change the waveform.
