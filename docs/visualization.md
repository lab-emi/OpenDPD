# Visualizing experiments

## Live plots in Studio

The guided Experiment view shows progress for PA training/testing and DPD training/testing. Training curves track the metrics emitted by the worker; signal panels compare time-domain waveforms and spectra. Read the metric profile and evidence label alongside every score.

- **PA modeling:** inspect NMSE and the model output against the measured PA response.
- **DPD:** inspect the available linearization metrics and the cascade response. A PA-surrogate result is labeled as simulated; it does not replace a physical measurement.
- **Batch context:** the view reports I/Q sequences per batch, complex samples per sequence and sample rate, so framing is explicit.
- **Refresh rate:** plots are throttled and previews have bounded point counts. The worker adapts its preview cadence to limit overhead; display refreshes do not change the stored formal metrics.
- **Terminal:** expand the bar at the bottom for worker output. Its tabs let you inspect an earlier step's log without leaving the current experiment step.
- **Pan and zoom:** plots support interaction and recover the fitted view when panning leaves too little data visible. Use the plot controls to reset the view manually.

See the [Studio walkthrough](tutorials/gui-quickstart.md) for the guided flow. Exported reports use the stored result and plot data, so changing a browser view does not recompute a metric.

## Saved figures with the original CLI

The original pipeline can generate plots during training and save figures, animations and an HTML dashboard. These are separate from Studio's live preview controls.

Train the PA first, then enable plotting for DPD:

```bash
python main.py --dataset_name DPA_200MHz --step train_pa --accelerator cpu
python main.py --dataset_name DPA_200MHz --step train_dpd --accelerator cpu --plot --plot_every 10 --gif_duration 10.0
```

`--plot_every N` controls **epochs** between saved plot sets. Saving a set every epoch (`N=1`) adds work; a larger interval is useful for long runs. `--gif_duration` sets the animation duration in seconds.

| Output | Purpose |
| --- | --- |
| PSD | Compare spectral regrowth and the reference signal. |
| AM/AM and AM/PM | Inspect amplitude compression and phase distortion. |
| Waveforms and prediction error | Compare time-domain behavior. |
| Constellation | Inspect demodulated symbols when the dataset has a suitable demodulator. |
| Training curves | Track loss, ACLR, EVM and NMSE across epochs. |
| GIFs and `dashboard.html` | Review the saved plot sequence after training. |

After training, saved epoch plots are re-rendered with consistent axis limits for comparison. Metrics and constellations must be interpreted with their signal assumptions; see the [signal-metric FAQ](faq.md#why-do-apa_200mhz-metadata-and-paper-descriptions-differ).

![Example DPD training animation](../pics/overview_test.gif)

## Compare without and with DPD

After training and exporting DPD output, generate the comparison figures:

```bash
python main.py --dataset_name DPA_200MHz --step run_dpd --accelerator cpu
python main.py --dataset_name DPA_200MHz --step plot --accelerator cpu
```

The comparison includes PSD, AM/AM, AM/PM, constellation, waveform and a metric summary. A comparison produced through a learned PA model remains simulation evidence.

The equivalent Python options are:

```python
import opendpd

# A matching PA checkpoint must already exist.
opendpd.train_dpd(dataset_name="DPA_200MHz", n_epochs=100,
                  plot=True, plot_every=10)
opendpd.run_dpd(dataset_name="DPA_200MHz")
opendpd.plot_dpd(dataset_name="DPA_200MHz")
```

See [training](training.md) for the stage dependencies and [API reference](api.md) for all parameters.
