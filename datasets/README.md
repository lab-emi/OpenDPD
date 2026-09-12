# Datasets

This directory contains PA (Power Amplifier) measurement datasets for training and evaluating DPD (Digital Pre-Distortion) models. Each dataset consists of time-aligned input/output I/Q samples captured from a real PA device.

## Quick Start

```bash
# Generate diagnostic plots for a dataset
python datasets/DPA_200MHz/plot_dataset.py
```

```python
# Load a dataset via the Python API
import opendpd
X_train, y_train, X_val, y_val, X_test, y_test = opendpd.load_dataset('DPA_200MHz')
```

## Directory Structure

Each dataset folder contains:

```
datasets/<name>/
  spec.json             # Signal parameters and metadata
  demod.py              # Dataset-specific OFDM demodulator
  plot_dataset.py       # Quick-look plot generation script
  train_input.csv       # Training input I/Q samples
  train_output.csv      # Training output I/Q samples
  val_input.csv         # Validation input I/Q samples
  val_output.csv        # Validation output I/Q samples
  test_input.csv        # Test input I/Q samples
  test_output.csv       # Test output I/Q samples
```

Single-CSV datasets (e.g. `MyCustomPA`) use a single `data.csv` with columns `I_in, Q_in, I_out, Q_out` instead of separate split files.

Shared utilities live at the top level:

```
datasets/
  demodulator.py        # Base demodulator classes (OFDMCPDemodulator, IFFTFrameDemodulator)
  plot_utils.py         # Shared plotting functions for all datasets
```

## CSV Format

Each CSV file has two columns: `I` (in-phase) and `Q` (quadrature), representing baseband complex I/Q samples. The input CSV is the signal fed to the PA; the output CSV is the PA's response, time-aligned sample-by-sample.

## spec.json Reference

Every dataset has a `spec.json` that describes the signal parameters. Fields fall into three categories.

### Core fields (all datasets)

| Field | Type | Description |
|-------|------|-------------|
| `description` | string | Human-readable summary of the dataset |
| `dataset_format` | string | `"split_csv"` (separate train/val/test files) or `"single_csv"` (one `data.csv`) |
| `split_ratios` | object | Fraction of data for `train`, `val`, `test` (must sum to 1.0) |
| `input_signal_fs` | float | Sampling rate of the I/Q data (Hz) |
| `bw_main_ch` | float | Total occupied bandwidth of the composite signal (Hz) |
| `bw_sub_ch` | float | Bandwidth per sub-channel / carrier spacing (Hz) |
| `n_sub_ch` | int | Number of sub-channels (carriers) |
| `nperseg` | int | Segment length in samples. For DPA datasets this is the IFFT frame size used during signal generation; for APA datasets and metrics it is the Welch PSD segment size. |
| `modulation` | string | QAM order of the data subcarriers (e.g. `"64QAM"`, `"256QAM"`, `"1024QAM"`) |

### LTE / OFDM fields (APA and some DPA datasets)

| Field | Type | Description |
|-------|------|-------------|
| `standard` | string | Wireless standard (e.g. `"LTE"`) |
| `scs` | float | Subcarrier spacing (Hz). For APA datasets this is the *effective* SCS at the capture rate. |
| `ofdm_nfft` | int | OFDM FFT size in samples |
| `n_active` | int | Number of active (data-bearing) subcarriers extracted per carrier for constellation demodulation |
| `cp_first` | int | Cyclic prefix length of the first OFDM symbol (samples) |
| `cp_other` | int | Cyclic prefix length of subsequent OFDM symbols (samples) |
| `test_model` | string | LTE test model (e.g. `"TM3.1a"`) |
| `papr_db` | float | Peak-to-average power ratio (dB) |

### Single-CSV fields (MyCustomPA)

| Field | Type | Description |
|-------|------|-------------|
| `csv_filename` | string | Name of the CSV file (default `"data.csv"`) |
| `train_end` | int | Sample index where training split ends |
| `val_end` | int | Sample index where validation split ends |

## Diagnostic Plots

Run `python datasets/<name>/plot_dataset.py` from the project root to generate five diagnostic plots saved in the dataset folder:

| Plot | Filename | Description |
|------|----------|-------------|
| Time-domain waveform | `waveform.png` | First ~1000 samples of I and Q channels, input vs output overlay |
| Power spectral density | `psd.png` | Frame-aligned PSD (averaged \|FFT\|^2 per `nperseg` frame), input vs output |
| Constellation | `constellation.png` | Demodulated QAM constellation for input (clean) and output (after PA, with equalization) |
| AM/AM | `amam.png` | Normalized input amplitude vs output amplitude (shows gain compression) |
| AM/PM | `ampm.png` | Normalized input amplitude vs phase difference (shows phase distortion) |

## Signal Generation and Demodulation

The datasets use two fundamentally different signal structures, each with a corresponding demodulator.

### DPA datasets: IFFT-concatenated frames (no cyclic prefix)

**Applies to:** DPA_100MHz, DPA_160MHz, DPA_200MHz, MyCustomPA

**How the signal is generated:**
The transmit signal is constructed by mapping random QAM symbols onto frequency-domain subcarriers across multiple carriers, taking the IFFT, and concatenating the resulting time-domain frames back-to-back without cyclic prefix insertion. Each frame is exactly `nperseg` samples long.

**How the signal is demodulated (`IFFTFrameDemodulator`):**

1. Chop the raw signal into non-overlapping frames of `nperseg` samples.
2. FFT each frame once (one FFT covers all carriers simultaneously).
3. For each carrier, read the `n_active` subcarrier bins centered on the carrier's frequency offset.
4. RMS-normalize and return the constellation points.

Because each frame is a complete IFFT output, the FFT perfectly inverts the generation process with no spectral leakage. No bandpass filtering or carrier isolation is needed.

**Key parameters:**

- `nperseg` must exactly match the IFFT frame size used during generation. Incorrect values produce a blurred constellation.
- `n_active` is auto-computed as `bw_sub_ch / (fs / nperseg)` if not specified.

### APA datasets: Standard OFDM with cyclic prefix

**Applies to:** APA_200MHz, APA_200MHz_b

**How the signal is generated:**
The signal is a standard LTE waveform (TM3.1a) generated at 491.52 MHz with SCS = 15 kHz, then transmitted and captured at 983.04 MHz (effectively doubling the SCS to 30 kHz). It consists of 5 independently-timed LTE carriers at 40 MHz spacing, each carrying 20 MHz of 256QAM data on PDSCH (Physical Downlink Shared Channel). Different OFDM symbols within the LTE frame carry different channels (PDCCH uses QPSK, PDSCH uses 256QAM).

**How the signal is demodulated (`OFDMCPDemodulator`):**

1. For each carrier, frequency-shift to baseband and bandpass-filter.
2. Find OFDM symbol boundaries via cyclic prefix correlation.
3. Fine-tune the FFT start offset by minimizing kurtosis over a wider subcarrier range (1200 bins) for timing sensitivity.
4. Use the first (earliest) detected symbol per carrier. Different OFDM symbols carry different LTE channels (PDCCH vs PDSCH) with different modulation, so mixing symbols creates constellation artifacts.
5. FFT the symbol (after skipping the cyclic prefix) and extract `n_active` subcarriers.
6. For output signals, apply per-subcarrier zero-forcing equalization using the clean input as reference to remove the PA's linear frequency response, revealing only nonlinear distortion.

**Key parameters:**

- `ofdm_nfft`: FFT size (32768). At the effective 30 kHz SCS, this equals `fs / scs`.
- `n_active`: Set to 600, covering the 18 MHz occupied bandwidth of each 20 MHz LTE carrier at 30 kHz bin spacing. Setting this too large (e.g. 1200) includes guard-band subcarriers that create circular artifacts on the constellation.
- `cp_other`: Cyclic prefix length (2304 samples) used to locate symbol boundaries.

**Note on sample rate:** The CSV data contains 98304 samples. The spec lists `input_signal_fs = 983.04 MHz` (the signal generator / capture rate), but the signal was originally generated at 491.52 MHz with `ofdm_nfft = 32768` and `SCS = 15 kHz`. The MATLAB reference code (`Matlab/calculate_200MHz_256QAM_evm.m`) operates at 491.52 MHz on the same samples. Both interpretations are valid since the frequency ratios are consistent; the spec uses the transmission rate.

## Dataset Details

### DPA_200MHz

10-carrier LTE 20MHz signal through a Doherty PA (DPA) device.

| Parameter | Value |
|-----------|-------|
| Carriers | 10 x 20 MHz |
| Total bandwidth | 200 MHz |
| Sampling rate | 800 MHz |
| Modulation | 64QAM |
| IFFT frame size (`nperseg`) | 2560 |
| Active subcarriers per carrier | 64 |
| Demodulator | `IFFTFrameDemodulator` |

### DPA_160MHz

4-carrier 40MHz signal through a DPA device.

| Parameter | Value |
|-----------|-------|
| Carriers | 4 x 40 MHz |
| Total bandwidth | 160 MHz |
| Sampling rate | 640 MHz |
| Modulation | 1024QAM |
| IFFT frame size (`nperseg`) | 16384 |
| Active subcarriers per carrier | 1024 |
| Demodulator | `IFFTFrameDemodulator` |

### DPA_100MHz

5-carrier LTE 20MHz signal through a DPA device.

| Parameter | Value |
|-----------|-------|
| Carriers | 5 x 20 MHz |
| Total bandwidth | 100 MHz |
| Sampling rate | 800 MHz |
| Modulation | 64QAM |
| IFFT frame size (`nperseg`) | 1280 |
| Active subcarriers per carrier | 32 |
| Demodulator | `IFFTFrameDemodulator` |

### APA_200MHz

5-carrier LTE 20MHz TM3.1a signal through an Auxiliary PA (APA) device.

| Parameter | Value |
|-----------|-------|
| Carriers | 5 x 20 MHz (40 MHz spacing) |
| Total bandwidth | 200 MHz |
| Sampling rate | 983.04 MHz |
| Generation rate | 491.52 MHz |
| Modulation | 256QAM (PDSCH) / QPSK (PDCCH) |
| LTE test model | TM3.1a |
| OFDM FFT size | 32768 |
| Active subcarriers | 600 per carrier |
| Cyclic prefix | 2304 samples (normal) |
| Subcarrier spacing | 30 kHz (effective) |
| PAPR | 10.0 dB |
| Demodulator | `OFDMCPDemodulator` |

### APA_200MHz_b

Second measurement of the same signal type as APA_200MHz on the same APA device. Identical signal parameters; different PA operating conditions or measurement instance.

### MyCustomPA

Template dataset for user-provided PA measurements. Uses single-CSV format with a single wideband carrier.

| Parameter | Value |
|-----------|-------|
| Carriers | 1 (single channel) |
| Total bandwidth | 200 MHz |
| Sampling rate | 800 MHz |
| IFFT frame size (`nperseg`) | 2560 |
| Demodulator | `IFFTFrameDemodulator` |

## Adding a Custom Dataset

1. Create a folder under `datasets/` with your dataset name.
2. Place your I/Q CSV files (either split or single format).
3. Create a `spec.json` with the signal parameters (see reference above).
4. Create a `demod.py` that subclasses the appropriate demodulator:

```python
# For IFFT-concatenated signals (no cyclic prefix):
from datasets.demodulator import IFFTFrameDemodulator

class Demodulator(IFFTFrameDemodulator):
    pass

# For standard OFDM with cyclic prefix:
from datasets.demodulator import OFDMCPDemodulator

class Demodulator(OFDMCPDemodulator):
    pass
```

5. Create a `plot_dataset.py`:

```python
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from datasets.plot_utils import plot_dataset

if __name__ == '__main__':
    plot_dataset('YourDatasetName')
```

6. Generate diagnostic plots to verify: `python datasets/YourDatasetName/plot_dataset.py`
