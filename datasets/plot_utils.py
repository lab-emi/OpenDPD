"""Shared dataset visualization with publication-quality styling.

Generates five diagnostic plots for any OpenDPD dataset:
  1. Time-domain waveform (I/Q)
  2. Power spectral density
  3. Input constellation (via per-dataset demodulator)
  4. AM/AM characteristic
  5. AM/PM characteristic

Usage from any dataset folder::

    python datasets/<name>/plot_dataset.py
"""

__author__ = "Chang Gao"
__license__ = "Apache-2.0 License"
__email__ = "chang.gao@tudelft.nl"

import os
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Load scientific-figure-pro styling (optional — graceful fallback)
# ---------------------------------------------------------------------------
from utils.plotting import publication_style


def _savefig(fig, path, dpi=300):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


# --- Palette ---------------------------------------------------------------
C_INPUT = '#0F4D92'   # blue_main
C_OUTPUT = '#B64342'  # red_strong
C_CONST = '#0F4D92'


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@publication_style
def plot_dataset(dataset_name: str):
    """Generate all quick-look plots for *dataset_name* and save them
    in ``datasets/<dataset_name>/``."""


    ds_dir = os.path.join(os.path.dirname(__file__), dataset_name)
    spec_path = os.path.join(ds_dir, 'spec.json')
    with open(spec_path) as f:
        spec = json.load(f)

    fmt = spec.get('dataset_format', 'split_csv')
    fs = spec.get('input_signal_fs', 800e6)
    nperseg = spec.get('nperseg', 2560)

    # --- Load data ----------------------------------------------------------
    if fmt == 'single_csv':
        csv_file = spec.get('csv_filename', 'data.csv')
        df = pd.read_csv(os.path.join(ds_dir, csv_file))
        inp = df[['I_in', 'Q_in']].values if 'I_in' in df.columns else df.iloc[:, :2].values
        out = df[['I_out', 'Q_out']].values if 'I_out' in df.columns else df.iloc[:, 2:4].values
    else:
        dfs_in, dfs_out = [], []
        for split in ['train', 'val', 'test']:
            p_in = os.path.join(ds_dir, f'{split}_input.csv')
            p_out = os.path.join(ds_dir, f'{split}_output.csv')
            if os.path.exists(p_in):
                dfs_in.append(pd.read_csv(p_in))
            if os.path.exists(p_out):
                dfs_out.append(pd.read_csv(p_out))
        inp = pd.concat(dfs_in, ignore_index=True).values
        out = pd.concat(dfs_out, ignore_index=True).values

    sig_in = inp[:, 0] + 1j * inp[:, 1]
    sig_out = out[:, 0] + 1j * out[:, 1]

    total = len(sig_in)
    desc = spec.get('description', dataset_name)
    print(f"Dataset: {dataset_name}")
    print(f"  {desc}")
    print(f"  {total} samples, fs={fs/1e6:.2f} MHz")

    # --- 1. Time-domain waveform -------------------------------------------
    n_show = min(1000, total)
    t_us = np.arange(n_show) / fs * 1e6

    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    axes[0].plot(t_us, inp[:n_show, 0], lw=0.8, color=C_INPUT, label='Input')
    axes[0].plot(t_us, out[:n_show, 0], lw=0.8, color=C_OUTPUT, alpha=0.7, label='Output')
    axes[0].set_ylabel('In-phase (I)')
    axes[0].legend(loc='upper right')
    axes[0].set_title(f'{dataset_name} — Time-Domain Waveform', loc='left', fontweight='bold')

    axes[1].plot(t_us, inp[:n_show, 1], lw=0.8, color=C_INPUT, label='Input')
    axes[1].plot(t_us, out[:n_show, 1], lw=0.8, color=C_OUTPUT, alpha=0.7, label='Output')
    axes[1].set_ylabel('Quadrature (Q)')
    axes[1].set_xlabel('Time (\u03bcs)')
    axes[1].legend(loc='upper right')

    for ax in axes:
        ax.grid(alpha=0.2, linestyle='--')
    _savefig(fig, os.path.join(ds_dir, 'waveform.png'))
    print('  Saved waveform.png')

    # --- 2. Power spectral density -----------------------------------------

    def _psd_frame_aligned(sig, fs, nperseg):
        """PSD by averaging |FFT|^2 over nperseg-aligned frames.

        For IFFT-concatenated signals each frame is a complete OFDM
        symbol, so rectangular windowing is exact — no spectral leakage.
        """
        n_frames = len(sig) // nperseg
        if n_frames == 0:
            n_frames = 1
        psd = np.zeros(nperseg)
        for i in range(n_frames):
            fd = np.fft.fftshift(np.fft.fft(sig[i * nperseg:(i + 1) * nperseg]))
            psd += np.abs(fd) ** 2
        psd /= n_frames
        f = np.fft.fftshift(np.fft.fftfreq(nperseg, d=1.0 / fs))
        return f, psd

    fig, ax = plt.subplots(figsize=(10, 4))
    for sig, lbl, clr, ls in [(sig_in, 'Input', C_INPUT, '-'),
                               (sig_out, 'Output', C_OUTPUT, '--')]:
        f, p = _psd_frame_aligned(sig, fs, nperseg)
        p_db = 10 * np.log10(p / np.max(p) + 1e-30)
        ax.plot(f / 1e6, p_db, lw=1.2, color=clr, linestyle=ls, label=lbl)
    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('Normalized PSD (dB)')
    ax.set_title(f'{dataset_name} — Power Spectral Density', loc='left', fontweight='bold')
    ax.set_ylim([-80, 5])
    ax.legend()
    ax.grid(alpha=0.2, linestyle='--')
    _savefig(fig, os.path.join(ds_dir, 'psd.png'))
    print('  Saved psd.png')

    # --- 3. Constellation ---------------------------------------------------
    from datasets.demodulator import Demodulator
    try:
        demod = Demodulator.from_dataset(dataset_name)
        re_in, im_in = demod.demodulate(sig_in)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

        axes[0].scatter(re_in, im_in, s=5, alpha=0.4, color=C_CONST, edgecolors='none')
        axes[0].set_xlabel('In-phase (I)')
        axes[0].set_ylabel('Quadrature (Q)')
        axes[0].set_title('Input', loc='left', fontweight='bold')
        axes[0].set_aspect('equal')
        axes[0].grid(alpha=0.2, linestyle='--')

        re_out, im_out = demod.demodulate(sig_out, sync_signal=sig_in,
                                          equalize=True)
        axes[1].scatter(re_out, im_out, s=5, alpha=0.4, color=C_OUTPUT, edgecolors='none')
        axes[1].set_xlabel('In-phase (I)')
        axes[1].set_ylabel('Quadrature (Q)')
        axes[1].set_title('Output (after PA)', loc='left', fontweight='bold')
        axes[1].set_aspect('equal')
        axes[1].grid(alpha=0.2, linestyle='--')

        # Use consistent axis limits based on the input constellation
        lim = max(np.abs(re_in).max(), np.abs(im_in).max()) * 1.3
        for ax in axes:
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)

        fig.suptitle(f'{dataset_name} — Constellation', fontweight='bold')
        _savefig(fig, os.path.join(ds_dir, 'constellation.png'))
        print('  Saved constellation.png')
    except Exception as e:
        print(f'  Constellation skipped: {e}')

    # --- 4. AM/AM -----------------------------------------------------------
    amp_in = np.sqrt(inp[:, 0]**2 + inp[:, 1]**2)
    amp_out = np.sqrt(out[:, 0]**2 + out[:, 1]**2)
    max_in = np.max(amp_in) + 1e-30

    n_pts = len(amp_in)
    idx = np.random.default_rng(42).choice(n_pts, min(n_pts, 20000), replace=False)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(amp_in[idx] / max_in, amp_out[idx] / max_in,
               s=1, alpha=0.3, color=C_INPUT, edgecolors='none')
    ax.set_xlabel('Normalized Input Amplitude')
    ax.set_ylabel('Normalized Output Amplitude')
    ax.set_title(f'{dataset_name} — AM/AM', loc='left', fontweight='bold')
    ax.grid(alpha=0.2, linestyle='--')
    _savefig(fig, os.path.join(ds_dir, 'amam.png'))
    print('  Saved amam.png')

    # --- 5. AM/PM -----------------------------------------------------------
    phase_in = np.arctan2(inp[:, 1], inp[:, 0])
    phase_out = np.arctan2(out[:, 1], out[:, 0])
    phase_diff = np.degrees(np.angle(np.exp(1j * (phase_out - phase_in))))

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(amp_in[idx] / max_in, phase_diff[idx],
               s=1, alpha=0.3, color=C_INPUT, edgecolors='none')
    ax.set_xlabel('Normalized Input Amplitude')
    ax.set_ylabel('Phase Difference (degrees)')
    ax.set_title(f'{dataset_name} — AM/PM', loc='left', fontweight='bold')
    ax.grid(alpha=0.2, linestyle='--')
    _savefig(fig, os.path.join(ds_dir, 'ampm.png'))
    print('  Saved ampm.png')

    print(f'  All plots saved to datasets/{dataset_name}/')
