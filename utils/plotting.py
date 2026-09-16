__author__ = "Chang Gao"
__license__ = "Apache-2.0 License"
__email__ = "chang.gao@tudelft.nl"

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from utils.metrics import IQ_to_complex, power_spectrum, moving_average, EVM, ACLR

# ===========================================================================
# Publication-quality styling
# ===========================================================================

PUBLICATION_STYLE = {
    'font.size': 12, 'axes.spines.right': False, 'axes.spines.top': False,
    'legend.frameon': False, 'savefig.bbox': 'tight',
}


def publication_style(function):
    """Apply deterministic figure defaults only for the duration of a call."""
    from functools import wraps
    @wraps(function)
    def wrapped(*args, **kwargs):
        with plt.rc_context(PUBLICATION_STYLE):
            return function(*args, **kwargs)
    return wrapped


# --- Palette constants -----------------------------------------------------
C_PRED = '#0F4D92'     # blue_main - predictions / with DPD
C_GT = '#B64342'        # red_strong - ground truth / without DPD
C_REF = '#8BCF8B'       # green - reference / linear target / input
C_SCATTER_ALPHA = 0.3

# Datasets that need full-sequence input for proper OFDM constellation
# (OFDM symbol length > individual val/test split size)
_FULL_SEQ_CONSTELLATION_DATASETS = {'APA_200MHz', 'APA_200MHz_b'}


def needs_full_seq_constellation(dataset_name):
    """Check if dataset needs full-sequence data for proper OFDM constellation."""
    return dataset_name in _FULL_SEQ_CONSTELLATION_DATASETS


def load_full_dataset_iq(dataset_name):
    """Load all splits combined for full-sequence constellation plotting.

    Returns:
        full_input: np.ndarray of shape [N, 2] (I, Q)
        full_output: np.ndarray of shape [N, 2] (I, Q)
    """
    from modules.data_collector import load_dataset
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(dataset_name)
    full_input = np.concatenate([X_train, X_val, X_test], axis=0)
    full_output = np.concatenate([y_train, y_val, y_test], axis=0)
    return full_input, full_output


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _iq_to_amp_phase(iq_signal):
    """Convert IQ signal [..., 2] to amplitude and phase."""
    I = iq_signal[..., 0]
    Q = iq_signal[..., 1]
    amp = np.sqrt(I**2 + Q**2)
    phase = np.arctan2(Q, I)
    return amp, phase


def _savefig(fig, filepath):
    _ensure_dir(os.path.dirname(filepath))
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)


def _epoch_title(base_title, epoch):
    """Append epoch number to a plot title if epoch is not None."""
    if epoch is not None:
        return f"{base_title}  [Epoch {epoch}]"
    return base_title


def _compute_evm_for_iq(pred_flat, gt_flat, fs, nperseg, bw_main_ch, n_sub_ch):
    """Compute EVM (dB) from IQ arrays [N, 2]."""
    try:
        n_total = pred_flat.shape[0]
        n_seg = n_total // nperseg
        if n_seg > 0:
            seg_len = n_seg * nperseg
            pred_seg = pred_flat[:seg_len].reshape(n_seg, nperseg, 2)
            gt_seg = gt_flat[:seg_len].reshape(n_seg, nperseg, 2)
        else:
            pred_seg = pred_flat[np.newaxis, :]
            gt_seg = gt_flat[np.newaxis, :]
        return EVM(pred_seg, gt_seg, sample_rate=int(fs),
                   bw_main_ch=bw_main_ch, n_sub_ch=n_sub_ch, nperseg=nperseg)
    except Exception:
        return None


# ===========================================================================
# Plot directory helpers
# ===========================================================================

def get_plot_dir_train_pa(dataset_name, pa_model_id):
    return os.path.join('plots', dataset_name, 'train_pa', pa_model_id)


def get_plot_dir_train_dpd(dataset_name, pa_model_id, dpd_model_id):
    return os.path.join('plots', dataset_name, 'train_dpd', pa_model_id, dpd_model_id)


def get_plot_dir_run_dpd(dataset_name, dpd_model_id):
    return os.path.join('plots', dataset_name, 'run_dpd', dpd_model_id)


def get_plot_dir_compare(dataset_name, dpd_model_id):
    return os.path.join('plots', dataset_name, 'compare', dpd_model_id)


# ===========================================================================
# ACLR annotation helper
# ===========================================================================

def _compute_aclr_for_complex(sig_complex, fs, nperseg, bw_main_ch, n_sub_ch):
    """Compute ACLR (left, right) for a 1-D complex signal.

    Converts to IQ segment form expected by ``metrics.ACLR``.
    Returns (aclr_left, aclr_right) in dB, or (None, None) on failure.
    """
    try:
        # Build [n_seg, nperseg, 2] from flat complex
        sig = sig_complex.flatten()
        n_seg = max(1, len(sig) // nperseg)
        sig = sig[:n_seg * nperseg].reshape(n_seg, nperseg)
        iq = np.stack([sig.real, sig.imag], axis=-1)  # [n_seg, nperseg, 2]
        aclr_l, aclr_r = ACLR(iq, fs=fs, nperseg=nperseg,
                               bw_main_ch=bw_main_ch, n_sub_ch=n_sub_ch)
        return aclr_l, aclr_r
    except Exception:
        return None, None


def _annotate_aclr(ax, lines_info, fs, nperseg, bw_main_ch, n_sub_ch,
                   evm_info=None):
    """Add ACLR and optional EVM text annotation to a PSD axes.

    Parameters
    ----------
    ax : matplotlib Axes
    lines_info : list of (label, complex_signal)
        Each entry produces one ACLR annotation line.
    fs, nperseg, bw_main_ch, n_sub_ch : signal parameters forwarded to ACLR.
    evm_info : list of (label, evm_db), optional
        Pre-computed EVM values to display alongside ACLR.
    """
    parts = []
    max_label_len = max(len(lbl) for lbl, _ in lines_info) if lines_info else 0
    if evm_info:
        max_label_len = max(max_label_len, max(len(lbl) for lbl, _ in evm_info))
    for lbl, sig_c in lines_info:
        al, ar = _compute_aclr_for_complex(sig_c, fs, nperseg, bw_main_ch, n_sub_ch)
        if al is not None and ar is not None:
            avg = (al + ar) / 2.0
            parts.append(f"{lbl:<{max_label_len}s}: ACLR={avg:+.1f} dB")
    if evm_info:
        for lbl, evm_db in evm_info:
            parts.append(f"{lbl:<{max_label_len}s}:  EVM={evm_db:+.1f} dB")
    if parts:
        txt = "\n".join(parts)
        ax.text(0.98, 0.95, txt, transform=ax.transAxes,
                ha='right', va='top', fontsize=9, family='monospace',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))


# ===========================================================================
# Core plotting functions
# ===========================================================================

def _ensure_2d_segments(sig, nperseg):
    """Reshape a 1D complex signal into 2D segments for power_spectrum.

    power_spectrum uses np.mean(ps, axis=0) to average over segments,
    which collapses a 1D array to a scalar. This ensures proper 2D input.
    """
    if sig.ndim == 1:
        n_seg = max(1, len(sig) // nperseg)
        sig = sig[:n_seg * nperseg].reshape(n_seg, nperseg)
    return sig


@publication_style
def plot_psd(signal_complex, save_path, label='Signal',
             ref_complex=None, ref_label='Reference',
             fs=800e6, nperseg=2560, smoothing_window=10):
    """Plot power spectral density. Optionally overlay a reference signal."""
    fig, ax = plt.subplots(figsize=(10, 5))

    def _plot_one(sig, lbl, color, ls='-'):
        sig_2d = _ensure_2d_segments(sig, nperseg)
        freq, ps = power_spectrum(sig_2d, fs=fs, nperseg=nperseg)
        ps_db = 10 * np.log10(ps / np.max(ps) + 1e-30)
        ps_smooth = moving_average(ps_db, smoothing_window)
        # moving_average drops (window_size - 1) elements from the front
        freq_adj = freq[smoothing_window - 1:]
        # Ensure matching lengths
        n = min(len(freq_adj), len(ps_smooth))
        ax.plot(freq_adj[:n] / 1e6, ps_smooth[:n], label=lbl, color=color,
                linestyle=ls, lw=1.5)

    _plot_one(signal_complex, label, C_PRED)
    if ref_complex is not None:
        _plot_one(ref_complex, ref_label, C_GT, '--')

    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('Normalized PSD (dB)')
    ax.set_title('Power Spectral Density', loc='left', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.2, linestyle='--')
    fig.tight_layout()
    _savefig(fig, save_path)


@publication_style
def plot_amam(input_iq, output_iq, save_path, label='Model',
              ref_output_iq=None, ref_label='Reference'):
    """AM/AM characteristic: input amplitude vs output amplitude."""
    amp_in, _ = _iq_to_amp_phase(input_iq)
    amp_out, _ = _iq_to_amp_phase(output_iq)
    # Flatten
    amp_in_flat = amp_in.flatten()
    amp_out_flat = amp_out.flatten()
    # Normalize
    max_in = np.max(amp_in_flat) + 1e-30
    max_out = np.max(amp_out_flat) + 1e-30

    fig, ax = plt.subplots(figsize=(6, 6))
    # Subsample for scatter
    n = len(amp_in_flat)
    idx = np.random.choice(n, min(n, 20000), replace=False)
    ax.scatter(amp_in_flat[idx] / max_in, amp_out_flat[idx] / max_out,
               s=1, alpha=C_SCATTER_ALPHA, color=C_PRED, label=label,
               edgecolors='none')

    if ref_output_iq is not None:
        amp_ref, _ = _iq_to_amp_phase(ref_output_iq)
        amp_ref_flat = amp_ref.flatten()
        max_ref = np.max(amp_ref_flat) + 1e-30
        ax.scatter(amp_in_flat[idx] / max_in, amp_ref_flat[idx] / max_ref,
                   s=1, alpha=C_SCATTER_ALPHA, color=C_GT, label=ref_label,
                   edgecolors='none')

    ax.set_xlabel('Normalized Input Amplitude')
    ax.set_ylabel('Normalized Output Amplitude')
    ax.set_title('AM/AM Characteristic', loc='left', fontweight='bold')
    ax.legend(markerscale=5)
    ax.grid(alpha=0.2, linestyle='--')
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    _savefig(fig, save_path)


@publication_style
def plot_ampm(input_iq, output_iq, save_path, label='Model',
              ref_output_iq=None, ref_label='Reference'):
    """AM/PM characteristic: input amplitude vs phase difference."""
    amp_in, phase_in = _iq_to_amp_phase(input_iq)
    _, phase_out = _iq_to_amp_phase(output_iq)
    phase_diff = np.degrees(np.angle(np.exp(1j * (phase_out - phase_in))))
    amp_in_flat = amp_in.flatten()
    phase_diff_flat = phase_diff.flatten()
    max_in = np.max(amp_in_flat) + 1e-30

    fig, ax = plt.subplots(figsize=(8, 5))
    n = len(amp_in_flat)
    idx = np.random.choice(n, min(n, 20000), replace=False)
    ax.scatter(amp_in_flat[idx] / max_in, phase_diff_flat[idx],
               s=1, alpha=C_SCATTER_ALPHA, color=C_PRED, label=label,
               edgecolors='none')

    if ref_output_iq is not None:
        _, phase_ref = _iq_to_amp_phase(ref_output_iq)
        phase_diff_ref = np.degrees(np.angle(np.exp(1j * (phase_ref - phase_in))))
        phase_diff_ref_flat = phase_diff_ref.flatten()
        ax.scatter(amp_in_flat[idx] / max_in, phase_diff_ref_flat[idx],
                   s=1, alpha=C_SCATTER_ALPHA, color=C_GT, label=ref_label,
                   edgecolors='none')

    ax.set_xlabel('Normalized Input Amplitude')
    ax.set_ylabel('Phase Difference (degrees)')
    ax.set_title('AM/PM Characteristic', loc='left', fontweight='bold')
    ax.legend(markerscale=5)
    ax.grid(alpha=0.2, linestyle='--')
    fig.tight_layout()
    _savefig(fig, save_path)


def _demod_signal(sig_complex, demod, sync_complex=None):
    """Demodulate a signal using the given :class:`Demodulator`.

    Flattens multi-dimensional (segmented) inputs into one continuous
    signal so that the demodulator has enough samples to find OFDM
    symbol boundaries via CP correlation.

    Returns (re, im) arrays of constellation points (may be empty if
    the signal is too short for OFDM demodulation).
    """
    sig = sig_complex.ravel()
    ref = None
    if sync_complex is not None:
        ref = sync_complex.ravel()
    return demod.demodulate(sig, sync_signal=ref)


@publication_style
def _plot_full_seq_constellation(full_const_data, demod, save_path,
                                  left_key, left_label, left_color,
                                  right_key, right_label, right_color,
                                  epoch=None):
    """Plot constellation using full-sequence data for proper OFDM demodulation.

    Used for APA datasets where val/test splits are shorter than one OFDM symbol.
    Demodulates the input for axis limits, and each output using input as sync signal.

    Args:
        full_const_data: dict with 'input_c' and signal complex arrays.
        demod: Demodulator instance.
        save_path: Path to save the figure.
        left_key, right_key: Keys into full_const_data for left/right panels.
        left_label, right_label: Panel titles.
        left_color, right_color: Scatter colors.
    """
    input_c = full_const_data['input_c']
    left_c = full_const_data[left_key]
    right_c = full_const_data[right_key]

    # Demodulate input for axis limits reference
    re_in, im_in = demod.demodulate(input_c)
    if len(re_in) == 0:
        return  # shouldn't happen with full data, but guard

    # Demodulate outputs using input as sync signal + equalization
    re_l, im_l = demod.demodulate(left_c, sync_signal=input_c, equalize=True)
    re_r, im_r = demod.demodulate(right_c, sync_signal=input_c, equalize=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    for ax, re_s, im_s, lbl, clr in [
        (axes[0], re_l, im_l, left_label, left_color),
        (axes[1], re_r, im_r, right_label, right_color),
    ]:
        if len(re_s) > 0:
            ax.scatter(re_s, im_s, s=8, alpha=0.5, color=clr, edgecolors='none')
        ax.set_xlabel('In-phase (I)')
        ax.set_ylabel('Quadrature (Q)')
        ax.set_title(lbl, loc='left', fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(alpha=0.2, linestyle='--')

    # Consistent axis limits based on input constellation
    lim = max(np.abs(re_in).max(), np.abs(im_in).max()) * 1.3
    for ax in axes:
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)

    fig.suptitle(_epoch_title('Constellation (Full Sequence)', epoch),
                 fontweight='bold')
    fig.tight_layout()
    _savefig(fig, save_path)


def _plot_constellation_density(ax, sig_complex, label, cmap='Blues',
                                 bins=120):
    """Plot a 2D density histogram of RMS-normalised I/Q samples.

    Used as a fallback when the signal is too short for OFDM
    demodulation (e.g. APA val/test splits < 1 OFDM symbol).
    """
    sig = sig_complex.ravel()
    rms = np.sqrt(np.mean(np.abs(sig) ** 2) + 1e-30)
    sig_n = sig / rms
    lim = np.percentile(np.abs(sig_n), 99.5) * 1.1
    ax.hist2d(sig_n.real, sig_n.imag, bins=bins,
              range=[[-lim, lim], [-lim, lim]], cmap=cmap, cmin=1)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel('In-phase (I)')
    ax.set_ylabel('Quadrature (Q)')
    ax.set_title(label, loc='left', fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(alpha=0.2, linestyle='--')


@publication_style
def plot_constellation(signal_complex, save_path, demod, label='Signal',
                       ref_complex=None, ref_label='Reference'):
    """Constellation diagram via dataset-specific OFDM demodulation.

    Falls back to density plot if signal is too short for OFDM demod.
    """
    re_s, im_s = _demod_signal(signal_complex, demod)
    if len(re_s) == 0:
        # Density fallback
        fig, ax = plt.subplots(figsize=(6, 6))
        _plot_constellation_density(ax, signal_complex, label)
        fig.tight_layout()
        _savefig(fig, save_path)
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(re_s, im_s, s=8, alpha=0.5, color=C_PRED, label=label,
               edgecolors='none')

    if ref_complex is not None:
        re_r, im_r = _demod_signal(ref_complex, demod)
        if len(re_r) > 0:
            ax.scatter(re_r, im_r, s=8, alpha=0.5, color=C_GT, label=ref_label,
                       edgecolors='none')

    ax.set_xlabel('In-phase (I)')
    ax.set_ylabel('Quadrature (Q)')
    ax.set_title('Constellation Diagram', loc='left', fontweight='bold')
    ax.legend(markerscale=5)
    ax.grid(alpha=0.2, linestyle='--')
    ax.set_aspect('equal')
    fig.tight_layout()
    _savefig(fig, save_path)


@publication_style
def plot_constellation_dual(left_complex, left_label, left_color,
                            right_complex, right_label, right_color,
                            ref_complex, ref_label, ref_color,
                            save_path, demod):
    """Dual constellation diagram: two panels sharing a reference signal.

    If the signal is too short for OFDM demod (returns empty), falls
    back to a 2D density histogram that clearly shows the PA distortion
    pattern (compression, phase rotation).

    Args:
        demod: :class:`datasets.demodulator.Demodulator` instance.
    """
    re_ref, im_ref = _demod_signal(ref_complex, demod)
    use_density = len(re_ref) == 0  # signal too short for OFDM demod

    if use_density:
        # Density fallback — show I/Q distribution shape
        cmaps = {left_color: 'Reds', right_color: 'Blues'}
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        for ax, sig_c, lbl, clr in [(axes[0], left_complex, left_label, left_color),
                                     (axes[1], right_complex, right_label, right_color)]:
            _plot_constellation_density(ax, sig_c, lbl,
                                        cmap=cmaps.get(clr, 'Blues'))
        fig.suptitle('I/Q Density', fontsize=14, fontweight='bold')
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        panels = [(axes[0], left_complex, left_label, left_color),
                  (axes[1], right_complex, right_label, right_color)]
        for ax, sig_c, lbl, clr in panels:
            re_s, im_s = _demod_signal(sig_c, demod)
            ax.scatter(re_s, im_s, s=10, alpha=0.6, color=clr,
                       label=lbl, edgecolors='none', zorder=2)
            if len(re_ref) > 0:
                ax.scatter(re_ref, im_ref, s=30, alpha=0.8, color=ref_color,
                           label=ref_label, marker='x', linewidths=1.0, zorder=3)
            ax.set_xlabel('In-phase (I)')
            ax.set_ylabel('Quadrature (Q)')
            ax.set_title(lbl, loc='left', fontweight='bold')
            ax.legend(markerscale=5)
            ax.grid(alpha=0.2, linestyle='--')
            ax.set_aspect('equal')
        fig.suptitle('Constellation Comparison', fontsize=14, fontweight='bold')
    fig.tight_layout()
    _savefig(fig, save_path)


@publication_style
def plot_waveform(input_iq, output_iq, save_path, pred_iq=None,
                  n_samples=500, label_in='Input', label_out='Output',
                  label_pred='Prediction', ylim=None, epoch=None):
    """Time-domain I/Q waveform comparison."""
    n = min(n_samples, input_iq.shape[0] if input_iq.ndim == 2 else input_iq.shape[-2])
    # Flatten to [samples, 2]
    inp = input_iq.reshape(-1, 2)[:n]
    out = output_iq.reshape(-1, 2)[:n]
    t = np.arange(n)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for ch, ch_name in enumerate(['I', 'Q']):
        ax = axes[ch]
        ax.plot(t, inp[:, ch], label=f'{label_in}', alpha=0.7, color=C_REF, lw=1.5)
        ax.plot(t, out[:, ch], label=f'{label_out}', alpha=0.7, color=C_GT, lw=1.5)
        if pred_iq is not None:
            pred = pred_iq.reshape(-1, 2)[:n]
            ax.plot(t, pred[:, ch], label=f'{label_pred}', alpha=0.7,
                    linestyle='--', color=C_PRED, lw=1.5)
        ax.set_ylabel(f'{ch_name} Amplitude')
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2, linestyle='--')
    axes[0].set_title(_epoch_title('Time-Domain Waveform', epoch),
                      loc='left', fontweight='bold')
    axes[1].set_xlabel('Sample Index')
    fig.tight_layout()
    _savefig(fig, save_path)


@publication_style
def plot_error_signal(pred_iq, target_iq, save_path, n_samples=500, ylim=None,
                      epoch=None):
    """Plot prediction error (residual) in time domain."""
    pred = pred_iq.reshape(-1, 2)
    target = target_iq.reshape(-1, 2)
    n = min(n_samples, pred.shape[0])
    error = pred[:n] - target[:n]
    t = np.arange(n)

    fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    for ch, ch_name in enumerate(['I', 'Q']):
        ax = axes[ch]
        ax.plot(t, error[:, ch], color=C_GT, alpha=0.7, lw=1.5)
        ax.set_ylabel(f'{ch_name} Error')
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.grid(alpha=0.2, linestyle='--')
    axes[0].set_title(_epoch_title('Prediction Error (Residual)', epoch),
                      loc='left', fontweight='bold')
    axes[1].set_xlabel('Sample Index')
    fig.tight_layout()
    _savefig(fig, save_path)


@publication_style
def plot_training_curves(history, save_path, metric_keys=None):
    """Plot training metric curves over epochs from a list of dicts."""
    if not history:
        return
    if metric_keys is None:
        metric_keys = ['NMSE', 'EVM', 'ACLR_AVG']

    for metric in metric_keys:
        val_key = f'VAL_{metric}'
        test_key = f'TEST_{metric}'
        has_val = val_key in history[0]
        has_test = test_key in history[0]
        if not has_val and not has_test:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        epochs = [h.get('EPOCH', i) for i, h in enumerate(history)]
        if has_val:
            vals = [h[val_key] for h in history]
            ax.plot(epochs, vals, label=f'Val {metric}', marker='.', markersize=3,
                    color=C_PRED, lw=1.5)
        if has_test:
            tests = [h[test_key] for h in history]
            ax.plot(epochs, tests, label=f'Test {metric}', marker='.', markersize=3,
                    color=C_GT, lw=1.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(f'{metric} (dB)')
        ax.set_title(f'{metric} vs Epoch', loc='left', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.2, linestyle='--')
        fig.tight_layout()
        fname = f'{metric.lower()}_vs_epoch.png'
        _savefig(fig, os.path.join(save_path, fname))


@publication_style
def plot_metrics_summary(metrics_wo, metrics_w, save_path, metric_names=None):
    """Bar chart comparing metrics with and without DPD."""
    if metric_names is None:
        metric_names = ['NMSE', 'EVM', 'ACLR_L', 'ACLR_R', 'ACLR_AVG']

    vals_wo = [metrics_wo.get(m, 0) for m in metric_names]
    vals_w = [metrics_w.get(m, 0) for m in metric_names]

    x = np.arange(len(metric_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width / 2, vals_wo, width, label='Without DPD', color=C_GT)
    bars2 = ax.bar(x + width / 2, vals_w, width, label='With DPD', color=C_PRED)

    ax.set_ylabel('Value (dB)')
    ax.set_title('Metrics Comparison: Without DPD vs With DPD', loc='left',
                 fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.legend()
    ax.grid(alpha=0.2, linestyle='--', axis='y')

    # Value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=7)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=7)

    fig.tight_layout()
    _savefig(fig, save_path)


# ===========================================================================
# 2x2 Overview plot
# ===========================================================================

@publication_style
def generate_overview_plot(epoch_dir, split, pred_flat, gt_flat, pa_flat,
                           pred_c, gt_c, pa_c, fs, nperseg, bw_main_ch,
                           n_sub_ch, demod=None, sw=10,
                           full_const_data=None, epoch=None):
    """Create a 2x2 overview figure comparing w/o DPD vs w/ DPD.

    Layout:
        (0,0) AM/AM   (0,1) AM/PM
        (1,0) PSD     (1,1) Constellation (dual sub-panels)
    """
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

    fig = plt.figure(figsize=(16, 14))
    fig.suptitle(_epoch_title('DPD Training Overview', epoch),
                 fontsize=16, fontweight='bold', y=0.98)
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3, top=0.94)

    # --- (0,0) AM/AM ---
    ax_amam = fig.add_subplot(gs[0, 0])
    amp_in = _iq_to_amp_phase(gt_flat)[0].flatten()
    max_in = np.max(amp_in) + 1e-30
    n_pts = len(amp_in)
    idx = np.random.choice(n_pts, min(n_pts, 20000), replace=False)
    for out_flat, lbl, clr in [(pa_flat, 'PA Only (w/o DPD)', C_GT),
                                (pred_flat, 'Cascaded (DPD+PA)', C_PRED)]:
        amp_out = _iq_to_amp_phase(out_flat)[0].flatten()
        max_out = np.max(amp_out) + 1e-30
        ax_amam.scatter(amp_in[idx] / max_in, amp_out[idx] / max_out,
                        s=1, alpha=C_SCATTER_ALPHA, color=clr, label=lbl,
                        edgecolors='none')
    ax_amam.set_xlabel('Normalized Input Amplitude')
    ax_amam.set_ylabel('Normalized Output Amplitude')
    ax_amam.set_title(_epoch_title('AM/AM Characteristic', epoch), loc='left', fontweight='bold')
    ax_amam.legend(markerscale=5)
    ax_amam.grid(alpha=0.2, linestyle='--')
    ax_amam.set_xlim(0, 1.05)
    ax_amam.set_ylim(0, 1.05)

    # --- (0,1) AM/PM ---
    ax_ampm = fig.add_subplot(gs[0, 1])
    _, phase_in = _iq_to_amp_phase(gt_flat)
    phase_in_flat = phase_in.flatten()
    for out_flat, lbl, clr in [(pa_flat, 'PA Only (w/o DPD)', C_GT),
                                (pred_flat, 'Cascaded (DPD+PA)', C_PRED)]:
        _, phase_out = _iq_to_amp_phase(out_flat)
        phase_diff = np.degrees(np.angle(np.exp(1j * (phase_out.flatten() - phase_in_flat))))
        ax_ampm.scatter(amp_in[idx] / max_in, phase_diff[idx],
                        s=1, alpha=C_SCATTER_ALPHA, color=clr, label=lbl,
                        edgecolors='none')
    ax_ampm.set_xlabel('Normalized Input Amplitude')
    ax_ampm.set_ylabel('Phase Difference (degrees)')
    ax_ampm.set_title(_epoch_title('AM/PM Characteristic', epoch), loc='left', fontweight='bold')
    ax_ampm.legend(markerscale=5)
    ax_ampm.grid(alpha=0.2, linestyle='--')

    # --- (1,0) PSD with ACLR annotation ---
    ax_psd = fig.add_subplot(gs[1, 0])
    for sig_c, lbl, color, ls in [(pa_c, 'PA Only (w/o DPD)', C_GT, '-'),
                                   (pred_c, 'Cascaded (DPD+PA)', C_PRED, '-')]:
        sig_2d = _ensure_2d_segments(sig_c, nperseg)
        freq, ps = power_spectrum(sig_2d, fs=fs, nperseg=nperseg)
        ps_db = 10 * np.log10(ps / np.max(ps) + 1e-30)
        ps_smooth = moving_average(ps_db, sw)
        freq_adj = freq[sw - 1:]
        n = min(len(freq_adj), len(ps_smooth))
        ax_psd.plot(freq_adj[:n] / 1e6, ps_smooth[:n], label=lbl, color=color,
                    linestyle=ls, lw=1.5)
    ax_psd.set_xlabel('Frequency (MHz)')
    ax_psd.set_ylabel('Normalized PSD (dB)')
    ax_psd.set_title(_epoch_title('Power Spectral Density', epoch), loc='left', fontweight='bold')
    ax_psd.legend()
    ax_psd.grid(alpha=0.2, linestyle='--')
    _annotate_aclr(ax_psd,
                   [('PA Only', pa_c), ('DPD+PA', pred_c)],
                   fs, nperseg, bw_main_ch, n_sub_ch)

    # --- (1,1) Constellation (Cascaded DPD+PA only) ---
    ax_const = fig.add_subplot(gs[1, 1])

    # Use full-sequence data for OFDM demod when available
    if full_const_data is not None and demod is not None:
        input_c = full_const_data['input_c']
        re_in, im_in = demod.demodulate(input_c)
        if len(re_in) > 0:
            sig_c = full_const_data['cascaded_c']
            re_s, im_s = demod.demodulate(sig_c, sync_signal=input_c,
                                          equalize=True)
            if len(re_s) > 0:
                ax_const.scatter(re_s, im_s, s=8, alpha=0.6, color=C_PRED,
                                 label='Cascaded (DPD+PA)', edgecolors='none', zorder=2)
            ax_const.set_xlabel('I')
            ax_const.set_ylabel('Q')
            ax_const.set_title(_epoch_title('Constellation — Cascaded (DPD+PA)', epoch),
                               loc='left', fontweight='bold', fontsize=10)
            ax_const.legend(markerscale=5, fontsize=7)
            ax_const.grid(alpha=0.2, linestyle='--')
            ax_const.set_aspect('equal')
            lim = max(np.abs(re_in).max(), np.abs(im_in).max()) * 1.3
            ax_const.set_xlim(-lim, lim)
            ax_const.set_ylim(-lim, lim)
        _savefig(fig, os.path.join(epoch_dir, f'overview_{split}.png'))
        return

    # Try OFDM demod; fall back to density if signal too short
    can_demod = demod is not None
    if can_demod:
        re_ref, im_ref = _demod_signal(gt_c, demod)
        can_demod = len(re_ref) > 0

    if can_demod:
        re_s, im_s = _demod_signal(pred_c, demod)
        ax_const.scatter(re_s, im_s, s=8, alpha=0.6, color=C_PRED,
                         label='Cascaded (DPD+PA)', edgecolors='none', zorder=2)
        ax_const.set_xlabel('I')
        ax_const.set_ylabel('Q')
        ax_const.set_title(_epoch_title('Constellation — Cascaded (DPD+PA)', epoch),
                           loc='left', fontweight='bold', fontsize=10)
        ax_const.legend(markerscale=5, fontsize=7)
        ax_const.grid(alpha=0.2, linestyle='--')
        ax_const.set_aspect('equal')
    else:
        # Density fallback for short signals
        _plot_constellation_density(ax_const, pred_c, 'Cascaded (DPD+PA)', cmap='Blues', bins=80)

    _savefig(fig, os.path.join(epoch_dir, f'overview_{split}.png'))


# ===========================================================================
# High-level epoch plotting dispatchers
# ===========================================================================

@publication_style
def generate_epoch_plots_train_pa(plot_dir, epoch, prediction, ground_truth,
                                  input_data, split, spec, demod=None,
                                  subfolder=None, fixed_limits=None,
                                  full_const_data=None, rerender=False):
    """Generate all per-epoch plots for PA training.

    Args:
        demod: :class:`datasets.demodulator.Demodulator` instance for
            constellation plotting.  If *None*, constellation is skipped.
        subfolder: If 'best', save to plots_dir/best/ instead of epoch dir.
        fixed_limits: Dict with optional fixed axis limits:
            'psd_ylim', 'ampm_ylim', 'error_ylim', 'waveform_ylim'
    """
    if subfolder == 'best':
        epoch_dir = os.path.join(plot_dir, 'best')
    else:
        epoch_dir = os.path.join(plot_dir, 'history', 'epochs', f'epoch_{epoch:04d}')
    fs = spec.get('input_signal_fs', 800e6)
    nperseg = spec.get('nperseg', 2560)
    n_sub_ch = spec.get('n_sub_ch', 10)
    bw_main_ch = spec.get('bw_main_ch', 200e6)
    bw_sub_ch = spec.get('bw_sub_ch', bw_main_ch / n_sub_ch)
    fl = fixed_limits or {}

    # Epoch label for titles (None for 'best' subfolder)
    ep = epoch if subfolder != 'best' else None

    # Flatten segments for plotting: [n_seg, seg_len, 2] -> [total, 2]
    pred_flat = prediction.reshape(-1, 2)
    gt_flat = ground_truth.reshape(-1, 2)

    pred_c = IQ_to_complex(pred_flat[np.newaxis, :])[0]
    gt_c = IQ_to_complex(gt_flat[np.newaxis, :])[0]

    # Pre-compute EVM for PSD annotation
    evm_pred = _compute_evm_for_iq(pred_flat, gt_flat, fs, nperseg, bw_main_ch, n_sub_ch)
    evm_info = []
    if evm_pred is not None:
        evm_info.append(('Predicted', evm_pred))

    # PSD: predicted vs ground truth (inline to support ylim)
    sw = 10
    fig, ax = plt.subplots(figsize=(10, 5))
    for sig_c, lbl, color, ls in [(pred_c, 'Predicted', C_PRED, '-'),
                                   (gt_c, 'Ground Truth', C_GT, '--')]:
        sig_2d = _ensure_2d_segments(sig_c, nperseg)
        freq, ps = power_spectrum(sig_2d, fs=fs, nperseg=nperseg)
        ps_db = 10 * np.log10(ps / np.max(ps) + 1e-30)
        ps_smooth = moving_average(ps_db, sw)
        freq_adj = freq[sw - 1:]
        n = min(len(freq_adj), len(ps_smooth))
        ax.plot(freq_adj[:n] / 1e6, ps_smooth[:n], label=lbl, color=color,
                linestyle=ls, lw=1.5)
    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('Normalized PSD (dB)')
    ax.set_title(_epoch_title('Power Spectral Density', ep),
                 loc='left', fontweight='bold')
    if 'psd_ylim' in fl:
        ax.set_ylim(fl['psd_ylim'])
    ax.legend()
    ax.grid(alpha=0.2, linestyle='--')
    # ACLR + EVM annotation
    _annotate_aclr(ax,
                   [('Predicted', pred_c), ('Ground Truth', gt_c)],
                   fs, nperseg, bw_main_ch, n_sub_ch,
                   evm_info=evm_info if evm_info else None)
    fig.tight_layout()
    _savefig(fig, os.path.join(epoch_dir, f'psd_{split}.png'))

    # AM/AM (limits already fixed at [0, 1.05])
    inp_flat = input_data.reshape(-1, 2) if input_data is not None else gt_flat
    plot_amam(inp_flat, pred_flat,
              os.path.join(epoch_dir, f'amam_{split}.png'),
              label='Predicted', ref_output_iq=gt_flat, ref_label='Ground Truth')

    # AM/PM (inline to support ylim)
    fig, ax = plt.subplots(figsize=(8, 5))
    amp_in, phase_in = _iq_to_amp_phase(inp_flat)
    amp_in_flat = amp_in.flatten()
    phase_in_flat = phase_in.flatten()
    max_in = np.max(amp_in_flat) + 1e-30
    n_pts = len(amp_in_flat)
    idx = np.random.choice(n_pts, min(n_pts, 20000), replace=False)
    for out, lbl, clr in [(pred_flat, 'Predicted', C_PRED),
                           (gt_flat, 'Ground Truth', C_GT)]:
        _, phase_out = _iq_to_amp_phase(out)
        phase_diff = np.degrees(np.angle(np.exp(1j * (phase_out.flatten() - phase_in_flat))))
        ax.scatter(amp_in_flat[idx] / max_in, phase_diff[idx],
                   s=1, alpha=C_SCATTER_ALPHA, color=clr, label=lbl,
                   edgecolors='none')
    ax.set_xlabel('Normalized Input Amplitude')
    ax.set_ylabel('Phase Difference (degrees)')
    ax.set_title(_epoch_title('AM/PM Characteristic', ep),
                 loc='left', fontweight='bold')
    if 'ampm_ylim' in fl:
        ax.set_ylim(fl['ampm_ylim'])
    ax.legend(markerscale=5)
    ax.grid(alpha=0.2, linestyle='--')
    fig.tight_layout()
    _savefig(fig, os.path.join(epoch_dir, f'ampm_{split}.png'))

    # Constellation
    if demod is not None and not rerender:
        if full_const_data is not None:
            _plot_full_seq_constellation(
                full_const_data, demod,
                os.path.join(epoch_dir, f'constellation_{split}.png'),
                left_key='gt_c', left_label='PA Output (Actual)',
                left_color=C_GT,
                right_key='pred_c', right_label='PA Output (Predicted)',
                right_color=C_PRED, epoch=ep)
        else:
            plot_constellation(pred_c, os.path.join(epoch_dir, f'constellation_{split}.png'),
                               demod, label='Predicted', ref_complex=gt_c,
                               ref_label='Ground Truth')

    # Waveform
    plot_waveform(inp_flat, gt_flat,
                  os.path.join(epoch_dir, f'waveform_{split}.png'),
                  pred_iq=pred_flat, label_in='PA Input', label_out='PA Output (Actual)',
                  label_pred='PA Output (Predicted)',
                  ylim=fl.get('waveform_ylim'), epoch=ep)

    # Error signal
    plot_error_signal(pred_flat, gt_flat,
                      os.path.join(epoch_dir, f'error_{split}.png'),
                      ylim=fl.get('error_ylim'), epoch=ep)


@publication_style
def generate_epoch_plots_train_dpd(plot_dir, epoch, prediction, ground_truth,
                                   split, spec, demod=None, subfolder=None,
                                   pa_only_prediction=None, fixed_limits=None,
                                   full_const_data=None, rerender=False):
    """Generate all per-epoch plots for DPD training.

    Args:
        subfolder: If 'best', save to plots_dir/best/ instead of epoch dir.
        pa_only_prediction: PA-only output (w/o DPD) for comparison.
        fixed_limits: Dict with optional fixed axis limits:
            'psd_ylim', 'ampm_ylim', 'error_ylim', 'waveform_ylim'
        rerender: If True, skip constellation/overview (they don't need
            axis normalization, and re-rendering would overwrite correct
            per-epoch data with a single model snapshot).
    """
    fl = fixed_limits or {}
    if subfolder == 'best':
        epoch_dir = os.path.join(plot_dir, 'best')
    else:
        epoch_dir = os.path.join(plot_dir, 'history', 'epochs', f'epoch_{epoch:04d}')
    fs = spec.get('input_signal_fs', 800e6)
    nperseg = spec.get('nperseg', 2560)
    n_sub_ch = spec.get('n_sub_ch', 10)
    bw_main_ch = spec.get('bw_main_ch', 200e6)
    bw_sub_ch = spec.get('bw_sub_ch', bw_main_ch / n_sub_ch)
    sw = 10  # smoothing window for PSD

    # prediction = cascaded output (DPD->PA), ground_truth = G*x (linear target)
    pred_flat = prediction.reshape(-1, 2)
    gt_flat = ground_truth.reshape(-1, 2)

    pred_c = IQ_to_complex(pred_flat[np.newaxis, :])[0]
    gt_c = IQ_to_complex(gt_flat[np.newaxis, :])[0]

    has_pa = pa_only_prediction is not None
    if has_pa:
        pa_flat = pa_only_prediction.reshape(-1, 2)
        pa_c = IQ_to_complex(pa_flat[np.newaxis, :])[0]

    # For APA datasets, use full-dataset signals for PSD, AM/AM, AM/PM
    # so the plots reflect the entire signal, not just a val/test split.
    use_full = full_const_data is not None
    if use_full:
        full_cas_c = full_const_data['cascaded_c']
        full_pa_c = full_const_data.get('pa_only_c')
        full_in_c = full_const_data['input_c']
        # Convert complex -> IQ [N, 2] for AM/AM and AM/PM
        full_cas_flat = np.stack([full_cas_c.real, full_cas_c.imag], axis=-1)
        full_pa_flat = np.stack([full_pa_c.real, full_pa_c.imag], axis=-1) if full_pa_c is not None else None
        full_in_flat = np.stack([full_in_c.real, full_in_c.imag], axis=-1)

    # Select data source: full dataset for APA, val/test split otherwise
    psd_pa_c = full_pa_c if (use_full and full_pa_c is not None) else (pa_c if has_pa else None)
    psd_cas_c = full_cas_c if use_full else pred_c
    amam_ref_flat = full_in_flat if use_full else gt_flat
    amam_pa_flat = full_pa_flat if (use_full and full_pa_flat is not None) else (pa_flat if has_pa else None)
    amam_cas_flat = full_cas_flat if use_full else pred_flat

    # Epoch label for titles (None for 'best' subfolder)
    ep = epoch if subfolder != 'best' else None

    # Pre-compute EVM for PSD annotation (always from val/test split for metrics accuracy)
    evm_info = []
    if has_pa:
        evm_pa = _compute_evm_for_iq(pa_flat, gt_flat, fs, nperseg, bw_main_ch, n_sub_ch)
        if evm_pa is not None:
            evm_info.append(('PA Only', evm_pa))
    evm_cas = _compute_evm_for_iq(pred_flat, gt_flat, fs, nperseg, bw_main_ch, n_sub_ch)
    if evm_cas is not None:
        evm_info.append(('DPD+PA', evm_cas))

    # --- PSD (2-trace: PA Only vs Cascaded, no Linear Target) ---
    if has_pa or use_full:
        fig, ax = plt.subplots(figsize=(10, 5))
        traces = []
        if psd_pa_c is not None:
            traces.append((psd_pa_c, 'PA Only (w/o DPD)', C_GT, '-'))
        traces.append((psd_cas_c, 'Cascaded (DPD+PA)', C_PRED, '-'))
        for sig_c, lbl, color, ls in traces:
            sig_2d = _ensure_2d_segments(sig_c, nperseg)
            freq, ps = power_spectrum(sig_2d, fs=fs, nperseg=nperseg)
            ps_db = 10 * np.log10(ps / np.max(ps) + 1e-30)
            ps_smooth = moving_average(ps_db, sw)
            freq_adj = freq[sw - 1:]
            n = min(len(freq_adj), len(ps_smooth))
            ax.plot(freq_adj[:n] / 1e6, ps_smooth[:n], label=lbl, color=color,
                    linestyle=ls, lw=1.5)
        ax.set_xlabel('Frequency (MHz)')
        ax.set_ylabel('Normalized PSD (dB)')
        ax.set_title(_epoch_title('Power Spectral Density', ep),
                     loc='left', fontweight='bold')
        if 'psd_ylim' in fl:
            ax.set_ylim(fl['psd_ylim'])
        ax.legend()
        ax.grid(alpha=0.2, linestyle='--')
        # ACLR + EVM annotation
        aclr_traces = []
        if psd_pa_c is not None:
            aclr_traces.append(('PA Only', psd_pa_c))
        aclr_traces.append(('DPD+PA', psd_cas_c))
        _annotate_aclr(ax, aclr_traces,
                       fs, nperseg, bw_main_ch, n_sub_ch,
                       evm_info=evm_info if evm_info else None)
        fig.tight_layout()
        _savefig(fig, os.path.join(epoch_dir, f'psd_{split}.png'))
    else:
        plot_psd(pred_c, os.path.join(epoch_dir, f'psd_{split}.png'),
                 label='Cascaded (DPD+PA)', ref_complex=gt_c, ref_label='Linear Target',
                 fs=fs, nperseg=nperseg)

    # --- AM/AM (2-scatter inline when PA-only available) ---
    if has_pa or use_full:
        fig, ax = plt.subplots(figsize=(6, 6))
        amp_in = _iq_to_amp_phase(amam_ref_flat)[0].flatten()
        max_in = np.max(amp_in) + 1e-30
        idx = np.random.choice(len(amp_in), min(len(amp_in), 20000), replace=False)
        scatter_pairs = []
        if amam_pa_flat is not None:
            scatter_pairs.append((amam_pa_flat, 'PA Only (w/o DPD)', C_GT))
        scatter_pairs.append((amam_cas_flat, 'Cascaded (DPD+PA)', C_PRED))
        for out_flat_i, lbl, clr in scatter_pairs:
            amp_out = _iq_to_amp_phase(out_flat_i)[0].flatten()
            max_out = np.max(amp_out) + 1e-30
            ax.scatter(amp_in[idx] / max_in, amp_out[idx] / max_out,
                       s=1, alpha=C_SCATTER_ALPHA, color=clr, label=lbl,
                       edgecolors='none')
        ax.set_xlabel('Normalized Input Amplitude')
        ax.set_ylabel('Normalized Output Amplitude')
        ax.set_title(_epoch_title('AM/AM Characteristic', ep),
                     loc='left', fontweight='bold')
        ax.legend(markerscale=5)
        ax.grid(alpha=0.2, linestyle='--')
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
        fig.tight_layout()
        _savefig(fig, os.path.join(epoch_dir, f'amam_{split}.png'))
    else:
        plot_amam(gt_flat, pred_flat, os.path.join(epoch_dir, f'amam_{split}.png'),
                  label='Cascaded (DPD+PA)')

    # --- AM/PM (2-scatter inline when PA-only available) ---
    if has_pa or use_full:
        fig, ax = plt.subplots(figsize=(8, 5))
        amp_in, phase_in = _iq_to_amp_phase(amam_ref_flat)
        amp_in_flat = amp_in.flatten()
        phase_in_flat = phase_in.flatten()
        max_in = np.max(amp_in_flat) + 1e-30
        idx = np.random.choice(len(amp_in_flat), min(len(amp_in_flat), 20000), replace=False)
        ampm_pairs = []
        if amam_pa_flat is not None:
            ampm_pairs.append((amam_pa_flat, 'PA Only (w/o DPD)', C_GT))
        ampm_pairs.append((amam_cas_flat, 'Cascaded (DPD+PA)', C_PRED))
        for out_flat_i, lbl, clr in ampm_pairs:
            _, phase_out = _iq_to_amp_phase(out_flat_i)
            phase_diff = np.degrees(np.angle(np.exp(1j * (phase_out.flatten() - phase_in_flat))))
            ax.scatter(amp_in_flat[idx] / max_in, phase_diff[idx],
                       s=1, alpha=C_SCATTER_ALPHA, color=clr, label=lbl,
                       edgecolors='none')
        ax.set_xlabel('Normalized Input Amplitude')
        ax.set_ylabel('Phase Difference (degrees)')
        ax.set_title(_epoch_title('AM/PM Characteristic', ep),
                     loc='left', fontweight='bold')
        if 'ampm_ylim' in fl:
            ax.set_ylim(fl['ampm_ylim'])
        ax.legend(markerscale=5)
        ax.grid(alpha=0.2, linestyle='--')
        fig.tight_layout()
        _savefig(fig, os.path.join(epoch_dir, f'ampm_{split}.png'))
    else:
        plot_ampm(gt_flat, pred_flat, os.path.join(epoch_dir, f'ampm_{split}.png'),
                  label='Cascaded (DPD+PA)')

    # --- Constellation (dual subplots when PA-only available) ---
    # Skip during re-rendering: constellation axes are already consistent
    # and re-rendering would overwrite per-epoch data with a single snapshot.
    if demod is not None and not rerender:
        if full_const_data is not None:
            _plot_full_seq_constellation(
                full_const_data, demod,
                os.path.join(epoch_dir, f'constellation_{split}.png'),
                left_key='pa_only_c', left_label='PA Only (w/o DPD)',
                left_color=C_GT,
                right_key='cascaded_c', right_label='Cascaded (DPD+PA)',
                right_color=C_PRED, epoch=ep)
        elif has_pa:
            plot_constellation_dual(
                left_complex=pa_c, left_label='PA Only (w/o DPD)', left_color=C_GT,
                right_complex=pred_c, right_label='Cascaded (DPD+PA)', right_color=C_PRED,
                ref_complex=gt_c, ref_label='Linear Target', ref_color=C_REF,
                save_path=os.path.join(epoch_dir, f'constellation_{split}.png'),
                demod=demod)
        else:
            plot_constellation(pred_c, os.path.join(epoch_dir, f'constellation_{split}.png'),
                               demod, label='Cascaded (DPD+PA)', ref_complex=gt_c,
                               ref_label='Linear Target')

    # --- Waveform (3-trace when PA-only available) ---
    if has_pa:
        plot_waveform(gt_flat, pa_flat,
                      os.path.join(epoch_dir, f'waveform_{split}.png'),
                      pred_iq=pred_flat,
                      label_in='Linear Target',
                      label_out='PA Only (w/o DPD)',
                      label_pred='Cascaded (DPD+PA)',
                      ylim=fl.get('waveform_ylim'), epoch=ep)
    else:
        plot_waveform(gt_flat, pred_flat,
                      os.path.join(epoch_dir, f'waveform_{split}.png'),
                      label_in='Linear Target', label_out='Cascaded (DPD+PA)',
                      ylim=fl.get('waveform_ylim'), epoch=ep)

    # --- Error ---
    plot_error_signal(pred_flat, gt_flat,
                      os.path.join(epoch_dir, f'error_{split}.png'),
                      ylim=fl.get('error_ylim'), epoch=ep)

    # --- Overview plot (when PA-only available) ---
    # Skip during re-rendering: overview includes constellation which
    # should not be overwritten with a single model snapshot.
    if has_pa and not rerender:
        try:
            generate_overview_plot(
                epoch_dir, split, pred_flat, gt_flat, pa_flat,
                pred_c, gt_c, pa_c, fs, nperseg, bw_main_ch, n_sub_ch,
                demod=demod, sw=sw, full_const_data=full_const_data,
                epoch=ep)
        except Exception as e:
            print(f"  Warning: Overview plot failed for epoch {epoch} {split}: {e}")


def generate_plots_run_dpd(plot_dir, dpd_input, dpd_output, spec, demod=None):
    """Generate plots for the run_dpd step (DPD inference output)."""
    fs = spec.get('input_signal_fs', 491.52e6)
    nperseg = spec.get('nperseg', 2560)

    inp_flat = dpd_input.reshape(-1, 2)
    out_flat = dpd_output.reshape(-1, 2)

    inp_c = IQ_to_complex(inp_flat[np.newaxis, :])[0]
    out_c = IQ_to_complex(out_flat[np.newaxis, :])[0]

    # PSD
    plot_psd(out_c, os.path.join(plot_dir, 'psd.png'),
             label='DPD Output (Pre-distorted)', ref_complex=inp_c,
             ref_label='Original Input', fs=fs, nperseg=nperseg)

    # AM/AM
    plot_amam(inp_flat, out_flat, os.path.join(plot_dir, 'amam.png'),
              label='DPD (Pre-distorted)')

    # AM/PM
    plot_ampm(inp_flat, out_flat, os.path.join(plot_dir, 'ampm.png'),
              label='DPD (Pre-distorted)')

    # Constellation
    if demod is not None:
        plot_constellation(out_c, os.path.join(plot_dir, 'constellation.png'),
                           demod, label='DPD Output', ref_complex=inp_c,
                           ref_label='Original Input')

    # Waveform
    plot_waveform(inp_flat, out_flat, os.path.join(plot_dir, 'waveform.png'),
                  label_in='Original Input', label_out='DPD Output (Pre-distorted)')


@publication_style
def generate_plots_compare(plot_dir, pa_input, pa_output_wo_dpd, pa_output_w_dpd,
                           spec, demod=None, metrics_wo=None, metrics_w=None,
                           full_const_data=None):
    """Generate comparison plots: without DPD vs with DPD."""
    fs = spec.get('input_signal_fs', 491.52e6)
    nperseg = spec.get('nperseg', 2560)
    n_sub_ch = spec.get('n_sub_ch', 10)
    bw_main_ch = spec.get('bw_main_ch', 200e6)

    inp_flat = pa_input.reshape(-1, 2)
    wo_flat = pa_output_wo_dpd.reshape(-1, 2)
    w_flat = pa_output_w_dpd.reshape(-1, 2)

    inp_c = IQ_to_complex(inp_flat[np.newaxis, :])[0]
    wo_c = IQ_to_complex(wo_flat[np.newaxis, :])[0]
    w_c = IQ_to_complex(w_flat[np.newaxis, :])[0]

    # PSD comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    for sig, lbl, color, ls in [(wo_c, 'Without DPD', C_GT, '-'),
                                 (w_c, 'With DPD', C_PRED, '--')]:
        sig_2d = _ensure_2d_segments(sig, nperseg)
        freq, ps = power_spectrum(sig_2d, fs=fs, nperseg=nperseg)
        ps_db = 10 * np.log10(ps / np.max(ps) + 1e-30)
        sw = 10
        ps_smooth = moving_average(ps_db, sw)
        freq_adj = freq[sw - 1:]
        n = min(len(freq_adj), len(ps_smooth))
        ax.plot(freq_adj[:n] / 1e6, ps_smooth[:n], label=lbl, color=color,
                linestyle=ls, lw=1.5)
    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('Normalized PSD (dB)')
    ax.set_title('PSD Comparison: Without DPD vs With DPD', loc='left',
                 fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.2, linestyle='--')
    # ACLR annotation
    _annotate_aclr(ax,
                   [('Without DPD', wo_c), ('With DPD', w_c)],
                   fs, nperseg, bw_main_ch, n_sub_ch)
    fig.tight_layout()
    _savefig(fig, os.path.join(plot_dir, 'psd_comparison.png'))

    # AM/AM comparison
    plot_amam(inp_flat, wo_flat, os.path.join(plot_dir, 'amam_comparison.png'),
              label='Without DPD', ref_output_iq=w_flat, ref_label='With DPD')

    # AM/PM comparison
    plot_ampm(inp_flat, wo_flat, os.path.join(plot_dir, 'ampm_comparison.png'),
              label='Without DPD', ref_output_iq=w_flat, ref_label='With DPD')

    # Constellation comparison - side by side
    if demod is not None:
        if full_const_data is not None:
            _plot_full_seq_constellation(
                full_const_data, demod,
                os.path.join(plot_dir, 'constellation_comparison.png'),
                left_key='wo_dpd_c', left_label='Without DPD',
                left_color=C_GT,
                right_key='w_dpd_c', right_label='With DPD',
                right_color=C_PRED)
        else:
            plot_constellation_dual(
                left_complex=wo_c, left_label='Without DPD', left_color=C_GT,
                right_complex=w_c, right_label='With DPD', right_color=C_PRED,
                ref_complex=inp_c, ref_label='Input', ref_color=C_REF,
                save_path=os.path.join(plot_dir, 'constellation_comparison.png'),
                demod=demod)

    # Waveform comparison
    n = min(500, inp_flat.shape[0])
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    t = np.arange(n)
    for ch, ch_name in enumerate(['I', 'Q']):
        ax = axes[ch]
        ax.plot(t, inp_flat[:n, ch], label='PA Input', alpha=0.7, color=C_REF, lw=1.5)
        ax.plot(t, wo_flat[:n, ch], label='PA Output (w/o DPD)', alpha=0.7,
                color=C_GT, lw=1.5)
        ax.plot(t, w_flat[:n, ch], label='PA Output (w/ DPD)', alpha=0.7,
                color=C_PRED, linestyle='--', lw=1.5)
        ax.set_ylabel(f'{ch_name} Amplitude')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2, linestyle='--')
    axes[0].set_title('Time-Domain Waveform Comparison', loc='left', fontweight='bold')
    axes[1].set_xlabel('Sample Index')
    fig.tight_layout()
    _savefig(fig, os.path.join(plot_dir, 'waveform_comparison.png'))

    # Metrics summary bar chart
    if metrics_wo is not None and metrics_w is not None:
        plot_metrics_summary(metrics_wo, metrics_w,
                             os.path.join(plot_dir, 'metrics_summary.png'))


# ===========================================================================
# Fixed-axis re-rendering
# ===========================================================================

def compute_global_limits(epoch_cache, constants, step, spec):
    """Compute global axis limits across all cached epochs.

    Args:
        epoch_cache: {epoch: {'val': prediction, 'test': prediction}}
        constants: {'val_ground_truth', 'test_ground_truth',
                    'input_data_val', 'input_data_test',  (PA only)
                    'pa_only_val', 'pa_only_test'}          (DPD only)
        step: 'train_pa' or 'train_dpd'
        spec: signal spec dict

    Returns:
        fixed_limits dict with 'psd_ylim', 'ampm_ylim', 'error_ylim', 'waveform_ylim'
    """
    fs = spec.get('input_signal_fs', 800e6)
    nperseg = spec.get('nperseg', 2560)
    sw = 10

    psd_mins = []
    ampm_mins, ampm_maxs = [], []
    err_mins, err_maxs = [], []
    wave_mins, wave_maxs = [], []

    for epoch, preds in epoch_cache.items():
        for split in ['val', 'test']:
            pred = preds.get(split)
            gt = constants.get(f'{split}_ground_truth')
            if pred is None or gt is None:
                continue

            pred_flat = pred.reshape(-1, 2)
            gt_flat = gt.reshape(-1, 2)

            # PSD range (from prediction; constants like PA-only/GT are fixed)
            pred_c = IQ_to_complex(pred_flat[np.newaxis, :])[0]
            sig_2d = _ensure_2d_segments(pred_c, nperseg)
            _, ps = power_spectrum(sig_2d, fs=fs, nperseg=nperseg)
            ps_db = 10 * np.log10(ps / np.max(ps) + 1e-30)
            ps_smooth = moving_average(ps_db, sw)
            psd_mins.append(np.min(ps_smooth))

            # Include GT PSD only for PA training (GT is actual PA output)
            # Skip for DPD training (GT is linear target with -160 dB floor)
            if step == 'train_pa':
                gt_c = IQ_to_complex(gt_flat[np.newaxis, :])[0]
                sig_2d = _ensure_2d_segments(gt_c, nperseg)
                _, ps = power_spectrum(sig_2d, fs=fs, nperseg=nperseg)
                ps_db = 10 * np.log10(ps / np.max(ps) + 1e-30)
                ps_smooth = moving_average(ps_db, sw)
                psd_mins.append(np.min(ps_smooth))

            if step == 'train_dpd':
                pa_only = constants.get(f'pa_only_{split}')
                if pa_only is not None:
                    pa_flat = pa_only.reshape(-1, 2)
                    pa_c = IQ_to_complex(pa_flat[np.newaxis, :])[0]
                    sig_2d = _ensure_2d_segments(pa_c, nperseg)
                    _, ps = power_spectrum(sig_2d, fs=fs, nperseg=nperseg)
                    ps_db = 10 * np.log10(ps / np.max(ps) + 1e-30)
                    ps_smooth = moving_average(ps_db, sw)
                    psd_mins.append(np.min(ps_smooth))

            # AM/PM range
            if step == 'train_pa':
                inp_flat = constants.get(f'input_data_{split}')
                inp_flat = inp_flat.reshape(-1, 2) if inp_flat is not None else gt_flat
            else:
                inp_flat = gt_flat
            _, phase_in = _iq_to_amp_phase(inp_flat)
            phase_in_flat = phase_in.flatten()
            for out in [pred_flat]:
                _, phase_out = _iq_to_amp_phase(out)
                pd = np.degrees(np.angle(np.exp(1j * (phase_out.flatten() - phase_in_flat))))
                ampm_mins.append(np.percentile(pd, 0.5))
                ampm_maxs.append(np.percentile(pd, 99.5))

            # Error range
            error = pred_flat - gt_flat
            err_mins.append(np.min(error[:500]))
            err_maxs.append(np.max(error[:500]))

            # Waveform range
            for sig in [pred_flat[:500], gt_flat[:500]]:
                wave_mins.append(np.min(sig))
                wave_maxs.append(np.max(sig))

    # Add margin (5%)
    def _with_margin(lo, hi, pct=0.05):
        span = hi - lo
        return (lo - pct * span, hi + pct * span)

    limits = {}
    if psd_mins:
        limits['psd_ylim'] = _with_margin(min(psd_mins), 0)
    if ampm_mins and ampm_maxs:
        limits['ampm_ylim'] = _with_margin(min(ampm_mins), max(ampm_maxs))
    if err_mins and err_maxs:
        limits['error_ylim'] = _with_margin(min(err_mins), max(err_maxs))
    if wave_mins and wave_maxs:
        limits['waveform_ylim'] = _with_margin(min(wave_mins), max(wave_maxs))
    return limits


def rerender_epochs_fixed_axes(plot_dir, epoch_cache, constants, step, spec,
                                fixed_limits, demod=None,
                                full_const_data=None):
    """Re-render all cached epoch plots and best plots with fixed axis limits.

    Args:
        plot_dir: Base plot directory.
        epoch_cache: {epoch: {'val': prediction, 'test': prediction}}
        constants: Dict with ground_truth, input_data, pa_only arrays.
        step: 'train_pa' or 'train_dpd'
        spec: Signal spec dict.
        fixed_limits: Dict from compute_global_limits.
        demod: Demodulator instance for constellation plots.
        full_const_data: Full-sequence constellation data for APA datasets.
    """
    print("Re-rendering epoch plots with fixed axis limits...")
    for epoch, preds in sorted(epoch_cache.items()):
        for split in ['val', 'test']:
            pred = preds.get(split)
            gt = constants.get(f'{split}_ground_truth')
            if pred is None or gt is None:
                continue
            try:
                if step == 'train_pa':
                    inp = constants.get(f'input_data_{split}')
                    generate_epoch_plots_train_pa(
                        plot_dir, epoch, pred, gt, inp, split, spec,
                        demod=demod, fixed_limits=fixed_limits,
                        full_const_data=full_const_data,
                        rerender=True)
                elif step == 'train_dpd':
                    pa_only = constants.get(f'pa_only_{split}')
                    generate_epoch_plots_train_dpd(
                        plot_dir, epoch, pred, gt, split, spec,
                        demod=demod, pa_only_prediction=pa_only,
                        fixed_limits=fixed_limits,
                        full_const_data=full_const_data,
                        rerender=True)
            except Exception as e:
                print(f"  Warning: Re-render failed for epoch {epoch} {split}: {e}")

    # Also re-render best plots with fixed limits
    best_epoch = None
    best_key = None
    for epoch in sorted(epoch_cache.keys(), reverse=True):
        # Use the last cached epoch as proxy for best (best is always a cached epoch)
        best_epoch = epoch
        break
    if best_epoch is not None:
        for split in ['val', 'test']:
            pred = epoch_cache[best_epoch].get(split)
            gt = constants.get(f'{split}_ground_truth')
            if pred is None or gt is None:
                continue
            try:
                if step == 'train_pa':
                    inp = constants.get(f'input_data_{split}')
                    generate_epoch_plots_train_pa(
                        plot_dir, best_epoch, pred, gt, inp, split, spec,
                        demod=demod, subfolder='best', fixed_limits=fixed_limits,
                        full_const_data=full_const_data,
                        rerender=True)
                elif step == 'train_dpd':
                    pa_only = constants.get(f'pa_only_{split}')
                    generate_epoch_plots_train_dpd(
                        plot_dir, best_epoch, pred, gt, split, spec,
                        demod=demod, subfolder='best', pa_only_prediction=pa_only,
                        fixed_limits=fixed_limits,
                        full_const_data=full_const_data,
                        rerender=True)
            except Exception as e:
                print(f"  Warning: Re-render best failed for {split}: {e}")
    print("Re-rendering complete.")


# ===========================================================================
# GIF Animation Generator
# ===========================================================================

def generate_epoch_gifs(plot_dir, gif_duration=5.0):
    """Generate GIF animations for each plot type across all epochs.

    Scans history/epochs/ for epoch directories, collects PNGs by plot type,
    and creates one animated GIF per plot type under history/.

    Args:
        plot_dir: Base plot directory (contains history/epochs/).
        gif_duration: Total animation duration in seconds.
    """
    from PIL import Image

    epochs_dir = os.path.join(plot_dir, 'history', 'epochs')
    history_dir = os.path.join(plot_dir, 'history')

    if not os.path.isdir(epochs_dir):
        return

    # Discover sorted epoch directories
    epoch_dirs = sorted([
        d for d in os.listdir(epochs_dir)
        if os.path.isdir(os.path.join(epochs_dir, d)) and d.startswith('epoch_')
    ])
    if len(epoch_dirs) < 2:
        return

    # Discover plot types from first epoch
    first_epoch_path = os.path.join(epochs_dir, epoch_dirs[0])
    plot_files = sorted([f for f in os.listdir(first_epoch_path) if f.endswith('.png')])

    if not plot_files:
        return

    # Frame duration in milliseconds
    n_frames = len(epoch_dirs)
    frame_ms = max(int(gif_duration * 1000 / n_frames), 20)  # min 20ms per frame

    for plot_file in plot_files:
        frames = []
        for epoch_dir in epoch_dirs:
            img_path = os.path.join(epochs_dir, epoch_dir, plot_file)
            if os.path.exists(img_path):
                img = Image.open(img_path)
                # Convert to RGB (GIF doesn't support RGBA well)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                frames.append(img)

        if len(frames) < 2:
            continue

        gif_name = plot_file.replace('.png', '.gif')
        gif_path = os.path.join(history_dir, gif_name)
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=frame_ms,
            loop=0  # infinite loop
        )

    # Also create overview GIFs if overview PNGs exist
    for split in ['val', 'test']:
        overview_file = f'overview_{split}.png'
        frames = []
        for epoch_dir in epoch_dirs:
            img_path = os.path.join(epochs_dir, epoch_dir, overview_file)
            if os.path.exists(img_path):
                img = Image.open(img_path)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                frames.append(img)
        if len(frames) >= 2:
            gif_path = os.path.join(history_dir, f'overview_{split}.gif')
            frames[0].save(
                gif_path,
                save_all=True,
                append_images=frames[1:],
                duration=frame_ms,
                loop=0
            )

    print(f"GIF animations saved to {history_dir}")


# ===========================================================================
# Interactive HTML Dashboard
# ===========================================================================

def generate_dashboard_html(plot_dir, step, metadata):
    """Generate interactive HTML dashboard for epoch plot browsing.

    Creates history/dashboard.html that displays all epoch plots with a
    slider to scrub through training history.
    """
    import base64
    import re as regex

    history_dir = os.path.join(plot_dir, 'history')
    epochs_dir = os.path.join(history_dir, 'epochs')

    if not os.path.isdir(epochs_dir):
        print("Warning: No epochs directory found, skipping dashboard generation.")
        return

    # Discover epoch directories
    epoch_dirs = sorted([
        d for d in os.listdir(epochs_dir)
        if os.path.isdir(os.path.join(epochs_dir, d)) and d.startswith('epoch_')
    ])
    if not epoch_dirs:
        print("Warning: No epoch directories found, skipping dashboard generation.")
        return

    # Extract epoch numbers for display
    epoch_numbers = []
    for d in epoch_dirs:
        m = regex.match(r'epoch_(\d+)', d)
        epoch_numbers.append(int(m.group(1)) if m else 0)

    # Discover plot types from first epoch
    first_epoch_path = os.path.join(epochs_dir, epoch_dirs[0])
    all_pngs = sorted([f for f in os.listdir(first_epoch_path) if f.endswith('.png')])

    # Group by plot type: e.g., psd_val.png -> type='psd', split='val'
    plot_types = {}
    for png in all_pngs:
        name = png.replace('.png', '')
        parts = name.rsplit('_', 1)
        if len(parts) == 2 and parts[1] in ('val', 'test'):
            ptype, split = parts
        else:
            ptype = name
            split = None
        if ptype not in plot_types:
            plot_types[ptype] = {}
        plot_types[ptype][split or 'single'] = png

    # Pretty names for plot types
    pretty_names = {
        'psd': 'Power Spectral Density',
        'amam': 'AM/AM Characteristic',
        'ampm': 'AM/PM Characteristic',
        'constellation': 'Constellation Diagram',
        'waveform': 'Time-Domain Waveform',
        'error': 'Prediction Error',
        'overview': 'Overview',
    }

    # Load and base64-encode the logo
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    logo_path = os.path.join(project_root, 'pics', 'OpenDPDlogo.png')
    logo_b64 = ''
    if os.path.exists(logo_path):
        with open(logo_path, 'rb') as f:
            logo_b64 = base64.b64encode(f.read()).decode('utf-8')

    # Build metadata display
    meta = metadata or {}
    meta_items = []
    if meta.get('dataset'):
        meta_items.append(f"Dataset: {meta['dataset']}")
    if meta.get('step'):
        meta_items.append(f"Step: {meta['step']}")
    if meta.get('backbone'):
        meta_items.append(f"Backbone: {meta['backbone']}")
    if meta.get('hidden_size'):
        meta_items.append(f"Hidden: {meta['hidden_size']}")
    if meta.get('n_params'):
        meta_items.append(f"Params: {meta['n_params']:,}")
    if meta.get('model_id'):
        meta_items.append(f"ID: {meta['model_id']}")

    # Build plot grid HTML
    plot_rows_html = ''
    for ptype in ['psd', 'amam', 'ampm', 'constellation', 'waveform', 'error', 'overview']:
        if ptype not in plot_types:
            continue
        splits = plot_types[ptype]
        title = pretty_names.get(ptype, ptype.upper())
        cards_html = ''

        if 'val' in splits:
            fname = splits['val']
            cards_html += (
                '<div class="plot-card">'
                '<h4>Validation</h4>'
                '<img data-pattern="epochs/EPOCH/' + fname + '"'
                ' src="epochs/' + epoch_dirs[0] + '/' + fname + '"'
                ' alt="' + title + ' (val)" loading="lazy"/>'
                '</div>'
            )
        if 'test' in splits:
            fname = splits['test']
            cards_html += (
                '<div class="plot-card">'
                '<h4>Test</h4>'
                '<img data-pattern="epochs/EPOCH/' + fname + '"'
                ' src="epochs/' + epoch_dirs[0] + '/' + fname + '"'
                ' alt="' + title + ' (test)" loading="lazy"/>'
                '</div>'
            )
        if 'single' in splits:
            fname = splits['single']
            cards_html += (
                '<div class="plot-card plot-card-full">'
                '<img data-pattern="epochs/EPOCH/' + fname + '"'
                ' src="epochs/' + epoch_dirs[0] + '/' + fname + '"'
                ' alt="' + title + '" loading="lazy"/>'
                '</div>'
            )

        plot_rows_html += (
            '<div class="plot-row">'
            '<h3 class="plot-type-title">' + title + '</h3>'
            '<div class="plot-pair">'
            + cards_html +
            '</div>'
            '</div>'
        )

    # JSON arrays for JS
    import json
    epoch_dirs_json = json.dumps(epoch_dirs)
    epoch_numbers_json = json.dumps(epoch_numbers)
    max_epoch = epoch_numbers[-1] if epoch_numbers else 0

    # Build the logo tag safely
    logo_tag = ''
    if logo_b64:
        logo_tag = "<img class='logo' src='data:image/png;base64," + logo_b64 + "' alt='OpenDPD'/>"

    # Build badge tags safely
    badge_tags = ''
    for item in meta_items:
        badge_tags += '<span class="badge">' + item + '</span>'

    html_parts = []
    html_parts.append('<!DOCTYPE html>\n<html lang="en">\n<head>\n')
    html_parts.append('<meta charset="UTF-8">\n')
    html_parts.append('<meta name="viewport" content="width=device-width, initial-scale=1.0">\n')
    html_parts.append('<title>OpenDPD Dashboard</title>\n')
    html_parts.append('<style>\n')
    html_parts.append('''  :root {
    --bg-primary: #0a0a1a;
    --bg-secondary: #12122a;
    --bg-card: #16213e;
    --bg-card-hover: #1a2744;
    --text-primary: #e8e8f0;
    --text-secondary: #8888aa;
    --text-muted: #555577;
    --accent: #4a9eff;
    --accent-glow: rgba(74, 158, 255, 0.3);
    --accent-dim: rgba(74, 158, 255, 0.15);
    --border: #1e2d4a;
    --gradient-start: #4a9eff;
    --gradient-end: #7c3aed;
    --control-bar-height: 80px;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    background: var(--bg-primary);
    color: var(--text-primary);
    min-height: 100vh;
    padding-bottom: calc(var(--control-bar-height) + 20px);
  }
  .header {
    background: linear-gradient(135deg, var(--bg-secondary), var(--bg-card));
    border-bottom: 1px solid var(--border);
    padding: 20px 32px;
    display: flex;
    align-items: center;
    gap: 20px;
    position: sticky;
    top: 0;
    z-index: 100;
    backdrop-filter: blur(10px);
  }
  .logo {
    height: 56px;
    width: 56px;
    border-radius: 12px;
    box-shadow: 0 0 20px var(--accent-glow);
  }
  .header-text h1 {
    font-size: 1.6rem;
    font-weight: 700;
    background: linear-gradient(135deg, var(--gradient-start), var(--gradient-end));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }
  .meta-badges {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 6px;
  }
  .badge {
    background: var(--accent-dim);
    color: var(--accent);
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 500;
    border: 1px solid rgba(74, 158, 255, 0.2);
  }
  .plot-grid {
    padding: 24px 32px;
    display: flex;
    flex-direction: column;
    gap: 28px;
  }
  .plot-row {
    background: var(--bg-secondary);
    border-radius: 16px;
    padding: 20px;
    border: 1px solid var(--border);
    transition: border-color 0.3s;
  }
  .plot-row:hover {
    border-color: rgba(74, 158, 255, 0.3);
  }
  .plot-type-title {
    font-size: 1rem;
    font-weight: 600;
    color: var(--text-secondary);
    margin-bottom: 14px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }
  .plot-pair {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
    gap: 16px;
  }
  .plot-card {
    background: var(--bg-card);
    border-radius: 12px;
    padding: 12px;
    border: 1px solid var(--border);
    transition: all 0.3s ease;
  }
  .plot-card:hover {
    background: var(--bg-card-hover);
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.3), 0 0 12px var(--accent-glow);
    transform: translateY(-1px);
  }
  .plot-card-full {
    grid-column: 1 / -1;
  }
  .plot-card h4 {
    font-size: 0.8rem;
    color: var(--text-muted);
    margin-bottom: 8px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }
  .plot-card img {
    width: 100%;
    height: auto;
    border-radius: 8px;
    display: block;
    background: #111;
  }
  .control-bar {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    height: var(--control-bar-height);
    background: linear-gradient(180deg, rgba(10, 10, 26, 0.95), rgba(18, 18, 42, 0.98));
    backdrop-filter: blur(20px);
    border-top: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 0 32px;
    z-index: 200;
    box-shadow: 0 -4px 30px rgba(0, 0, 0, 0.4);
  }
  .play-btn {
    width: 44px;
    height: 44px;
    border-radius: 50%;
    border: 2px solid var(--accent);
    background: transparent;
    color: var(--accent);
    font-size: 1.1rem;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.2s;
    flex-shrink: 0;
  }
  .play-btn:hover {
    background: var(--accent);
    color: var(--bg-primary);
    box-shadow: 0 0 16px var(--accent-glow);
  }
  .play-btn.active {
    background: var(--accent);
    color: var(--bg-primary);
  }
  .slider-container {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  .epoch-slider {
    -webkit-appearance: none;
    appearance: none;
    width: 100%;
    height: 6px;
    border-radius: 3px;
    background: var(--bg-card);
    outline: none;
    cursor: pointer;
  }
  .epoch-slider::-webkit-slider-thumb {
    -webkit-appearance: none;
    appearance: none;
    width: 20px;
    height: 20px;
    border-radius: 50%;
    background: var(--accent);
    cursor: pointer;
    box-shadow: 0 0 10px var(--accent-glow);
    transition: transform 0.15s;
  }
  .epoch-slider::-webkit-slider-thumb:hover {
    transform: scale(1.2);
  }
  .epoch-slider::-moz-range-thumb {
    width: 20px;
    height: 20px;
    border-radius: 50%;
    background: var(--accent);
    cursor: pointer;
    border: none;
    box-shadow: 0 0 10px var(--accent-glow);
  }
  .slider-labels {
    display: flex;
    justify-content: space-between;
    font-size: 0.75rem;
    color: var(--text-muted);
  }
  .epoch-display {
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--text-primary);
    min-width: 160px;
    text-align: center;
    flex-shrink: 0;
  }
  .epoch-display .epoch-num {
    color: var(--accent);
    font-variant-numeric: tabular-nums;
  }
  .speed-btn {
    padding: 4px 10px;
    border-radius: 8px;
    border: 1px solid var(--border);
    background: transparent;
    color: var(--text-secondary);
    font-size: 0.75rem;
    cursor: pointer;
    transition: all 0.2s;
    flex-shrink: 0;
  }
  .speed-btn:hover, .speed-btn.active {
    border-color: var(--accent);
    color: var(--accent);
  }
  .kbd-hint {
    font-size: 0.65rem;
    color: var(--text-muted);
    text-align: center;
  }
  .kbd {
    background: var(--bg-card);
    padding: 1px 5px;
    border-radius: 3px;
    border: 1px solid var(--border);
    font-family: monospace;
  }
''')
    html_parts.append('</style>\n</head>\n<body>\n\n')

    # Header
    html_parts.append('<div class="header">\n')
    if logo_tag:
        html_parts.append('  ' + logo_tag + '\n')
    html_parts.append('  <div class="header-text">\n')
    html_parts.append('    <h1>OpenDPD Dashboard</h1>\n')
    html_parts.append('    <div class="meta-badges">\n')
    html_parts.append('      ' + badge_tags + '\n')
    html_parts.append('    </div>\n')
    html_parts.append('  </div>\n')
    html_parts.append('</div>\n\n')

    # Main content
    html_parts.append('<main class="plot-grid">\n')
    html_parts.append(plot_rows_html)
    html_parts.append('\n</main>\n\n')

    # Control bar
    html_parts.append('<div class="control-bar">\n')
    html_parts.append('  <button class="play-btn" id="playBtn" title="Play/Pause (Space)">&#9654;</button>\n')
    html_parts.append('  <div class="slider-container">\n')
    html_parts.append('    <input type="range" class="epoch-slider" id="epochSlider"'
                       ' min="0" max="' + str(len(epoch_dirs) - 1) + '" value="0" step="1"/>\n')
    html_parts.append('    <div class="slider-labels">\n')
    html_parts.append('      <span>Epoch ' + str(epoch_numbers[0]) + '</span>\n')
    html_parts.append('      <span class="kbd-hint"><span class="kbd">&larr;</span>'
                       ' <span class="kbd">&rarr;</span> step &middot;'
                       ' <span class="kbd">Space</span> play</span>\n')
    html_parts.append('      <span>Epoch ' + str(max_epoch) + '</span>\n')
    html_parts.append('    </div>\n')
    html_parts.append('  </div>\n')
    html_parts.append('  <div class="epoch-display">\n')
    html_parts.append('    Epoch <span class="epoch-num" id="epochNum">'
                       + str(epoch_numbers[0]) + '</span> / ' + str(max_epoch) + '\n')
    html_parts.append('  </div>\n')
    html_parts.append('  <button class="speed-btn" id="speedBtn" title="Playback speed">1x</button>\n')
    html_parts.append('</div>\n\n')

    # Script - use textContent for safe DOM updates
    html_parts.append('<script>\n(function() {\n')
    html_parts.append('  var epochDirs = ' + epoch_dirs_json + ';\n')
    html_parts.append('  var epochNums = ' + epoch_numbers_json + ';\n')
    html_parts.append('''  var slider = document.getElementById('epochSlider');
  var epochNum = document.getElementById('epochNum');
  var playBtn = document.getElementById('playBtn');
  var speedBtn = document.getElementById('speedBtn');
  var images = document.querySelectorAll('img[data-pattern]');

  var playing = false;
  var timer = null;
  var speeds = [1000, 500, 250, 2000];
  var speedLabels = ['1x', '2x', '4x', '0.5x'];
  var speedIdx = 0;

  function updateEpoch(idx) {
    var dir = epochDirs[idx];
    var num = epochNums[idx];
    epochNum.textContent = num;
    images.forEach(function(img) {
      var pattern = img.getAttribute('data-pattern');
      img.src = pattern.replace('EPOCH', dir);
    });
  }

  slider.addEventListener('input', function() {
    updateEpoch(parseInt(this.value));
  });

  function togglePlay() {
    playing = !playing;
    playBtn.textContent = playing ? '\\u23F8' : '\\u25B6';
    playBtn.classList.toggle('active', playing);
    if (playing) {
      timer = setInterval(function() {
        var v = parseInt(slider.value);
        if (v >= epochDirs.length - 1) {
          v = 0;
        } else {
          v++;
        }
        slider.value = v;
        updateEpoch(v);
      }, speeds[speedIdx]);
    } else {
      clearInterval(timer);
      timer = null;
    }
  }

  playBtn.addEventListener('click', togglePlay);

  speedBtn.addEventListener('click', function() {
    speedIdx = (speedIdx + 1) % speeds.length;
    this.textContent = speedLabels[speedIdx];
    if (playing) {
      clearInterval(timer);
      timer = setInterval(function() {
        var v = parseInt(slider.value);
        if (v >= epochDirs.length - 1) v = 0; else v++;
        slider.value = v;
        updateEpoch(v);
      }, speeds[speedIdx]);
    }
  });

  document.addEventListener('keydown', function(e) {
    if (e.key === 'ArrowRight') {
      e.preventDefault();
      var v = Math.min(parseInt(slider.value) + 1, epochDirs.length - 1);
      slider.value = v;
      updateEpoch(v);
    } else if (e.key === 'ArrowLeft') {
      e.preventDefault();
      var v = Math.max(parseInt(slider.value) - 1, 0);
      slider.value = v;
      updateEpoch(v);
    } else if (e.key === ' ') {
      e.preventDefault();
      togglePlay();
    }
  });
''')
    html_parts.append('})();\n</script>\n\n</body>\n</html>')

    html = ''.join(html_parts)

    dashboard_path = os.path.join(history_dir, 'dashboard.html')
    _ensure_dir(history_dir)
    with open(dashboard_path, 'w') as f:
        f.write(html)
    print(f"Dashboard saved to {dashboard_path}")
