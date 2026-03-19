#!/usr/bin/env python3
"""Evaluate independently generated 5G NR signals against all 8 target .mat files.

For each target file, generates an independent signal using per-channel baseband
processing, computes comparison metrics, and saves a 3x2 comparison plot.
A final summary dashboard aggregates all 8 results in a single figure.
"""

import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

# ── project style ──────────────────────────────────────────────────────────
from pathlib import Path
import importlib.util

_SFP_CANDIDATES = [
    Path(__file__).resolve().parents[3] / (
        ".claude/plugins/cache/figures4papers/figures4papers/unknown/"
        "skills/scientific-figure-pro/scripts/scientific_figure_pro.py"),
    Path.home() / (
        ".claude/plugins/cache/figures4papers/figures4papers/unknown/"
        "skills/scientific-figure-pro/scripts/scientific_figure_pro.py"),
]
for _p in _SFP_CANDIDATES:
    if _p.exists():
        _spec = importlib.util.spec_from_file_location("scientific_figure_pro", _p)
        _sfp = importlib.util.module_from_spec(_spec)
        sys.modules[_spec.name] = _sfp
        _spec.loader.exec_module(_sfp)
        _sfp.apply_publication_style(_sfp.FigureStyle(font_size=12, axes_linewidth=1.5))
        break
else:
    plt.rcParams.update({
        'font.size': 11,
        'axes.spines.right': False,
        'axes.spines.top': False,
        'legend.frameon': False,
        'savefig.bbox': 'tight',
    })

C_GEN  = '#0F4D92'   # blue  – generated
C_TGT  = '#B64342'   # red   – target
C_ERR  = '#8BCF8B'   # green – error


# ── target files ───────────────────────────────────────────────────────────

TARGET_FILES = [
    'Precook_Signal_100WDevice_[1cLTE20MHz_iBW20MHz_SR491p52MHz_200uS_TM3p1_PAPR10p0_IQ].mat',
    'Precook_Signal_100WDevice_[1cLTE20MHz_iBW20MHz_SR491p52MHz_200uS_TM3p1a_PAPR10p0_IQ].mat',
    'Precook_Signal_100WDevice_[2cLTE20MHz_iBW40MHz_SR491p52MHz_200uS_TM3p1_PAPR10p0_IQ].mat',
    'Precook_Signal_100WDevice_[2cLTE20MHz_iBW40MHz_SR491p52MHz_200uS_TM3p1a_PAPR10p0_IQ].mat',
    'Precook_Signal_100WDevice_[5cLTE20MHz_iBW100MHz_SR491p52MHz_200uS_TM3p1_PAPR10p0_IQ].mat',
    'Precook_Signal_100WDevice_[5cLTE20MHz_iBW100MHz_SR491p52MHz_200uS_TM3p1a_PAPR10p0_IQ].mat',
    'Precook_Signal_100WDevice_[10cLTE20MHz_iBW200MHz_SR983p04MHz_200uS_TM3p1_PAPR10p0_IQ].mat',
    'Precook_Signal_100WDevice_[10cLTE20MHz_iBW200MHz_SR983p04MHz_200uS_TM3p1a_PAPR10p0_IQ].mat',
]

# Directory containing the .mat files (one level up from this script)
MAT_DIR = os.path.join(os.path.dirname(__file__), '..')


# ── helpers ────────────────────────────────────────────────────────────────

def _papr(s):
    return 10 * np.log10(np.max(np.abs(s)**2) / np.mean(np.abs(s)**2))


def _annotate(ax, txt, loc='upper right'):
    ha = 'right' if 'right' in loc else 'left'
    va = 'top' if 'upper' in loc else 'bottom'
    x = 0.97 if ha == 'right' else 0.03
    y = 0.95 if va == 'top' else 0.05
    ax.text(x, y, txt, transform=ax.transAxes, ha=ha, va=va,
            fontsize=9, family='monospace',
            bbox=dict(boxstyle='round,pad=0.4', fc='white', alpha=0.85,
                      ec='#999999', lw=0.5))


def _short_name(p):
    """Short label like '5c-256Q' from parse_mat_params dict."""
    return f"{p['num_channels']}c-{p['mod_order']}Q"


# ── metric computation ─────────────────────────────────────────────────────

def compute_metrics(gen_sig, tgt_sig, sr, num_channels, bw_mhz, mod_order):
    """Compute comparison metrics between generated and target signals.

    Parameters
    ----------
    gen_sig, tgt_sig : 1-D complex arrays
    sr : float  (Hz)
    num_channels : int
    bw_mhz : float  (per-channel bandwidth in MHz)
    mod_order : int  (64 or 256)

    Returns
    -------
    dict with keys: psd_mae, d_papr, ccdf_dev, ch_err, evm_gen, evm_tgt
    """
    from generate_signal import demodulate_all, _ideal_qam_grid, compute_evm

    # Normalise to unit RMS
    gen = gen_sig / np.sqrt(np.mean(np.abs(gen_sig) ** 2))
    tgt = tgt_sig / np.sqrt(np.mean(np.abs(tgt_sig) ** 2))
    n = min(len(gen), len(tgt))
    gen, tgt = gen[:n], tgt[:n]

    bw_hz = bw_mhz * 1e6
    freqs = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / sr))
    ft_gen = np.fft.fftshift(np.fft.fft(gen))
    ft_tgt = np.fft.fftshift(np.fft.fft(tgt))

    kern = max(1, int(200e3 / (sr / n)))
    pg = uniform_filter1d(20 * np.log10(np.abs(ft_gen) + 1e-30), kern)
    pt = uniform_filter1d(20 * np.log10(np.abs(ft_tgt) + 1e-30), kern)

    # PSD MAE – smoothed, in-band only
    ib_mask = (np.abs(freqs) < (num_channels * bw_hz / 2 + 5e6)) & \
              (np.abs(ft_tgt) > 1e-2 * np.max(np.abs(ft_tgt)))
    psd_mae = float(np.mean(np.abs(pg[ib_mask] - pt[ib_mask]))) \
        if np.any(ib_mask) else 0.0

    # |PAPR_gen - PAPR_tgt|
    d_papr = abs(_papr(gen) - _papr(tgt))

    # CCDF deviation at {4,6,8,10} dB thresholds
    def _ccdf(s, th):
        inst = np.abs(s) ** 2 / (np.mean(np.abs(s) ** 2) + 1e-30)
        return np.mean(10 * np.log10(inst + 1e-30) > th)

    ccdf_dev = max(abs(_ccdf(gen, th) - _ccdf(tgt, th))
                   for th in (4, 6, 8, 10))

    # Max per-channel power error
    carrier_centers = [(k - (num_channels - 1) / 2) * bw_hz
                       for k in range(num_channels)]
    ch_err = 0.0
    for fc in carrier_centers:
        m = (freqs >= fc - bw_hz / 2) & (freqs <= fc + bw_hz / 2)
        pg_c = 10 * np.log10(np.mean(np.abs(ft_gen[m]) ** 2) + 1e-30)
        pt_c = 10 * np.log10(np.mean(np.abs(ft_tgt[m]) ** 2) + 1e-30)
        ch_err = max(ch_err, abs(pg_c - pt_c))

    # EVM vs ideal QAM (generated and target independently)
    ideal = _ideal_qam_grid(mod_order)

    re_gen, im_gen = demodulate_all(gen_sig, sr,
                                    num_channels=num_channels,
                                    bw_mhz=bw_mhz)
    re_tgt, im_tgt = demodulate_all(tgt_sig, sr,
                                    num_channels=num_channels,
                                    bw_mhz=bw_mhz)

    if re_gen.size > 0:
        pts_gen = re_gen + 1j * im_gen
        evm_gen_val, _, _ = compute_evm(pts_gen, ideal)
    else:
        evm_gen_val = 100.0

    if re_tgt.size > 0:
        pts_tgt = re_tgt + 1j * im_tgt
        evm_tgt_val, _, _ = compute_evm(pts_tgt, ideal)
    else:
        evm_tgt_val = 100.0

    return dict(
        psd_mae=psd_mae,
        d_papr=d_papr,
        ccdf_dev=ccdf_dev,
        ch_err=ch_err,
        evm_gen=evm_gen_val,
        evm_tgt=evm_tgt_val,
    )


# ── comparison plot ────────────────────────────────────────────────────────

def plot_comparison(gen_sig, tgt_sig, sr, save_path,
                    num_channels, bw_mhz, mod_order, metrics, short_name=''):
    """Create 3x2 comparison plot for generated vs target signal.

    Panels
    ------
    (a) Normalised PSD overlay
    (b) PSD difference in-band
    (c) Per-channel power bars
    (d) CCDF comparison
    (e) Amplitude distribution histogram
    (f) QAM constellation (generated signal)
    """
    from generate_signal import demodulate_all, _ideal_qam_grid, compute_evm

    # Normalise to unit RMS
    gen = gen_sig / np.sqrt(np.mean(np.abs(gen_sig) ** 2))
    tgt = tgt_sig / np.sqrt(np.mean(np.abs(tgt_sig) ** 2))
    n = min(len(gen), len(tgt))
    gen, tgt = gen[:n], tgt[:n]

    bw_hz = bw_mhz * 1e6
    carrier_centers = [(k - (num_channels - 1) / 2) * bw_hz
                       for k in range(num_channels)]

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    title = f'Independent: {short_name} ({"TM3.1" if mod_order == 64 else "TM3.1a"})'
    fig.suptitle(title, fontweight='bold', fontsize=14, y=0.995)

    ft_gen = np.fft.fftshift(np.fft.fft(gen))
    ft_tgt = np.fft.fftshift(np.fft.fft(tgt))
    f_hz   = np.linspace(-sr / 2, sr / 2, n, endpoint=False)
    freqs  = f_hz

    kern = max(1, int(200e3 / (sr / n)))
    pg = uniform_filter1d(20 * np.log10(np.abs(ft_gen) + 1e-30), kern)
    pt = uniform_filter1d(20 * np.log10(np.abs(ft_tgt) + 1e-30), kern)
    pg_norm = pg - np.max(pg)
    pt_norm = pt - np.max(pt)

    # ── (a) Normalised PSD ────────────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(f_hz / 1e6, pt_norm, color=C_TGT, lw=1.0, alpha=0.85,
            label='Target')
    ax.plot(f_hz / 1e6, pg_norm, color=C_GEN, lw=1.0, alpha=0.85,
            ls='--', label='Generated')
    ax.axhline(-60, color='#888888', ls=':', lw=0.8, label='-60 dB ref')

    # Shade inter-channel gap regions
    for k in range(num_channels - 1):
        gap_center = (carrier_centers[k] + carrier_centers[k + 1]) / 2
        ax.axvspan((gap_center - 0.4e6) / 1e6, (gap_center + 0.4e6) / 1e6,
                   alpha=0.08, color='orange', zorder=0)

    # Annotate inter-channel floor for multi-channel signals
    if num_channels > 1:
        gap_floors = []
        for k in range(num_channels - 1):
            gap_center = (carrier_centers[k] + carrier_centers[k + 1]) / 2
            gap_mask = (f_hz >= gap_center - 0.3e6) & (f_hz <= gap_center + 0.3e6)
            if np.any(gap_mask):
                gen_floor = float(np.mean(pg_norm[gap_mask]))
                tgt_floor = float(np.mean(pt_norm[gap_mask]))
                gap_floors.append((gap_center / 1e6, gen_floor, tgt_floor))

        if gap_floors:
            floor_txt = 'Inter-ch floor (dB):\n'
            for gc, gf, tf in gap_floors:
                floor_txt += f'  {gc:+.0f} MHz: gen={gf:.1f} tgt={tf:.1f}\n'
            _annotate(ax, floor_txt.rstrip(), loc='lower left')

    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('Normalised PSD (dB)')
    ax.set_title('(a)  Power spectral density', loc='left', fontweight='bold')
    ax.set_ylim(-100, 5)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(alpha=0.15, ls='--')

    # ── (b) PSD difference in-band ────────────────────────────────────────
    ax = axes[0, 1]
    ib_mask = np.abs(f_hz) < (num_channels * bw_hz / 2 + 5e6)
    sig_mask = np.abs(ft_tgt) > 1e-4 * np.max(np.abs(ft_tgt))
    mask = ib_mask & sig_mask

    psd_diff = pg - pt

    if np.any(mask):
        f_m = f_hz[mask]
        diff_m = psd_diff[mask]
        if len(diff_m) > kern:
            diff_m = uniform_filter1d(diff_m, kern)
        ax.fill_between(f_m / 1e6, diff_m, alpha=0.35, color=C_ERR)
        ax.plot(f_m / 1e6, diff_m, color='#2d7d2d', lw=0.5)
        ax.axhline(0, color='k', lw=0.5, ls=':')

        mae = np.mean(np.abs(diff_m))
        max_diff = np.max(np.abs(diff_m))
        std_diff = np.std(diff_m)
        _annotate(ax, f'MAE  = {mae:.4f} dB\nMax  = {max_diff:.4f} dB\n'
                      f'Std  = {std_diff:.4f} dB')

    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('PSD difference (dB)')
    ax.set_title('(b)  PSD difference in-band (Gen \u2212 Target)', loc='left',
                 fontweight='bold')
    ax.grid(alpha=0.15, ls='--')

    # ── (c) Per-channel power ─────────────────────────────────────────────
    ax = axes[1, 0]
    ch_labels, pow_gen, pow_tgt = [], [], []
    for ch in range(num_channels):
        fc = carrier_centers[ch]
        ch_mask = (f_hz >= fc - bw_hz / 2) & (f_hz <= fc + bw_hz / 2)
        pg_ch = 10 * np.log10(np.mean(np.abs(ft_gen[ch_mask]) ** 2) + 1e-30)
        pt_ch = 10 * np.log10(np.mean(np.abs(ft_tgt[ch_mask]) ** 2) + 1e-30)
        pow_gen.append(pg_ch)
        pow_tgt.append(pt_ch)
        ch_labels.append(f'Ch{ch+1}\n{fc/1e6:+.0f} MHz')

    x_pos = np.arange(num_channels)
    w = 0.35
    ax.bar(x_pos - w / 2, pow_tgt, w, label='Target', color=C_TGT,
           alpha=0.8, edgecolor='white', lw=0.5)
    ax.bar(x_pos + w / 2, pow_gen, w, label='Generated', color=C_GEN,
           alpha=0.8, edgecolor='white', lw=0.5)
    for i in range(num_channels):
        diff = pow_gen[i] - pow_tgt[i]
        y_top = max(pow_gen[i], pow_tgt[i]) + 0.15
        ax.text(x_pos[i], y_top, f'{diff:+.2f} dB', ha='center', va='bottom',
                fontsize=8, fontweight='bold',
                color='#2d7d2d' if abs(diff) < 0.5 else '#cc3333')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(ch_labels, fontsize=9)
    ax.set_ylabel('Power (dB)')
    ax.set_title('(c)  Per-channel power', loc='left', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.15, ls='--', axis='y')

    max_ch_err = max(abs(pow_gen[i] - pow_tgt[i])
                     for i in range(num_channels))
    _annotate(ax, f'Max |\u0394P| = {max_ch_err:.3f} dB')

    # ── (d) CCDF ──────────────────────────────────────────────────────────
    ax = axes[1, 1]
    thresholds = np.linspace(0, 12, 200)
    inst_gen = np.abs(gen) ** 2 / np.mean(np.abs(gen) ** 2)
    inst_tgt = np.abs(tgt) ** 2 / np.mean(np.abs(tgt) ** 2)
    inst_gen_db = 10 * np.log10(inst_gen + 1e-30)
    inst_tgt_db = 10 * np.log10(inst_tgt + 1e-30)
    ccdf_gen = np.array([np.mean(inst_gen_db > th) for th in thresholds])
    ccdf_tgt = np.array([np.mean(inst_tgt_db > th) for th in thresholds])

    ax.semilogy(thresholds, ccdf_tgt, color=C_TGT, lw=1.5, label='Target')
    ax.semilogy(thresholds, np.maximum(ccdf_gen, 1e-7), color=C_GEN, lw=1.5,
                ls='--', label='Generated')
    ax.set_xlabel('PAPR threshold (dB)')
    ax.set_ylabel('CCDF  P(PAPR > x)')
    ax.set_title('(d)  Complementary CDF', loc='left', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.15, ls='--', which='both')
    ax.set_ylim(1e-5, 1)

    vals_txt = []
    for th in (6, 8, 10):
        cg = np.mean(inst_gen_db > th)
        ct = np.mean(inst_tgt_db > th)
        vals_txt.append(f'>{th} dB: gen={cg:.5f} tgt={ct:.5f}')
    _annotate(ax, '\n'.join(vals_txt))

    # ── (e) Amplitude distribution ────────────────────────────────────────
    ax = axes[2, 0]
    bins = np.linspace(0, 4, 200)
    ax.hist(np.abs(tgt), bins=bins, density=True, alpha=0.6, color=C_TGT,
            label='Target', edgecolor='none')
    ax.hist(np.abs(gen), bins=bins, density=True, alpha=0.6, color=C_GEN,
            label='Generated', edgecolor='none')
    ax.set_xlabel('|signal| (unit RMS)')
    ax.set_ylabel('Probability density')
    ax.set_title('(e)  Amplitude distribution', loc='left', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.15, ls='--')

    _annotate(ax,
              f'PAPR gen = {_papr(gen):.2f} dB\n'
              f'PAPR tgt = {_papr(tgt):.2f} dB',
              loc='upper right')

    # ── (f) QAM constellation (generated signal) ──────────────────────────
    ax = axes[2, 1]
    ideal = _ideal_qam_grid(mod_order)

    re_gen, im_gen = demodulate_all(gen_sig, sr,
                                    num_channels=num_channels,
                                    bw_mhz=bw_mhz)

    if re_gen.size > 0:
        pts = re_gen + 1j * im_gen
        evm_val, snapped, err = compute_evm(pts, ideal)
        dist = np.abs(pts - snapped)

        # Sub-sample if very large to keep plot manageable
        max_pts = 20000
        if len(pts) > max_pts:
            idx = np.random.default_rng(0).choice(len(pts), max_pts,
                                                   replace=False)
            pts_plot   = pts[idx]
            dist_plot  = dist[idx]
        else:
            pts_plot  = pts
            dist_plot = dist

        sc = ax.scatter(pts_plot.real, pts_plot.imag,
                        c=dist_plot, cmap='plasma',
                        s=1.5, alpha=0.4, rasterized=True,
                        vmin=0, vmax=np.percentile(dist, 95))
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Distance to ideal (a.u.)', fontsize=8)

        # Overlay ideal grid
        ax.scatter(ideal.real, ideal.imag,
                   s=10, c='k', marker='+', zorder=5, label='Ideal QAM')

        _annotate(ax,
                  f'EVM (gen vs ideal) = {evm_val:.2f}%\n'
                  f'Points plotted: {len(pts_plot):,}',
                  loc='upper right')
    else:
        ax.text(0.5, 0.5, 'No symbols demodulated',
                ha='center', va='center', transform=ax.transAxes)
        _annotate(ax, 'EVM = N/A')

    ax.set_xlabel('In-phase')
    ax.set_ylabel('Quadrature')
    ax.set_title('(f)  QAM constellation (generated)', loc='left',
                 fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(alpha=0.15, ls='--')

    # ── global summary ────────────────────────────────────────────────────
    evm_g = metrics.get('evm_gen', float('nan'))
    evm_t = metrics.get('evm_tgt', float('nan'))
    fig.text(0.5, 0.005,
             f'PAPR: gen {_papr(gen):.2f} dB | tgt {_papr(tgt):.2f} dB      '
             f'PSD MAE: {metrics["psd_mae"]:.4f} dB      '
             f'EVM gen={evm_g:.1f}% tgt={evm_t:.1f}%',
             ha='center', va='bottom', fontsize=9, family='monospace',
             style='italic', color='#555555')

    fig.tight_layout(rect=[0, 0.025, 1, 0.97])
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved -> {save_path}')


# ── summary dashboard ──────────────────────────────────────────────────────

def plot_summary_dashboard(all_metrics, save_path):
    """5-panel bar chart dashboard summarising metrics across all 8 signals.

    Parameters
    ----------
    all_metrics : list of dicts, each with keys:
        short_name, psd_mae, d_papr, ccdf_dev, ch_err, evm_gen
    save_path : str
    """
    names     = [m['short_name'] for m in all_metrics]
    psd_maes  = [m['psd_mae']   for m in all_metrics]
    d_paprs   = [m['d_papr']    for m in all_metrics]
    ccdf_devs = [m['ccdf_dev']  for m in all_metrics]
    ch_errs   = [m['ch_err']    for m in all_metrics]
    evms      = [m['evm_gen']   for m in all_metrics]

    fig, axes = plt.subplots(1, 5, figsize=(16, 6))
    fig.suptitle('Independent Generation: Summary Dashboard (All 8 Signals)',
                 fontweight='bold', fontsize=13)

    bar_kw = dict(alpha=0.8, edgecolor='white', lw=0.5)

    panels = [
        (axes[0], 'PSD MAE (dB)',        psd_maes,  0.5,   C_GEN),
        (axes[1], '|dPAPR| (dB)',         d_paprs,   0.2,   C_TGT),
        (axes[2], 'CCDF Deviation',       ccdf_devs, 0.02,  '#8B5E3C'),
        (axes[3], 'Max Ch Power Err (dB)', ch_errs,  0.1,   '#6A4C93'),
        (axes[4], 'EVM vs ideal QAM (%)', evms,      None,  C_ERR),
    ]

    x = np.arange(len(names))
    for ax, ylabel, vals, thr, color in panels:
        bars = ax.bar(x, vals, color=color, **bar_kw)
        if thr is not None:
            ax.axhline(thr, color='crimson', ls='--', lw=1.2,
                       label=f'Thr={thr}')
            ax.legend(fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(alpha=0.15, ls='--', axis='y')

        # Annotate bar values
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + ax.get_ylim()[1] * 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=7,
                    rotation=90)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Summary dashboard saved -> {save_path}')


# ── main ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description='Evaluate independently generated signals vs all 8 target files')
    ap.add_argument('--single', type=str, default=None, metavar='PATTERN',
                    help='Only process files whose filename contains PATTERN '
                         '(for quick testing)')
    ap.add_argument('--seed', type=int, default=42,
                    help='Random seed for generate_independent (default 42)')
    args = ap.parse_args()

    from generate_signal import (
        generate_independent, load_mat, parse_mat_params,
        demodulate_all, _ideal_qam_grid, compute_evm,
    )

    out_dir = os.path.join(os.path.dirname(__file__), '..', 'plots')
    os.makedirs(out_dir, exist_ok=True)

    all_metrics = []

    for fname in TARGET_FILES:
        # Optional filter
        if args.single and args.single not in fname:
            continue

        fpath = os.path.join(MAT_DIR, fname)
        if not os.path.exists(fpath):
            print(f'WARNING: file not found, skipping: {fpath}')
            continue

        try:
            p = parse_mat_params(fname)
        except ValueError as e:
            print(f'WARNING: cannot parse params from {fname}: {e}')
            continue

        sname = _short_name(p)
        tm = 'TM3.1a' if p['mod_order'] == 256 else 'TM3.1'
        print(f'\n{"="*70}')
        print(f'Processing: {sname} ({tm})')
        print(f'  File: {fname}')
        print(f'  Channels={p["num_channels"]}, BW={p["bw_mhz"]} MHz/ch, '
              f'SR={p["sample_rate"]/1e6:.2f} MHz')

        print('  Loading target ...')
        tgt_sig, sr = load_mat(fpath)
        print(f'  Loaded: {len(tgt_sig)} samples, '
              f'{len(tgt_sig)/sr*1e6:.1f} us @ {sr/1e6:.2f} MHz')

        print('  Generating independent signal ...')
        gen_sig, info = generate_independent(
            tgt_sig, sr,
            num_channels=p['num_channels'],
            bw_mhz=p['bw_mhz'],
            mod_order=p['mod_order'],
            seed=args.seed,
        )

        print('  Computing metrics ...')
        met = compute_metrics(gen_sig, tgt_sig, sr,
                              num_channels=p['num_channels'],
                              bw_mhz=p['bw_mhz'],
                              mod_order=p['mod_order'])
        met['short_name'] = sname

        # Plot comparison
        plot_fname = f'eval_{sname.replace("-", "_")}_{tm.replace(".", "")}.png'
        plot_path  = os.path.join(out_dir, plot_fname)
        print('  Plotting comparison ...')
        plot_comparison(gen_sig, tgt_sig, sr, plot_path,
                        num_channels=p['num_channels'],
                        bw_mhz=p['bw_mhz'],
                        mod_order=p['mod_order'],
                        metrics=met,
                        short_name=sname)

        all_metrics.append(met)

        print(f'  Metrics for {sname}:')
        print(f'    PSD MAE  = {met["psd_mae"]:.4f} dB')
        print(f'    |dPAPR|  = {met["d_papr"]:.4f} dB')
        print(f'    CCDF dev = {met["ccdf_dev"]:.6f}')
        print(f'    Ch Err   = {met["ch_err"]:.4f} dB')
        print(f'    EVM gen  = {met["evm_gen"]:.2f}%')
        print(f'    EVM tgt  = {met["evm_tgt"]:.2f}%')

    if not all_metrics:
        print('\nNo files were processed.')
        return

    # ── Summary table ─────────────────────────────────────────────────────
    hdr = (f'\n{"Signal":<12} {"PSD MAE":>10} {"dPAPR":>10} '
           f'{"CCDF dev":>12} {"Ch Err":>10} '
           f'{"EVM gen%":>10} {"EVM tgt%":>10}')
    sep = '-' * len(hdr)
    print(f'\n{"="*70}')
    print('SUMMARY TABLE')
    print('="*70')
    print(hdr)
    print(sep)
    for m in all_metrics:
        print(f'{m["short_name"]:<12} '
              f'{m["psd_mae"]:>10.4f} '
              f'{m["d_papr"]:>10.4f} '
              f'{m["ccdf_dev"]:>12.6f} '
              f'{m["ch_err"]:>10.4f} '
              f'{m["evm_gen"]:>10.2f} '
              f'{m["evm_tgt"]:>10.2f}')
    print(sep)

    # ── Summary dashboard ─────────────────────────────────────────────────
    if len(all_metrics) > 1:
        dashboard_path = os.path.join(out_dir, 'eval_summary_dashboard.png')
        plot_summary_dashboard(all_metrics, dashboard_path)
    else:
        print('\n(Single file mode: skipping dashboard)')


if __name__ == '__main__':
    main()
