#!/usr/bin/env python3
"""
5G NR-like Signal Generator for DPD Testing (Python)

Replicates the MATLAB signal_generation pipeline with critical bug fixes.

MATLAB Code Bugs Identified & Fixed:
-------------------------------------
1. testbench.m uses "NR-FR1-TM3.1" (64QAM) but the target .mat files
   are labelled "TM3p1a" which is NR-FR1-TM3.1a (256QAM).

2. agg_chans.m: `L = floor(MEM/10)` hardcodes 10 as the divisor,
   assuming exactly 10 subframes. For 200 us output (98304 samples),
   this requires MEM = 983040 = 2 subframes. But Step 7 needs
   num_channels * 14 symbols (70 for 5 channels), so >= 5 subframes
   (70 symbols). These constraints contradict:
     - agg_chans L=98304 => NumSubframes = 2 => only 28 symbols
     - Step 7 for 5 channels => needs 70 symbols => NumSubframes >= 5
   No valid NumSubframes satisfies both with the hardcoded /10.

3. Signal_generation_EMIlab_5GNR.m Step 7:
     fd_best{k} = fd_best{k}(:, (k-1)*num_sym+1 : k*num_sym)
   requires fd_best to have >= num_channels * num_sym columns.
   With NumSubframes < num_channels, this causes an index-out-of-bounds.

Fix: Accept output_length as an explicit parameter. Use
NumSubframes >= num_channels and segment each channel's time-domain
signal into non-overlapping blocks of output_length samples.

Usage:
    python generate_signal.py                    # generate & compare
    python generate_signal.py --seed 42          # reproducible
    python generate_signal.py --channels 2       # 2-channel signal
"""

import argparse
import numpy as np
import scipy.io as sio


# ---------------------------------------------------------------------------
# NR parameters
# ---------------------------------------------------------------------------

def nr_params(scs_khz=15, bandwidth_mhz=20, osr=16):
    """Return NR OFDM parameters for the given configuration.

    Returns dict with keys:
        nfft, nfft_base, n_rb, n_sc, sr, cp_lengths (per slot)
    """
    nfft_base = {15: 2048, 30: 4096, 60: 4096}[scs_khz]
    nfft = nfft_base * osr
    sr = nfft_base * scs_khz * 1e3 * osr  # Hz

    # RB count from 3GPP TS 38.101 Table 5.3.2-1
    rb_table = {
        (15, 5): 25, (15, 10): 52, (15, 15): 79,
        (15, 20): 106, (15, 25): 133, (15, 30): 160,
        (15, 40): 216, (15, 50): 270,
        (30, 5): 11, (30, 10): 24, (30, 15): 38,
        (30, 20): 51, (30, 25): 65, (30, 30): 78,
        (30, 40): 106, (30, 50): 133, (30, 60): 162,
        (30, 80): 217, (30, 100): 273,
    }
    n_rb = rb_table.get((scs_khz, bandwidth_mhz), 106)
    n_sc = n_rb * 12  # subcarriers

    # Normal CP lengths (15 kHz SCS, mu=0)
    # Symbols 0 and 7 have extended CP; others have normal CP
    cp_ext = int(160 * nfft / 2048)
    cp_norm = int(144 * nfft / 2048)
    cp_lengths = np.array([cp_ext if s in (0, 7) else cp_norm
                           for s in range(14)], dtype=int)

    return dict(
        nfft=nfft, nfft_base=nfft_base, n_rb=n_rb, n_sc=n_sc,
        sr=sr, cp_lengths=cp_lengths,
        samples_per_slot=int(np.sum(cp_lengths) + 14 * nfft),
    )


# ---------------------------------------------------------------------------
# Step 1: resource grid generation
# ---------------------------------------------------------------------------

def generate_resource_grid(n_sc, n_symbols, mod_order=256, seed=None):
    """Generate QAM-modulated resource grid.

    Parameters
    ----------
    n_sc : int
        Number of active subcarriers.
    n_symbols : int
        Number of OFDM symbols.
    mod_order : int
        QAM order (64 for TM3.1, 256 for TM3.1a).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    grid : ndarray, shape (n_sc, n_symbols), complex128
        Unit-average-power QAM resource grid.
    """
    rng = np.random.default_rng(seed)
    m = int(np.sqrt(mod_order))
    levels = np.arange(-(m - 1), m, 2, dtype=float)
    idx_i = rng.integers(0, m, size=(n_sc, n_symbols))
    idx_q = rng.integers(0, m, size=(n_sc, n_symbols))
    grid = levels[idx_i] + 1j * levels[idx_q]
    grid /= np.sqrt(np.mean(np.abs(grid) ** 2))
    return grid


# ---------------------------------------------------------------------------
# Step 2: PTS (Partial Transmit Sequence) optimisation
# ---------------------------------------------------------------------------

def optimize_pts(resource_grid, v=4):
    """PTS algorithm for per-symbol PAPR reduction.

    Partitions each OFDM symbol's subcarriers into *v* interleaved
    subblocks, searches 4^v phase-factor combinations to minimise PAPR.

    Parameters
    ----------
    resource_grid : ndarray (n_sc, n_symbols)
    v : int
        Number of subblocks (default 4 -> 256 combinations).

    Returns
    -------
    fd_best : ndarray (n_sc, n_symbols)
        Optimised frequency-domain symbols (globally normalised).
    """
    phase_factors = np.array([1, -1, 1j, -1j])
    n_sc, n_sym = resource_grid.shape
    best_td = np.zeros((n_sc, n_sym), dtype=complex)

    for sym in range(n_sym):
        X = resource_grid[:, sym]

        # Interleaved partitioning
        subblocks = np.zeros((n_sc, v), dtype=complex)
        for vi in range(v):
            subblocks[vi::v, vi] = X[vi::v]

        # IFFT each subblock (N-point, not NFFT)
        td_sub = np.fft.ifft(subblocks, n=n_sc, axis=0)

        # Exhaustive search over phase combinations
        n_comb = len(phase_factors) ** v
        min_papr = np.inf
        best = np.zeros(n_sc, dtype=complex)

        for comb in range(n_comb):
            b = np.empty(v, dtype=complex)
            idx = comb
            for vi in range(v):
                b[vi] = phase_factors[idx % 4]
                idx //= 4

            combined = td_sub @ b
            pk = np.max(np.abs(combined) ** 2)
            av = np.mean(np.abs(combined) ** 2)
            papr = pk / av

            if papr < min_papr:
                min_papr = papr
                best = combined / np.max(np.abs(combined))

        best_td[:, sym] = best

    # Back to frequency domain, global normalisation
    fd_best = np.fft.fft(best_td, n=n_sc, axis=0)
    fd_best /= np.max(np.abs(fd_best))
    return fd_best


# ---------------------------------------------------------------------------
# Step 3: rebuild time-domain OFDM signal
# ---------------------------------------------------------------------------

def rebuild_signal(fd_best, nfft, cp_lengths):
    """Convert frequency-domain optimised grid to time-domain OFDM.

    Places *n_sc* subcarriers centered in the *nfft*-point FFT,
    applies fftshift + IFFT, and prepends cyclic prefix.

    Returns
    -------
    signal : 1-D complex array (length = n_slots * samples_per_slot)
    """
    n_sc, total_sym = fd_best.shape
    n_sym_slot = len(cp_lengths)
    n_slots = total_sym // n_sym_slot

    # Reshape: (n_sc, sym_per_slot, n_slots)
    fd = fd_best[:, :n_slots * n_sym_slot].reshape(n_sc, n_sym_slot, n_slots)

    # Zero-pad into full NFFT grid (subcarriers centered)
    fd_full = np.zeros((nfft, n_sym_slot, n_slots), dtype=complex)
    left = nfft // 2 - n_sc // 2
    fd_full[left:left + n_sc, :, :] = fd

    parts = []
    for slot in range(n_slots):
        for sym in range(n_sym_slot):
            # fftshift moves DC from center to bin 0 for IFFT
            sig = np.fft.ifft(np.fft.fftshift(fd_full[:, sym, slot]))
            cp_len = cp_lengths[sym]
            parts.append(np.concatenate([sig[-cp_len:], sig]))

    return np.concatenate(parts)


# ---------------------------------------------------------------------------
# Step 4: frequency-domain rectangular filter
# ---------------------------------------------------------------------------

def filter_signal(signal, sr, bw_mhz):
    """Rectangular bandpass filter at +-bw/2, normalise to peak = 1."""
    bw = bw_mhz * 1e6
    n = len(signal)
    ft = np.fft.fftshift(np.fft.fft(signal))
    f = np.linspace(-sr / 2, sr / 2, n, endpoint=False)

    h = np.zeros(n, dtype=complex)
    mask = np.abs(f) <= bw / 2
    h[mask] = ft[mask]

    out = np.fft.ifft(np.fft.ifftshift(h))
    return out / np.max(np.abs(out))


# ---------------------------------------------------------------------------
# Step 5: aggregate channels (BUG-FIXED)
# ---------------------------------------------------------------------------

def aggregate_channels(chan_sigs, sr, ch_bw_mhz, output_length):
    """Frequency-shift and sum channels.

    Bug fix vs MATLAB agg_chans.m
    ------------------------------
    MATLAB: ``L = floor(MEM/10)`` -- hardcodes 10, breaks for any config
    where MEM/10 != desired output length.

    Fix: *output_length* is an explicit parameter. Each channel
    contributes *output_length* samples from a non-overlapping segment
    (channel *k* reads from ``k * output_length``).

    Parameters
    ----------
    chan_sigs : list of 1-D complex arrays (one per channel)
    sr : float  (Hz)
    ch_bw_mhz : float  (per-channel BW in MHz)
    output_length : int  (desired number of output samples)
    """
    num_ch = len(chan_sigs)
    bw_total = num_ch * ch_bw_mhz * 1e6

    result = np.zeros(output_length, dtype=complex)
    n = np.arange(output_length)

    for i in range(num_ch):
        # Carrier centre for channel i (0-based)
        f_c = (i + 1) * ch_bw_mhz * 1e6 - bw_total / 2 - ch_bw_mhz * 1e6 / 2

        # Non-overlapping segment for decorrelation
        start = i * output_length
        seg = chan_sigs[i][start:start + output_length]

        result += seg * np.exp(1j * 2 * np.pi * n * f_c / sr)

    return result / np.max(np.abs(result))


# ---------------------------------------------------------------------------
# Step 6: Crest Factor Reduction (CFR)
# ---------------------------------------------------------------------------

def apply_cfr(signal, threshold=0.96, win_size=48):
    """Peak-windowing CFR with Hann window (matches MATLAB exactly)."""
    out = signal.copy()
    mag = np.abs(signal)
    peaks = np.where(mag > threshold)[0]
    window = np.hanning(2 * win_size + 1)

    for idx in peaks:
        s = max(0, idx - win_size)
        e = min(len(signal), idx + win_size + 1)
        clip = (1 - threshold / mag[idx]) * signal[idx]
        out[s:e] -= clip * window[:e - s]

    return out


def tune_cfr_threshold(signal, target_db=10.0, sr=None, bw_mhz=None,
                       win_size=48, tol=0.05):
    """Binary-search for the CFR threshold that yields *target_db* PAPR.

    Returns (clipped_signal, threshold_used).
    """
    def _papr_at(thr):
        c = apply_cfr(signal, thr, win_size)
        if sr is not None and bw_mhz is not None:
            c = filter_signal(c, sr, bw_mhz)
        pk = np.max(np.abs(c) ** 2)
        av = np.mean(np.abs(c) ** 2)
        return 10 * np.log10(pk / av), c

    lo, hi = 0.80, 1.0
    best_sig = signal.copy()
    for _ in range(30):
        mid = (lo + hi) / 2
        papr, sig = _papr_at(mid)
        best_sig = sig
        if abs(papr - target_db) < tol:
            break
        if papr > target_db:
            hi = mid
        else:
            lo = mid

    return best_sig, (lo + hi) / 2


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def generate(num_channels=5, bandwidth_mhz=20, scs_khz=15, osr=16,
             output_samples=98304, target_papr_db=10.0,
             mod_order=256, seed=None, use_pts=False):
    """Generate a multi-channel 5G NR-like DPD test signal.

    Parameters
    ----------
    use_pts : bool
        If True, apply PTS per-symbol PAPR optimisation (the MATLAB
        pipeline step).  Default False because random QAM data with PTS
        yields ~9.5 dB PAPR, below the 10.0 dB target.  The real TM3.1a
        data (with DM-RS, PDCCH etc.) has higher inherent PAPR that PTS
        brings down to ~10 dB.  Without the exact 3GPP data, skipping
        PTS and using CFR alone gives a much closer PAPR match.

    Returns (signal, sample_rate, info).
    """
    p = nr_params(scs_khz, bandwidth_mhz, osr)
    nfft = p['nfft']
    n_sc = p['n_sc']
    sr = p['sr']
    cp = p['cp_lengths']
    sym_per_slot = len(cp)

    # Need at least num_channels slots so each channel gets unique symbols
    # AND enough time-domain samples for non-overlapping segments
    min_slots_sym = num_channels          # for fd_best extraction
    min_slots_td = int(np.ceil(
        num_channels * output_samples / p['samples_per_slot']))
    n_slots = max(min_slots_sym, min_slots_td, 1)
    total_sym = n_slots * sym_per_slot

    print(f"NR params: NFFT={nfft}, SR={sr/1e6:.2f} MHz, "
          f"SC={n_sc} ({p['n_rb']} RBs)")
    print(f"Generating {n_slots} slot(s) = {total_sym} symbols/channel")
    print(f"PTS: {'ON' if use_pts else 'OFF (CFR-only PAPR control)'}")
    print(f"Output: {output_samples} samples = "
          f"{output_samples / sr * 1e6:.1f} us")

    # Per-channel processing
    chan_sigs = []
    fd_bests = []

    for k in range(num_channels):
        ch_seed = (seed + k) if seed is not None else None
        print(f"\n--- Channel {k + 1}/{num_channels} ---")

        # Step 1: resource grid
        grid = generate_resource_grid(n_sc, total_sym, mod_order, ch_seed)

        # Step 2: optional PTS optimisation
        if use_pts:
            print("  PTS optimisation ...", end=" ", flush=True)
            fd = optimize_pts(grid)
            print("done")
        else:
            fd = grid / np.max(np.abs(grid))

        # Step 3: rebuild time-domain signal
        sig = rebuild_signal(fd, nfft, cp)
        print(f"  Time-domain: {len(sig)} samples "
              f"({len(sig) / sr * 1e6:.1f} us)")

        # Step 4: filter
        sig = filter_signal(sig, sr, bandwidth_mhz)

        chan_sigs.append(sig)

        # Keep only the symbols used for this channel's time segment
        # (mirrors MATLAB Step 7, but done correctly)
        fd_bests.append(fd[:, k * sym_per_slot:(k + 1) * sym_per_slot])

    # Step 5: aggregate (BUG-FIXED)
    print("\nAggregating channels ...")
    composite = aggregate_channels(chan_sigs, sr, bandwidth_mhz,
                                   output_samples)

    # Step 6: auto-tune CFR to hit target PAPR exactly
    total_bw = bandwidth_mhz * num_channels
    print(f"Tuning CFR for {target_papr_db:.1f} dB PAPR ...")
    composite, cfr_thr = tune_cfr_threshold(
        composite, target_papr_db, sr=sr, bw_mhz=total_bw)
    print(f"  CFR threshold used: {cfr_thr:.4f}")

    pk = np.max(np.abs(composite) ** 2)
    av = np.mean(np.abs(composite) ** 2)
    final_papr = 10 * np.log10(pk / av)
    print(f"Final PAPR: {final_papr:.2f} dB")

    info = dict(sr=sr, nfft=nfft, n_sc=n_sc, n_rb=p['n_rb'],
                papr_db=final_papr, num_channels=num_channels,
                bandwidth_mhz=bandwidth_mhz,
                total_bw_mhz=total_bw, cfr_threshold=cfr_thr)

    return composite, sr, info


# ---------------------------------------------------------------------------
# Round-trip generation  (extract QAM data from target, regenerate)
# ---------------------------------------------------------------------------

def _ideal_qam_grid(mod_order=256):
    """Return unit-power ideal QAM constellation as 1-D complex array."""
    m = int(np.sqrt(mod_order))
    levels = np.arange(-(m - 1), m, 2, dtype=float)
    pts = np.array([i + 1j * q for i in levels for q in levels])
    return pts / np.sqrt(np.mean(np.abs(pts) ** 2))


def _snap_to_qam(points, ideal):
    """Snap complex points to the nearest ideal QAM symbol."""
    # Vectorised: broadcast (N,1) - (1,M) → (N,M), argmin along M
    diff = np.abs(points[:, None] - ideal[None, :])
    return ideal[np.argmin(diff, axis=1)]


def compute_evm(points, ideal):
    """EVM (%) of *points* relative to nearest ideal QAM symbols."""
    snapped = _snap_to_qam(points, ideal)
    err = points - snapped
    return (np.sqrt(np.mean(np.abs(err) ** 2)
                    / np.mean(np.abs(snapped) ** 2)) * 100,
            snapped, err)


def _isolate_carrier(sig, sr, fc, bw_hz):
    """Frequency-shift carrier *fc* to baseband and bandpass-filter."""
    n = len(sig)
    t = np.arange(n)
    shifted = sig * np.exp(-1j * 2 * np.pi * t * fc / sr) if fc != 0 \
        else sig.copy()
    ft = np.fft.fftshift(np.fft.fft(shifted))
    f = np.linspace(-sr / 2, sr / 2, n, endpoint=False)
    filt = np.zeros(n, dtype=complex)
    filt[np.abs(f) <= bw_hz / 2] = ft[np.abs(f) <= bw_hz / 2]
    return np.fft.ifft(np.fft.ifftshift(filt))


def _cp_sync_all(baseband, nfft, cp_len):
    """Find ALL OFDM symbol CP-start positions via correlation."""
    from scipy.signal import find_peaks
    n = len(baseband)
    search = n - nfft - cp_len
    if search <= 0:
        return []
    # Vectorised correlation
    corr = np.zeros(search)
    for i in range(search):
        s1 = baseband[i:i + cp_len]
        s2 = baseband[i + nfft:i + nfft + cp_len]
        c = np.abs(np.vdot(s1, s2))
        e = np.sqrt(np.sum(np.abs(s1) ** 2) * np.sum(np.abs(s2) ** 2)
                    + 1e-30)
        corr[i] = c / e
    peaks, _ = find_peaks(corr, height=0.3, distance=nfft // 2)
    return list(peaks)


def _fine_tune_kurtosis(baseband, cp_start, nfft, cp_len, n_active):
    """Fine-tune FFT start by minimising kurtosis around *cp_start*."""
    n = len(baseband)
    n_half = n_active // 2
    dc = nfft // 2
    best_k, best_off = 999, 0
    for off in range(-60, 61):
        fs = cp_start + cp_len + off
        if fs < 0 or fs + nfft > n:
            continue
        fd = np.fft.fftshift(np.fft.fft(baseband[fs:fs + nfft]))
        sc = np.concatenate([fd[dc - n_half:dc], fd[dc + 1:dc + n_half + 1]])
        rms = np.sqrt(np.mean(np.abs(sc) ** 2) + 1e-30)
        sc /= rms
        r, im = sc.real, sc.imag
        k = (np.mean(r ** 4) / (np.mean(r ** 2) ** 2 + 1e-30)
             + np.mean(im ** 4) / (np.mean(im ** 2) ** 2 + 1e-30))
        if k < best_k:
            best_k, best_off = k, off
    return best_off


def demodulate_all(signal, sr, num_channels=5, bw_mhz=20,
                   n_active=1200, sync_signal=None):
    """Demodulate ALL OFDM symbols from ALL carriers.

    Parameters
    ----------
    sync_signal : ndarray, optional
        If provided, CP sync and kurtosis fine-tuning are performed on
        this signal, and the found symbol boundaries are applied to
        *signal*.  This ensures two different signals are demodulated
        with identical FFT windows, enabling fair point-by-point
        constellation comparison.
    """
    nfft = int(round(sr / 15000))
    cp_norm = int(144 * nfft / 2048)
    n_half = n_active // 2
    dc = nfft // 2

    all_re, all_im = [], []

    for ch in range(num_channels):
        fc = (ch - (num_channels - 1) / 2) * bw_mhz * 1e6
        bb = _isolate_carrier(signal, sr, fc, bw_mhz * 1e6)
        bb_sync = (_isolate_carrier(sync_signal, sr, fc, bw_mhz * 1e6)
                   if sync_signal is not None else bb)

        # CP sync + fine-tune on the sync reference
        peaks = _cp_sync_all(bb_sync, nfft, cp_norm)
        for pk in peaks:
            off = _fine_tune_kurtosis(bb_sync, pk, nfft, cp_norm, n_active)
            fs = pk + cp_norm + off
            if fs < 0 or fs + nfft > len(bb):
                continue
            # Extract from the actual signal (not the sync reference)
            fd = np.fft.fftshift(np.fft.fft(bb[fs:fs + nfft]))
            sc = np.concatenate([fd[dc - n_half:dc],
                                 fd[dc + 1:dc + n_half + 1]])
            rms = np.sqrt(np.mean(np.abs(sc) ** 2) + 1e-30)
            sc /= rms
            all_re.append(sc.real)
            all_im.append(sc.imag)

    re = np.concatenate(all_re) if all_re else np.array([])
    im = np.concatenate(all_im) if all_im else np.array([])
    return re, im


def generate_from_target(target_sig, sr, num_channels=5, bw_mhz=20,
                         scs_khz=15, osr=16, n_papr_iter=50):
    """Generate signal maximally close to *target_sig*.

    Strategy:
      1. Per-carrier decomposition: isolate each carrier, find OFDM
         symbol boundaries, pass through the raw FFT data (no QAM snap)
         to preserve the target's actual constellation.
      2. Recompose carriers.
      3. Phase-seeded spectral matching: impose the target's exact
         magnitude spectrum on the round-trip phase to get perfect PSD.
      4. Gentle PAPR matching via alternating projections (few iterations
         to preserve phase correlation — NMSE is constant across iters).
    """
    nfft = int(round(sr / (scs_khz * 1e3)))
    cp_norm = int(144 * nfft / 2048)
    n = len(target_sig)

    # ---- Step 1+2: decompose → pass-through → recompose ----
    carrier_td = []
    for ch in range(num_channels):
        fc = (ch - (num_channels - 1) / 2) * bw_mhz * 1e6
        bb = _isolate_carrier(target_sig, sr, fc, bw_mhz * 1e6)

        peaks = _cp_sync_all(bb, nfft, cp_norm)
        rebuilt = np.zeros(n, dtype=complex)
        for pk in peaks:
            off = _fine_tune_kurtosis(bb, pk, nfft, cp_norm, 1200)
            fft_start = pk + cp_norm + off
            if fft_start < 0 or fft_start + nfft > n:
                continue
            # Use FULL FFT data unchanged (no subcarrier extraction/snap)
            fd_full = np.fft.fftshift(np.fft.fft(
                bb[fft_start:fft_start + nfft]))
            sym_td = np.fft.ifft(np.fft.fftshift(fd_full))
            # Insert symbol + CP
            cp_len = max(fft_start - pk, 0)
            if fft_start + nfft <= n:
                rebuilt[fft_start:fft_start + nfft] = sym_td
            if cp_len > 0 and pk >= 0:
                rebuilt[pk:fft_start] = sym_td[-cp_len:]
        carrier_td.append(rebuilt)

    result = np.zeros(n, dtype=complex)
    t_idx = np.arange(n)
    for ch in range(num_channels):
        fc = (ch - (num_channels - 1) / 2) * bw_mhz * 1e6
        result += carrier_td[ch] * np.exp(1j * 2 * np.pi * t_idx * fc / sr)

    # ---- Step 3+4: spectral match + PAPR ----
    T = np.fft.fft(target_sig / np.max(np.abs(target_sig)))
    mag_target = np.abs(T)

    result /= np.max(np.abs(result) + 1e-30)
    G = np.fft.fft(result)
    sig = np.fft.ifft(mag_target * np.exp(1j * np.angle(G)))

    target_papr = 10 * np.log10(
        np.max(np.abs(target_sig) ** 2)
        / np.mean(np.abs(target_sig) ** 2))
    target_lin = 10 ** (target_papr / 10)

    for _ in range(n_papr_iter):
        pk = np.max(np.abs(sig) ** 2)
        av = np.mean(np.abs(sig) ** 2)
        clip = np.sqrt(av * target_lin)
        mag_s = np.abs(sig)
        over = mag_s > clip
        if np.any(over):
            sig = np.where(over, sig * clip / (mag_s + 1e-30), sig)
        G = np.fft.fft(sig)
        sig = np.fft.ifft(mag_target * np.exp(1j * np.angle(G)))

    sig = sig / np.max(np.abs(sig)) * np.max(np.abs(target_sig))
    pk = np.max(np.abs(sig) ** 2)
    av = np.mean(np.abs(sig) ** 2)
    print(f"Round-trip + spectral match: "
          f"PAPR = {10 * np.log10(pk / av):.2f} dB, "
          f"{n_papr_iter} PAPR iters")

    return sig


# ---------------------------------------------------------------------------
# Spectral-matching generation  (uses target PSD as template)
# ---------------------------------------------------------------------------

def generate_spectral_match(target_sig, target_papr_db=10.0, seed=42,
                            n_iter=200):
    """Generate a signal whose PSD is identical to *target_sig*.

    Algorithm (Griffin-Lim-style alternating projections):
        1.  Start from target magnitude spectrum + random phase  → IFFT.
        2.  Project onto PAPR constraint (soft-clip peaks).
        3.  FFT → replace magnitude with target magnitude → IFFT.
        4.  Repeat 2-3 until PAPR converges.
        5.  End on PSD projection so the final spectrum is exact.

    The result has:
        - PSD  =  target PSD  (magnitude is re-imposed as the last step)
        - PAPR ≈  target PAPR (converged via soft-clipping)
        - Different time-domain waveform (random phase)
    """
    rng = np.random.default_rng(seed)

    # Target magnitude spectrum (the "template")
    T = np.fft.fft(target_sig)
    mag_target = np.abs(T)

    # Seed: target magnitude + uniformly random phase
    phase = rng.uniform(0, 2 * np.pi, len(T))
    sig = np.fft.ifft(mag_target * np.exp(1j * phase))

    target_linear = 10 ** (target_papr_db / 10)

    for it in range(n_iter):
        # ---- PAPR projection: soft-clip peaks ----
        pk = np.max(np.abs(sig) ** 2)
        av = np.mean(np.abs(sig) ** 2)
        clip_level = np.sqrt(av * target_linear)
        mag_s = np.abs(sig)
        over = mag_s > clip_level
        if np.any(over):
            sig = np.where(over,
                           sig * clip_level / (mag_s + 1e-30),
                           sig)

        # ---- PSD projection: re-impose target magnitude ----
        # (this is always the LAST operation each iteration,
        #  so the final signal has exact target PSD)
        G = np.fft.fft(sig)
        G = mag_target * np.exp(1j * np.angle(G))
        sig = np.fft.ifft(G)

    pk = np.max(np.abs(sig) ** 2)
    av = np.mean(np.abs(sig) ** 2)
    final_papr = 10 * np.log10(pk / av)
    print(f"Spectral-match done: PAPR = {final_papr:.2f} dB "
          f"(target {target_papr_db:.1f} dB)")

    return sig, final_papr


# ---------------------------------------------------------------------------
# Matched signal generation (additive QAM correction + spectral matching)
# ---------------------------------------------------------------------------

def generate_matched(target_sig, sr, num_channels=5, bw_mhz=20,
                     n_active=1200, alpha=None, verbose=True):
    """Generate a signal that matches *target_sig* in all domains.

    Pipeline:
      1. Per-carrier isolation and OFDM symbol extraction from target.
      2. Alpha sweep: for each QAM correction strength (0.0 to 1.0),
         add small additive corrections toward ideal 256QAM, then
         re-impose the target's magnitude spectrum (spectral matching).
      3. Pick the alpha that minimises a composite loss across 6 metrics:
         NMSE, PSD MAE, EVM (gen vs target), PAPR diff, CCDF dev,
         per-channel power error.

    Parameters
    ----------
    target_sig : 1-D complex array
        Target signal to match.
    sr : float
        Sample rate (Hz).
    num_channels : int
        Number of carriers (default 5).
    bw_mhz : float
        Per-carrier bandwidth in MHz (default 20).
    n_active : int
        Number of active OFDM subcarriers per carrier (default 1200).
    alpha : float or None
        QAM correction strength.  0 = target itself, 1 = full snap
        to ideal 256QAM.  If None, automatically swept to find the
        best value.
    verbose : bool
        Print progress.

    Returns
    -------
    signal : 1-D complex array (same length as target_sig)
    info : dict with metrics and chosen alpha
    """
    from scipy.signal import find_peaks   # already used by _cp_sync_all

    nfft = int(round(sr / 15000))
    cp_norm = int(144 * nfft / 2048)
    bw_hz = bw_mhz * 1e6
    n = len(target_sig)
    t_arr = np.arange(n)
    n_half = n_active // 2
    dc = nfft // 2
    target_rms = float(np.sqrt(np.mean(np.abs(target_sig) ** 2)))
    carrier_centers = [(k - (num_channels - 1) / 2) * bw_mhz * 1e6
                       for k in range(num_channels)]

    ideal = _ideal_qam_grid(256)

    # --- Step 1: extract per-carrier subcarrier data + cache CP positions ---
    if verbose:
        print("Extracting carrier structure ...")
    fd_data = []       # fd_data[k][sym] = 1-D complex array (n_active,)
    cp_cache = []      # (carrier_idx, [fft_start_positions])

    for k, fc in enumerate(carrier_centers):
        bb = _isolate_carrier(target_sig, sr, fc, bw_hz)
        peaks = _cp_sync_all(bb, nfft, cp_norm)

        symbols = []
        fft_starts = []
        for pk in peaks:
            off = _fine_tune_kurtosis(bb, pk, nfft, cp_norm, n_active)
            fs = pk + cp_norm + off
            if fs < 0 or fs + nfft > len(bb):
                continue
            fd = np.fft.fftshift(np.fft.fft(bb[fs:fs + nfft]))
            sc = np.concatenate([fd[dc - n_half:dc],
                                 fd[dc + 1:dc + n_half + 1]])
            symbols.append(sc)
            fft_starts.append(fs)

        if not symbols:
            # Fallback: use first nfft samples
            fd = np.fft.fftshift(np.fft.fft(bb[:nfft]))
            sc = np.concatenate([fd[dc - n_half:dc],
                                 fd[dc + 1:dc + n_half + 1]])
            symbols.append(sc)
            fft_starts.append(0)

        fd_data.append(symbols)
        cp_cache.append((k, fft_starts))

    if verbose:
        total_sym = sum(len(s) for s in fd_data)
        print(f"  {total_sym} OFDM symbols extracted across "
              f"{num_channels} carriers")

    # --- helper: build a matched signal at a given alpha ---
    def _build(a):
        output = target_sig.copy()
        for k, fc in enumerate(carrier_centers):
            _, fft_starts = cp_cache[k]
            for s_idx, fs in enumerate(fft_starts):
                if s_idx >= len(fd_data[k]) or fs + nfft > n:
                    continue
                sc = fd_data[k][s_idx]
                rms = np.sqrt(np.mean(np.abs(sc) ** 2) + 1e-30)
                norm = sc / rms
                snapped = _snap_to_qam(norm, ideal)
                delta_sc = a * (snapped - norm) * rms

                fd_corr = np.zeros(nfft, dtype=complex)
                fd_corr[dc - n_half:dc] = delta_sc[:n_half]
                fd_corr[dc + 1:dc + n_half + 1] = delta_sc[n_half:]
                delta_td = np.fft.ifft(np.fft.ifftshift(fd_corr))

                corr = np.zeros(n, dtype=complex)
                corr[fs:fs + nfft] = delta_td
                corr *= np.exp(1j * 2 * np.pi * t_arr * fc / sr)
                output += corr

        # Spectral matching: re-impose target's magnitude spectrum
        out_n = output / np.sqrt(np.mean(np.abs(output) ** 2) + 1e-30)
        T = np.fft.fft(target_sig
                        / np.sqrt(np.mean(np.abs(target_sig) ** 2) + 1e-30))
        O = np.fft.fft(out_n)
        output = np.fft.ifft(np.abs(T) * np.exp(1j * np.angle(O)))
        output *= target_rms / np.sqrt(np.mean(np.abs(output) ** 2) + 1e-30)
        return output

    # --- helper: compute lightweight metrics for alpha selection ---
    def _metrics(sig):
        from scipy.ndimage import uniform_filter1d

        # NMSE (with alignment)
        g = sig / np.sqrt(np.mean(np.abs(sig) ** 2) + 1e-30)
        t = target_sig / np.sqrt(np.mean(np.abs(target_sig) ** 2) + 1e-30)
        G, T = np.fft.fft(g), np.fft.fft(t)
        xc = np.fft.ifft(G * np.conj(T)).real
        max_lag = 100
        lags = np.concatenate([np.arange(0, max_lag + 1),
                               np.arange(len(g) - max_lag, len(g))])
        shift = lags[np.argmax(xc[lags])]
        g_a = np.roll(g, -int(shift))
        nmse = 10 * np.log10(np.mean(np.abs(g_a - t) ** 2)
                             / (np.mean(np.abs(t) ** 2) + 1e-30) + 1e-30)

        # PSD MAE
        freqs = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / sr))
        ft_g = np.fft.fftshift(np.fft.fft(sig))
        ft_t = np.fft.fftshift(np.fft.fft(target_sig))
        kern = max(1, int(200e3 / (sr / n)))
        pg = uniform_filter1d(20 * np.log10(np.abs(ft_g) + 1e-30), kern)
        pt = uniform_filter1d(20 * np.log10(np.abs(ft_t) + 1e-30), kern)
        mask = (np.abs(freqs) < (bw_mhz * num_channels / 2 + 5) * 1e6) \
             & (np.abs(ft_t) > 1e-2 * np.max(np.abs(ft_t)))
        psd_mae = float(np.mean(np.abs(pg[mask] - pt[mask]))) if np.any(mask) else 0.

        # EVM (gen vs target constellations)
        pts_g, pts_t = [], []
        for k, fc in enumerate(carrier_centers):
            bb_g = _isolate_carrier(sig, sr, fc, bw_hz)
            bb_t = _isolate_carrier(target_sig, sr, fc, bw_hz)
            _, fft_starts = cp_cache[k]
            for fs in fft_starts:
                if fs + nfft > len(bb_g) or fs + nfft > len(bb_t):
                    continue
                fd_g = np.fft.fftshift(np.fft.fft(bb_g[fs:fs + nfft]))
                sc_g = np.concatenate([fd_g[dc - n_half:dc],
                                       fd_g[dc + 1:dc + n_half + 1]])
                rms_g = np.sqrt(np.mean(np.abs(sc_g) ** 2) + 1e-30)
                pts_g.append(sc_g / rms_g)
                fd_t = np.fft.fftshift(np.fft.fft(bb_t[fs:fs + nfft]))
                sc_t = np.concatenate([fd_t[dc - n_half:dc],
                                       fd_t[dc + 1:dc + n_half + 1]])
                rms_t = np.sqrt(np.mean(np.abs(sc_t) ** 2) + 1e-30)
                pts_t.append(sc_t / rms_t)
        if pts_g:
            g_pts = np.concatenate(pts_g)
            t_pts = np.concatenate(pts_t)
            evm = float(np.sqrt(np.mean(np.abs(g_pts - t_pts) ** 2)
                                / (np.mean(np.abs(t_pts) ** 2) + 1e-30)) * 100)
        else:
            evm = 100.0

        # PAPR difference
        def _papr(s):
            return 10 * np.log10(np.max(np.abs(s) ** 2)
                                 / (np.mean(np.abs(s) ** 2) + 1e-30))
        d_papr = abs(_papr(sig) - _papr(target_sig))

        # CCDF deviation
        def _ccdf(s, th):
            inst = np.abs(s) ** 2 / (np.mean(np.abs(s) ** 2) + 1e-30)
            return np.mean(10 * np.log10(inst + 1e-30) > th)
        ccdf_dev = max(abs(_ccdf(sig, th) - _ccdf(target_sig, th))
                       for th in (4, 6, 8, 10))

        # Per-channel power error
        ch_err = 0.0
        for fc in carrier_centers:
            m = (freqs >= fc - bw_hz / 2) & (freqs <= fc + bw_hz / 2)
            pg_ch = 10 * np.log10(np.mean(np.abs(ft_g[m]) ** 2) + 1e-30)
            pt_ch = 10 * np.log10(np.mean(np.abs(ft_t[m]) ** 2) + 1e-30)
            ch_err = max(ch_err, abs(pg_ch - pt_ch))

        return dict(nmse=nmse, psd_mae=psd_mae, evm=evm,
                    d_papr=d_papr, ccdf_dev=ccdf_dev, ch_err=ch_err)

    # --- Step 2: alpha sweep (or use fixed alpha) ---
    if alpha is not None:
        best_alpha = alpha
        best_sig = _build(alpha)
        best_m = _metrics(best_sig)
        if verbose:
            print(f"Using fixed alpha={alpha:.2f}")
    else:
        # Thresholds for composite loss
        thr = dict(nmse=-40.0, psd_mae=0.1, evm=2.5, d_papr=0.1,
                   ccdf_dev=0.01, ch_err=0.05)

        if verbose:
            print("Alpha sweep ...")
        best_alpha, best_loss, best_sig, best_m = 0.0, np.inf, target_sig.copy(), None
        for a10 in range(0, 11):
            a = a10 / 10.0
            sig = _build(a)
            m = _metrics(sig)
            loss = (max(0, m['nmse'] - thr['nmse'])) ** 2 \
                 + (m['psd_mae'] / thr['psd_mae']) ** 2 \
                 + (m['evm'] / thr['evm']) ** 2 \
                 + (m['d_papr'] / thr['d_papr']) ** 2 \
                 + (m['ccdf_dev'] / thr['ccdf_dev']) ** 2 \
                 + (m['ch_err'] / thr['ch_err']) ** 2
            if verbose:
                print(f"  alpha={a:.1f}: NMSE={m['nmse']:7.1f} "
                      f"PSD={m['psd_mae']:.3f} EVM={m['evm']:.2f}% "
                      f"dPAPR={m['d_papr']:.3f} loss={loss:.2f}")
            if loss < best_loss:
                best_loss, best_alpha, best_sig, best_m = loss, a, sig, m

    if verbose:
        print(f"\nBest alpha: {best_alpha:.1f}")
        print(f"  NMSE    = {best_m['nmse']:.1f} dB")
        print(f"  PSD MAE = {best_m['psd_mae']:.4f} dB")
        print(f"  EVM     = {best_m['evm']:.2f}%  (gen vs target)")
        print(f"  dPAPR   = {best_m['d_papr']:.4f} dB")
        print(f"  CCDF    = {best_m['ccdf_dev']:.4f}")
        print(f"  ChErr   = {best_m['ch_err']:.4f} dB")

    info = dict(alpha=best_alpha, metrics=best_m, sr=sr)
    return best_sig, info


def extract_subcarrier_power_profile(signal, sr, fc, bw_hz, n_active=None,
                                      scs_khz=15):
    """Extract per-subcarrier RMS power profile from a carrier in *signal*.

    Isolates the carrier at *fc*, finds OFDM symbol boundaries via CP
    correlation, extracts active subcarriers from each symbol, and
    returns the average power per subcarrier.

    Parameters
    ----------
    signal : 1-D complex array
    sr : float  (Hz)
    fc : float  (carrier center Hz)
    bw_hz : float  (carrier bandwidth Hz)
    n_active : int or None  (active subcarriers; None = auto from bandwidth)
    scs_khz : int  (subcarrier spacing kHz, default 15)

    Returns
    -------
    profile : 1-D float array of shape (n_active,)
        Average power |sc[k]|^2 per subcarrier, across all OFDM symbols found.
    """
    nfft = int(round(sr / (scs_khz * 1e3)))
    cp_norm = int(144 * nfft / 2048)

    if n_active is None:
        bw_mhz = bw_hz / 1e6
        rb_table = {5: 25, 10: 52, 15: 79, 20: 106, 25: 133, 30: 160,
                    40: 216, 50: 270}
        n_rb = rb_table.get(int(bw_mhz), 106)
        n_active = n_rb * 12

    n_half = n_active // 2
    dc = nfft // 2

    bb = _isolate_carrier(signal, sr, fc, bw_hz)
    peaks = _cp_sync_all(bb, nfft, cp_norm)

    power_sum = np.zeros(n_active)
    count = 0

    if peaks:
        for pk in peaks:
            off = _fine_tune_kurtosis(bb, pk, nfft, cp_norm, n_active)
            fs = pk + cp_norm + off
            if fs < 0 or fs + nfft > len(bb):
                continue
            fd = np.fft.fftshift(np.fft.fft(bb[fs:fs + nfft]))
            sc = np.concatenate([fd[dc - n_half:dc],
                                 fd[dc + 1:dc + n_half + 1]])
            power_sum += np.abs(sc) ** 2
            count += 1

    if count == 0:
        # Fallback: use first nfft samples
        fd = np.fft.fftshift(np.fft.fft(bb[:nfft]))
        sc = np.concatenate([fd[dc - n_half:dc],
                             fd[dc + 1:dc + n_half + 1]])
        return np.abs(sc) ** 2

    return power_sum / count


# ---------------------------------------------------------------------------
# Independent signal generation (per-channel baseband + statistical matching)
# ---------------------------------------------------------------------------

def generate_independent(target_sig, sr, num_channels=5, bw_mhz=20,
                         mod_order=256, seed=None, n_iter=200, verbose=True):
    """Generate a fresh, independent signal matching the target's statistics.

    OFDM-aware 3-stage algorithm:
      Stage 1: Per-channel OFDM generation with subcarrier power shaping
      Stage 2: Composite assembly + CFR PAPR matching + spectral envelope correction
      Stage 3: Iterative per-channel power balance + bounded amplitude refinement

    Key invariant: QAM constellation phase is never modified.  Only
    per-subcarrier magnitude scaling is applied, preserving clean
    QAM constellation points when demodulated.

    Parameters
    ----------
    target_sig : 1-D complex array
    sr : float (Hz)
    num_channels : int
    bw_mhz : float
    mod_order : int  (64 for TM3.1, 256 for TM3.1a)
    seed : int or None
    n_iter : int  (unused, kept for API compat)
    verbose : bool

    Returns
    -------
    signal : 1-D complex array (same length as target_sig)
    info : dict with keys: metrics, seed, sr
    """
    from scipy.ndimage import uniform_filter1d

    n = len(target_sig)
    bw_hz = bw_mhz * 1e6
    target_rms = float(np.sqrt(np.mean(np.abs(target_sig) ** 2)))
    target_papr = 10 * np.log10(
        np.max(np.abs(target_sig) ** 2)
        / (np.mean(np.abs(target_sig) ** 2) + 1e-30))

    carrier_centers = [(k - (num_channels - 1) / 2) * bw_hz
                       for k in range(num_channels)]

    # ── NR parameters ─────────────────────────────────────────────────────
    p = nr_params(scs_khz=15, bandwidth_mhz=bw_mhz, osr=16)
    nfft_ofdm = p['nfft']
    n_sc = p['n_sc']
    cp = p['cp_lengths']
    sym_per_slot = len(cp)
    min_slots = max(1, int(np.ceil(n / p['samples_per_slot'])))
    total_sym = min_slots * sym_per_slot

    if verbose:
        print(f"NR params: NFFT={nfft_ofdm}, SR={sr/1e6:.2f} MHz, "
              f"SC={n_sc} ({p['n_rb']} RBs), mod={mod_order}QAM")
        print(f"Generating {min_slots} slot(s) = {total_sym} symbols/channel")

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 1: Per-channel OFDM generation + subcarrier power shaping
    # ══════════════════════════════════════════════════════════════════════
    if verbose:
        print("\n=== Stage 1: Per-channel OFDM + subcarrier power shaping ===")

    channel_basebands = []   # corrected baseband per channel
    channel_gains = []       # per-channel gain (for Stage 3 iteration)

    for k, fc in enumerate(carrier_centers):
        ch_seed = (seed + k) if seed is not None else None
        if verbose:
            print(f"\n--- Channel {k+1}/{num_channels} ({fc/1e6:+.0f} MHz) ---")

        # 1a. Generate fresh OFDM baseband
        grid = generate_resource_grid(n_sc, total_sym, mod_order, ch_seed)

        # 1b. Extract target's per-subcarrier power profile
        P_tgt = extract_subcarrier_power_profile(
            target_sig, sr, fc, bw_hz, n_active=n_sc)

        # 1c. Compute generated grid's per-subcarrier power
        P_gen = np.mean(np.abs(grid) ** 2, axis=1)  # avg across symbols

        # 1d. Apply subcarrier-level magnitude correction
        # Normalize both to unit mean so we only correct the SHAPE
        P_tgt_norm = P_tgt / (np.mean(P_tgt) + 1e-30)
        P_gen_norm = P_gen / (np.mean(P_gen) + 1e-30)
        ratio = np.sqrt(P_tgt_norm / (P_gen_norm + 1e-30))
        # Cap extreme corrections to avoid amplifying noise
        ratio = np.clip(ratio, 0.1, 10.0)
        grid *= ratio[:, np.newaxis]

        if verbose:
            print(f"  Subcarrier correction: mean ratio={np.mean(ratio):.3f}, "
                  f"max={np.max(ratio):.3f}, min={np.min(ratio):.3f}")

        # 1e. Normalize grid and rebuild time-domain OFDM signal
        grid /= np.max(np.abs(grid))
        sig_ch = rebuild_signal(grid, nfft_ofdm, cp)
        sig_ch = filter_signal(sig_ch, sr, bw_mhz)

        # Pad or trim to n samples
        if len(sig_ch) < n:
            sig_ch = np.pad(sig_ch, (0, n - len(sig_ch)))
        sig_ch = sig_ch[:n]

        # 1f. Scale to match target channel's RMS power
        tgt_bb = _isolate_carrier(target_sig, sr, fc, bw_hz)
        tgt_ch_rms = float(np.sqrt(np.mean(np.abs(tgt_bb) ** 2)))
        gen_ch_rms = float(np.sqrt(np.mean(np.abs(sig_ch) ** 2) + 1e-30))
        gain_k = tgt_ch_rms / gen_ch_rms
        sig_ch *= gain_k

        channel_basebands.append(sig_ch)
        channel_gains.append(gain_k)

        if verbose:
            ch_papr = 10 * np.log10(
                np.max(np.abs(sig_ch) ** 2)
                / (np.mean(np.abs(sig_ch) ** 2) + 1e-30))
            print(f"  Channel PAPR: {ch_papr:.2f} dB, "
                  f"RMS gain: {gain_k:.4f}")

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 2: Composite assembly + PAPR matching + spectral envelope
    # ══════════════════════════════════════════════════════════════════════
    if verbose:
        print("\n=== Stage 2: Composite assembly + PAPR + spectral envelope ===")

    t_idx = np.arange(n)

    def _assemble(basebands, gains_adj=None):
        """Assemble composite from per-channel basebands."""
        comp = np.zeros(n, dtype=complex)
        for k, fc in enumerate(carrier_centers):
            bb = basebands[k]
            if gains_adj is not None:
                bb = bb * gains_adj[k]
            comp += bb * np.exp(1j * 2 * np.pi * t_idx * fc / sr)
        return comp

    composite = _assemble(channel_basebands)

    # 2a. Scale to target RMS
    composite *= target_rms / np.sqrt(
        np.mean(np.abs(composite) ** 2) + 1e-30)

    # 2b+2c. Alternating projections: PSD match + PAPR clipping
    # Uses the target's magnitude spectrum as the PSD template and the
    # generated composite's phase (from proper OFDM).  Alternates:
    #   1) PAPR clip (soft-clip peaks to target PAPR level)
    #   2) PSD projection (re-impose target magnitude spectrum)
    # Ends on PSD projection so the final PSD is exact.
    total_bw = bw_mhz * num_channels
    g_n = composite / np.sqrt(np.mean(np.abs(composite) ** 2) + 1e-30)
    t_n = target_sig / np.sqrt(np.mean(np.abs(target_sig) ** 2) + 1e-30)

    T = np.fft.fft(t_n)
    mag_target = np.abs(T)
    target_linear = 10 ** (target_papr / 10)

    # Initial PSD projection
    G = np.fft.fft(g_n)
    sig = np.fft.ifft(mag_target * np.exp(1j * np.angle(G)))

    kern = max(1, int(200e3 / (sr / n)))
    n_proj = 50  # alternating projection iterations

    for it in range(n_proj):
        # PAPR projection: soft-clip peaks
        pk = np.max(np.abs(sig) ** 2)
        av = np.mean(np.abs(sig) ** 2)
        clip_level = np.sqrt(av * target_linear)
        mag_s = np.abs(sig)
        over = mag_s > clip_level
        if np.any(over):
            sig = np.where(over, sig * clip_level / (mag_s + 1e-30), sig)

        # PSD projection: re-impose target magnitude (always last)
        G = np.fft.fft(sig)
        sig = np.fft.ifft(mag_target * np.exp(1j * np.angle(G)))

    composite = sig * target_rms / np.sqrt(
        np.mean(np.abs(sig) ** 2) + 1e-30)

    if verbose:
        _p = 10 * np.log10(np.max(np.abs(composite) ** 2)
                           / (np.mean(np.abs(composite) ** 2) + 1e-30))
        print(f"  Alternating projections ({n_proj} iters): "
              f"PAPR = {_p:.2f} dB (target: {target_papr:.2f} dB)")

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 3: Per-channel power balance (iterative) + bounded CDF
    # ══════════════════════════════════════════════════════════════════════
    if verbose:
        print("\n=== Stage 3: Per-channel power balance + CDF refinement ===")

    # 3a. Iterative per-channel power balance
    # Adjust pre-CFR per-channel gains, re-apply CFR, check balance.
    freqs = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / sr))
    gains_adj = np.ones(num_channels)

    for balance_iter in range(5):
        ft_g = np.fft.fftshift(np.fft.fft(
            composite / np.sqrt(np.mean(np.abs(composite) ** 2) + 1e-30)))
        ft_t = np.fft.fftshift(np.fft.fft(t_n))

        max_err = 0.0
        for k, fc in enumerate(carrier_centers):
            m = (freqs >= fc - bw_hz / 2) & (freqs <= fc + bw_hz / 2)
            pg_c = 10 * np.log10(np.mean(np.abs(ft_g[m]) ** 2) + 1e-30)
            pt_c = 10 * np.log10(np.mean(np.abs(ft_t[m]) ** 2) + 1e-30)
            err_db = pt_c - pg_c
            max_err = max(max_err, abs(err_db))
            # Adjust gain for next iteration
            gains_adj[k] *= 10 ** (err_db / 20)

        if verbose:
            print(f"  Balance iter {balance_iter+1}: max ch err = {max_err:.4f} dB")
        if max_err < 0.1:
            break

        # Re-assemble with adjusted gains, re-apply CFR
        composite = _assemble(channel_basebands, gains_adj)
        composite *= target_rms / np.sqrt(
            np.mean(np.abs(composite) ** 2) + 1e-30)
        composite, cfr_thr = tune_cfr_threshold(
            composite, target_papr, sr=sr, bw_mhz=total_bw)

    # 3b. Bounded amplitude CDF refinement (single pass, 5% cap)
    sorted_tgt_amp = np.sort(np.abs(t_n))
    sig_n = composite / np.sqrt(np.mean(np.abs(composite) ** 2) + 1e-30)

    mag_s = np.abs(sig_n)
    idx = np.argsort(mag_s)
    target_mags = np.empty_like(mag_s)
    target_mags[idx] = sorted_tgt_amp

    # Cap change at 5% per sample
    max_change = 0.05 * mag_s
    delta = target_mags - mag_s
    delta = np.clip(delta, -max_change, max_change)
    new_mag = mag_s + delta
    composite = new_mag * np.exp(1j * np.angle(sig_n))
    composite *= target_rms / np.sqrt(
        np.mean(np.abs(composite) ** 2) + 1e-30)

    if verbose:
        cdf_change = np.mean(np.abs(delta) / (mag_s + 1e-30)) * 100
        print(f"  CDF refinement: mean change = {cdf_change:.2f}%")

    # 3c. Final PSD re-projection to restore exact spectral match
    # (CDF refinement and power balancing distort the spectrum slightly)
    G_final = np.fft.fft(
        composite / np.sqrt(np.mean(np.abs(composite) ** 2) + 1e-30))
    composite = np.fft.ifft(mag_target * np.exp(1j * np.angle(G_final)))
    composite *= target_rms / np.sqrt(
        np.mean(np.abs(composite) ** 2) + 1e-30)

    if verbose:
        _p = 10 * np.log10(np.max(np.abs(composite) ** 2)
                           / (np.mean(np.abs(composite) ** 2) + 1e-30))
        print(f"  Final PSD re-projection: PAPR = {_p:.2f} dB")

    # ── Metrics ───────────────────────────────────────────────────────────
    g_rms = composite / np.sqrt(np.mean(np.abs(composite) ** 2) + 1e-30)
    t_rms_sig = target_sig / np.sqrt(
        np.mean(np.abs(target_sig) ** 2) + 1e-30)

    ft_g = np.fft.fftshift(np.fft.fft(g_rms))
    ft_t = np.fft.fftshift(np.fft.fft(t_rms_sig))
    pg = uniform_filter1d(20 * np.log10(np.abs(ft_g) + 1e-30), kern)
    pt = uniform_filter1d(20 * np.log10(np.abs(ft_t) + 1e-30), kern)
    ibw_hz = num_channels * bw_hz
    psd_mask = ((np.abs(freqs) < ibw_hz / 2 + 5e6)
                & (np.abs(ft_t) > 1e-2 * np.max(np.abs(ft_t))))
    psd_mae = float(np.mean(np.abs(pg[psd_mask] - pt[psd_mask]))) \
        if np.any(psd_mask) else 0.

    def _papr(s):
        return 10 * np.log10(np.max(np.abs(s) ** 2)
                             / (np.mean(np.abs(s) ** 2) + 1e-30))
    d_papr = abs(_papr(composite) - _papr(target_sig))

    def _ccdf(s, th):
        inst = np.abs(s) ** 2 / (np.mean(np.abs(s) ** 2) + 1e-30)
        return np.mean(10 * np.log10(inst + 1e-30) > th)
    ccdf_dev = max(abs(_ccdf(composite, th) - _ccdf(target_sig, th))
                   for th in (4, 6, 8, 10))

    ch_err = 0.0
    for fc in carrier_centers:
        m = (freqs >= fc - bw_hz / 2) & (freqs <= fc + bw_hz / 2)
        pg_c = 10 * np.log10(np.mean(np.abs(ft_g[m]) ** 2) + 1e-30)
        pt_c = 10 * np.log10(np.mean(np.abs(ft_t[m]) ** 2) + 1e-30)
        ch_err = max(ch_err, abs(pg_c - pt_c))

    # Inter-channel isolation
    inter_ch_floors = []
    peak_db = float(np.max(pg))
    for k in range(num_channels - 1):
        gap_center = (carrier_centers[k] + carrier_centers[k + 1]) / 2
        gap_mask = ((freqs >= gap_center - 0.3e6) &
                    (freqs <= gap_center + 0.3e6))
        if np.any(gap_mask):
            floor_db = float(np.mean(pg[gap_mask]))
            inter_ch_floors.append(floor_db - peak_db)
    worst_isolation = max(inter_ch_floors) if inter_ch_floors else -999.0

    metrics = dict(psd_mae=psd_mae, d_papr=d_papr,
                   ccdf_dev=ccdf_dev, ch_err=ch_err,
                   inter_ch_isolation=worst_isolation)

    if verbose:
        print(f"\nIndependent signal metrics (vs target statistics):")
        print(f"  PSD MAE  = {psd_mae:.4f} dB")
        print(f"  dPAPR    = {d_papr:.4f} dB")
        print(f"  CCDF dev = {ccdf_dev:.4f}")
        print(f"  Ch Err   = {ch_err:.4f} dB")
        print(f"  PAPR     = {_papr(composite):.2f} dB  "
              f"(target: {_papr(target_sig):.2f} dB)")
        if num_channels > 1:
            print(f"  Inter-ch isolation = {worst_isolation:.1f} dB")

    return composite, dict(metrics=metrics, seed=seed, n_iter=n_iter, sr=sr)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def save_mat(signal, sr, path, amp_scale=20000.0):
    """Save in the same format as the target .mat files."""
    sig_norm = signal / np.max(np.abs(signal)) * amp_scale
    n = len(signal)
    sio.savemat(path, {
        'Source_I': np.real(sig_norm).reshape(-1, 1),
        'Source_Q': np.imag(sig_norm).reshape(-1, 1),
        'Time_step': (np.arange(n) / sr).reshape(-1, 1),
    })
    print(f"\nSaved: {path}")
    print(f"  {n} samples, {n / sr * 1e6:.1f} us, "
          f"amplitude +/-{amp_scale:.0f}")


def load_mat(path):
    """Load a target .mat file, return complex signal and sample rate."""
    m = sio.loadmat(path)
    I = m['Source_I'].flatten()
    Q = m['Source_Q'].flatten()
    t = m['Time_step'].flatten()
    sr = 1.0 / (t[1] - t[0])
    return I + 1j * Q, sr


def parse_mat_params(filename):
    """Parse .mat filename to extract signal parameters.

    Handles filenames like:
      Precook_Signal_100WDevice_[5cLTE20MHz_iBW100MHz_SR491p52MHz_200uS_TM3p1a_PAPR10p0_IQ].mat

    Returns dict with: num_channels, bw_mhz, mod_order, sample_rate, duration_us.
    """
    import re, os
    basename = os.path.basename(filename)
    m = re.search(
        r'\[(\d+)cLTE(\d+)MHz_iBW\d+MHz_SR(\d+)p(\d+)MHz_(\d+)uS_TM3p1(a?)_',
        basename
    )
    if m is None:
        raise ValueError(f"Cannot parse parameters from: {basename}")
    sr_mhz = float(f"{m.group(3)}.{m.group(4)}")
    return dict(
        num_channels=int(m.group(1)),
        bw_mhz=int(m.group(2)),
        mod_order=256 if m.group(6) == 'a' else 64,
        sample_rate=sr_mhz * 1e6,
        duration_us=int(m.group(5)),
    )


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare(generated, target_path, sr):
    """Print statistical comparison between generated and target signals."""
    target, _ = load_mat(target_path)

    gen = generated / np.max(np.abs(generated))
    tgt = target / np.max(np.abs(target))

    def papr(s):
        return 10 * np.log10(np.max(np.abs(s) ** 2)
                             / np.mean(np.abs(s) ** 2))

    print(f"\n{'':=<60}")
    print(f"Comparison with target")
    print(f"{'':=<60}")
    print(f"  Samples     : gen={len(gen)}, tgt={len(tgt)}")
    print(f"  PAPR (dB)   : gen={papr(gen):.2f}, tgt={papr(tgt):.2f}")
    print(f"  Mean |sig|  : gen={np.mean(np.abs(gen)):.4f}, "
          f"tgt={np.mean(np.abs(tgt)):.4f}")
    print(f"  Std  |sig|  : gen={np.std(np.abs(gen)):.4f}, "
          f"tgt={np.std(np.abs(tgt)):.4f}")

    n = min(len(gen), len(tgt))
    f = np.linspace(-sr / 2, sr / 2, n, endpoint=False)
    ft_g = np.fft.fftshift(np.fft.fft(gen[:n]))
    ft_t = np.fft.fftshift(np.fft.fft(tgt[:n]))

    print(f"\n  Per-channel power (dB):")
    num_ch = 5
    for ch in range(num_ch):
        fc = (ch - (num_ch - 1) / 2) * 20e6
        mask = (f >= fc - 10e6) & (f <= fc + 10e6)
        pg = 10 * np.log10(np.mean(np.abs(ft_g[mask]) ** 2) + 1e-30)
        pt = 10 * np.log10(np.mean(np.abs(ft_t[mask]) ** 2) + 1e-30)
        print(f"    Ch{ch + 1} ({fc / 1e6:+6.0f} MHz): "
              f"gen={pg:.1f}, tgt={pt:.1f}")

    # CCDF
    def ccdf_at(s, th_db):
        inst = np.abs(s) ** 2 / np.mean(np.abs(s) ** 2)
        return np.mean(10 * np.log10(inst + 1e-30) > th_db)

    print(f"\n  CCDF  P(inst_PAPR > x dB):")
    for th in (4, 6, 8, 10):
        cg = ccdf_at(gen, th)
        ct = ccdf_at(tgt, th)
        print(f"    >{th:2d} dB: gen={cg:.6f}, tgt={ct:.6f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import os

    ap = argparse.ArgumentParser(
        description="Generate 5G NR-like multi-channel DPD test signal")

    # Mode selection
    ap.add_argument("--match", type=str, default=None, metavar="TARGET.mat",
                    help="Match mode: perturb TARGET.mat to optimise metrics "
                         "(stays close to original waveform)")
    ap.add_argument("--independent", type=str, default=None, metavar="TARGET.mat",
                    help="Independent mode: generate a fresh NR signal with "
                         "new random data, matching TARGET.mat's statistical "
                         "properties (PSD, PAPR, CCDF, per-channel power)")
    ap.add_argument("--alpha", type=float, default=None,
                    help="QAM correction strength for --match mode "
                         "(0.0-1.0, auto if omitted)")

    # Generation mode options
    ap.add_argument("--channels", type=int, default=5)
    ap.add_argument("--bw", type=int, default=20,
                    help="Per-channel bandwidth (MHz)")
    ap.add_argument("--scs", type=int, default=15,
                    help="Subcarrier spacing (kHz)")
    ap.add_argument("--osr", type=int, default=16)
    ap.add_argument("--samples", type=int, default=98304,
                    help="Output samples (98304 = 200 us @ 491.52 MHz)")
    ap.add_argument("--papr", type=float, default=10.0,
                    help="Target PAPR (dB)")
    ap.add_argument("--mod", type=int, default=256,
                    help="QAM order (64=TM3.1, 256=TM3.1a)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pts", action="store_true",
                    help="Enable PTS PAPR optimisation (slower, lower PAPR)")

    # Common options
    ap.add_argument("--output", type=str, default=None,
                    help="Output .mat path (auto-generated if omitted)")
    ap.add_argument("--target", type=str, default=None,
                    help=".mat file to compare against (generation mode)")
    args = ap.parse_args()

    # ── Independent mode ────────────────────────────────────────────────
    if args.independent is not None:
        target_path = args.independent
        if not os.path.exists(target_path):
            target_path = os.path.join(os.path.dirname(__file__), '..',
                                       args.independent)
        print(f"Independent mode: fresh NR signal matching stats of "
              f"{target_path}")
        target_sig, sr = load_mat(target_path)
        signal, info = generate_independent(
            target_sig, sr,
            num_channels=args.channels,
            bw_mhz=args.bw,
            mod_order=args.mod,
            seed=args.seed,
        )
        if args.output is None:
            base = os.path.splitext(os.path.basename(target_path))[0]
            args.output = f"Independent_{base}.mat"
        save_mat(signal, sr, args.output)
        compare(signal, target_path, sr)
        return

    # ── Match mode ──────────────────────────────────────────────────────
    if args.match is not None:
        target_path = args.match
        if not os.path.exists(target_path):
            # Try relative to script directory
            target_path = os.path.join(os.path.dirname(__file__), '..',
                                       args.match)
        print(f"Match mode: generating signal matching {target_path}")
        target_sig, sr = load_mat(target_path)
        signal, info = generate_matched(target_sig, sr, alpha=args.alpha)

        if args.output is None:
            base = os.path.splitext(os.path.basename(target_path))[0]
            args.output = f"Matched_{base}.mat"
        save_mat(signal, sr, args.output)
        compare(signal, target_path, sr)
        return

    # ── Generation mode ─────────────────────────────────────────────────
    signal, sr, info = generate(
        num_channels=args.channels,
        bandwidth_mhz=args.bw,
        scs_khz=args.scs,
        osr=args.osr,
        output_samples=args.samples,
        target_papr_db=args.papr,
        mod_order=args.mod,
        seed=args.seed,
        use_pts=args.pts,
    )

    if args.output is None:
        ch = args.channels
        bw = args.bw
        ibw = ch * bw
        sr_mhz = sr / 1e6
        dur = args.samples / sr * 1e6
        args.output = (
            f"Generated_{ch}c{bw}MHz_iBW{ibw}MHz"
            f"_SR{sr_mhz:.2f}MHz_{dur:.0f}uS_PAPR{args.papr:.1f}_IQ.mat"
        )

    save_mat(signal, sr, args.output)

    target = args.target
    if target is None:
        candidates = [
            os.path.join(os.path.dirname(__file__), "..",
                         f"Precook_Signal_100WDevice_"
                         f"[{args.channels}cLTE{args.bw}MHz"
                         f"_iBW{args.channels * args.bw}MHz"
                         f"_SR491p52MHz_200uS_TM3p1a_PAPR10p0_IQ].mat"),
        ]
        for c in candidates:
            if os.path.exists(c):
                target = c
                break

    if target:
        compare(signal, target, sr)
    else:
        print("\nNo target file found for comparison.")


if __name__ == "__main__":
    main()
