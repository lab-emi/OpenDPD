"""OFDM modulation of test-model grids (TS 38.211 5.3.1 without windowing) and carrier accumulation.

Conventions: row k of a K-row grid (k = 0 the lowest subcarrier) sits at signed FFT bin k - K/2, so the subcarrier
K/2 is exactly at 0 Hz and is used (nothing nulls the DC subcarrier). The cyclic prefix is the tail of the useful part
(no extra phase ramp), normal CP: 144 N/2048 samples, and 16 x 2^mu N/2048 more for the symbols l = 0 and l = 7 x 2^mu of
every subframe. All carriers of a signal share one sample rate and start at t = 0 on a subframe boundary.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np


def nfft_for_nrb(nrb: int, minimum: int = 128, max_occupancy: float = 0.85) -> int:
    """Smallest power of two with 12 N_RB / N <= 0.85 and N >= 128 (the FFT rule of MATLAB nrOFDMInfo; the occupancy
    is therefore not constant: 46 % ... 85 %)."""
    need = 12 * nrb / max_occupancy
    n = minimum
    while n < need:
        n *= 2
    return n


def cp_lengths(mu: int, n_fft: int, n_slots: int) -> np.ndarray:
    """Cyclic-prefix length in samples of every OFDM symbol of n_slots slots that start on a subframe boundary."""
    if n_fft % 128:
        raise ValueError("The normal cyclic prefix needs an FFT size that is a multiple of 128.")
    symbols = np.arange(14 * n_slots)
    in_subframe = symbols % (14 * 2 ** mu)
    cp = np.full(len(symbols), 9 * n_fft // 128, dtype=np.int64)                          # 144 N / 2048
    cp[(in_subframe == 0) | (in_subframe == 7 * 2 ** mu)] += (2 ** mu * n_fft) // 128      # + 16 x 2^mu N / 2048
    return cp


def symbol_starts(mu: int, n_fft: int, n_slots: int) -> tuple[np.ndarray, np.ndarray]:
    """(CP lengths, start sample of the useful part) of every OFDM symbol."""
    cp = cp_lengths(mu, n_fft, n_slots)
    ends = np.cumsum(cp + n_fft)
    return cp, ends - n_fft


def ofdm_modulate(grid: np.ndarray, mu: int, n_fft: int, gain: float = 1.0) -> np.ndarray:
    """K x L grid -> samples, x = gain x ifft(spectrum) per symbol with the cyclic prefix prepended."""
    k_total, n_symbols = grid.shape
    if k_total > n_fft:
        raise ValueError("The FFT is smaller than the number of subcarriers.")
    cp = cp_lengths(mu, n_fft, n_symbols // 14)
    bins = (np.arange(k_total) - k_total // 2) % n_fft
    spectrum = np.zeros((n_symbols, n_fft), dtype=np.complex128)
    spectrum[:, bins] = grid.T
    useful = np.fft.ifft(spectrum, axis=1)
    if gain != 1.0:
        useful *= gain
    out = np.empty(int(cp.sum()) + n_symbols * n_fft, dtype=np.complex128)
    pos = 0
    for symbol in range(n_symbols):
        c = int(cp[symbol])
        out[pos:pos + c] = useful[symbol, n_fft - c:]
        out[pos + c:pos + c + n_fft] = useful[symbol]
        pos += c + n_fft
    return out


def ofdm_demodulate(samples: np.ndarray, mu: int, n_fft: int, n_rows: int, n_slots: int, gain: float = 1.0) -> np.ndarray:
    """Inverse of ofdm_modulate: the FFT of every useful part, rows k -> bins k - n_rows/2, divided by `gain`."""
    cp, starts = symbol_starts(mu, n_fft, n_slots)
    windows = samples[starts[:, None] + np.arange(n_fft)[None, :]]
    spectrum = np.fft.fft(windows, axis=1)
    bins = (np.arange(n_rows) - n_rows // 2) % n_fft
    return spectrum[:, bins].T / gain


def phasor(shift_hz: Fraction, sample_rate_hz: int, n: int) -> np.ndarray:
    """exp(j 2 pi shift n / fs), n = 0..N-1, with the phase reduced exactly in integers before the exponential."""
    ratio = Fraction(shift_hz) / sample_rate_hz
    phase = (np.arange(n, dtype=np.int64) * ratio.numerator) % ratio.denominator
    return np.exp(2j * np.pi * (phase / ratio.denominator))
