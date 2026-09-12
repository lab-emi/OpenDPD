"""Aligning a measured PA output to the signal that was played (plan S16).

Pure numpy/scipy: an integer-sample delay by circular cross-correlation with
the played signal, a complex least-squares gain of the aligned capture onto
the *target* input ``x`` (what a linear PA would have amplified), and level
statistics in capture units. Nothing here rescales a capture: the gain is
reported, and the samples are scored as captured.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Optional, Tuple

import numpy as np

MAX_RESAMPLE_DENOMINATOR = 10_000
MIN_CORRELATION = 0.5          # below this the capture is not the played signal (wrong file, no signal, clipping)


def to_complex(iq: np.ndarray) -> np.ndarray:
    iq = np.asarray(iq)
    if np.iscomplexobj(iq):
        return iq.astype(np.complex128).reshape(-1)
    iq = iq.reshape(-1, 2).astype(np.float64)
    return iq[:, 0] + 1j * iq[:, 1]


def to_iq(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.complex128).reshape(-1)
    return np.stack([z.real, z.imag], axis=1).astype(np.float32)


def rate_ratio(fs_from: float, fs_to: float) -> Optional[Tuple[int, int]]:
    """(up, down) with ``fs_from * up / down == fs_to`` exactly, or None when no small rational does."""
    if fs_from <= 0 or fs_to <= 0:
        return None
    frac = Fraction(fs_to / fs_from).limit_denominator(MAX_RESAMPLE_DENOMINATOR)
    if abs(float(frac) - fs_to / fs_from) > 1e-12:
        return None
    return frac.numerator, frac.denominator


def resample(z: np.ndarray, fs_from: float, fs_to: float) -> np.ndarray:
    """Polyphase rate conversion of a complex signal; raises ValueError when the ratio is not a small rational."""
    ratio = rate_ratio(fs_from, fs_to)
    if ratio is None:
        raise ValueError(f"cannot convert {fs_from:g} Hz to {fs_to:g} Hz with a rational ratio of denominator "
                         f"<= {MAX_RESAMPLE_DENOMINATOR}")
    up, down = ratio
    z = np.asarray(z, dtype=np.complex128).reshape(-1)
    if (up, down) == (1, 1):
        return z
    from scipy.signal import resample_poly

    return resample_poly(z, up, down)


def rms(z: np.ndarray) -> float:
    z = np.asarray(z)
    return float(np.sqrt(np.mean(np.abs(z) ** 2))) if z.size else 0.0


def peak_abs(z: np.ndarray) -> float:
    z = np.asarray(z)
    return float(np.max(np.abs(z))) if z.size else 0.0


def level_db(a: np.ndarray, b: np.ndarray) -> float:
    """20 log10(rms(a) / rms(b)); ±inf when one side is silent."""
    ra, rb = rms(a), rms(b)
    if ra == 0.0 or rb == 0.0:
        return float("-inf") if ra == 0.0 else float("inf")
    return float(20.0 * np.log10(ra / rb))


@dataclass(frozen=True)
class Alignment:
    delay_samples: int
    wrapped: bool
    correlation: float
    gain: complex


def align(capture: np.ndarray, played: np.ndarray, target: np.ndarray, *, loop: bool = True) -> Tuple[np.ndarray, Alignment]:
    """Cut the window of ``capture`` that corresponds to ``played`` and fit its gain onto ``target``.

    ``played`` is what left the generator (``u`` with DPD, ``x`` without); ``target`` is ``x`` in both cases,
    because a linearised PA should output a scaled ``x``. The delay comes from the peak of the circular
    cross-correlation over one period; with ``loop`` the window may wrap around the file (the file was
    repeated on air), otherwise the capture must hold the whole period after the delay. Raises ValueError
    when the capture is shorter than the played signal or does not correlate with it.
    """
    y = np.asarray(capture, dtype=np.complex128).reshape(-1)
    p = np.asarray(played, dtype=np.complex128).reshape(-1)
    t = np.asarray(target, dtype=np.complex128).reshape(-1)
    n = p.size
    if n == 0 or t.size != n:
        raise ValueError("played and target signals must be non-empty and of equal length")
    if y.size < n:
        raise ValueError(f"the capture holds {y.size} samples, the played signal {n}: capture at least one full period")
    if not (np.all(np.isfinite(y)) and np.all(np.isfinite(p))):
        raise ValueError("capture or played signal contains non-finite samples")
    corr = np.fft.ifft(np.fft.fft(y[:n]) * np.conj(np.fft.fft(p)))
    delay = int(np.argmax(np.abs(corr)))
    if y.size >= delay + n:
        window, wrapped = y[delay:delay + n], False
    elif loop:
        window, wrapped = np.roll(y[:n], -delay), True
    else:
        raise ValueError(f"the capture ends {delay + n - y.size} samples before the played period does (delay "
                         f"{delay}); capture longer, or declare looped playback")
    denom = float(np.linalg.norm(p) * np.linalg.norm(window))
    rho = float(abs(np.vdot(p, window)) / denom) if denom > 0 else 0.0
    if rho < MIN_CORRELATION:
        raise ValueError(f"the capture does not correlate with the played signal (correlation {rho:.3f} < "
                         f"{MIN_CORRELATION}); wrong file, no signal or a different waveform")
    energy = float(np.vdot(t, t).real)
    gain = complex(np.vdot(t, window) / energy) if energy > 0 else 0j
    return window, Alignment(delay_samples=delay, wrapped=wrapped, correlation=rho, gain=gain)
