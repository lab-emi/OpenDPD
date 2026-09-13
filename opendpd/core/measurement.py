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


FRACTIONAL_HALF_TAPS = 64


@dataclass(frozen=True)
class FractionalAlignment:
    integer: Alignment
    fractional_delay_samples: float
    valid_sample_range: Tuple[int, int]
    boundary_method: str
    diagnostics: list
    correlation: float
    gain: complex


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return min(1.0, float(abs(np.vdot(a, b)) / denom)) if denom else 0.0


def _diagnostic(stage: str, played: np.ndarray, target: np.ndarray, window: np.ndarray) -> dict:
    energy = float(np.vdot(target, target).real)
    gain = complex(np.vdot(target, window) / energy) if energy else 0j

    def nmse(g):
        reference = g * target
        power = float(np.vdot(reference, reference).real)
        error = window - reference
        ratio = float(np.vdot(error, error).real) / power if power else 0
        # Exact zero or an undefined reference is represented as unavailable, not JSON infinity.
        return float(10 * np.log10(ratio)) if ratio > 0 else None

    return dict(stage=stage, correlation=_correlation(played, window), gain_abs=abs(gain),
                gain_phase_deg=float(np.degrees(np.angle(gain))), magnitude_fit_nmse_db=nmse(abs(gain)),
                complex_fit_nmse_db=nmse(gain))


def align_fractional(capture: np.ndarray, played: np.ndarray, target: np.ndarray, *, loop: bool = True):
    """Version 2: retain the legacy integer search, then fit a residual in [-0.5, 0.5].

    Loop playback uses the existing preprocessing FFT phase ramp on exactly one
    declared period. Single-shot playback uses a 129-tap Kaiser-windowed sinc
    (beta=8.6) and trims 64 samples on each edge: none of the retained samples
    depends on circular wrapping or padding. This finite filter approximates an
    ideal delay; near-Nyquist signals need separate validation.

    Delay maximises correlation with the *played* waveform; the reported complex
    gain fits the target x. Captured amplitudes are never normalised. Raw, integer
    and fractional diagnostics use the identical retained target interval.
    """
    from scipy.optimize import minimize_scalar
    from opendpd.core.preprocess import _fractional_shift

    integer_window, coarse = align(capture, played, target, loop=loop)
    p, t = to_complex(played), to_complex(target)
    n = p.size
    edge = 0 if loop else FRACTIONAL_HALF_TAPS
    if n <= 2 * edge + 16:
        raise ValueError("fractional single-shot alignment needs more than 144 played samples for its valid interval")
    start, stop = edge, n - edge
    p_valid, t_valid = p[start:stop], t[start:stop]
    if loop:
        iq = np.column_stack([integer_window.real, integer_window.imag])
        spectrum = np.fft.fft(integer_window) * np.conj(np.fft.fft(p))
        freqs = np.fft.fftfreq(n)

        def objective(delay):
            return -float(abs(np.sum(spectrum * np.exp(2j * np.pi * freqs * delay))))

        def shift(delay):
            return to_complex(_fractional_shift(iq, delay))

        boundary = "periodic FFT phase ramp; full declared playback period; no edge trimming"
    else:
        indices = np.arange(-edge, edge + 1)
        taper = np.kaiser(2 * edge + 1, 8.6)

        def shift(delay):
            kernel = np.sinc(indices + delay) * taper
            kernel /= np.sum(kernel)
            return np.convolve(integer_window, kernel, mode="same")[start:stop]

        def objective(delay):
            return -_correlation(p_valid, shift(delay))

        boundary = "129-tap Kaiser sinc beta=8.6; 64 samples trimmed at each edge; no circular wrapping"
    fit = minimize_scalar(objective, bounds=(-.5, .5), method="bounded", options={"xatol": 1e-8})
    if not fit.success:
        raise ValueError("fractional delay fit did not converge")
    # Include the exact integer and endpoints; zero delay must not acquire numerical motion.
    candidates = [0., -.5, .5, float(fit.x)]
    delay = min(candidates, key=objective)
    window = shift(delay)
    rho = _correlation(p_valid, window)
    if rho < MIN_CORRELATION:
        raise ValueError(f"fractional aligned correlation {rho:.3f} is below {MIN_CORRELATION}")
    energy = float(np.vdot(t_valid, t_valid).real)
    gain = complex(np.vdot(t_valid, window) / energy) if energy else 0j
    diagnostics = [_diagnostic(stage, p_valid, t_valid, samples) for stage, samples in (
        ("raw_start", to_complex(capture)[start:stop]), ("integer_aligned", integer_window[start:stop]),
        ("fractional_aligned", window))]
    return window, FractionalAlignment(coarse, delay, (start, stop), boundary, diagnostics, rho, gain)
