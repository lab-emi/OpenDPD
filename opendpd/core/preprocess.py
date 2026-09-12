"""Traceable preprocessing ``preprocess-v1``: raw data is never modified.

Operations (applied in this order):
1. interpolate non-finite samples (linear, per I/Q component)
2. replace isolated outliers by interpolation (same robust rule as the doctor)
3. delay correction: shift the output earlier by ``delay_samples`` (fractional
   delays use an FFT phase ramp); the affected edge samples are trimmed from
   *both* signals so pairs stay aligned
4. complex gain/phase correction of the output: ``y / (10^(gain_db/20) e^{j phase})``
5. normalisation by the input peak measured on the *fit range* only (the
   training split), never on validation/test samples

The function returns the processed arrays and a record of what was fitted on
which sample range, so a version manifest can state it.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from opendpd.schemas import PreprocessingParams

PREPROCESS_VERSION = "preprocess-v1"
OUTLIER_SIGMAS = 25.0


def _interpolate(values: np.ndarray, bad: np.ndarray) -> np.ndarray:
    if not bad.any():
        return values
    out = values.copy()
    idx = np.arange(len(values))
    good = ~bad
    if good.sum() == 0:
        return np.zeros_like(values)
    for comp in range(values.shape[1]):
        out[bad, comp] = np.interp(idx[bad], idx[good], values[good, comp])
    return out


def _fractional_shift(y: np.ndarray, shift: float) -> np.ndarray:
    """Advance ``y`` by ``shift`` samples (positive = earlier) using an FFT phase ramp."""
    n = len(y)
    z = y[:, 0].astype(np.float64) + 1j * y[:, 1].astype(np.float64)
    freqs = np.fft.fftfreq(n)
    z = np.fft.ifft(np.fft.fft(z) * np.exp(2j * np.pi * freqs * shift))
    return np.stack([z.real, z.imag], -1)


def apply(x: np.ndarray, y: np.ndarray, params: PreprocessingParams,
          fit_range: Optional[Tuple[int, int]] = None) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """Return processed ``(x, y, record)``; ``record`` states fitted values and ranges."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 2 or x.shape[1] != 2:
        raise ValueError(f"x and y must both be (n, 2) arrays, got {x.shape} and {y.shape}")
    record: Dict[str, object] = {"code_version": PREPROCESS_VERSION, "steps": []}

    if params.interpolate_non_finite:
        bad_x, bad_y = ~np.isfinite(x).all(axis=1), ~np.isfinite(y).all(axis=1)
        x, y = _interpolate(x, bad_x), _interpolate(y, bad_y)
        record["steps"].append({"interpolate_non_finite": {"input": int(bad_x.sum()), "output": int(bad_y.sum())}})
    if params.remove_outliers:
        mag = np.hypot(y[:, 0], y[:, 1])
        med = float(np.median(mag))
        mad = float(np.median(np.abs(mag - med))) * 1.4826
        bad = mag > med + OUTLIER_SIGMAS * max(mad, 1e-12)
        y = _interpolate(y, bad)
        record["steps"].append({"remove_outliers": {"count": int(bad.sum()), "threshold": med + OUTLIER_SIGMAS * mad}})
    trimmed = 0
    if params.delay_samples:
        d = float(params.delay_samples)
        whole, frac = int(np.trunc(d)), d - np.trunc(d)
        if frac:
            y = _fractional_shift(y, frac)
        if whole > 0:        # output lags: drop its first `whole` samples, align input by trimming its tail
            y, x = y[whole:], x[:-whole]
        elif whole < 0:
            y, x = y[:whole], x[-whole:]
        trimmed = abs(whole) + (1 if frac else 0)
        if frac:             # the phase-ramp edges wrap around; drop one sample at each end
            x, y = x[1:-1], y[1:-1]
        record["steps"].append({"delay_correction": {"delay_samples": d, "edge_samples_trimmed": trimmed}})
    if params.gain_db or params.phase_deg:
        g = 10 ** (params.gain_db / 20) * np.exp(1j * np.radians(params.phase_deg))
        z = (y[:, 0] + 1j * y[:, 1]) / g
        y = np.stack([z.real, z.imag], -1)
        record["steps"].append({"gain_phase_correction": {"gain_db": params.gain_db, "phase_deg": params.phase_deg}})
    if params.normalize == "peak_input":
        if fit_range is None:
            raise ValueError("normalisation needs a fit_range (the training split); it must not see test samples")
        start, end = fit_range
        end = min(end, len(x))
        peak = float(np.hypot(x[start:end, 0], x[start:end, 1]).max())
        if peak <= 0:
            raise ValueError("input peak on the fit range is zero")
        x, y = x / peak, y / peak
        record["steps"].append({"normalize": {"method": "peak_input", "scale": peak, "fit_range": [int(start), int(end)]}})
        record["fit_range"] = [int(start), int(end)]
    record["n_samples"] = int(len(x))
    return x.astype(np.float32), y.astype(np.float32), record
