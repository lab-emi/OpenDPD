"""``plots-v1``: derived plot data computed by the worker, never by the browser.

Spectrum, time-domain excerpt and AM-AM / AM-PM point clouds are computed from
the same arrays the metrics were scored on and written next to the result.
The PSD estimator is the one of ``general-spectral-v1`` (Welch, Hann,
``nperseg`` from the dataset, ``noverlap = nperseg // 2``, density scaling),
so what is drawn is what was measured. Arrays are decimated / subsampled to a
fixed budget so a page never receives a whole capture. Charts only display
these numbers; nothing here feeds back into any metric.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from opendpd.core.metrics.general_v1 import FLOOR_DB, psd, to_complex

PLOTS_VERSION = "plots-v1"
TIME_EXCERPT_SAMPLES = 1000
AM_POINTS = 4000
DB_DECIMALS = 3
AMP_DECIMALS = 6

TraceRole = str   # "input" | "reference" | "primary" | "baseline" | "predistorted"


def _round(values: np.ndarray, decimals: int) -> List[float]:
    return [float(v) for v in np.round(np.asarray(values, dtype=np.float64), decimals)]


def spectrum(signals: Dict[str, np.ndarray], roles: Dict[str, TraceRole], *, sample_rate_hz: Optional[float],
             nperseg: Optional[int], bandwidth_hz: Optional[float], valid_samples: Optional[int] = None) -> Dict:
    """Welch PSD (dB) of every signal on one frequency axis. Without a sample rate the axis is in
    cycles per sample and no channel bands are drawn; nothing is guessed."""
    z = {name: to_complex(arr, valid_samples) for name, arr in signals.items()}
    n = min(len(v) for v in z.values())
    fs = float(sample_rate_hz) if sample_rate_hz else 1.0
    seg = int(nperseg) if nperseg else min(1024, n)
    seg = max(2, min(seg, n))
    axis = "hz" if sample_rate_hz else "normalized"
    traces = []
    freq = None
    for name, arr in z.items():
        f, p = psd(arr[:n], fs, seg)
        freq = f
        with np.errstate(divide="ignore"):
            p_db = np.where(p > 0, 10.0 * np.log10(np.maximum(p, 1e-300)), FLOOR_DB)
        traces.append({"name": name, "role": roles.get(name, "primary"), "psd_db": _round(np.maximum(p_db, FLOOR_DB), DB_DECIMALS)})
    bands = None
    if sample_rate_hz and bandwidth_hz:
        bw = float(bandwidth_hz)
        bands = {"main": [-bw / 2, bw / 2], "adjacent": [[-3 * bw / 2, -bw / 2], [bw / 2, 3 * bw / 2]]}
    return {
        "version": PLOTS_VERSION, "kind": "spectrum", "axis": axis, "sample_rate_hz": sample_rate_hz, "nperseg": seg,
        "n_samples": int(n), "frequency": _round(freq, 3) if freq is not None else [], "traces": traces, "bands": bands,
        "estimator": "scipy.signal.welch, window=hann, noverlap=nperseg//2, detrend off, density scaling, "
                     "two-sided; dB = 10*log10(PSD); identical to general-spectral-v1",
    }


def time_excerpt(signals: Dict[str, np.ndarray], roles: Dict[str, TraceRole], *, start: int = 0,
                 n: int = TIME_EXCERPT_SAMPLES, valid_samples: Optional[int] = None) -> Dict:
    """A short window of I/Q samples, identical sample indices for every trace."""
    traces = []
    total = None
    for name, arr in signals.items():
        z = to_complex(arr, valid_samples)
        total = len(z) if total is None else min(total, len(z))
        window = z[start:start + n]
        traces.append({"name": name, "role": roles.get(name, "primary"),
                       "i": _round(window.real, AMP_DECIMALS), "q": _round(window.imag, AMP_DECIMALS)})
    return {"version": PLOTS_VERSION, "kind": "time", "start": int(start), "n": int(min(n, total or 0)),
            "n_samples": int(total or 0), "traces": traces}


def am_am_pm(x: np.ndarray, outputs: Dict[str, np.ndarray], roles: Dict[str, TraceRole], *,
             max_points: int = AM_POINTS, valid_samples: Optional[int] = None) -> Dict:
    """|out| and angle(out * conj(in)) against |in|, subsampled with a fixed stride over the valid range."""
    zx = to_complex(x, valid_samples)
    n = len(zx)
    stride = max(1, int(np.ceil(n / max_points)))
    idx = np.arange(0, n, stride)
    amp_in = np.abs(zx[idx])
    traces = []
    for name, arr in outputs.items():
        zy = to_complex(arr, valid_samples)[:n][idx]
        with np.errstate(divide="ignore", invalid="ignore"):
            phase = np.degrees(np.angle(zy * np.conj(zx[idx])))
        traces.append({"name": name, "role": roles.get(name, "primary"), "amp_out": _round(np.abs(zy), AMP_DECIMALS),
                       "phase_deg": _round(np.where(np.isfinite(phase), phase, 0.0), 3)})
    return {"version": PLOTS_VERSION, "kind": "am", "stride": int(stride), "n_points": int(len(idx)),
            "n_samples": int(n), "amp_in": _round(amp_in, AMP_DECIMALS), "traces": traces,
            "note": "AM-AM: |out| vs |in|; AM-PM: angle(out * conj(in)) in degrees; every trace uses the same "
                    "input samples (fixed stride over the valid range)"}
