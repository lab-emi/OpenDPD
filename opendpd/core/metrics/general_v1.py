"""``general-spectral-v1``: pooled NMSE, in-band error and adjacent-channel power ratio.

Everything that changes a number is written into ``PROFILE.parameters`` and
checked by analytic references in ``tests/unit/test_metrics_general.py``:
zero error reaches the documented floor, a complex gain error gives exactly
``20*log10|g-1|``, white noise of known power gives its SNR, a tone of known
power in the adjacent channel gives its power ratio. None of this is a
demodulated EVM or a 3GPP conformance measurement.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from opendpd.schemas.common import BetterDirection, MetricStatus, MetricValue
from opendpd.schemas.dataset import SignalSpec
from opendpd.schemas.metrics import MetricDefinition, MetricProfile, ProfileValidation

PROFILE_ID = "general-spectral-v1"
FLOOR_DB = -300.0          # 10*log10 of a zero power ratio is reported as this floor, never as -inf
WINDOW = "hann"

PROFILE = MetricProfile(
    profile_id=PROFILE_ID,
    version=1,
    frozen=False,
    validation=ProfileValidation.analytic,
    description="General complex-baseband error and spectral-leakage metrics with explicit conventions: "
                "pooled NMSE, in-band error ratio and adjacent-channel power ratio (leakage, dBc).",
    parameters={
        "valid_range": "real samples of the evaluated split only; the zero padding of the last segment is excluded",
        "aggregation": "pooled over the whole valid range (sums of powers); never the mean of per-segment dB values",
        "psd": "scipy.signal.welch, two-sided, window=hann, nperseg=dataset nperseg, noverlap=nperseg//2, "
               "detrend off, scaling=density",
        "band_power": "sum over PSD bins whose centre frequency lies in [lo, hi) times the bin width fs/nperseg",
        "main_channel": "[-bandwidth_hz/2, +bandwidth_hz/2) around 0 Hz (complex baseband)",
        "adjacent_channels": "[+bandwidth_hz/2, +3*bandwidth_hz/2) (right) and [-3*bandwidth_hz/2, -bandwidth_hz/2) "
                             "(left): same width as the main channel, immediately adjacent",
        "capture_requirement": "a band must lie inside [-fs/2, +fs/2); otherwise the metric is not_applicable",
        "normalization": "none: prediction and reference are compared as given (target = gain * input for DPD)",
        "floor_db": FLOOR_DB,
        "units": "NMSE and IBE are 10*log10 of power ratios in dB; ACPR is dBc relative to the main-channel "
                 "power of the same signal. Ratio and percent forms are documented per metric, never mixed.",
    },
    metrics=[
        MetricDefinition(name="NMSE", display_name="NMSE (pooled)", unit="dB", better=BetterDirection.lower,
                         formula="10*log10( sum|y - r|^2 / sum|r|^2 ) over the valid range",
                         aggregation="pooled over all valid samples"),
        MetricDefinition(name="IBE", display_name="In-band error", unit="dB", better=BetterDirection.lower,
                         formula="10*log10( P_main(y - r) / P_main(r) ) with band powers from the Welch PSD",
                         aggregation="pooled PSD over the valid range",
                         notes="Spectral error ratio inside the main channel; not a demodulated EVM. "
                               "As an amplitude ratio it is 10^(IBE/20), as a percentage 100*10^(IBE/20)."),
        MetricDefinition(name="ACPR_L", display_name="ACPR left (leakage)", unit="dBc", better=BetterDirection.lower,
                         formula="10*log10( P_adjacent_left(y) / P_main(y) )",
                         aggregation="pooled PSD over the valid range", requires_reference=False,
                         notes="Leakage convention: adjacent-channel power relative to the main channel of the same "
                               "signal, negative dBc, lower is better. The positive 'suppression' convention "
                               "(main / adjacent, higher is better) equals -ACPR_L and is not reported."),
        MetricDefinition(name="ACPR_R", display_name="ACPR right (leakage)", unit="dBc", better=BetterDirection.lower,
                         formula="10*log10( P_adjacent_right(y) / P_main(y) )",
                         aggregation="pooled PSD over the valid range", requires_reference=False,
                         notes="Leakage convention: adjacent-channel power relative to the main channel of the same "
                               "signal, negative dBc, lower is better. The positive 'suppression' convention "
                               "(main / adjacent, higher is better) equals -ACPR_R and is not reported."),
    ],
)

_NEEDS_METADATA = ("sample_rate_hz", "bandwidth_hz", "nperseg")


def to_complex(iq: np.ndarray, valid_samples: Optional[int] = None) -> np.ndarray:
    arr = np.asarray(iq, dtype=np.float64)
    if arr.ndim == 3:
        arr = arr.reshape(-1, arr.shape[-1])
    z = arr[:, 0] + 1j * arr[:, 1]
    return z[:valid_samples] if valid_samples is not None else z


def db(ratio: float) -> float:
    """10*log10 with the documented floor instead of -inf."""
    return max(10.0 * math.log10(ratio), FLOOR_DB) if ratio > 0 else FLOOR_DB


def psd(z: np.ndarray, fs: float, nperseg: int) -> Tuple[np.ndarray, np.ndarray]:
    from scipy.signal import welch

    f, p = welch(z, fs=fs, window=WINDOW, nperseg=nperseg, noverlap=nperseg // 2, detrend=False,
                 return_onesided=False, scaling="density")
    order = np.argsort(f)
    return f[order], p[order]


def band_power(freq: np.ndarray, p: np.ndarray, lo: float, hi: float, fs: float, nperseg: int) -> Tuple[float, int]:
    mask = (freq >= lo) & (freq < hi)
    return float(p[mask].sum() * fs / nperseg), int(mask.sum())


def compute(prediction: np.ndarray, reference: Optional[np.ndarray], signal: SignalSpec, *,
            valid_samples: Optional[int] = None) -> List[MetricValue]:
    defs = {m.name: m for m in PROFILE.metrics}

    def value(name: str, v: Optional[float] = None, status: MetricStatus = MetricStatus.ok,
              reason: Optional[str] = None) -> MetricValue:
        d = defs[name]
        if status == MetricStatus.ok and (v is None or not math.isfinite(v)):
            return MetricValue(name=name, unit=d.unit, better=d.better, status=MetricStatus.invalid,
                               reason=f"non-finite value ({v!r})")
        return MetricValue(name=name, unit=d.unit, better=d.better, value=v, status=status, reason=reason)

    y = to_complex(prediction, valid_samples)
    r = to_complex(reference, valid_samples) if reference is not None else None
    out: Dict[str, MetricValue] = {}

    if y.size == 0:
        return [value(n, status=MetricStatus.not_applicable, reason="no samples in the valid range") for n in defs]
    finite_y = bool(np.isfinite(y).all())
    finite_r = r is None or bool(np.isfinite(r).all())

    # --- pooled NMSE -----------------------------------------------------------------
    if r is None:
        out["NMSE"] = value("NMSE", status=MetricStatus.missing_reference, reason="no reference signal")
    elif not (finite_y and finite_r):
        out["NMSE"] = value("NMSE", status=MetricStatus.invalid,
                            reason="non-finite samples in the prediction or reference")
    else:
        p_ref = float(np.sum(np.abs(r) ** 2))
        if p_ref == 0.0:
            out["NMSE"] = value("NMSE", status=MetricStatus.invalid, reason="reference has zero energy; the ratio is undefined")
        else:
            out["NMSE"] = value("NMSE", db(float(np.sum(np.abs(y - r) ** 2)) / p_ref))

    # --- spectral metrics: shared preconditions ---------------------------------------
    missing = [k for k in _NEEDS_METADATA if getattr(signal, k) is None]
    spectral: List[str] = ["IBE", "ACPR_L", "ACPR_R"]
    if missing:
        reason = f"dataset signal metadata missing: {', '.join(missing)}; spectral metrics undefined"
        for n in spectral:
            out[n] = value(n, status=MetricStatus.not_applicable, reason=reason)
        return [out[m.name] for m in PROFILE.metrics]
    fs, bw, nperseg = float(signal.sample_rate_hz), float(signal.bandwidth_hz), int(signal.nperseg)
    if y.size < nperseg:
        reason = f"{y.size} valid samples is fewer than one PSD segment of {nperseg}"
        for n in spectral:
            out[n] = value(n, status=MetricStatus.not_applicable, reason=reason)
        return [out[m.name] for m in PROFILE.metrics]
    if not finite_y:
        for n in spectral:
            out[n] = value(n, status=MetricStatus.invalid, reason="non-finite samples in the prediction")
        return [out[m.name] for m in PROFILE.metrics]

    def band_ok(lo: float, hi: float) -> Optional[str]:
        if lo < -fs / 2 or hi > fs / 2:
            return (f"band [{lo / 1e6:.1f}, {hi / 1e6:.1f}] MHz exceeds the captured range "
                    f"[{-fs / 2e6:.1f}, {fs / 2e6:.1f}] MHz")
        return None

    main = (-bw / 2, bw / 2)
    f_y, p_y = psd(y, fs, nperseg)
    p_main_y, bins_main = band_power(f_y, p_y, *main, fs, nperseg)

    # --- in-band error -----------------------------------------------------------------
    if r is None:
        out["IBE"] = value("IBE", status=MetricStatus.missing_reference, reason="no reference signal")
    elif not finite_r:
        out["IBE"] = value("IBE", status=MetricStatus.invalid, reason="non-finite samples in the reference")
    elif band_ok(*main):
        out["IBE"] = value("IBE", status=MetricStatus.not_applicable, reason="main channel " + band_ok(*main))
    elif bins_main == 0:
        out["IBE"] = value("IBE", status=MetricStatus.not_applicable, reason="no PSD bin lies inside the main channel")
    else:
        f_r, p_r = psd(r, fs, nperseg)
        p_main_r, _ = band_power(f_r, p_r, *main, fs, nperseg)
        if p_main_r == 0.0:
            out["IBE"] = value("IBE", status=MetricStatus.invalid, reason="reference has no in-band power")
        else:
            f_e, p_e = psd(y - r, fs, nperseg)
            p_main_e, _ = band_power(f_e, p_e, *main, fs, nperseg)
            out["IBE"] = value("IBE", db(p_main_e / p_main_r))

    # --- adjacent-channel power ratio (leakage) ---------------------------------------
    for name, (lo, hi) in (("ACPR_L", (-3 * bw / 2, -bw / 2)), ("ACPR_R", (bw / 2, 3 * bw / 2))):
        problem = band_ok(lo, hi) or band_ok(*main)
        if problem:
            out[name] = value(name, status=MetricStatus.not_applicable, reason="adjacent channel " + problem)
            continue
        p_adj, bins_adj = band_power(f_y, p_y, lo, hi, fs, nperseg)
        if bins_main == 0 or bins_adj == 0:
            out[name] = value(name, status=MetricStatus.not_applicable, reason="no PSD bin lies inside the band")
        elif p_main_y == 0.0:
            out[name] = value(name, status=MetricStatus.invalid, reason="evaluated signal has no in-band power")
        else:
            out[name] = value(name, db(p_adj / p_main_y))
    return [out[m.name] for m in PROFILE.metrics]
