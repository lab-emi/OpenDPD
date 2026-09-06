"""``legacy-opendpd-v1``: the historical OpenDPD metrics, frozen.

The numbers are produced by ``utils.metrics`` exactly as
``modules.train_funcs.calculate_metrics`` calls it; ``tests/golden`` pins them.
This module only adds the descriptor and explicit statuses around them.
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np

from opendpd.schemas.common import BetterDirection, MetricStatus, MetricValue
from opendpd.schemas.dataset import SignalSpec
from opendpd.schemas.metrics import MetricDefinition, MetricProfile

PROFILE_ID = "legacy-opendpd-v1"

PROFILE = MetricProfile(
    profile_id=PROFILE_ID,
    version=1,
    frozen=True,
    description="Historical OpenDPD metrics as implemented in utils/metrics.py "
                "(OpenDPDv1/v2 papers and benchmark_report.md).",
    parameters={
        "segmenting": "independent nperseg segments from the dataset spec; the last segment is zero-padded",
        "psd": "scipy.signal.welch, nperseg=nperseg, scaling=spectrum, two-sided, mean over segments",
        "main_channel": "[-bw_main_ch/2, +bw_main_ch/2] split into n_sub_ch sub-channels",
        "normalization": "none; signals compared as stored (target = gain * input for DPD)",
        "sample_range": "whole split, zero-padded last segment",
        "aggregation": "NMSE and EVM are means of per-segment values (dB and ratio respectively), not pooled",
    },
    metrics=[
        MetricDefinition(name="NMSE", display_name="NMSE (mean of segment dB)", unit="dB", better=BetterDirection.lower,
                         formula="mean over segments of 10*log10(sum|e|^2 / sum|y|^2)",
                         aggregation="mean of per-segment dB (not pooled)"),
        MetricDefinition(name="EVM", display_name="Spectral EVM (repo-specific)", unit="dB",
                         better=BetterDirection.lower,
                         formula="20*log10(mean over segments of mean over sub-channels of "
                                 "mean|X_pred - X_ref| / mean|X_ref|), FFT of nperseg samples",
                         aggregation="mean of sub-channel ratios, then dB",
                         notes="Not a demodulated constellation EVM. Do not compare with standard EVM."),
        MetricDefinition(name="ACLR_L", display_name="ACLR left", unit="dBc", better=BetterDirection.lower,
                         formula="10*log10(P_adjacent_left / max sub-channel power)",
                         aggregation="Welch PSD averaged over segments", requires_reference=False,
                         notes="Leakage convention (adjacent / strongest main sub-channel): negative dBc, lower is better."),
        MetricDefinition(name="ACLR_R", display_name="ACLR right", unit="dBc", better=BetterDirection.lower,
                         formula="10*log10(P_adjacent_right / max sub-channel power)",
                         aggregation="Welch PSD averaged over segments", requires_reference=False,
                         notes="Leakage convention (adjacent / strongest main sub-channel): negative dBc, lower is better."),
        MetricDefinition(name="ACLR_AVG", display_name="ACLR average", unit="dBc",
                         better=BetterDirection.lower, formula="(ACLR_L + ACLR_R) / 2",
                         aggregation="arithmetic mean of dB values", requires_reference=False),
    ],
)

_NEEDS_METADATA = ("sample_rate_hz", "bandwidth_hz", "n_sub_ch", "nperseg")


def _segments(iq: np.ndarray, nperseg: Optional[int]) -> np.ndarray:
    arr = np.asarray(iq, dtype=np.float64)
    if arr.ndim == 3:
        return arr
    if nperseg is None:
        return arr[None, ...]
    n = int(math.ceil(len(arr) / nperseg))
    padded = np.zeros((n * nperseg, 2), dtype=np.float64)
    padded[:len(arr)] = arr
    return padded.reshape(n, nperseg, 2)


def compute(prediction: np.ndarray, reference: Optional[np.ndarray], signal: SignalSpec) -> List[MetricValue]:
    from utils import metrics as legacy

    defs = {m.name: m for m in PROFILE.metrics}

    def value(name: str, v: Optional[float] = None, status: MetricStatus = MetricStatus.ok,
              reason: Optional[str] = None) -> MetricValue:
        d = defs[name]
        if status == MetricStatus.ok and (v is None or not math.isfinite(v)):
            return MetricValue(name=name, unit=d.unit, better=d.better, status=MetricStatus.invalid,
                               reason=f"non-finite value ({v!r})")
        return MetricValue(name=name, unit=d.unit, better=d.better, value=v, status=status, reason=reason)

    missing = [k for k in _NEEDS_METADATA if getattr(signal, k) is None]
    pred = _segments(prediction, signal.nperseg)
    ref = _segments(reference, signal.nperseg) if reference is not None else None
    out: List[MetricValue] = []
    finite_pred = bool(np.isfinite(pred).all())
    finite_ref = ref is None or bool(np.isfinite(ref).all())

    if ref is None:
        out.append(value("NMSE", status=MetricStatus.missing_reference, reason="no reference signal"))
    elif not (finite_pred and finite_ref):
        out.append(value("NMSE", status=MetricStatus.invalid, reason="non-finite samples in the prediction or reference"))
    else:
        try:
            out.append(value("NMSE", float(legacy.NMSE(pred, ref))))
        except Exception as err:  # noqa: BLE001 - reported, never swallowed into a fake number
            out.append(value("NMSE", status=MetricStatus.failed, reason=f"{type(err).__name__}: {err}"))

    spectral_block = None
    if missing:
        spectral_block = (MetricStatus.not_applicable,
                          f"dataset signal metadata missing: {', '.join(missing)}; spectral metrics undefined")
    elif not finite_pred:
        spectral_block = (MetricStatus.invalid, "non-finite samples in the prediction")

    if ref is None:
        out.append(value("EVM", status=MetricStatus.missing_reference, reason="no reference signal"))
    elif spectral_block:
        out.append(value("EVM", status=spectral_block[0], reason=spectral_block[1]))
    elif not finite_ref:
        out.append(value("EVM", status=MetricStatus.invalid, reason="non-finite samples in the reference"))
    else:
        try:
            out.append(value("EVM", float(legacy.EVM(pred, ref, sample_rate=int(signal.sample_rate_hz),
                                                    bw_main_ch=signal.bandwidth_hz, n_sub_ch=signal.n_sub_ch,
                                                    nperseg=signal.nperseg))))
        except Exception as err:  # noqa: BLE001
            out.append(value("EVM", status=MetricStatus.failed, reason=f"{type(err).__name__}: {err}"))

    if spectral_block:
        for name in ("ACLR_L", "ACLR_R", "ACLR_AVG"):
            out.append(value(name, status=spectral_block[0], reason=spectral_block[1]))
    else:
        try:
            left, right = legacy.ACLR(pred, fs=signal.sample_rate_hz, nperseg=signal.nperseg,
                                      bw_main_ch=signal.bandwidth_hz, n_sub_ch=signal.n_sub_ch)
            out.append(value("ACLR_L", float(left)))
            out.append(value("ACLR_R", float(right)))
            out.append(value("ACLR_AVG", float((left + right) / 2)))
        except Exception as err:  # noqa: BLE001
            for name in ("ACLR_L", "ACLR_R", "ACLR_AVG"):
                out.append(value(name, status=MetricStatus.failed, reason=f"{type(err).__name__}: {err}"))
    return out
