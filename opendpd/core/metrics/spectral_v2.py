"""OpenDPD carrier ACLR on real samples, without the padded-tail spectral edge.

The historical profile remains frozen. V2 uses a Welch estimate over the
valid signal, explicit half-open frequency bands and a pooled error ratio.
Carrier ACLR retains the OpenDPD convention: adjacent carrier power relative
to the strongest in-band carrier, in negative dBc. This is not a standards
conformance measurement or the full-band ACPR of general-spectral-v1.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from opendpd.schemas.common import BetterDirection, MetricStatus, MetricValue
from opendpd.schemas.dataset import SignalSpec
from opendpd.schemas.metrics import MetricDefinition, MetricProfile, ProfileValidation
from . import general_v1

PROFILE_ID = "opendpd-spectral-v2"
PROFILE = MetricProfile(
    profile_id=PROFILE_ID, version=2, frozen=True, validation=ProfileValidation.analytic,
    description="OpenDPD carrier ACLR and pooled error on valid samples; excludes evaluation padding.",
    parameters={
        **general_v1.PROFILE.parameters,
        "main_channel": "[-B/2, B/2), divided into N = n_sub_ch equal carrier bands of width W = B/N",
        "adjacent_channels": "[-B/2-W, -B/2) and [B/2, B/2+W)",
        "aclr_reference": "strongest integrated in-band carrier power; adjacent/reference in negative dBc",
        "aclr_average": "arithmetic mean of left and right dBc, for historical OpenDPD checkpoint selection",
        "units": "NMSE and IBE are power-error ratios in dB. ACLR is adjacent-carrier / strongest in-band carrier power in negative dBc; lower is better.",
    },
    metrics=[
        *[m.model_copy(deep=True) for m in general_v1.PROFILE.metrics if m.name in ("NMSE", "IBE")],
        *[MetricDefinition(name=f"ACLR_{side}", display_name=f"Carrier ACLR {label}", unit="dBc",
              better=BetterDirection.lower, requires_reference=False,
              formula=f"10*log10(P_adjacent_{label} / max in-band carrier power)",
              aggregation="Welch PSD over valid samples",
              notes="Equal carrier bands of width bandwidth_hz / n_sub_ch. Negative leakage dBc; lower is better.")
          for side, label in (("L", "left"), ("R", "right"))],
        MetricDefinition(name="ACLR_AVG", display_name="Carrier ACLR average", unit="dBc",
            better=BetterDirection.lower, requires_reference=False, formula="(ACLR_L + ACLR_R) / 2",
            aggregation="arithmetic mean of dB values"),
    ],
)


def compute(prediction: np.ndarray, reference: Optional[np.ndarray], signal: SignalSpec, *,
            valid_samples: Optional[int] = None) -> list[MetricValue]:
    errors = [m for m in general_v1.compute(prediction, reference, signal, valid_samples=valid_samples)
              if m.name in ("NMSE", "IBE")]

    def value(name, v=None, status=MetricStatus.ok, reason=None):
        return MetricValue(name=name, value=v, unit="dBc", better=BetterDirection.lower,
                           status=status, reason=reason)

    def unavailable(status, reason):
        return errors + [value(name, status=status, reason=reason) for name in ("ACLR_L", "ACLR_R", "ACLR_AVG")]

    missing = [key for key in ("sample_rate_hz", "bandwidth_hz", "n_sub_ch", "nperseg")
               if getattr(signal, key) is None]
    if missing:
        return unavailable(MetricStatus.not_applicable, "dataset signal metadata missing: " + ", ".join(missing))
    y = general_v1.to_complex(prediction, valid_samples)
    fs, bw, n, size = float(signal.sample_rate_hz), float(signal.bandwidth_hz), int(signal.n_sub_ch), int(signal.nperseg)
    if len(y) < size:
        return unavailable(MetricStatus.not_applicable, f"{len(y)} valid samples is fewer than one PSD segment of {size}")
    if not np.isfinite(y).all():
        return unavailable(MetricStatus.invalid, "non-finite samples in the prediction")
    width = bw / n
    if bw / 2 + width > fs / 2:
        return unavailable(MetricStatus.not_applicable, "adjacent carrier bands exceed the captured Nyquist range")
    f, p = general_v1.psd(y, fs, size)
    bands = [(-bw / 2 + i * width, -bw / 2 + (i + 1) * width) for i in range(n)]
    bands += [(-bw / 2 - width, -bw / 2), (bw / 2, bw / 2 + width)]
    powers = [general_v1.band_power(f, p, lo, hi, fs, size) for lo, hi in bands]
    if any(count == 0 for _, count in powers):
        return unavailable(MetricStatus.not_applicable, "no PSD bin lies inside at least one carrier band")
    reference_power = max(power for power, _ in powers[:n])
    if not math.isfinite(reference_power) or reference_power <= 0:
        return unavailable(MetricStatus.invalid, "evaluated signal has no finite in-band carrier power")
    left, right = (general_v1.db(power / reference_power) for power, _ in powers[-2:])
    return errors + [value("ACLR_L", left), value("ACLR_R", right), value("ACLR_AVG", (left + right) / 2)]
