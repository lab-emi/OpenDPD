"""Versioned LTE-numerology profiles with explicit, matched PSD configurations.

The original LTE/general/legacy profiles remain numerically unchanged. These
additional profiles implement docs/protocols/waveform-profiles-v2.md.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from opendpd.core.waveforms.ofdm import demodulate, generate, to_baseband_rate
from opendpd.schemas.common import BetterDirection, MetricStatus, MetricValue
from opendpd.schemas.dataset import SignalSpec
from opendpd.schemas.metrics import ProfileValidation
from . import general_v1, ofdm_evm_v1

PROFILE_ID = "ofdm-lte20-evm-v2"
AUTO_PROFILE_ID = "ofdm-lte20-evm-auto-v2"
PROTOCOL = "docs/protocols/waveform-profiles-v2.md"


def _profile(auto):
    profile = ofdm_evm_v1.PROFILE.model_copy(deep=True)
    profile.profile_id = AUTO_PROFILE_ID if auto else PROFILE_ID
    profile.version = 2
    profile.frozen = True
    profile.validation = ProfileValidation.cross_validated
    profile.description = ("Frequency-tolerant data-aided LTE-numerology EVM and "
                           + ("full-record Hamming" if auto else "dataset-segment Hann")
                           + " ACLR, with enclosing FFT-bin bands. Not a conformance measurement.")
    profile.parameters.update({
        "waveform": "ofdm-lte20-v1 known-symbol test waveform: LTE 20 MHz numerology, 15 kHz spacing, "
                    "2048 FFT at 30.72 MS/s, normal CP and 1200 occupied subcarriers; no conformance claim",
        "cross_validation": "docs/performance/matlab-parity-v2.md; registered digital signals against MATLAB R2026a",
        "timing": "complex circular correlation; below 0.3 use magnitude timing only as a candidate, "
                  "estimate CP coarse frequency offset, compensate, then require complex correlation >=0.3",
        "aclr": "10*log10(P_adjacent/P_main); main [-9,+9] MHz, adjacent [-29,-11] and [+11,+29] MHz; "
                "enclosing FFT bins inclusive, clipped at the captured Nyquist endpoints; fs >=58 MS/s",
        "band_power": "sum PSD from last bin <= lower edge through first bin >= upper edge, inclusive, times fs/nfft",
        "psd": ("full valid record, symmetric Hamming, no overlap, next-power-of-two nfft >=256, no detrend; "
                "dataset nperseg is not used" if auto else
                "Welch, symmetric Hann, dataset nperseg >=3, floor(nperseg/2) overlap, nfft=nperseg, no detrend"),
        "matlab_comparator": "comm.ACPR Auto" if auto else "comm.ACPR Specify window parameters (Hann, 50% overlap)",
        "protocol": PROTOCOL,
    })
    return profile


PROFILE = _profile(False)
AUTO_PROFILE = _profile(True)


def enclosing_band_power(freq, density, lo, hi, fs):
    """Integrate the enclosing range on the actual FFT grid, including endpoints."""
    if len(freq) < 2 or lo < -fs / 2 or hi > fs / 2 or lo >= hi:
        return 0.0, 0
    left = max(0, int(np.searchsorted(freq, lo, side="right")) - 1)
    right = min(len(freq) - 1, int(np.searchsorted(freq, hi, side="left")))
    if right < left:
        return 0.0, 0
    return float(np.sum(density[left:right + 1]) * fs / len(freq)), right - left + 1


def _psd(y, fs, size, auto):
    from scipy.signal import periodogram, welch

    if auto:
        nfft = 1 << (max(256, len(y)) - 1).bit_length()
        freq, density = periodogram(y, fs=fs, window=np.hamming(len(y)), nfft=nfft,
                                   detrend=False, return_onesided=False, scaling="density")
    else:
        freq, density = welch(y, fs=fs, window=np.hanning(size), nperseg=size,
                              noverlap=size // 2, nfft=size, detrend=False,
                              return_onesided=False, scaling="density")
    order = np.argsort(freq)
    return freq[order], density[order]


def compute(prediction: np.ndarray, reference: Optional[np.ndarray], signal: SignalSpec, *,
            valid_samples: Optional[int] = None, auto: bool = False) -> list[MetricValue]:
    profile = AUTO_PROFILE if auto else PROFILE
    out = {}

    def put(names, status, reason):
        for name in names:
            definition = profile.metric(name)
            out[name] = MetricValue(name=name, value=None, unit=definition.unit,
                                   better=BetterDirection.lower, status=status, reason=reason)

    names = [m.name for m in profile.metrics]
    if signal.waveform is None:
        put(names, MetricStatus.missing_reference, "dataset is not bound to the ofdm-lte20-v1 reference waveform")
        return [out[name] for name in names]
    if signal.sample_rate_hz is None:
        put(names, MetricStatus.not_applicable, "dataset signal metadata missing: sample_rate_hz")
        return [out[name] for name in names]
    y = general_v1.to_complex(prediction, valid_samples)
    fs = float(signal.sample_rate_hz)
    with np.errstate(over="ignore", invalid="ignore"):
        energy = float(np.vdot(y, y).real)
    if not np.isfinite(y).all() or not math.isfinite(energy):
        put(names, MetricStatus.invalid, "non-finite samples or overflowing signal energy")
        return [out[name] for name in names]
    baseband = to_baseband_rate(y, fs)
    if baseband is None:
        put(ofdm_evm_v1.EVM_METRICS, MetricStatus.not_applicable, "capture rate cannot be converted to 30.72 MS/s")
    else:
        result = demodulate(baseband, generate(signal.waveform.spec), frequency_tolerant=True)
        if result is None:
            put(ofdm_evm_v1.EVM_METRICS, MetricStatus.missing_reference,
                "no verified complex reference correlation >=0.3 or no complete OFDM symbol")
        else:
            for name, number in (("EVM_RMS", result.evm_rms_pct), ("EVM_DB", result.evm_db)):
                out[name] = MetricValue(name=name, value=number, unit=profile.metric(name).unit, better=BetterDirection.lower)
    reason = None
    size = len(y) if auto else signal.nperseg
    if fs < ofdm_evm_v1.MIN_FS_ACLR_HZ:
        reason = "capture rate does not contain the first adjacent channels (needs at least 58 MS/s)"
    elif size is None:
        reason = "dataset signal metadata missing: nperseg"
    elif size < (2 if auto else 3) or len(y) < size:
        reason = "signal is shorter than one valid PSD segment"
    if reason:
        put(ofdm_evm_v1.ACLR_METRICS, MetricStatus.not_applicable, reason)
    else:
        freq, density = _psd(y, fs, int(size), auto)
        main, bins = enclosing_band_power(freq, density, -9e6, 9e6, fs)
        if not bins or not math.isfinite(main) or main <= 0:
            put(ofdm_evm_v1.ACLR_METRICS, MetricStatus.invalid, "evaluated signal has no finite in-band power")
        else:
            for name, lo, hi in (("ACLR_L", -29e6, -11e6), ("ACLR_R", 11e6, 29e6)):
                power, count = enclosing_band_power(freq, density, lo, hi, fs)
                if not count or not math.isfinite(power):
                    put([name], MetricStatus.not_applicable, "PSD grid does not represent the complete adjacent band")
                else:
                    out[name] = MetricValue(name=name, value=general_v1.db(power / main), unit="dBc",
                                           better=BetterDirection.lower)
    return [out[name] for name in names]
