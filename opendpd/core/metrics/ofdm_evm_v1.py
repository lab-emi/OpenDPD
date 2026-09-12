"""``ofdm-lte20-evm-v1``: data-aided EVM and E-UTRA-style ACLR of a signal captured from the ``ofdm-lte20-v1`` waveform.

Applies only to datasets bound to the reference waveform (``SignalSpec.waveform``); everything else gets an
explicit status, never a number under this name. The procedure, its deviations from a standard's EVM
definition and the error budget are in ``docs/protocols/waveform-profiles.md``; until the cross-validation
recorded there has been run, ``PROFILE.validation`` stays ``pending_cross_validation`` and the GUI hides
the profile.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np

from opendpd.core.metrics.general_v1 import band_power, db, psd, to_complex
from opendpd.core.waveforms import demodulate, generate, to_baseband_rate
from opendpd.core.waveforms.ofdm import FS as WAVEFORM_FS, MIN_CORRELATION
from opendpd.schemas.common import BetterDirection, MetricStatus, MetricValue
from opendpd.schemas.dataset import SignalSpec
from opendpd.schemas.metrics import MetricDefinition, MetricProfile, ProfileValidation

PROFILE_ID = "ofdm-lte20-evm-v1"
MAIN_HALF_BW_HZ = 9.0e6          # 18 MHz transmission bandwidth configuration of a 20 MHz E-UTRA carrier
ADJACENT_OFFSET_HZ = 20.0e6      # first adjacent channel centre
MEASUREMENT_BW_HZ = 18.0e6       # rectangular measurement bandwidth of the adjacent channel
MIN_FS_ACLR_HZ = 2 * (ADJACENT_OFFSET_HZ + MEASUREMENT_BW_HZ / 2)   # 58 MS/s: the adjacent channel must be captured

PROFILE = MetricProfile(
    profile_id=PROFILE_ID,
    version=1,
    frozen=False,
    validation=ProfileValidation.pending_cross_validation,
    description="Data-aided RMS EVM and adjacent-channel leakage ratio of a capture bound to the ofdm-lte20-v1 "
                "reference waveform (LTE 20 MHz numerology, known 64QAM symbols). Not a conformance measurement.",
    parameters={
        "waveform": "ofdm-lte20-v1: CP-OFDM, 15 kHz spacing, 2048-point FFT at 30.72 MS/s, normal cyclic prefix, "
                    "1200 occupied subcarriers, 64QAM with known symbols (3GPP TS 36.211 §6.12 numerology; version fixed "
                    "at cross-validation)",
        "applicability": "only datasets bound to the waveform (signal.waveform); other inputs are missing_reference / "
                         "not_applicable, never re-scored under another definition",
        "rate_conversion": "polyphase resampling to 30.72 MS/s with the exact rational ratio (denominator <= 10000); "
                           "capture rates below 30.72 MS/s are not applicable",
        "timing": "circular cross-correlation with the regenerated waveform; integer offset; normalised peak >= 0.3 required",
        "frequency_offset": "coarse from the cyclic-prefix correlation (first 32 prefix samples skipped), fine from the "
                            "slope of the per-symbol residual phase of the known symbols; no per-symbol phase tracking",
        "equalizer": "one least-squares complex gain per subcarrier over all complete symbols in the capture (data aided); "
                     "bias factor sqrt(1 - 1/L) on noise-like error with L symbols",
        "evm": "RMS over every occupied subcarrier of every complete symbol at the nominal FFT position (end of the cyclic "
               "prefix); percent of the reference symbol power; also 20*log10 in dB",
        "aclr": "Welch PSD (Hann, dataset nperseg, 50 % overlap) at the capture rate; main channel [-9, +9] MHz, first "
                "adjacent channels centred at +/-20 MHz with an 18 MHz rectangular measurement bandwidth; "
                "10*log10(P_adjacent / P_main), leakage convention (negative dBc, lower is better); requires fs >= 58 MS/s "
                "(3GPP TS 36.104 §6.6.2 definition; version fixed at cross-validation)",
        "aggregation": "EVM: pooled over every occupied subcarrier of every complete OFDM symbol in the valid range; "
                       "ACLR: pooled Welch PSD over the valid range",
        "normalization": "none for ACLR (a ratio of the same signal); the data-aided equaliser normalises each "
                         "subcarrier's complex gain, so the absolute scale of the evaluated signal does not enter the EVM",
        "floors": "EVM < 1e-9 % without rate conversion, 0.037 % after conversion from 122.88 or 800 MS/s",
        "protocol": "docs/protocols/waveform-profiles.md",
    },
    metrics=[
        MetricDefinition(name="EVM_RMS", display_name="EVM (RMS)", unit="%", better=BetterDirection.lower,
                         formula="100 * sqrt( sum|S_eq - S_ref|^2 / sum|S_ref|^2 ) over occupied subcarriers and complete symbols",
                         aggregation="pooled over every complete OFDM symbol in the capture",
                         notes="Data-aided (known symbols); single FFT timing position; see the protocol for deviations "
                               "from TS 36.104 Annex E."),
        MetricDefinition(name="EVM_DB", display_name="EVM (dB)", unit="dB", better=BetterDirection.lower,
                         formula="20 * log10( EVM_RMS / 100 )", aggregation="same window as EVM_RMS"),
        MetricDefinition(name="ACLR_L", display_name="ACLR lower adjacent (leakage)", unit="dBc",
                         better=BetterDirection.lower, requires_reference=False,
                         formula="10 * log10( P[-29, -11 MHz] / P[-9, +9 MHz] ) from the Welch PSD at the capture rate",
                         aggregation="pooled PSD over the valid range",
                         notes="Leakage convention: adjacent over main, negative dBc; the positive suppression convention "
                               "equals -ACLR_L and is not reported."),
        MetricDefinition(name="ACLR_R", display_name="ACLR upper adjacent (leakage)", unit="dBc",
                         better=BetterDirection.lower, requires_reference=False,
                         formula="10 * log10( P[+11, +29 MHz] / P[-9, +9 MHz] ) from the Welch PSD at the capture rate",
                         aggregation="pooled PSD over the valid range",
                         notes="Leakage convention: adjacent over main, negative dBc; the positive suppression convention "
                               "equals -ACLR_R and is not reported."),
    ],
)

EVM_METRICS = ("EVM_RMS", "EVM_DB")
ACLR_METRICS = ("ACLR_L", "ACLR_R")


def compute(prediction: np.ndarray, reference: Optional[np.ndarray], signal: SignalSpec, *,
            valid_samples: Optional[int] = None) -> List[MetricValue]:
    """Score ``prediction`` (the evaluated signal). ``reference`` (the dataset's target signal) is not used: the
    reference of this profile is the bound waveform's known symbol grid."""
    defs = {m.name: m for m in PROFILE.metrics}

    def value(name: str, v: Optional[float] = None, status: MetricStatus = MetricStatus.ok,
              reason: Optional[str] = None) -> MetricValue:
        d = defs[name]
        if status == MetricStatus.ok and (v is None or not math.isfinite(v)):
            return MetricValue(name=name, unit=d.unit, better=d.better, status=MetricStatus.invalid,
                               reason=f"non-finite value ({v!r})")
        return MetricValue(name=name, unit=d.unit, better=d.better, value=v, status=status, reason=reason)

    def all_with(status: MetricStatus, reason: str) -> List[MetricValue]:
        return [value(m.name, status=status, reason=reason) for m in PROFILE.metrics]

    binding = signal.waveform
    if binding is None:
        return all_with(MetricStatus.missing_reference,
                        "dataset is not bound to a reference waveform (opendpd datasets import --waveform); the channel "
                        "layout and the reference symbols of this profile are undefined for it")
    if signal.sample_rate_hz is None:
        return all_with(MetricStatus.not_applicable, "dataset signal metadata missing: sample_rate_hz")
    fs = float(signal.sample_rate_hz)
    y = to_complex(prediction, valid_samples)
    if y.size == 0:
        return all_with(MetricStatus.not_applicable, "no samples in the valid range")
    if not np.isfinite(y).all():
        return all_with(MetricStatus.invalid, "non-finite samples in the evaluated signal")
    out: Dict[str, MetricValue] = {}

    # --- EVM (data aided) ------------------------------------------------------------------------------
    baseband = to_baseband_rate(y, fs)
    if baseband is None:
        reason = (f"capture rate {fs / 1e6:.6g} MS/s cannot be converted to the waveform clock "
                  f"({WAVEFORM_FS / 1e6:.2f} MS/s) with a small exact ratio, or is below it")
        for n in EVM_METRICS:
            out[n] = value(n, status=MetricStatus.not_applicable, reason=reason)
    else:
        demod = demodulate(baseband, generate(binding.spec))
        if demod is None:
            reason = (f"signal does not correlate with the bound waveform {binding.spec.waveform_id} seed "
                      f"{binding.spec.seed} (normalised peak below {MIN_CORRELATION}) or holds no complete OFDM symbol")
            for n in EVM_METRICS:
                out[n] = value(n, status=MetricStatus.missing_reference, reason=reason)
        else:
            out["EVM_RMS"] = value("EVM_RMS", demod.evm_rms_pct)
            out["EVM_DB"] = value("EVM_DB", demod.evm_db)

    # --- ACLR at the capture rate ---------------------------------------------------------------------
    if signal.nperseg is None:
        for n in ACLR_METRICS:
            out[n] = value(n, status=MetricStatus.not_applicable, reason="dataset signal metadata missing: nperseg")
    elif fs < MIN_FS_ACLR_HZ:
        reason = (f"capture rate {fs / 1e6:.6g} MS/s does not contain the first adjacent channel "
                  f"(needs at least {MIN_FS_ACLR_HZ / 1e6:.0f} MS/s)")
        for n in ACLR_METRICS:
            out[n] = value(n, status=MetricStatus.not_applicable, reason=reason)
    elif y.size < int(signal.nperseg):
        reason = f"{y.size} valid samples is fewer than one PSD segment of {signal.nperseg}"
        for n in ACLR_METRICS:
            out[n] = value(n, status=MetricStatus.not_applicable, reason=reason)
    else:
        nperseg = int(signal.nperseg)
        f, p = psd(y, fs, nperseg)
        p_main, bins_main = band_power(f, p, -MAIN_HALF_BW_HZ, MAIN_HALF_BW_HZ, fs, nperseg)
        half = MEASUREMENT_BW_HZ / 2
        for name, centre in (("ACLR_L", -ADJACENT_OFFSET_HZ), ("ACLR_R", ADJACENT_OFFSET_HZ)):
            p_adj, bins_adj = band_power(f, p, centre - half, centre + half, fs, nperseg)
            if bins_main == 0 or bins_adj == 0:
                out[name] = value(name, status=MetricStatus.not_applicable, reason="no PSD bin lies inside the band")
            elif p_main == 0.0:
                out[name] = value(name, status=MetricStatus.invalid, reason="evaluated signal has no in-band power")
            else:
                out[name] = value(name, db(p_adj / p_main))
    return [out[m.name] for m in PROFILE.metrics]
