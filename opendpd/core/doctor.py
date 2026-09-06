"""Dataset Doctor ``dataset-doctor-v1``: evidence-based checks on an I/Q capture.

Every finding carries the numbers behind it and, when a fix exists, a
*suggested* preprocessing step with a confidence in [0, 1]. The doctor never
claims a measurement is "wrong": smooth PA compression is physics, not a
defect, so only hard plateaus are reported as possible clipping and every
delay/gain estimate is reported as an estimate.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

from opendpd.schemas import DiagnosticItem, DiagnosticReport, Severity, SignalSpec

DOCTOR_VERSION = "dataset-doctor-v1"
MIN_SAMPLES = 1024
MAX_LAG = 512
OUTLIER_SIGMAS = 25.0
CLIP_WINDOW = 1e-3          # relative band below the peak counted as "at the peak"
CLIP_MIN_FRACTION = 1e-3    # of samples at the peak before a plateau is reported
CLIP_MIN_COUNT = 20
ANALYSIS_SAMPLES = 1 << 17  # samples used for correlation / spectrum estimates


@dataclass(frozen=True)
class Estimates:
    delay_samples: float
    delay_confidence: float
    gain_db: float
    phase_deg: float
    gain_confidence: float


def _complex(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if np.iscomplexobj(a):
        return a.astype(np.complex128)
    if a.ndim != 2 or a.shape[1] != 2:
        raise ValueError(f"expected an (n, 2) I/Q array, got shape {a.shape}")
    return a[:, 0].astype(np.float64) + 1j * a[:, 1].astype(np.float64)


def estimate_delay(x: np.ndarray, y: np.ndarray, max_lag: int = MAX_LAG) -> Tuple[float, float]:
    """Delay of ``y`` relative to ``x`` (positive = output lags) with sub-sample
    parabolic refinement; confidence is the normalised correlation peak."""
    n = min(len(x), len(y), ANALYSIS_SAMPLES)
    xs, ys = x[:n], y[:n]
    size = 1 << int(np.ceil(np.log2(2 * n)))
    corr = np.fft.ifft(np.fft.fft(ys, size) * np.conj(np.fft.fft(xs, size)))
    lags = np.concatenate([np.arange(0, max_lag + 1), np.arange(-max_lag, 0)])
    mag = np.abs(np.concatenate([corr[: max_lag + 1], corr[-max_lag:]]))
    k = int(np.argmax(mag))
    energy = np.sqrt(np.sum(np.abs(xs) ** 2) * np.sum(np.abs(ys) ** 2))
    confidence = float(mag[k] / energy) if energy > 0 else 0.0
    lag = float(lags[k])
    # parabolic interpolation on the three points around the peak (circular index)
    left, right = mag[(k - 1) % len(mag)], mag[(k + 1) % len(mag)]
    denom = left - 2 * mag[k] + right
    if denom < 0:
        lag += 0.5 * (left - right) / denom
    return lag, max(0.0, min(1.0, confidence))


def estimate_gain(x: np.ndarray, y: np.ndarray, delay: int, small_signal_quantile: float = 0.4) -> Tuple[complex, float]:
    """Least-squares complex gain on small-signal samples after integer alignment."""
    if delay > 0:
        xa, ya = x[:-delay] if delay else x, y[delay:]
    elif delay < 0:
        xa, ya = x[-delay:], y[:delay]
    else:
        xa, ya = x, y
    n = min(len(xa), len(ya), ANALYSIS_SAMPLES)
    xa, ya = xa[:n], ya[:n]
    mag = np.abs(xa)
    small = mag <= np.quantile(mag, small_signal_quantile)
    if small.sum() < 16 or np.sum(np.abs(xa[small]) ** 2) == 0:
        return 1.0 + 0j, 0.0
    g = np.sum(np.conj(xa[small]) * ya[small]) / np.sum(np.abs(xa[small]) ** 2)
    resid = np.linalg.norm(ya[small] - g * xa[small]) / max(np.linalg.norm(ya[small]), 1e-30)
    return complex(g), float(max(0.0, min(1.0, 1.0 - resid)))


def occupied_bandwidth(x: np.ndarray, fs: float, power_fraction: float = 0.99, nfft: int = 1024) -> float:
    """Bandwidth containing ``power_fraction`` of the power (averaged periodograms)."""
    n = min(len(x), ANALYSIS_SAMPLES)
    segs = n // nfft
    if segs < 1:
        return float("nan")
    frames = x[: segs * nfft].reshape(segs, nfft) * np.hanning(nfft)
    psd = np.mean(np.abs(np.fft.fftshift(np.fft.fft(frames, axis=1), axes=1)) ** 2, axis=0)
    total = psd.sum()
    if total <= 0:
        return float("nan")
    order = np.argsort(psd)[::-1]
    cumulative = np.cumsum(psd[order])
    k = int(np.searchsorted(cumulative, power_fraction * total)) + 1
    return float(k / nfft * fs)


def _item(code: str, severity: Severity, title: str, message: str, evidence: Dict, suggestion: Optional[str] = None,
          confidence: Optional[float] = None, blocking: bool = False) -> DiagnosticItem:
    clean = {k: (float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer,)) else v)
             for k, v in evidence.items()}
    return DiagnosticItem(code=code, severity=severity, title=title, message=message, evidence=clean,
                          suggestion=suggestion, confidence=confidence, blocking=blocking)


def diagnose(x_iq: np.ndarray, y_iq: np.ndarray, signal: SignalSpec, *, dataset_id: str,
             raw_sha256: Optional[str] = None) -> DiagnosticReport:
    items: List[DiagnosticItem] = []
    x, y = _complex(x_iq), _complex(y_iq)

    # --- structural ------------------------------------------------------------
    if len(x) != len(y):
        items.append(_item("length_mismatch", Severity.error, "Input and output lengths differ",
                           f"input has {len(x)} samples, output has {len(y)}; pairs cannot be formed reliably",
                           {"n_input": len(x), "n_output": len(y)},
                           "check the column mapping or trim both signals to the paired range", blocking=True))
        n = min(len(x), len(y))
        x, y = x[:n], y[:n]
    n = len(x)
    if n < MIN_SAMPLES:
        items.append(_item("too_few_samples", Severity.error, "Too few samples",
                           f"{n} samples is below the minimum of {MIN_SAMPLES} for training and evaluation",
                           {"n_samples": n, "minimum": MIN_SAMPLES}, blocking=True))
    bad_x, bad_y = ~np.isfinite(x), ~np.isfinite(y)
    if bad_x.any() or bad_y.any():
        idx = np.flatnonzero(bad_x | bad_y)
        items.append(_item("non_finite_samples", Severity.error, "NaN or Inf samples",
                           f"{int(bad_x.sum())} input and {int(bad_y.sum())} output samples are not finite",
                           {"input_non_finite": int(bad_x.sum()), "output_non_finite": int(bad_y.sum()),
                            "first_indices": [int(i) for i in idx[:10]]},
                           "enable 'interpolate_non_finite' in preprocessing or fix the capture", blocking=True))
        x = np.where(bad_x, 0, x)
        y = np.where(bad_y, 0, y)

    mag_x, mag_y = np.abs(x), np.abs(y)
    rms_x = float(np.sqrt(np.mean(mag_x ** 2)))
    rms_y = float(np.sqrt(np.mean(mag_y ** 2)))
    if rms_x == 0 or rms_y == 0:
        items.append(_item("silent_signal", Severity.error, "All-zero signal",
                           "input or output has zero power; nothing can be modelled",
                           {"rms_input": rms_x, "rms_output": rms_y}, blocking=True))
        return _report(dataset_id, raw_sha256, items)

    # --- outliers (isolated spikes) -------------------------------------------------
    med = float(np.median(mag_y))
    mad = float(np.median(np.abs(mag_y - med))) * 1.4826
    threshold = med + OUTLIER_SIGMAS * max(mad, 1e-12)
    outliers = mag_y > threshold
    if outliers.any():
        idx = np.flatnonzero(outliers)
        items.append(_item("output_outliers", Severity.warning, "Isolated outliers in the output",
                           f"{int(outliers.sum())} output samples exceed {threshold:.3g} "
                           f"(median {med:.3g} + {OUTLIER_SIGMAS:g} robust sigmas)",
                           {"count": int(outliers.sum()), "fraction": float(outliers.mean()),
                            "threshold": threshold, "max_magnitude": float(mag_y.max()),
                            "first_indices": [int(i) for i in idx[:10]]},
                           "enable 'remove_outliers' in preprocessing (spikes are replaced by interpolation)",
                           confidence=float(min(1.0, (mag_y[outliers].max() / threshold - 1) / 4 + 0.5))))
    keep = ~outliers
    rms_y = float(np.sqrt(np.mean(mag_y[keep] ** 2)))   # spikes excluded: they would dominate the RMS

    # --- clipping: a hard plateau at the peak, not smooth compression -----------------
    for name, mag in (("output", mag_y[keep]), ("input", mag_x)):
        peak = float(mag.max())
        at_peak = mag >= peak * (1 - CLIP_WINDOW)
        frac = float(at_peak.mean())
        if at_peak.sum() >= CLIP_MIN_COUNT and frac >= CLIP_MIN_FRACTION:
            items.append(_item(f"possible_{name}_clipping", Severity.warning, f"Possible clipping of the {name}",
                               f"{frac:.2%} of {name} samples sit within {CLIP_WINDOW:.1%} of the peak magnitude "
                               f"{peak:.4g}: a flat plateau, unlike smooth compression",
                               {"fraction_at_peak": frac, "count_at_peak": int(at_peak.sum()), "peak_magnitude": peak,
                                "rms": rms_y if name == "output" else rms_x},
                               "check the capture range / ADC full scale; if the saturation is intentional, keep the "
                               "data and note it in the manifest",
                               confidence=float(min(1.0, frac / 0.01))))

    # --- time alignment ----------------------------------------------------------------
    delay, delay_conf = estimate_delay(np.where(keep, x, 0), np.where(keep, y, 0))
    if abs(delay) >= 0.5:
        items.append(_item("time_misalignment", Severity.warning, "Output appears delayed relative to input",
                           f"cross-correlation peaks at {delay:+.2f} samples (correlation {delay_conf:.2f})",
                           {"delay_samples": delay, "correlation": delay_conf},
                           f"apply delay correction of {delay:+.2f} samples in preprocessing", confidence=delay_conf))
    else:
        items.append(_item("alignment_ok", Severity.info, "Input and output are time aligned",
                           f"cross-correlation peaks at {delay:+.2f} samples (correlation {delay_conf:.2f})",
                           {"delay_samples": delay, "correlation": delay_conf}, confidence=delay_conf))

    # --- linear gain / phase ---------------------------------------------------------------
    g, gain_conf = estimate_gain(np.where(keep, x, 0), np.where(keep, y, 0), int(round(delay)))
    gain_db = float(20 * np.log10(abs(g))) if abs(g) > 0 else float("-inf")
    phase_deg = float(np.degrees(np.angle(g)))
    evidence = {"gain_db": gain_db, "phase_deg": phase_deg, "fit_residual": 1 - gain_conf,
                "fitted_on": "samples with |x| below the 40th percentile, after integer alignment"}
    if abs(phase_deg) > 10 or abs(gain_db) > 3:
        items.append(_item("linear_gain_phase", Severity.warning, "Large linear gain or phase offset",
                           f"small-signal complex gain is {gain_db:+.2f} dB at {phase_deg:+.1f} degrees",
                           evidence, "apply gain/phase correction in preprocessing if you want a unity-gain reference; "
                           "otherwise keep it and let the evaluation profile derive its reference gain",
                           confidence=gain_conf))
    else:
        items.append(_item("linear_gain_phase", Severity.info, "Small-signal gain and phase",
                           f"small-signal complex gain is {gain_db:+.2f} dB at {phase_deg:+.1f} degrees",
                           evidence, confidence=gain_conf))

    # --- amplitude range ---------------------------------------------------------------------
    papr_x = float(20 * np.log10(mag_x.max() / rms_x))
    papr_y = float(20 * np.log10(mag_y[keep].max() / rms_y))
    amp = {"peak_input": float(mag_x.max()), "rms_input": rms_x, "papr_input_db": papr_x,
           "peak_output": float(mag_y[keep].max()), "rms_output": rms_y, "papr_output_db": papr_y,
           "amplitude_units": signal.amplitude_units}
    if signal.amplitude_units == "unknown":
        items.append(_item("amplitude_units_unconfirmed", Severity.info, "Amplitude units not confirmed", 
                           "values are treated as dimensionless; confirm 'normalized' or 'volts' in the manifest", amp,
                           "set amplitude_units in the dataset manifest"))
    elif signal.amplitude_units == "normalized" and mag_x.max() > 1.05:
        items.append(_item("input_exceeds_unit_range", Severity.warning, "Input peak above 1.0",
                           f"peak input magnitude is {mag_x.max():.3g} although units are declared normalized", amp,
                           "rescale in preprocessing (normalize='peak_input', fitted on the training split) or "
                           "correct the units"))
    else:
        items.append(_item("amplitude_range", Severity.info, "Amplitude range", 
                           f"input PAPR {papr_x:.1f} dB, output PAPR {papr_y:.1f} dB", amp))

    # --- metadata and spectrum ------------------------------------------------------------------
    missing = signal.missing_for_legacy_evaluation()
    if missing:
        items.append(_item("metadata_missing", Severity.error, "Metadata required for evaluation is missing",
                           "the legacy metric profile needs " + ", ".join(missing) + "; without them ACLR/EVM "
                           "would be computed on guesses", {"missing": missing},
                           "fill the signal metadata in the dataset manifest", blocking=True))
    fs = signal.sample_rate_hz
    if fs:
        obw = occupied_bandwidth(x, fs)
        spectrum = {"occupied_bandwidth_hz": obw, "sample_rate_hz": fs, "declared_bandwidth_hz": signal.bandwidth_hz}
        if np.isfinite(obw) and obw > 0.6 * fs:
            items.append(_item("insufficient_oversampling", Severity.warning, "Signal nearly fills the sampling bandwidth",
                               f"99% of the input power occupies {obw / 1e6:.1f} MHz of the {fs / 1e6:.1f} MHz capture; "
                               "adjacent-channel bands needed for ACLR fall outside it (need main + 2 adjacent bands < fs)", spectrum,
                               "capture at a higher sample rate (>= 4x the signal bandwidth for ACLR)"))
        elif signal.bandwidth_hz and np.isfinite(obw) and not (0.7 <= obw / signal.bandwidth_hz <= 1.3):
            items.append(_item("bandwidth_metadata_mismatch", Severity.warning, "Declared bandwidth disagrees with the signal",
                               f"99% occupied bandwidth is {obw / 1e6:.1f} MHz but bandwidth_hz says "
                               f"{signal.bandwidth_hz / 1e6:.1f} MHz", spectrum,
                               "check bandwidth_hz (main channel) in the manifest; ACLR bands depend on it"))
        else:
            items.append(_item("spectrum_summary", Severity.info, "Spectrum", 
                               f"99% occupied bandwidth {obw / 1e6:.1f} MHz at {fs / 1e6:.1f} MHz sampling", spectrum))
    return _report(dataset_id, raw_sha256, items)


def _report(dataset_id: str, raw_sha256: Optional[str], items: List[DiagnosticItem]) -> DiagnosticReport:
    order = {Severity.error: 0, Severity.warning: 1, Severity.info: 2}
    items = sorted(items, key=lambda i: order[i.severity])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return DiagnosticReport(report_id=f"doc-{stamp}-{(raw_sha256 or 'nohash')[:8]}", dataset_id=dataset_id,
                            dataset_raw_sha256=raw_sha256, doctor_version=DOCTOR_VERSION, items=items,
                            evaluation_blocked=any(i.blocking for i in items))


def estimates_from_report(report: DiagnosticReport) -> Estimates:
    """Pull the numeric estimates back out of a report (for preprocessing suggestions)."""
    by_code = {i.code: i for i in report.items}
    align = by_code.get("time_misalignment") or by_code.get("alignment_ok")
    gain = by_code.get("linear_gain_phase")
    return Estimates(
        delay_samples=float(align.evidence["delay_samples"]) if align else 0.0,
        delay_confidence=float(align.confidence or 0.0) if align else 0.0,
        gain_db=float(gain.evidence["gain_db"]) if gain else 0.0,
        phase_deg=float(gain.evidence["phase_deg"]) if gain else 0.0,
        gain_confidence=float(gain.confidence or 0.0) if gain else 0.0,
    )
