"""Compose existing core measurements and plot payloads for dataset inspection.

No dataset, split, reference gain, or evaluation profile is changed here.
Unreviewed scientific definitions remain explicitly unavailable. In particular,
raw output-versus-input NMSE is not presented as PA nonlinearity.
"""

import math

import numpy as np

from opendpd.core import plots
from opendpd.core.doctor import ANALYSIS_SAMPLES, DOCTOR_VERSION, MIN_SAMPLES, diagnose
from opendpd.core.metrics import evaluate, get_profile
from opendpd.schemas.analysis import DatasetAnalysis, InspectionMeasurement, InspectionReading
from opendpd.schemas.diagnostics import DiagnosticItem, DiagnosticReport
from opendpd.schemas.metrics import ProfileValidation
from opendpd.services.datasets import analysis_window, load_version_arrays
from opendpd.services.dataset_constellation import dataset_constellation
from opendpd.services.workspace import Workspace


def _unavailable(reason: str, status: str = "not_applicable") -> InspectionReading:
    return InspectionReading(status=status, reason=reason)


def _evidence_reading(value, *, finite: bool, reason: str) -> InspectionReading:
    if not finite:
        return _unavailable("Capture contains non-finite samples; run preprocessing before measurement", "invalid")
    if value is None or not math.isfinite(float(value)):
        return _unavailable(reason)
    return InspectionReading(value=float(value), status="ok")


def analyze_dataset(ws: Workspace, dataset_id: str, version: str = "raw-v1") -> DatasetAnalysis:
    """Read a version and inspect the doctor's bounded, central contiguous window.

    The window is descriptive capture inspection, not a train/validation/test
    score. Metric estimators see the complete window; chart points are bounded.
    """
    manifest = ws.get_dataset(dataset_id)
    x, y, _ = load_version_arrays(ws, dataset_id, version)
    total = len(x)
    start, end = analysis_window(total, ANALYSIS_SAMPLES)
    xw, yw = np.asarray(x[start:end]), np.asarray(y[start:end])
    paired = len(xw) == len(yw)
    finite = bool(np.isfinite(xw).all() and np.isfinite(yw).all())
    enough = min(len(xw), len(yw)) >= MIN_SAMPLES
    if enough:
        report = diagnose(xw, yw, manifest.signal, dataset_id=dataset_id, raw_sha256=manifest.raw_sha256)
    else:
        report = DiagnosticReport(
            report_id="inspection-too-short", dataset_id=dataset_id, doctor_version=DOCTOR_VERSION,
            dataset_raw_sha256=manifest.raw_sha256, evaluation_blocked=True,
            items=[DiagnosticItem(code="too_few_samples", severity="error", title="Too few samples",
                                  message=f"At least {MIN_SAMPLES} paired samples are needed for dataset inspection",
                                  blocking=True)],
        )
    by_code = {item.code: item for item in report.items}
    amplitude = next((item.evidence for item in report.items if "papr_input_db" in item.evidence), {})
    measurements = []
    for name, label, unit, field in (("PAPR", "PAPR", "dB", "papr_{side}_db"),
                                   ("RMS", "RMS amplitude", manifest.signal.amplitude_units, "rms_{side}"),
                                   ("PEAK", "Peak amplitude", manifest.signal.amplitude_units, "peak_{side}")):
        readings = {side: _evidence_reading(amplitude.get(field.format(side=side)), finite=finite,
                                          reason="Amplitude measurement unavailable for this capture")
                    for side in ("input", "output")}
        measurements.append(InspectionMeasurement(
            name=name, label=label, unit=unit, **readings,
            method="dataset-doctor-v1; output statistics exclude isolated outliers identified by the doctor. "
                   "Capture amplitude is not calibrated PA output power.",
        ))

    profile = get_profile("general-spectral-v1")
    scores = {side: {m.name: m for m in evaluate(profile.profile_id, a, None, manifest.signal)}
              for side, a in (("input", xw), ("output", yw))} if enough else {}
    for definition in profile.metrics:
        if definition.name not in ("ACPR_L", "ACPR_R"):
            continue
        readings = {}
        for side in ("input", "output"):
            metric = scores.get(side, {}).get(definition.name)
            readings[side] = (InspectionReading(value=metric.value, status=metric.status.value, reason=metric.reason)
                              if metric else _unavailable("Too few samples"))
        measurements.append(InspectionMeasurement(
            name=definition.name, label=definition.display_name, unit=definition.unit, **readings,
            method=f"{profile.profile_id}: {definition.formula}. {definition.notes}",
        ))

    for name, label, unit, code, field in (
        ("DELAY", "Delay estimate", "samples", "time_misalignment" if "time_misalignment" in by_code else "alignment_ok", "delay_samples"),
        ("GAIN", "Capture gain (small signal)", "dB", "linear_gain_phase", "gain_db"),
        ("PHASE", "Capture phase (small signal)", "deg", "linear_gain_phase", "phase_deg"),
    ):
        item = by_code.get(code)
        measurements.append(InspectionMeasurement(
            name=name, label=label, unit=unit,
            output=_evidence_reading(item.evidence.get(field) if item else None, finite=finite,
                                     reason="Estimate unavailable"),
            method="dataset-doctor-v1: " + (item.message if item else "estimate unavailable") +
                   (". Gain/phase fit uses the lowest 40% of input amplitudes after integer alignment; "
                    "this includes the capture/attenuator scale, not calibrated PA gain." if name != "DELAY" else ""),
        ))

    measurements.append(InspectionMeasurement(
        name="BLA_NMSE", label="BLA residual NMSE", unit="dB",
        output=_unavailable("A reviewed linear-reference profile is required (alignment, fit window, "
                            "scalar/FIR reference and denominator). Raw input/output NMSE is not PA nonlinearity.", "review_required"),
        method="Residual relative to a best linear approximation; includes noise and unmodelled memory. "
               "No universal nonlinearity grade is assigned.",
    ))
    ofdm = get_profile("ofdm-lte20-evm-v1")
    eligible = ofdm.validation != ProfileValidation.pending_cross_validation
    ofdm_scores = {m.name: m for m in evaluate(ofdm.profile_id, yw, None, manifest.signal)} if eligible and enough else {}
    for definition in ofdm.metrics:
        if definition.name == "EVM_DB":
            continue
        if manifest.signal.waveform is None:
            reading = _unavailable("Requires a waveform binding and a reviewed EVM receiver profile; "
                                   "a modulation label or quick-look constellation does not define an EVM protocol", "missing_reference")
        elif not eligible:
            reading = _unavailable("ofdm-lte20-evm-v1 is pending independent cross-validation", "review_required")
        else:
            metric = ofdm_scores.get(definition.name)
            reading = (InspectionReading(value=metric.value, status=metric.status.value, reason=metric.reason)
                       if metric else _unavailable("Too few samples"))
        measurements.append(InspectionMeasurement(name=definition.name, label=definition.display_name,
                                                  unit=definition.unit, output=reading,
                                                  method=f"{ofdm.profile_id}: {definition.formula}. Not a conformance result."))

    data = DatasetAnalysis(
        dataset_id=dataset_id, data_version=version, total_samples=total,
        sample_range=(start, end), metadata_complete=not manifest.signal.missing_for_legacy_evaluation(),
        inspection_ready=not report.evaluation_blocked, diagnostics=report, measurements=measurements,
        notes=["Measurements describe the displayed capture window, not a held-out model evaluation.",
               "I/Q scatter shows sampled complex baseband, not demodulated constellation symbols. "
               "AM/AM and AM/PM use the selected version as captured, without automatic gain/delay correction.",
               "No calibrated dBm, noise-only SNR, spectral mask or standards pass/fail is inferred from these samples."],
    )
    if enough and finite and paired:
        signals, roles = {"Input": xw, "PA output": yw}, {"Input": "input", "PA output": "primary"}
        data.spectrum = plots.spectrum(signals, roles, sample_rate_hz=manifest.signal.sample_rate_hz,
                                       nperseg=manifest.signal.nperseg, bandwidth_hz=manifest.signal.bandwidth_hz)
        data.time = plots.time_excerpt(signals, roles, n=256)
        data.time.start = data.sample_range[0]
        data.iq = plots.iq_scatter(signals, roles)
        data.am = plots.am_am_pm(xw, {"PA output": yw}, roles)
    data.constellation = dataset_constellation(manifest, version, x, y)
    return data
