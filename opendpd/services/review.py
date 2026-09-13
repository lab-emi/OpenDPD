"""RF review of stored evidence. Display helpers never re-score a signal."""

import hashlib
import json
import math

from opendpd.core.metrics import get_profile
from opendpd.schemas.dataset import SignalSpec
from opendpd.schemas.review import ReviewBand, ReviewContext, ReviewFact
from opendpd.schemas.rf import RFConditions
from opendpd.services import experiments
from opendpd.services.workspace import WorkspaceError, read_json, sha256_file, write_json_atomic


def object_sha(value):
    data = value.model_dump(mode="json") if hasattr(value, "model_dump") else value
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def plot_artifact(ws, run_id, kind):
    manifest = experiments.load_artifacts(ws, run_id)
    artifact = next((a for a in manifest.artifacts if a.artifact_id == f"plot-{kind}"), None) if manifest else None
    if artifact is None:
        raise WorkspaceError(f"run '{run_id}' has no stored {kind} plot")
    path = ws.run_dir(run_id) / artifact.file.path
    if not path.is_file() or not path.resolve().is_relative_to(ws.run_dir(run_id).resolve()):
        raise WorkspaceError("plot artifact is missing or outside its run")
    digest = sha256_file(path)
    if artifact.file.sha256 and artifact.file.sha256 != digest:
        raise WorkspaceError(f"plot artifact hash mismatch: {run_id}/{kind}")
    return read_json(path), artifact.file.path, digest


def save_conditions(ws, run_id, conditions: RFConditions):
    if experiments.load_result(ws, run_id) is None:
        raise WorkspaceError("RF conditions require a stored result")
    # A revision is retained even when the current declaration is replaced.
    root = ws.run_dir(run_id)
    history = root / "rf-conditions"
    history.mkdir(exist_ok=True)
    write_json_atomic(history / f"{object_sha(conditions)}.json", conditions)
    write_json_atomic(root / "rf-conditions.json", conditions)
    return conditions


def integration_bands(profile_id, signal):
    if signal is None or signal.sample_rate_hz is None:
        return [], "Integration bands unavailable: the evaluated sample rate was not recorded."
    fs, bw = signal.sample_rate_hz, signal.bandwidth_hz
    bands = []

    def band(label, role, lo, hi):
        available = lo >= -fs / 2 and hi <= fs / 2 and lo < hi
        bands.append(ReviewBand(label=label, role=role, edges_hz=(lo, hi), available=available,
                                reason=None if available else "Band exceeds the captured range or contains no bins."))

    if profile_id == "ofdm-lte20-evm-v1":
        band("Main", "main", -9e6, 9e6)
        band("Left", "adjacent", -29e6, -11e6)
        band("Right", "adjacent", 11e6, 29e6)
        return bands, "Profile integration intervals in Hz; stored PSD is a display, not a demodulated constellation."
    if not bw:
        return [], "Integration bands unavailable: the evaluated bandwidth was not recorded."
    if profile_id == "general-spectral-v1":
        band("Main", "main", -bw / 2, bw / 2)
        band("Left", "adjacent", -3 * bw / 2, -bw / 2)
        band("Right", "adjacent", bw / 2, 3 * bw / 2)
        return bands, "General spectral profile: bin centres in [low, high); adjacent/main power, pooled Welch PSD."
    if profile_id == "legacy-opendpd-v1":
        if not signal.nperseg or not signal.n_sub_ch:
            return [], "Legacy integration bands unavailable: nperseg and n_sub_ch were not recorded. The display PSD uses the general estimator."
        # Map the frozen implementation's index slices to frequency boundaries.
        # This is coordinate metadata only; no PSD powers or metrics are computed.
        n = signal.nperseg
        df = fs / n
        f0 = -(n // 2) * df
        left = max(0, math.ceil((-bw / 2 - f0) / df))
        right = min(n - 1, math.floor((bw / 2 - f0) / df))
        width = (right - left) // signal.n_sub_ch
        band("Main envelope", "main", -bw / 2, bw / 2)
        for i in range(signal.n_sub_ch):
            band(f"Subchannel {i + 1}", "subchannel", f0 + (left + i * width) * df, f0 + (left + (i + 1) * width) * df)
        band("Left", "adjacent", f0 + (left - width) * df, f0 + left * df)
        band("Right", "adjacent", f0 + right * df, f0 + (right + width) * df)
        return bands, "Legacy bin slices: adjacent/strongest subchannel power. Display PSD uses Welch density over valid samples; it does not reproduce legacy segment aggregation."
    return [], "No integration-band renderer registered for this profile."


def review_result(ws, run_id, profile_id=None):
    result = experiments.load_result(ws, run_id, profile_id)
    if result is None:
        raise WorkspaceError(f"run '{run_id}' has no stored result")
    profile = get_profile(result.metric_profile_id)
    if profile.version != result.metric_profile_version:
        raise WorkspaceError("stored profile version is not available in this installation")
    signal = result.evaluated_signal
    source = "evaluation snapshot" if signal else "not recorded"
    if signal is None:
        try:
            plot, _, _ = plot_artifact(ws, run_id, "spectrum")
        except WorkspaceError:
            plot = None
        if plot:
            main = (plot.get("bands") or {}).get("main")
            signal = SignalSpec(sample_rate_hz=plot.get("sample_rate_hz"),
                                bandwidth_hz=main[1] - main[0] if main else None)
            source = "historical stored display metadata; complete evaluation metadata was not recorded"
    facts = []

    def fact(key, label, value, unit=None, origin="not provided"):
        facts.append(ReviewFact(key=key, label=label, value=None if value is None else str(value), unit=unit, source=origin))

    try:
        dataset = ws.get_dataset(result.dataset.dataset_id)
        origin = dataset.origin.value if dataset.raw_sha256 == result.dataset.raw_sha256 else None
    except WorkspaceError:
        origin = None
    fact('dataset_origin', 'Dataset origin', origin, origin='dataset declaration matched to the result raw SHA256')

    rf = result.rf_conditions
    measurement = result.measurement
    c = measurement.conditions if measurement else None
    declared = "user-provided measurement" if c else "not provided"
    rf_source = rf.source if rf else "not provided"
    fact("dut", "PA / DUT", rf.dut if rf and rf.dut else c.pa if c else None, origin=rf_source if rf and rf.dut else declared)
    for key, label, unit in (("carrier_frequency_hz", "Carrier frequency", "Hz"),
                             ("average_output_power_dbm", "Average modulated output power", "dBm"),
                             ("input_power_dbm", "RF input power", "dBm"), ("pa_dc_power_w", "PA DC power", "W"),
                             ("power_reference_plane", "Power reference plane", None),
                             ("temperature_c", "Temperature", "°C"), ("supply_v", "Supply", "V"),
                             ("bias", "Bias", None), ("mode", "Operating mode", None), ("load", "Load", None),
                             ("vswr", "VSWR", None), ("reflection_phase_deg", "Reflection phase", "°"),
                             ("calibration_id", "Calibration record", None), ("fixture", "Fixture", None),
                             ("deembedding", "De-embedding", None), ("backoff_reference", "Back-off reference", None)):
        value = getattr(rf, key, None)
        origin = rf_source
        if value is None and measurement:
            if key == "average_output_power_dbm":
                value = measurement.captures[0].declared_output_power_dbm
            elif key == "temperature_c":
                value = c.temperature_c
            origin = declared
        fact(key, label, value, unit, origin)
    for key, label, unit in (("sample_rate_hz", "Baseband sample rate", "Hz"), ("bandwidth_hz", "Signal bandwidth", "Hz"),
                             ("modulation", "Modulation", None), ("amplitude_units", "Amplitude units", None)):
        fact(key, label, getattr(signal, key, None), unit, source)
    for stage in result.signal_chain:
        papr = 20 * math.log10(stage.peak_abs / stage.rms) if stage.peak_abs and stage.rms else None
        fact(f"papr_{stage.symbol}", f"PAPR ({stage.symbol})", papr, "dB", "stored peak/RMS over evaluated samples")
    # Only user-declared physical powers enter these formulas; normalized samples never do.
    pout = 10 ** ((rf.average_output_power_dbm - 30) / 10) if rf and rf.average_output_power_dbm is not None else None
    pin = 10 ** ((rf.input_power_dbm - 30) / 10) if rf and rf.input_power_dbm is not None else None
    pa_dc = rf.pa_dc_power_w if rf else None
    rails = sum(rf.dc_rails_w[name] for name in rf.included_rails) if rf else 0
    fact("de", "Drain efficiency", 100 * pout / pa_dc if pout is not None and pa_dc else None, "%", "declared powers: P_RF_out / P_DC_PA")
    fact("pae", "Power-added efficiency", 100 * (pout - pin) / pa_dc if pout is not None and pin is not None and pa_dc else None,
         "%", "declared powers: (P_RF_out − P_RF_in) / P_DC_PA")
    fact("tx_efficiency", "TX efficiency", 100 * pout / rails if pout is not None and rails else None,
         "%", "declared powers: P_RF_out / sum included DC rails: " + ", ".join(rf.included_rails if rf else []))
    bands, note = integration_bands(result.metric_profile_id, signal)
    provenance = {"raw_data_sha256": result.dataset.raw_sha256, "processed_data_sha256": result.dataset.processed_sha256}
    for name in ("config.resolved.json", "provenance.json", "rf-conditions.json"):
        path = ws.run_dir(run_id) / name
        provenance[name + " sha256"] = sha256_file(path) if path.is_file() else None
    for model in result.models:
        provenance[f"{model.role} weights sha256"] = model.weights_sha256
    return ReviewContext(result=result, signal=signal, signal_source=source, profile=profile, facts=facts,
                         bands=bands, band_note=note, provenance=provenance)
