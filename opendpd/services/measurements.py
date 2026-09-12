"""Measured DPD evaluation (plan S16): the ``evaluate_measured`` task.

A ``run_dpd`` run exported ``u = DPD(x)``; the operator played it (and ``x``)
through a physical PA and captured the output. This service binds such a
submission to the run whose signal was played, copies the captures into the
new run so it is self-contained, aligns every capture to what was played
(``opendpd.core.measurement``), scores it under every registered profile and
records the operator's conditions next to the numbers. Captures from the mock
instrument adapter follow the same path and are marked as mock evidence.
"""

from __future__ import annotations

import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from opendpd.core.measurement import align, level_db, peak_abs, rate_ratio, resample, rms, to_iq
from opendpd.core.metrics import evaluate as score, get_profile, list_profiles
from opendpd.schemas import (
    ATTESTATION,
    MOCK_ATTESTATION,
    ArtifactKind,
    ArtifactManifest,
    BaselineScore,
    CaptureAlignment,
    CaptureRef,
    DatasetManifest,
    EvaluationResult,
    ExperimentConfig,
    MeasurementConditions,
    MeasurementConfig,
    MeasurementEvidence,
    ResolvedExperimentConfig,
    RunStatus,
    ScalingInfo,
    SignalStage,
    TaskType,
)
from opendpd.services.config import ConfigError, ConfigIssue
from opendpd.services.workspace import Workspace, WorkspaceError, read_json, sha256_file, write_json_atomic

CAPTURES_DIR = "captures"
PLAYED_FILE = "played.csv"
MEASUREMENT_FILE = "measurement.json"
LEVEL_TOLERANCE_DB = 0.5          # larger with/without level differences are reported as a limitation
_CSV_PAIRS = (("I", "Q"), ("I_out", "Q_out"), ("i", "q"), ("re", "im"))
_ROLES = ("with_dpd", "without_dpd")


class MeasurementError(WorkspaceError):
    """A capture could not be used (unreadable, too short, does not correlate); the run fails with the reason."""


# --- reading captures -------------------------------------------------------------------

def _as_complex(arr: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(arr)
    if np.iscomplexobj(arr) and arr.ndim == 1:
        return arr.astype(np.complex128)
    if arr.ndim == 2 and arr.shape[1] == 2:
        return arr[:, 0].astype(np.float64) + 1j * arr[:, 1].astype(np.float64)
    if arr.ndim == 2 and arr.shape[0] == 2:
        return arr[0].astype(np.float64) + 1j * arr[1].astype(np.float64)
    raise MeasurementError(f"{name}: expected complex samples or an (n, 2) I/Q array, got shape {arr.shape}")


def read_capture(path: Path, columns: Optional[Tuple[str, str]] = None) -> np.ndarray:
    """Complex samples of a capture file: CSV (I/Q columns), .npy ((n, 2) or complex) or .npz (I/Q keys)."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        import pandas as pd

        frame = pd.read_csv(path)
        pair = columns or next((c for c in _CSV_PAIRS if c[0] in frame.columns and c[1] in frame.columns), None)
        if pair is None or any(c not in frame.columns for c in pair):
            raise MeasurementError(f"{path.name}: no I/Q columns {pair or 'I,Q'} among {list(frame.columns)}; "
                                   "name them with the capture's columns field")
        return frame[pair[0]].to_numpy(dtype=np.float64) + 1j * frame[pair[1]].to_numpy(dtype=np.float64)
    if suffix == ".npy":
        return _as_complex(np.load(path, allow_pickle=False), path.name)
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            keys = columns or (("I", "Q") if "I" in data and "Q" in data else None)
            if keys is not None:
                if any(k not in data for k in keys):
                    raise MeasurementError(f"{path.name}: keys {keys} not all present (has {data.files})")
                return data[keys[0]].astype(np.float64) + 1j * data[keys[1]].astype(np.float64)
            if len(data.files) == 1:
                return _as_complex(data[data.files[0]], path.name)
            raise MeasurementError(f"{path.name}: name the I/Q keys (has {data.files})")
    raise MeasurementError(f"{path.name}: unsupported capture format (use .csv, .npy or .npz)")


def read_played(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """``x`` and ``u`` (complex) from a run_dpd export (columns I, Q, I_dpd, Q_dpd)."""
    import pandas as pd

    frame = pd.read_csv(path)
    for col in ("I", "Q", "I_dpd", "Q_dpd"):
        if col not in frame.columns:
            raise MeasurementError(f"{path.name} is not a run_dpd export (missing column {col})")
    x = frame["I"].to_numpy(dtype=np.float64) + 1j * frame["Q"].to_numpy(dtype=np.float64)
    u = frame["I_dpd"].to_numpy(dtype=np.float64) + 1j * frame["Q_dpd"].to_numpy(dtype=np.float64)
    return x, u


# --- binding ------------------------------------------------------------------------------

def capture_path(ws: Workspace, ref: CaptureRef, field: str) -> Path:
    """The file behind a capture reference: inside the workspace imports directory, never elsewhere."""
    from opendpd.services.datasets import ImportError_, resolve_in_root

    try:
        path = resolve_in_root(ws, "imports", ref.path)
    except ImportError_ as err:
        raise ConfigError([ConfigIssue(field, str(err), "upload the capture through the service or copy it under "
                                                          "the workspace imports directory")]) from None
    if not path.is_file():
        raise ConfigError([ConfigIssue(field, f"no file at imports/{ref.path}",
                                       "upload the capture (POST /datasets/upload) or `opendpd measurements import`")])
    return path


def stage_capture(ws: Workspace, path: Path) -> str:
    """The imports-relative path of a capture file: as is when it already lives under imports/, otherwise a copy
    named by its hash under imports/captures/. Never a path outside the workspace."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise MeasurementError(f"no capture file at {path}")
    base = ws.imports_dir.resolve()
    if base in path.parents:
        return path.relative_to(base).as_posix()
    digest = sha256_file(path)
    target_dir = ws.imports_dir / "captures"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{digest[:12]}-{path.name}"
    if not target.exists():
        shutil.copy2(path, target)
    return target.relative_to(base).as_posix()


def bind_measurement(ws: Workspace, config: ExperimentConfig) -> ExperimentConfig:
    """Bind an ``evaluate_measured`` submission: the played run_dpd run (dataset, DPD weights, exported signal
    hash) and the capture files (hashed now, verified again when the run executes)."""
    from opendpd.services.experiments import load_artifacts, load_resolved, load_run

    m = config.measurement
    assert m is not None
    field = "measurement.apply_run_id"
    try:
        record = load_run(ws, m.apply_run_id)
    except WorkspaceError as err:
        raise ConfigError([ConfigIssue(field, str(err), "apply the DPD first (`opendpd apply <dpd-run>`) and play "
                                                          "its exported dpd-output file")]) from None
    if record.task != TaskType.run_dpd:
        raise ConfigError([ConfigIssue(field, f"run '{m.apply_run_id}' is a {record.task.value} run; the played "
                                              "signal comes from a run_dpd run's dpd-output artifact",
                                       "apply the DPD (`opendpd apply`) and play that run's export")])
    if record.status != RunStatus.succeeded:
        raise ConfigError([ConfigIssue(field, f"run '{m.apply_run_id}' has status {record.status.value}")])
    manifest = load_artifacts(ws, m.apply_run_id)
    played = manifest.by_kind(ArtifactKind.dpd_output) if manifest else []
    if not played or played[0].file.sha256 is None:
        raise ConfigError([ConfigIssue(field, f"run '{m.apply_run_id}' has no verified dpd-output artifact")])
    apply_resolved = load_resolved(ws, m.apply_run_id)
    dataset = ws.get_dataset(apply_resolved.dataset.id)
    refs: Dict[str, CaptureRef] = {}
    for role, ref in zip(_ROLES, (m.with_dpd, m.without_dpd)):
        if ref is None:
            continue
        path = capture_path(ws, ref, f"measurement.{role}.path")
        digest = sha256_file(path)
        if ref.sha256 is not None and ref.sha256 != digest:
            raise ConfigError([ConfigIssue(f"measurement.{role}.sha256", f"imports/{ref.path} changed since it was "
                                                                           "referenced", "upload it again")])
        refs[role] = ref.model_copy(update={"sha256": digest})
    fs_d = dataset.signal.sample_rate_hz
    if fs_d is not None and rate_ratio(m.conditions.sample_rate_hz, fs_d) is None:
        raise ConfigError([ConfigIssue("measurement.conditions.sample_rate_hz",
                                       f"cannot convert {m.conditions.sample_rate_hz:g} Hz to the dataset rate "
                                       f"{fs_d:g} Hz with a small rational ratio",
                                       "capture at the dataset rate or a rational multiple of it")])
    measurement = m.model_copy(update={"played_sha256": played[0].file.sha256, "with_dpd": refs["with_dpd"],
                                       "without_dpd": refs.get("without_dpd")})
    return config.model_copy(update={"measurement": measurement, "dataset": apply_resolved.dataset,
                                     "model": apply_resolved.model, "training": apply_resolved.training,
                                     "quantization": apply_resolved.quantization,
                                     "dpd_reference": apply_resolved.dpd_reference, "pa_reference": None})


def measurement_config(ws: Workspace, apply_run_id: str, *, with_dpd: CaptureRef, without_dpd: Optional[CaptureRef],
                       conditions: MeasurementConditions, source: str = "manual", playback: str = "loop",
                       profile_id: Optional[str] = None, name: Optional[str] = None) -> ExperimentConfig:
    """A submission for ``create_run``; dataset and model are placeholders that binding replaces."""
    from opendpd.schemas import DatasetRef, EvaluationConfig, EvidenceType, ModelSpec
    from opendpd.services.experiments import load_run

    record = load_run(ws, apply_run_id)
    evaluation = EvaluationConfig(evidence_type=EvidenceType.dpd_measured,
                                  **({"profile_id": profile_id} if profile_id else {}))
    return ExperimentConfig(
        task=TaskType.evaluate_measured, name=name or f"measured {apply_run_id}" + (" (mock)" if source != "manual" else ""),
        dataset=DatasetRef(id=record.dataset_id or "unbound"), model=ModelSpec(key=record.model_key or "gru"),
        evaluation=evaluation,
        measurement=MeasurementConfig(apply_run_id=apply_run_id, with_dpd=with_dpd, without_dpd=without_dpd,
                                      conditions=conditions, source=source, playback=playback),
    )


# --- execution --------------------------------------------------------------------------------

@dataclass
class MeasuredSignals:
    x: np.ndarray                      # target input, complex
    u: np.ndarray                      # what was played with DPD, complex
    with_dpd: np.ndarray               # aligned capture, complex, capture units
    without_dpd: Optional[np.ndarray]  # aligned capture of x played directly, or None
    gains: Dict[str, complex]          # least-squares gain onto x per role
    evidence: MeasurementEvidence


def _artifact_id(role: str) -> str:
    return "capture-" + role.replace("_", "-")


def _capture_copy(run_dir: Path, role: str, suffix: str) -> Path:
    return run_dir / CAPTURES_DIR / f"{role}{suffix}"


def materialise(ws: Workspace, run_dir: Path, resolved: ResolvedExperimentConfig) -> None:
    """Copy the played export and the captures into the run (verifying every hash) so the run is self-contained."""
    from opendpd.services.experiments import load_artifacts

    m = resolved.measurement
    assert m is not None
    manifest = load_artifacts(ws, m.apply_run_id)
    played = next((a for a in (manifest.artifacts if manifest else []) if a.kind == ArtifactKind.dpd_output), None)
    if played is None:
        raise FileNotFoundError(f"run {m.apply_run_id} no longer has a dpd-output artifact")
    src = ws.run_dir(m.apply_run_id) / played.file.path
    if not src.exists() or sha256_file(src) != m.played_sha256:
        raise FileNotFoundError(f"the exported signal of run {m.apply_run_id} is missing or its hash changed")
    (run_dir / CAPTURES_DIR).mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, run_dir / CAPTURES_DIR / PLAYED_FILE)
    for role, ref in zip(_ROLES, (m.with_dpd, m.without_dpd)):
        if ref is None:
            continue
        path = capture_path(ws, ref, f"measurement.{role}.path")
        if sha256_file(path) != ref.sha256:
            raise FileNotFoundError(f"capture imports/{ref.path} changed since submission (hash mismatch)")
        shutil.copy2(path, _capture_copy(run_dir, role, path.suffix.lower()))


def _stored_capture(run_dir: Path, role: str) -> Path:
    matches = sorted((run_dir / CAPTURES_DIR).glob(f"{role}.*"))
    if not matches:
        raise FileNotFoundError(f"run has no stored {role} capture")
    return matches[0]


def align_stored(run_dir: Path, resolved: ResolvedExperimentConfig, dataset: DatasetManifest) -> MeasuredSignals:
    """Deterministic: align the stored copies to the stored export and build the measurement record."""
    m = resolved.measurement
    assert m is not None
    x, u = read_played(run_dir / CAPTURES_DIR / PLAYED_FILE)
    fs_d = dataset.signal.sample_rate_hz
    fs_c = m.conditions.sample_rate_hz
    ratio = rate_ratio(fs_c, fs_d) if fs_d is not None else None
    if ratio == (1, 1):
        ratio = None
    aligned: Dict[str, np.ndarray] = {}
    gains: Dict[str, complex] = {}
    captures: List[CaptureAlignment] = []
    for role, ref in zip(_ROLES, (m.with_dpd, m.without_dpd)):
        if ref is None:
            continue
        path = _stored_capture(run_dir, role)
        raw = read_capture(path, ref.columns)
        if raw.size == 0:
            raise MeasurementError(f"{role} capture is empty")
        z = resample(raw, fs_c, fs_d) if ratio is not None else raw
        played = u if role == "with_dpd" else x
        try:
            window, al = align(z, played, x, loop=m.playback == "loop")
        except ValueError as err:
            raise MeasurementError(f"{role} capture: {err}") from None
        aligned[role], gains[role] = window, al.gain
        captures.append(CaptureAlignment(
            role=role, artifact_id=_artifact_id(role), raw_sha256=ref.sha256, n_samples_raw=int(raw.size),
            sample_rate_hz=fs_c, resample_ratio=ratio, delay_samples=al.delay_samples, wrapped=al.wrapped,
            correlation=al.correlation, gain_abs=abs(al.gain),
            gain_db=float(20 * math.log10(abs(al.gain))) if abs(al.gain) > 0 else float("-inf"),
            gain_phase_deg=float(np.degrees(np.angle(al.gain))), rms=rms(window), peak_abs=peak_abs(window),
            declared_output_power_dbm=ref.declared_output_power_dbm))
    without = aligned.get("without_dpd")
    with_ref, without_ref = m.with_dpd, m.without_dpd
    declared = None
    if without_ref is not None and with_ref.declared_output_power_dbm is not None \
            and without_ref.declared_output_power_dbm is not None:
        declared = with_ref.declared_output_power_dbm - without_ref.declared_output_power_dbm
    evidence = MeasurementEvidence(
        attestation=MOCK_ATTESTATION if m.source == "mock_adapter" else ATTESTATION,
        apply_run_id=m.apply_run_id, played_sha256=m.played_sha256,
        conditions=m.conditions, captures=captures,
        level_difference_db=level_db(aligned["with_dpd"], without) if without is not None else None,
        declared_power_difference_db=declared)
    return MeasuredSignals(x=x, u=u, with_dpd=aligned["with_dpd"], without_dpd=without, gains=gains, evidence=evidence)


def execute_measurement(ws: Workspace, run_dir: Path, resolved: ResolvedExperimentConfig) -> MeasuredSignals:
    """The compute phase of an ``evaluate_measured`` run: materialise, align, record."""
    materialise(ws, run_dir, resolved)
    signals = align_stored(run_dir, resolved, ws.get_dataset(resolved.dataset.id))
    write_json_atomic(run_dir / MEASUREMENT_FILE, signals.evidence)
    return signals


def load_signals(ws: Workspace, run_id: str, resolved: ResolvedExperimentConfig) -> MeasuredSignals:
    """Rebuild the aligned signals of a finished run from its stored copies (re-evaluation); the stored record
    must agree with what the alignment produces again."""
    run_dir = ws.run_dir(run_id)
    signals = align_stored(run_dir, resolved, ws.get_dataset(resolved.dataset.id))
    stored = MeasurementEvidence.model_validate(read_json(run_dir / MEASUREMENT_FILE))
    if [c.raw_sha256 for c in stored.captures] != [c.raw_sha256 for c in signals.evidence.captures]:
        raise WorkspaceError(f"run '{run_id}': the stored captures do not match its measurement record")
    return signals


# --- scoring ------------------------------------------------------------------------------------

def _limitations(signals: MeasuredSignals) -> List[str]:
    ev = signals.evidence
    with_ref = ev.captures[0]
    declared = ", ".join(f"{c.role.replace('_', ' ')}: " + (f"{c.declared_output_power_dbm:g} dBm"
                                                             if c.declared_output_power_dbm is not None else "not declared")
                         for c in ev.captures)
    out = [ev.attestation,
           f"no physical calibration: output power is the operator's declaration ({declared}), not measured by OpenDPD; "
           "capture units are the analyser's",
           "integer-sample alignment and one complex gain per capture: residual timing error and receiver impairments "
           "(IQ imbalance, DC offset, phase noise) are scored as distortion",
           "one capture per condition: no repeatability statistics"]
    if ev.level_difference_db is not None and abs(ev.level_difference_db) > LEVEL_TOLERANCE_DB:
        out.append(f"output level with DPD differs from the capture without DPD by {ev.level_difference_db:+.2f} dB "
                   f"(declared powers: {declared}); the difference between the two captures is not attributable to "
                   "the DPD alone")
    if any(c.wrapped for c in ev.captures):
        out.append("the aligned window wraps around the capture file (looped playback assumed)")
    if with_ref.resample_ratio is not None:
        up, down = with_ref.resample_ratio
        out.append(f"captures resampled by {up}/{down} to the dataset rate before scoring")
    return out


def result_for_measured(ws: Workspace, run_id: str, resolved: ResolvedExperimentConfig, manifest: ArtifactManifest,
                        signals: MeasuredSignals, profile_id: str) -> EvaluationResult:
    from opendpd.services.experiments import build_result

    dataset = ws.get_dataset(resolved.dataset.id)
    profile = get_profile(profile_id)
    ev = signals.evidence
    n = int(signals.x.size)
    g = signals.gains["with_dpd"]
    prediction, reference = to_iq(signals.with_dpd), to_iq(g * signals.x)
    metrics = score(profile_id, prediction, reference, dataset.signal, valid_samples=n)
    baselines: List[BaselineScore] = []
    if signals.without_dpd is not None:
        g0 = signals.gains["without_dpd"]
        baselines.append(BaselineScore(
            kind="measured_without_dpd",
            description=f"the same PA driven by x directly (no DPD), aligned with its own delay and least-squares gain "
                        f"|g0| = {abs(g0):.4g} (with DPD |g| = {abs(g):.4g}); level difference "
                        f"{ev.level_difference_db:+.2f} dB is reported, not normalised",
            metrics=score(profile_id, to_iq(signals.without_dpd), to_iq(g0 * signals.x), dataset.signal, valid_samples=n)))
    dpd = resolved.dpd_reference
    with_al = ev.captures[0]
    chain = [
        SignalStage(symbol="x", role="target input: the PA output should equal g * x",
                    source=f"dataset {dataset.dataset_id} version {resolved.dataset.preprocessing_version}, test split "
                           f"(columns I/Q of the played export)", n_samples=n, peak_abs=peak_abs(signals.x), rms=rms(signals.x)),
        SignalStage(symbol="u", role="pre-distorted PA input, u = DPD(x), as played",
                    source=f"DPD {resolved.model.key} weights {(dpd.checkpoint_sha256 or '')[:12]} from run {dpd.run_id}; "
                           f"exported by run_dpd {ev.apply_run_id} (sha256 {ev.played_sha256[:12]})",
                    n_samples=n, peak_abs=peak_abs(signals.u), rms=rms(signals.u), artifact_id="played-signal"),
        SignalStage(symbol="y", role="measured PA output while u was played, aligned (capture units)",
                    source=f"capture {with_al.artifact_id} sha256 {with_al.raw_sha256[:12]}, delay {with_al.delay_samples} "
                           f"samples, correlation {with_al.correlation:.3f}; {ev.attestation}",
                    simulated=False, n_samples=n, peak_abs=with_al.peak_abs, rms=with_al.rms,
                    artifact_id=with_al.artifact_id),
    ]
    scaling = ScalingInfo(amplitude_units="unknown",
                          input_scaling=f"capture units as received from the analyser ({ev.conditions.capture_chain})",
                          reference_gain=abs(g), physical_calibration=False)
    return build_result(ws, run_id, resolved, manifest, profile=profile, metrics=metrics, n_valid=n, target_gain=abs(g),
                        signal_chain=chain, baselines=baselines, scaling=scaling, measurement=ev,
                        limitations=_limitations(signals))


def write_measurement_plots(ws: Workspace, run_id: str, resolved: ResolvedExperimentConfig, signals: MeasuredSignals) -> List[Path]:
    from opendpd.core import plots
    from opendpd.services.evaluation import PLOTS_DIR

    dataset = ws.get_dataset(resolved.dataset.id)
    sig = dataset.signal
    g = signals.gains["with_dpd"]
    x, u = to_iq(signals.x), to_iq(signals.u)
    with_dpd = to_iq(signals.with_dpd)
    all_signals = {"target input x": x, "u = DPD(x) as played": u, "linear target g*x": to_iq(g * signals.x),
                   "measured PA output with DPD": with_dpd}
    roles = {"target input x": "input", "u = DPD(x) as played": "predistorted", "linear target g*x": "reference",
             "measured PA output with DPD": "primary"}
    outputs = {"measured PA output with DPD": with_dpd}
    if signals.without_dpd is not None:
        without = to_iq(signals.without_dpd)
        all_signals["measured PA output without DPD"] = without
        roles["measured PA output without DPD"] = "baseline"
        outputs["measured PA output without DPD"] = without
    n = int(signals.x.size)
    out_dir = ws.run_dir(run_id) / PLOTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, data in (
        ("spectrum", plots.spectrum(all_signals, roles, sample_rate_hz=sig.sample_rate_hz, nperseg=sig.nperseg,
                                    bandwidth_hz=sig.bandwidth_hz, valid_samples=n)),
        ("time", plots.time_excerpt(all_signals, roles, valid_samples=n)),
        ("amam", plots.am_am_pm(x, outputs, roles, valid_samples=n)),
    ):
        write_json_atomic(out_dir / f"{name}.json", data)
        written.append(out_dir / f"{name}.json")
    return written


def evaluate_measured(ws: Workspace, run_id: str, resolved: ResolvedExperimentConfig, manifest: ArtifactManifest,
                      signals: MeasuredSignals) -> Dict[str, EvaluationResult]:
    """Score the aligned captures under every registered profile; the configured profile is the primary result."""
    from opendpd.services.evaluation import store_results

    results = {p.profile_id: result_for_measured(ws, run_id, resolved, manifest, signals, p.profile_id)
               for p in list_profiles()}
    store_results(ws.run_dir(run_id), results, resolved.evaluation.profile_id)
    write_measurement_plots(ws, run_id, resolved, signals)
    return results
