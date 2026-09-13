"""Grouping and statistics for independently declared acquisitions.

Single captures never receive repeatability numbers. Seeds, files and slices
are never treated as independent hardware repeats. All metric values originate
from existing results; sample statistics do not replace metric algorithms.
"""

from collections import defaultdict
import statistics
import uuid

from opendpd.schemas.measurement_session import CaptureReview, MeasurementSession, MeasurementSessionSpec, RepeatStatistic
from opendpd.schemas import TERMINAL_STATUSES
from opendpd.services import experiments
from opendpd.services.review import object_sha
from opendpd.services.workspace import WorkspaceError, read_json, sha256_file, write_json_atomic


def create_session(ws, spec: MeasurementSessionSpec):
    captures, hashes, warnings = [], {}, []
    acquisitions = {}
    raw_ids = {}
    signatures = set()
    for capture in spec.captures:
        record = experiments.load_run(ws, capture.run_id)
        if record.status not in TERMINAL_STATUSES:
            raise WorkspaceError("a capture evaluation must finish before it joins a measurement session")
        result = experiments.load_result(ws, capture.run_id, spec.profile_id)
        if result is None:
            captures.append(CaptureReview(capture=capture, status="failed", reason=record.error.message if record.error else f"no stored result ({record.status.value})"))
            continue
        if not result.measurement:
            raise WorkspaceError(f"{capture.run_id}: a session requires measured-capture evidence")
        if result.is_mock != (spec.source == "mock"):
            raise WorkspaceError("mock and real measurement evidence cannot share a session")
        evidence = result.measurement
        if evidence.conditions.pa != spec.dut:
            raise WorkspaceError(f"{capture.run_id}: declared DUT differs from the session")
        aligned = next((c for c in evidence.captures if c.role == capture.role), None)
        if aligned is None:
            raise WorkspaceError(f"{capture.run_id}: no {capture.role} capture")
        manifest = experiments.load_artifacts(ws, capture.run_id)
        artifact = next((a for a in manifest.artifacts if a.artifact_id == aligned.artifact_id), None) if manifest else None
        if artifact is None:
            raise WorkspaceError(f"{capture.run_id}: raw capture artifact is missing")
        raw_path = ws.run_dir(capture.run_id) / artifact.file.path
        if (not raw_path.resolve().is_relative_to(ws.run_dir(capture.run_id).resolve()) or not raw_path.is_file()
                or sha256_file(raw_path) != aligned.raw_sha256):
            raise WorkspaceError(f"{capture.run_id}: raw capture hash mismatch")
        prior = raw_ids.setdefault(aligned.raw_sha256, capture.acquisition_id)
        if prior != capture.acquisition_id:
            raise WorkspaceError("the same raw capture cannot be declared as independent acquisitions")
        # A single acquisition may be evaluated more than once, but no repeat statistic
        # can select among its seeds or slices. Such duplicates require explicit exclusion.
        key = (capture.acquisition_id, capture.role)
        if not capture.excluded_reason:
            if key in acquisitions:
                raise WorkspaceError("multiple evaluations of one acquisition/role need an explicit exclusion reason")
            acquisitions[key] = aligned.raw_sha256
        metrics = result.metrics if capture.role == "with_dpd" else next((b.metrics for b in result.baselines if b.kind == "measured_without_dpd"), [])
        if not metrics and not capture.excluded_reason:
            raise WorkspaceError("the selected capture has no stored baseline metrics under this profile")
        hashes[capture.run_id] = object_sha(result)
        if not capture.excluded_reason:
            # Freeze the measurement method and model, independently of capture time/gain.
            signatures.add(object_sha({"profile": [result.metric_profile_id, result.metric_profile_version],
                                       "dataset": result.dataset.model_dump(mode="json"),
                                       "models": [m.model_dump(mode="json") for m in result.models],
                                       "reference": result.reference.kind,
                                       "chain": evidence.conditions.capture_chain,
                                       "calibration": evidence.conditions.calibration,
                                       "session_calibration": capture.calibration_id,
                                       "physical_conditions": result.rf_conditions.model_dump(mode="json", exclude={"recorded_at", "note"}) if result.rf_conditions else None,
                                       "temperature": evidence.conditions.temperature_c,
                                       "processing": aligned.processing_version,
                                       "valid_range": result.valid_sample_range,
                                       "rate": evidence.conditions.sample_rate_hz}))
        processing = aligned.model_dump(mode="json")
        processing["version"] = aligned.processing_version
        processing["valid_sample_range"] = result.valid_sample_range
        rate = result.evaluated_signal.sample_rate_hz if result.evaluated_signal else None
        if rate is None:
            ratio = aligned.resample_ratio or (1, 1)
            rate = aligned.sample_rate_hz * ratio[0] / ratio[1]
        processing["delay_ns"] = (aligned.delay_samples + aligned.fractional_delay_samples) * 1e9 / rate
        model_run = next((m.run_id for m in result.models if m.role == "dpd"), None)
        try:
            seed = experiments.load_resolved(ws, model_run).training.seed if model_run else None
        except (WorkspaceError, FileNotFoundError):
            seed = None
        captures.append(CaptureReview(capture=capture, status="excluded" if capture.excluded_reason else "included",
                                     reason=capture.excluded_reason, raw_sha256=aligned.raw_sha256, played_sha256=evidence.played_sha256,
                                     declared_output_power_dbm=aligned.declared_output_power_dbm,
                                     processing=processing, metrics=metrics, seed=seed))
    # Differences remain inspectable but cannot become one repeatability number.
    matching = len(signatures) <= 1
    if not matching:
        warnings.append("Capture methods, calibration, models or conditions differ; repeat statistics are unavailable.")
    included = [c for c in captures if c.status == "included"]
    powers = [c.declared_output_power_dbm for c in included]
    power_ok = bool(powers) and all(p is not None for p in powers) and spec.power_tolerance_db is not None
    if power_ok:
        spread = max(powers) - min(powers)
        power_ok = spread <= spec.power_tolerance_db
        power_matching = f"Declared output-power span {spread:g} dB; protocol tolerance {spec.power_tolerance_db:g} dB: {'matched' if power_ok else 'not matched'}."
    else:
        power_matching = "Power matching unverified: provide capture powers and a protocol tolerance."
    if not power_ok:
        warnings.append("Repeat statistics require declared output powers within the session tolerance; amplitude fitting is not power matching.")
    if not spec.reference_plane or not spec.calibrations or not spec.instruments:
        warnings.append("Reference plane, structured calibration or instrument-chain records are incomplete.")
    repeats = []
    groups = defaultdict(list)
    for c in included:
        for metric in c.metrics:
            if metric.value is not None and metric.status.value == "ok":
                groups[(c.capture.role, metric.name, metric.unit)].append((c, metric.value))
    for (role, name, unit), values in groups.items():
        ordered = sorted(values, key=lambda x: x[0].capture.acquired_at)
        numbers = [v for _, v in ordered]
        n = len(numbers)
        sufficient = n >= 2 and matching and power_ok
        mean = statistics.mean(numbers) if sufficient else None
        std = statistics.stdev(numbers) if sufficient else None
        ci = None
        if sufficient and spec.interval == "student_t_95":
            from scipy.stats import t
            radius = float(t.ppf(.975, n - 1)) * std / n ** .5
            ci = [mean - radius, mean + radius]
        repeats.append(RepeatStatistic(role=role, metric=name, unit=unit, n_independent_captures=n,
                                       n_seeds=len({c.seed for c, _ in ordered}) if all(c.seed is not None for c, _ in ordered) else None,
                                       mean=mean, median=statistics.median(numbers) if sufficient else None, sample_std=std,
                                       first_to_last_drift=numbers[-1] - numbers[0] if sufficient else None, ci95=ci,
                                       method="arithmetic statistics of stored capture metric values (including dB); sample SD with ddof=1; "
                                              + ("Student t 95% CI assumes independent stationary captures; " if ci else "")
                                              + "repeatability only, not total measurement uncertainty"))
    session = MeasurementSession(session_id=f"ms-{uuid.uuid4().hex}", spec=spec, result_hashes=hashes,
                                 captures=captures, repeats=repeats, power_matching=power_matching, warnings=warnings)
    root = ws.root / "measurement-sessions"
    root.mkdir(exist_ok=True)
    write_json_atomic(root / f"{session.session_id}.json", session)
    return session


def list_sessions(ws, run_id=None):
    records = [MeasurementSession.model_validate(read_json(p)) for p in (ws.root / "measurement-sessions").glob("ms-*.json")]
    return sorted((s for s in records if run_id is None or any(c.run_id == run_id for c in s.spec.captures)), key=lambda s: s.created_at, reverse=True)


def load_session(ws, session_id):
    from pydantic import TypeAdapter
    from opendpd.schemas.common import Slug
    TypeAdapter(Slug).validate_python(session_id)
    path = ws.root / "measurement-sessions" / f"{session_id}.json"
    if not path.is_file():
        raise WorkspaceError("measurement session not found")
    session = MeasurementSession.model_validate(read_json(path))
    changed = [run for run, digest in session.result_hashes.items()
               if (result := experiments.load_result(ws, run, session.spec.profile_id)) is None or object_sha(result) != digest]
    if changed:
        session.warnings.append("Stored session snapshot retained; source results changed: " + ", ".join(changed))
    for capture in session.captures:
        if not capture.raw_sha256:
            continue
        run_id = capture.capture.run_id
        manifest = experiments.load_artifacts(ws, run_id)
        artifact_id = f"capture-{capture.capture.role.replace('_', '-')}"
        artifact = next((a for a in manifest.artifacts if a.artifact_id == artifact_id), None) if manifest else None
        path = ws.run_dir(run_id) / artifact.file.path if artifact else None
        if (path is None or not path.resolve().is_relative_to(ws.run_dir(run_id).resolve()) or not path.is_file()
                or sha256_file(path) != capture.raw_sha256):
            session.warnings.append(f"Stored session snapshot retained; raw capture missing or changed: {run_id}/{artifact_id}")
    return session
