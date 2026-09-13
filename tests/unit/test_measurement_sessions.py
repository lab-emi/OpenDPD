"""Acquisition independence, missing-power and statistical boundaries of session records."""

from datetime import timedelta
from types import SimpleNamespace

import pytest

from opendpd.schemas import Artifact, ArtifactManifest, FileRef, RunStatus
from opendpd.schemas.examples import T0, result_dpd_measured_mock
from opendpd.schemas.measurement_session import MeasurementSessionSpec, SessionCapture
from opendpd.services import experiments, measurement_sessions as sessions
from opendpd.services.workspace import Workspace, WorkspaceError, sha256_file


@pytest.fixture
def captures(tmp_path, monkeypatch):
    ws = Workspace.create(tmp_path / "ws")
    results, artifacts = {}, {}
    for i in range(3):
        run = f"run-meas-{i}"
        root = ws.root / "runs" / run
        root.mkdir(parents=True)
        result = result_dpd_measured_mock()
        result.run_id, result.result_id = run, f"res-{i}"
        result.metrics[0].value = -20 + i * 2
        manifest = ArtifactManifest(run_id=run)
        for j, cap in enumerate(result.measurement.captures):
            raw = root / f"raw-{j}.csv"
            raw.write_text(f"synthetic test fixture {i} {j}")
            cap.raw_sha256 = sha256_file(raw)
            manifest.artifacts.append(Artifact(artifact_id=cap.artifact_id, kind="other", file=FileRef(path=raw.name, sha256=cap.raw_sha256)))
        results[run], artifacts[run] = result, manifest
    monkeypatch.setattr(experiments, "load_result", lambda _ws, run, profile=None: results.get(run))
    monkeypatch.setattr(experiments, "load_run", lambda _ws, run: SimpleNamespace(status=RunStatus.succeeded, error=None))
    monkeypatch.setattr(experiments, "load_artifacts", lambda _ws, run: artifacts[run])
    monkeypatch.setattr(experiments, "load_resolved", lambda *_: SimpleNamespace(training=SimpleNamespace(seed=7)))
    return ws, results


def spec(count=3, **overrides):
    value = dict(title="MOCK repeated captures", dut=result_dpd_measured_mock().measurement.conditions.pa,
                 source="mock", profile_id="legacy-opendpd-v1", power_tolerance_db=.1, interval="student_t_95",
                 captures=[SessionCapture(capture_id=f"cap-{i}", acquisition_id=f"acq-{i}", run_id=f"run-meas-{i}",
                                          role="with_dpd", acquired_at=T0 + timedelta(minutes=i)) for i in range(count)])
    value.update(overrides)
    return MeasurementSessionSpec(**value)


def test_repeat_statistics_have_declared_count_units_method_and_separate_seed_count(captures):
    ws, _ = captures
    saved = sessions.create_session(ws, spec())
    nmse = next(r for r in saved.repeats if r.metric == "NMSE")
    assert nmse.n_independent_captures == 3 and nmse.n_seeds == 1
    assert nmse.unit == "dB" and nmse.mean == -18 and nmse.median == -18 and nmse.sample_std == 2
    assert nmse.first_to_last_drift == 4
    assert nmse.ci95[0] < -18 < nmse.ci95[1]
    assert "not total measurement uncertainty" in nmse.method and "Student t" in nmse.method
    assert sessions.load_session(ws, saved.session_id).spec == saved.spec


def test_single_capture_does_not_get_repeatability_numbers(captures):
    ws, _ = captures
    saved = sessions.create_session(ws, spec(1))
    assert all(r.n_independent_captures == 1 and r.mean is None and r.sample_std is None and r.ci95 is None for r in saved.repeats)


def test_same_capture_cannot_be_renamed_into_independent_repeats(captures):
    ws, results = captures
    a, b = results["run-meas-0"], results["run-meas-1"]
    src = ws.run_dir(a.run_id) / "raw-0.csv"
    dst = ws.run_dir(b.run_id) / "raw-0.csv"
    dst.write_bytes(src.read_bytes())
    b.measurement.captures[0].raw_sha256 = a.measurement.captures[0].raw_sha256
    with pytest.raises(WorkspaceError, match="same raw capture"):
        sessions.create_session(ws, spec(2))


def test_two_segments_or_evaluations_of_one_acquisition_require_exclusion(captures):
    ws, _ = captures
    s = spec(2)
    s.captures[1].acquisition_id = s.captures[0].acquisition_id
    with pytest.raises(WorkspaceError, match="explicit exclusion"):
        sessions.create_session(ws, s)
    s.captures[1].excluded_reason = "second slice of the same physical acquisition"
    saved = sessions.create_session(ws, s)
    assert saved.captures[1].status == "excluded"
    assert all(r.n_independent_captures == 1 and r.sample_std is None for r in saved.repeats)


def test_power_mismatch_and_different_calibration_prevent_aggregation(captures):
    ws, results = captures
    results["run-meas-1"].measurement.captures[0].declared_output_power_dbm = 25
    saved = sessions.create_session(ws, spec(2))
    assert "not matched" in saved.power_matching
    assert all(r.mean is None for r in saved.repeats)
    results["run-meas-1"].measurement.captures[0].declared_output_power_dbm = 30
    results["run-meas-1"].measurement.conditions.calibration = "different calibration"
    saved = sessions.create_session(ws, spec(2))
    assert all(r.mean is None for r in saved.repeats)
    assert any("methods, calibration" in warning for warning in saved.warnings)


def test_no_tolerance_and_missing_power_never_imply_matching(captures):
    ws, results = captures
    for s in (spec(2, power_tolerance_db=None), spec(2)):
        results["run-meas-1"].measurement.captures[0].declared_output_power_dbm = None
        saved = sessions.create_session(ws, s)
        assert "unverified" in saved.power_matching and all(r.sample_std is None for r in saved.repeats)


def test_mock_data_cannot_enter_a_measured_session_and_raw_tampering_is_refused(captures):
    ws, _ = captures
    with pytest.raises(WorkspaceError, match="mock and real"):
        sessions.create_session(ws, spec(2, source="measured"))
    (ws.run_dir("run-meas-0") / "raw-0.csv").write_text("tampered")
    with pytest.raises(WorkspaceError, match="raw capture hash mismatch"):
        sessions.create_session(ws, spec(2))


def test_changed_results_leave_the_saved_session_snapshot_intact(captures):
    ws, results = captures
    saved = sessions.create_session(ws, spec(2))
    results["run-meas-0"].metrics[0].value = -100
    restored = sessions.load_session(ws, saved.session_id)
    assert restored.repeats == saved.repeats
    assert any("source results changed" in warning for warning in restored.warnings)


def test_stored_session_warns_if_raw_data_changes_after_creation(captures):
    ws, _ = captures
    saved = sessions.create_session(ws, spec())
    (ws.run_dir('run-meas-0') / 'raw-0.csv').write_text('changed after session')
    loaded = sessions.load_session(ws, saved.session_id)
    assert loaded.repeats == saved.repeats
    assert any('raw capture missing or changed' in w for w in loaded.warnings)
