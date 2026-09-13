"""Contract tests for opendpd.schemas (S01)."""

import json
import math
from datetime import timedelta

import pytest
from pydantic import ValidationError

from opendpd.schemas import (
    ArtifactManifest,
    BetterDirection,
    DatasetManifest,
    DiagnosticReport,
    EvaluationResult,
    ExperimentConfig,
    FileRef,
    MetricStatus,
    MetricValue,
    RunEvent,
    RunRecord,
    RunStatus,
    TERMINAL_STATUSES,
    can_transition,
    heartbeat_is_stale,
)
from opendpd.schemas.examples import T0, all_examples, run_running


# --- every example validates and survives a JSON round trip ------------------

@pytest.mark.parametrize("name", sorted(all_examples()))
def test_example_round_trips_through_json(name):
    example = all_examples()[name]
    items = example if isinstance(example, list) else [example]
    for item in items:
        payload = item.model_dump_json()
        assert "NaN" not in payload and "Infinity" not in payload
        rebuilt = type(item).model_validate_json(payload)
        assert rebuilt == item


def test_examples_cover_required_scenarios():
    names = set(all_examples())
    assert {"dataset_missing_metadata", "result_metric_not_applicable_mock", "run_interrupted",
            "run_log_disconnected", "result_legacy_import"} <= names


# --- metric values: no silent zeros, no NaN -----------------------------------

def test_metric_ok_requires_finite_value():
    with pytest.raises(ValidationError):
        MetricValue(name="NMSE", value=float("nan"), unit="dB", better=BetterDirection.lower)
    with pytest.raises(ValidationError):
        MetricValue(name="NMSE", value=None, unit="dB", better=BetterDirection.lower)


def test_metric_non_ok_requires_reason_and_no_value():
    with pytest.raises(ValidationError):
        MetricValue(name="EVM", unit="dB", better=BetterDirection.lower, status=MetricStatus.not_applicable)
    with pytest.raises(ValidationError):
        MetricValue(name="EVM", value=1.0, unit="dB", better=BetterDirection.lower,
                    status=MetricStatus.failed, reason="boom")
    ok = MetricValue(name="EVM", unit="dB", better=BetterDirection.lower,
                     status=MetricStatus.missing_reference, reason="no reference")
    assert ok.value is None


# --- file references never escape their container ------------------------------

@pytest.mark.parametrize("bad", ["/etc/passwd", "../x.pt", "a/../b", "a//b", "C:/x", "~/x", "a\\b", "./a"])
def test_fileref_rejects_escaping_paths(bad):
    with pytest.raises(ValidationError):
        FileRef(path=bad)


def test_fileref_accepts_relative_posix():
    assert FileRef(path="save/x/y.pt").path == "save/x/y.pt"


# --- experiment config task rules ---------------------------------------------

def test_train_pa_rejects_wrong_evidence_type():
    cfg = all_examples()["experiment_train_pa_smoke"].model_dump()
    cfg["evaluation"]["evidence_type"] = "dpd_surrogate"
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(cfg)


def test_train_dpd_requires_pa_reference():
    cfg = all_examples()["experiment_train_dpd_smoke"].model_dump()
    cfg["pa_reference"] = None
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(cfg)


def test_unknown_fields_are_rejected():
    cfg = all_examples()["experiment_train_pa_smoke"].model_dump()
    cfg["training"]["lr"] = 0.1  # legacy name; the contract name is learning_rate
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(cfg)


def test_training_defaults_match_legacy_recipe():
    from opendpd.schemas import TrainingConfig
    t = TrainingConfig()
    assert (t.epochs, t.batch_size, t.learning_rate, t.frame_length, t.frame_stride) == (150, 64, 5e-3, 200, 1)
    assert (t.optimizer, t.loss, t.lr_end, t.decay_factor, t.patience) == ("adamw", "l2", 5e-5, 0.5, 5)


# --- run state machine ---------------------------------------------------------

def test_state_machine_matches_plan():
    assert can_transition(RunStatus.queued, RunStatus.running)
    assert can_transition(RunStatus.queued, RunStatus.cancelled)
    assert can_transition(RunStatus.running, RunStatus.cancel_requested)
    assert can_transition(RunStatus.cancel_requested, RunStatus.cancelled)
    assert can_transition(RunStatus.running, RunStatus.interrupted)
    assert not can_transition(RunStatus.cancel_requested, RunStatus.running)
    for terminal in TERMINAL_STATUSES:
        for target in RunStatus:
            assert not can_transition(terminal, target), f"{terminal} must be terminal"
    # a late completion event can never resurrect a cancelled run
    assert not can_transition(RunStatus.cancelled, RunStatus.succeeded)


def test_run_record_lifecycle_rules():
    base = run_running().model_dump()
    base.update(status="failed", finished_at=T0 + timedelta(seconds=5))
    with pytest.raises(ValidationError):   # failed needs an error
        RunRecord.model_validate(base)
    base = run_running().model_dump()
    base.update(status="succeeded")        # terminal needs finished_at
    with pytest.raises(ValidationError):
        RunRecord.model_validate(base)
    base = run_running().model_dump()
    base.update(status="interrupted", finished_at=T0 + timedelta(seconds=5))  # needs reason
    with pytest.raises(ValidationError):
        RunRecord.model_validate(base)


def test_heartbeat_staleness():
    record = run_running()
    assert not heartbeat_is_stale(record, T0 + timedelta(seconds=40), timedelta(seconds=30))
    assert heartbeat_is_stale(record, T0 + timedelta(minutes=5), timedelta(seconds=30))
    done = all_examples()["run_interrupted"]
    assert not heartbeat_is_stale(done, T0 + timedelta(days=1), timedelta(seconds=1))


def test_events_have_increasing_seq():
    events = all_examples()["events_running"]
    seqs = [e.seq for e in events]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)
    with pytest.raises(ValidationError):
        RunEvent(seq=0, run_id="r", ts=T0, type="log")


# --- results carry evidence ----------------------------------------------------

def test_mock_results_are_flagged_consistently():
    res = all_examples()["result_pa_modeling_mock"]
    assert res.is_mock and res.source == "mock"
    data = res.model_dump()
    data["is_mock"] = False
    with pytest.raises(ValidationError):
        EvaluationResult.model_validate(data)


def test_surrogate_result_cannot_pose_as_measured():
    data = all_examples()["result_dpd_surrogate_mock"].model_dump()
    data["evidence_type"] = "dpd_measured"
    data["reference"]["kind"] = "pa_surrogate_output"
    with pytest.raises(ValidationError):
        EvaluationResult.model_validate(data)


def test_surrogate_result_must_state_limitation():
    data = all_examples()["result_dpd_surrogate_mock"].model_dump()
    data["limitations"] = ["MOCK DATA"]
    with pytest.raises(ValidationError):
        EvaluationResult.model_validate(data)


def test_signal_chain_marks_the_pa_output_as_simulated_exactly_for_surrogate_evidence():
    data = all_examples()["result_dpd_surrogate_mock"].model_dump()
    assert [s["symbol"] for s in data["signal_chain"]] == ["x", "u", "y"]
    y = next(s for s in data["signal_chain"] if s["symbol"] == "y")
    y["simulated"] = False
    with pytest.raises(ValidationError, match="simulated exactly for dpd_surrogate"):
        EvaluationResult.model_validate(data)
    data = all_examples()["result_dpd_surrogate_mock"].model_dump()
    data["signal_chain"].append(dict(data["signal_chain"][1]))
    with pytest.raises(ValidationError, match="appears once"):
        EvaluationResult.model_validate(data)


def test_baselines_share_the_metric_set_and_calibration_cannot_be_claimed():
    data = all_examples()["result_dpd_surrogate_mock"].model_dump()
    data["baselines"][0]["metrics"] = data["baselines"][0]["metrics"][:2]
    with pytest.raises(ValidationError, match="same metrics"):
        EvaluationResult.model_validate(data)
    data = all_examples()["result_dpd_surrogate_mock"].model_dump()
    data["scaling"]["physical_calibration"] = True
    with pytest.raises(ValidationError, match="physical calibration"):
        EvaluationResult.model_validate(data)


def test_lineage_example_names_relations_and_weights():
    graph = all_examples()["run_lineage_dpd"]
    assert graph.parents[0].relation.value == "pa_surrogate" and graph.parents[0].checkpoint_sha256
    assert {c.relation.value for c in graph.children} == {"dpd_model"}


def test_legacy_import_must_list_unknowns():
    data = all_examples()["result_legacy_import"].model_dump()
    data["limitations"] = []
    with pytest.raises(ValidationError):
        EvaluationResult.model_validate(data)


def test_not_applicable_metric_has_no_value():
    res = all_examples()["result_metric_not_applicable_mock"]
    evm = res.metric("EVM")
    assert evm.status == MetricStatus.not_applicable and evm.value is None and evm.reason
    assert math.isfinite(res.metric("NMSE").value)


# --- datasets, diagnostics, artifacts ------------------------------------------

def test_missing_metadata_is_reported_not_guessed():
    ds = all_examples()["dataset_missing_metadata"]
    assert ds.missing_metadata() == ["sample_rate_hz", "bandwidth_hz", "n_sub_ch", "nperseg"]
    assert all_examples()["dataset_builtin"].missing_metadata() == []


def test_split_ratios_must_sum_to_one():
    data = all_examples()["dataset_builtin"].model_dump()
    data["split"]["ratios"] = {"train": 0.7, "val": 0.2, "test": 0.2}
    with pytest.raises(ValidationError):
        DatasetManifest.model_validate(data)


def test_diagnostic_blocking_consistency():
    data = all_examples()["diagnostics_missing_metadata"].model_dump()
    data["evaluation_blocked"] = False
    with pytest.raises(ValidationError):
        DiagnosticReport.model_validate(data)
    report = all_examples()["diagnostics_missing_metadata"]
    assert report.counts() == {"error": 1, "warning": 1, "info": 1}


def test_artifact_manifest_complete_requires_hashes():
    data = all_examples()["artifact_manifest_complete"].model_dump()
    data["artifacts"][0]["file"]["sha256"] = None
    with pytest.raises(ValidationError):
        ArtifactManifest.model_validate(data)


def test_mock_export_writes_labelled_files(tmp_path):
    from opendpd.schemas.__main__ import export_mocks
    files = export_mocks(tmp_path)
    assert len(files) == len(all_examples())
    for f in files:
        payload = json.loads(f.read_text())
        assert payload["_mock"] is True and "data" in payload
