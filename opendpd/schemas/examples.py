"""Contract examples.

They serve three purposes: documentation of the vocabulary, unit-test inputs,
and the fixed **mock** fixtures the frontend uses before real endpoints exist.
Every result produced here is marked ``source="mock"`` so it can never be
exported as a scientific result. Timestamps are fixed for reproducible files.

    python -m opendpd.schemas export-mocks --out frontend/mocks
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Dict

from .artifacts import Artifact, ArtifactKind, ArtifactManifest
from .common import (
    BetterDirection,
    EvidenceType,
    FileRef,
    MetricStatus,
    MetricValue,
    Severity,
    SoftwareProvenance,
)
from .dataset import DatasetManifest, DatasetOrigin, DatasetSource, DatasetSourceKind, SignalSpec, SplitSpec
from .diagnostics import DiagnosticItem, DiagnosticReport
from .experiment import (
    DatasetRef,
    DPDReference,
    EvaluationConfig,
    ExecutionConfig,
    ExperimentConfig,
    ModelSpec,
    PAReference,
    ResolutionInfo,
    ResolvedExperimentConfig,
    TaskType,
    TrainingConfig,
)
from .measurement import MOCK_ATTESTATION, CaptureAlignment, CaptureRef, MeasurementConditions, MeasurementConfig, MeasurementEvidence
from .metrics import MetricProfile
from .results import (
    BaselineScore,
    ComparisonPair,
    ComparisonReport,
    DatasetEvidence,
    EvaluationResult,
    HistoryPoint,
    ModelEvidence,
    ScalingInfo,
    SignalReference,
    SignalStage,
    SurrogateCoverage,
)
from .run import LineageLink, LineageRelation, RunError, RunEvent, RunEventType, RunLineage, RunRecord, RunStatus, WorkerInfo

T0 = datetime(2026, 9, 6, 8, 0, 0, tzinfo=timezone.utc)
SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64

SOFTWARE = SoftwareProvenance(
    opendpd_version="2.2.0.dev0", python_version="3.13.14", platform="Linux-x86_64",
    torch_version="2.13.0+cu132", git_commit="6f026b1", git_dirty=False,
)


def legacy_metric_profile() -> MetricProfile:
    """Descriptor of the historical OpenDPD metrics (values frozen in tests/golden)."""
    from opendpd.core.metrics.legacy_v1 import PROFILE
    return PROFILE


def general_metric_profile() -> MetricProfile:
    """Pooled NMSE, in-band error and ACPR with explicit conventions (S08)."""
    from opendpd.core.metrics.general_v1 import PROFILE
    return PROFILE


def ofdm_metric_profile() -> MetricProfile:
    """Data-aided EVM and ACLR on captures bound to the ofdm-lte20-v1 waveform (S15); pending cross-validation."""
    from opendpd.core.metrics.ofdm_evm_v1 import PROFILE
    return PROFILE


def dataset_builtin() -> DatasetManifest:
    return DatasetManifest(
        dataset_id="dpa-200mhz",
        display_name="DPA_200MHz (built-in, measured)",
        origin=DatasetOrigin.measured,
        source=DatasetSource(kind=DatasetSourceKind.builtin, name="DPA_200MHz", imported_at=T0),
        signal=SignalSpec(sample_rate_hz=800e6, bandwidth_hz=200e6, sub_channel_bandwidth_hz=20e6,
                          n_sub_ch=10, nperseg=2560, modulation="64QAM", standard="LTE",
                          amplitude_units="normalized"),
        files=[FileRef(path="raw/train_input.csv", sha256=SHA_A, size_bytes=1_000_000),
               FileRef(path="raw/train_output.csv", sha256=SHA_B, size_bytes=1_000_000)],
        n_samples=38400,
        split=SplitSpec(ratios={"train": 0.6, "val": 0.2, "test": 0.2},
                        boundaries={"train": (0, 23040), "val": (23040, 30720), "test": (30720, 38400)}),
        raw_sha256=SHA_C,
    )


def dataset_missing_metadata() -> DatasetManifest:
    """A user CSV without sample rate / bandwidth: importable, not evaluable."""
    return DatasetManifest(
        dataset_id="my-pa-capture",
        display_name="my_pa_capture.csv",
        origin=DatasetOrigin.unknown,
        source=DatasetSource(kind=DatasetSourceKind.csv_import, original_path="/home/user/my_pa_capture.csv",
                             imported_at=T0),
        signal=SignalSpec(),
        files=[FileRef(path="raw/my_pa_capture.csv", sha256=SHA_A, size_bytes=52_000)],
        n_samples=4096,
        columns={"input_i": "I_in", "input_q": "Q_in", "output_i": "I_out", "output_q": "Q_out"},
        split=SplitSpec(ratios={"train": 0.6, "val": 0.2, "test": 0.2}),
        raw_sha256=SHA_A,
    )


def diagnostics_missing_metadata() -> DiagnosticReport:
    return DiagnosticReport(
        report_id="diag-my-pa-capture-1",
        dataset_id="my-pa-capture",
        dataset_raw_sha256=SHA_A,
        generated_at=T0,
        evaluation_blocked=True,
        items=[
            DiagnosticItem(code="missing_metadata", severity=Severity.error, blocking=True,
                           title="Sample rate and bandwidth unknown",
                           message="ACLR and spectral EVM need sample_rate_hz, bandwidth_hz, n_sub_ch and nperseg.",
                           evidence={"missing": ["sample_rate_hz", "bandwidth_hz", "n_sub_ch", "nperseg"]},
                           suggestion="Enter the signal parameters in the dataset manifest, then re-run the Doctor."),
            DiagnosticItem(code="delay_estimate", severity=Severity.info,
                           title="Output lags input by about 3 samples",
                           message="Cross-correlation peak at +3 samples (fractional refinement +0.12).",
                           evidence={"integer_delay": 3, "fractional_delay": 0.12, "peak_ratio": 0.93},
                           confidence=0.85,
                           suggestion="Review and apply the alignment as a new preprocessing version."),
            DiagnosticItem(code="possible_clipping", severity=Severity.warning,
                           title="Flat-top samples detected",
                           message="0.4% of output samples sit within 0.1% of the maximum amplitude.",
                           evidence={"fraction_at_max": 0.004, "max_amplitude": 0.998},
                           confidence=0.6,
                           suggestion="Natural PA compression can look similar; check the capture range."),
        ],
    )


def experiment_train_pa_smoke() -> ExperimentConfig:
    return ExperimentConfig(
        task=TaskType.train_pa,
        recipe_id="pa-gru-smoke-v1",
        name="PA GRU smoke",
        dataset=DatasetRef(id="dpa-200mhz"),
        model=ModelSpec(key="gru", parameters={"hidden_size": 23, "num_layers": 1}),
        training=TrainingConfig(epochs=3, frame_length=50, frame_stride=16, batch_size_eval=256),
        evaluation=EvaluationConfig(evidence_type=EvidenceType.pa_modeling),
        execution=ExecutionConfig(device="cpu"),
        notes="Smoke recipe: 3 epochs. Not a research benchmark.",
    )


def experiment_train_dpd_smoke() -> ExperimentConfig:
    return ExperimentConfig(
        task=TaskType.train_dpd,
        recipe_id="dpd-gru-smoke-v1",
        dataset=DatasetRef(id="dpa-200mhz"),
        model=ModelSpec(key="gru", parameters={"hidden_size": 15, "num_layers": 1}),
        training=TrainingConfig(epochs=3, frame_length=50, frame_stride=16, batch_size_eval=256),
        evaluation=EvaluationConfig(evidence_type=EvidenceType.dpd_surrogate),
        pa_reference=PAReference(run_id="run-pa-0001"),
    )


def experiment_run_dpd() -> ExperimentConfig:
    return ExperimentConfig(
        task=TaskType.run_dpd,
        dataset=DatasetRef(id="dpa-200mhz"),
        model=ModelSpec(key="gru", parameters={"hidden_size": 15, "num_layers": 1}),
        evaluation=EvaluationConfig(evidence_type=EvidenceType.dpd_surrogate),
        pa_reference=PAReference(run_id="run-pa-0001"),
        dpd_reference=DPDReference(run_id="run-dpd-0001"),
    )


def measurement_conditions() -> MeasurementConditions:
    return MeasurementConditions(pa="example GaN Doherty PA, unit 2", capture_chain="SMW200A -> PA -> 30 dB pad -> FSW (I/Q analyser)",
                                 sample_rate_hz=800e6, drive="generator -12 dBm, PA input +8 dBm", gain_db=28.5,
                                 calibration="none", measured_at=T0, temperature_c=25.0, operator="example operator")


def experiment_evaluate_measured() -> ExperimentConfig:
    return ExperimentConfig(
        task=TaskType.evaluate_measured,
        dataset=DatasetRef(id="dpa-200mhz"),
        model=ModelSpec(key="gru", parameters={"hidden_size": 15, "num_layers": 1}),
        evaluation=EvaluationConfig(evidence_type=EvidenceType.dpd_measured),
        dpd_reference=DPDReference(run_id="run-dpd-0001"),
        measurement=MeasurementConfig(apply_run_id="run-apply-0001",
                                      with_dpd=CaptureRef(path="uploads/with_dpd.npy", declared_output_power_dbm=30.0),
                                      without_dpd=CaptureRef(path="uploads/without_dpd.npy", declared_output_power_dbm=30.0),
                                      conditions=measurement_conditions()),
    )


def resolved_train_pa_smoke() -> ResolvedExperimentConfig:
    base = experiment_train_pa_smoke().model_dump()
    base["evaluation"]["checkpoint_selection_metric"] = "NMSE"
    base["model"]["parameters"] = {"hidden_size": 23, "num_layers": 1}
    return ResolvedExperimentConfig(
        **base,
        resolution=ResolutionInfo(resolver_version="config-resolver-v1", resolved_at=T0,
                                  defaults_source="opendpd 2.2.0.dev0 / schema 1 / registry gru",
                                  config_sha256=SHA_B),
    )


def run_queued() -> RunRecord:
    return RunRecord(run_id="run-pa-0002", task=TaskType.train_pa, name="PA GRU smoke", dataset_id="dpa-200mhz",
                     model_key="gru", status=RunStatus.queued, created_at=T0, config_sha256=SHA_B, device="cpu",
                     idempotency_key="gui-7f3e1c")


def run_running() -> RunRecord:
    return RunRecord(run_id="run-pa-0001", task=TaskType.train_pa, name="PA GRU smoke", dataset_id="dpa-200mhz",
                     model_key="gru", status=RunStatus.running, created_at=T0,
                     started_at=T0 + timedelta(seconds=2), config_sha256=SHA_B, device="cpu",
                     worker=WorkerInfo(pid=41235, create_time=1788000002.5, host="lab-laptop"),
                     last_event_seq=17, last_heartbeat_at=T0 + timedelta(seconds=32),
                     progress_epoch=1, progress_total_epochs=3)


def run_log_disconnected() -> RunRecord:
    """Server-side truth for the 'log stream disconnected' UI state: the run
    is still running but the heartbeat is old; the UI must say *stale*, not
    *running fine*, and must not invent progress."""
    record = run_running()
    return record.model_copy(update={"last_heartbeat_at": T0 + timedelta(seconds=32),
                                     "last_event_seq": 17})


def run_interrupted() -> RunRecord:
    return RunRecord(run_id="run-pa-0000", task=TaskType.train_pa, dataset_id="dpa-200mhz", model_key="gru",
                     status=RunStatus.interrupted,
                     status_reason="service restarted while the run was in progress; the worker was not found alive",
                     created_at=T0 - timedelta(hours=1), started_at=T0 - timedelta(hours=1),
                     finished_at=T0, config_sha256=SHA_B, device="cpu", last_event_seq=41)


def run_failed() -> RunRecord:
    return RunRecord(run_id="run-dpd-0009", task=TaskType.train_dpd, dataset_id="dpa-200mhz", model_key="gru",
                     status=RunStatus.failed, created_at=T0, started_at=T0 + timedelta(seconds=1),
                     finished_at=T0 + timedelta(seconds=4), config_sha256=SHA_C, device="cpu", exit_code=3,
                     error=RunError(code="pa_checkpoint_missing", stage="prepare",
                                    message="PA surrogate checkpoint for run-pa-0007 is not registered.",
                                    hint="Train a PA model first or pick a run whose checkpoint exists."))


def events_running() -> list[RunEvent]:
    r = "run-pa-0001"
    return [
        RunEvent(seq=1, run_id=r, ts=T0 + timedelta(seconds=2), type=RunEventType.status,
                 payload={"from": "queued", "to": "running"}),
        RunEvent(seq=2, run_id=r, ts=T0 + timedelta(seconds=3), type=RunEventType.log,
                 payload={"level": "info", "line": "::: Number of PA Model Parameters:  1911"}),
        RunEvent(seq=3, run_id=r, ts=T0 + timedelta(seconds=12), type=RunEventType.progress,
                 payload={"epoch": 0, "total_epochs": 3, "phase": "train"}),
        RunEvent(seq=4, run_id=r, ts=T0 + timedelta(seconds=15), type=RunEventType.metric,
                 payload={"epoch": 0, "split": "val", "values": {"NMSE": -16.09, "EVM": -16.73, "ACLR_AVG": -28.07}}),
        RunEvent(seq=5, run_id=r, ts=T0 + timedelta(seconds=15), type=RunEventType.checkpoint,
                 payload={"epoch": 0, "metric": "NMSE", "value": -16.09,
                          "path": "save/DPA_200MHz/train_pa/PA_S_0_M_GRU_H_23_F_50_P_1911.pt"}),
        RunEvent(seq=6, run_id=r, ts=T0 + timedelta(seconds=30), type=RunEventType.heartbeat, payload={}),
    ]


def _metrics_ok() -> list[MetricValue]:
    lower = BetterDirection.lower
    return [
        MetricValue(name="NMSE", value=-16.092247, unit="dB", better=lower),
        MetricValue(name="EVM", value=-16.731844, unit="dB", better=lower),
        MetricValue(name="ACLR_L", value=-29.788186, unit="dBc", better=lower),
        MetricValue(name="ACLR_R", value=-26.347937, unit="dBc", better=lower),
        MetricValue(name="ACLR_AVG", value=-28.068062, unit="dBc", better=lower),
    ]


def result_pa_modeling_mock() -> EvaluationResult:
    return EvaluationResult(
        result_id="res-pa-0001", run_id="run-pa-0001", generated_at=T0 + timedelta(minutes=1),
        source="mock", is_mock=True,
        evidence_type=EvidenceType.pa_modeling, metric_profile_id="legacy-opendpd-v1", metric_profile_version=1,
        dataset=DatasetEvidence(dataset_id="dpa-200mhz", split="test", raw_sha256=SHA_C,
                                preprocessing_version="raw-v1", split_version="contiguous-v1", n_samples=7680),
        models=[ModelEvidence(role="pa", model=ModelSpec(key="gru", parameters={"hidden_size": 23, "num_layers": 1}),
                              run_id="run-pa-0001", weights_sha256=SHA_A, n_parameters=1911)],
        reference=SignalReference(kind="measured_pa_output", description="measured PA output of the test split"),
        n_segments=3, nperseg=2560, valid_sample_range=(0, 7680),
        metrics=_metrics_ok(), selected_epoch=1,
        history=FileRef(path="log/DPA_200MHz/train_pa/history/PA_S_0_M_GRU_H_23_F_50_P_1911.csv"),
        software=SOFTWARE, device="cpu", seed=0,
        limitations=["MOCK DATA for UI development; not a computed result", "3-epoch smoke recipe"],
    )


def result_metric_not_applicable_mock() -> EvaluationResult:
    """Sample rate unknown: NMSE is still defined, spectral metrics are not."""
    lower = BetterDirection.lower
    reason = "sample_rate_hz, bandwidth_hz, n_sub_ch and nperseg are unknown for dataset my-pa-capture"
    return EvaluationResult(
        result_id="res-pa-0002", run_id="run-pa-0003", generated_at=T0, source="mock", is_mock=True,
        evidence_type=EvidenceType.pa_modeling, metric_profile_id="legacy-opendpd-v1", metric_profile_version=1,
        dataset=DatasetEvidence(dataset_id="my-pa-capture", split="test", raw_sha256=SHA_A,
                                preprocessing_version="raw-v1", split_version="contiguous-v1", n_samples=820),
        models=[ModelEvidence(role="pa", model=ModelSpec(key="gru", parameters={"hidden_size": 8}), run_id="run-pa-0003")],
        reference=SignalReference(kind="measured_pa_output", description="measured output column of the CSV"),
        metrics=[
            MetricValue(name="NMSE", value=-21.4, unit="dB", better=lower),
            MetricValue(name="EVM", unit="dB", better=lower, status=MetricStatus.not_applicable, reason=reason),
            MetricValue(name="ACLR_L", unit="dBc", better=lower, status=MetricStatus.not_applicable, reason=reason),
            MetricValue(name="ACLR_R", unit="dBc", better=lower, status=MetricStatus.not_applicable, reason=reason),
            MetricValue(name="ACLR_AVG", unit="dBc", better=lower, status=MetricStatus.not_applicable, reason=reason),
        ],
        software=SOFTWARE, device="cpu", seed=0,
        limitations=["MOCK DATA for UI development", "signal metadata missing: spectral metrics not applicable"],
    )


def result_dpd_surrogate_mock() -> EvaluationResult:
    lower = BetterDirection.lower
    return EvaluationResult(
        result_id="res-dpd-0001", run_id="run-dpd-0001", generated_at=T0, source="mock", is_mock=True,
        evidence_type=EvidenceType.dpd_surrogate, metric_profile_id="legacy-opendpd-v1", metric_profile_version=1,
        dataset=DatasetEvidence(dataset_id="dpa-200mhz", split="test", raw_sha256=SHA_C,
                                preprocessing_version="raw-v1", split_version="contiguous-v1", n_samples=7680),
        models=[
            ModelEvidence(role="dpd", model=ModelSpec(key="gru", parameters={"hidden_size": 15}), run_id="run-dpd-0001",
                          weights_sha256=SHA_B, n_parameters=887),
            ModelEvidence(role="pa", model=ModelSpec(key="gru", parameters={"hidden_size": 23}), run_id="run-pa-0001",
                          weights_sha256=SHA_A, n_parameters=1911),
        ],
        reference=SignalReference(kind="linear_gain_target", description="target = gain * input",
                                  gain_rule="max|y_train| / max|x_train| (legacy set_target_gain)", gain_value=1.0273),
        n_segments=3, nperseg=2560,
        metrics=[
            MetricValue(name="NMSE", value=-14.99, unit="dB", better=lower),
            MetricValue(name="EVM", value=-16.25, unit="dB", better=lower),
            MetricValue(name="ACLR_L", value=-25.99, unit="dBc", better=lower),
            MetricValue(name="ACLR_R", value=-27.65, unit="dBc", better=lower),
            MetricValue(name="ACLR_AVG", value=-26.82, unit="dBc", better=lower),
        ],
        software=SOFTWARE, device="cpu", seed=0,
        limitations=["MOCK DATA for UI development",
                     "simulated through the learned PA surrogate run-pa-0001, not a measured PA output",
                     "no physical calibration: absolute output power (dBm) and efficiency are not derived"],
        signal_chain=[
            SignalStage(symbol="x", role="target input: the PA output should equal reference_gain * x",
                        source="dataset dpa-200mhz version raw-v1, test split", n_samples=7680, peak_abs=0.842, rms=0.301),
            SignalStage(symbol="u", role="pre-distorted PA input, u = DPD(x)",
                        source=f"DPD gru weights {SHA_B[:12]} from run run-dpd-0001", n_samples=7680, peak_abs=0.913, rms=0.318),
            SignalStage(symbol="y", role="PA output, y = PA(u)",
                        source=f"PA surrogate gru weights {SHA_A[:12]} from run run-pa-0001; simulated, not measured",
                        simulated=True, n_samples=7680, peak_abs=0.861, rms=0.309),
        ],
        baselines=[
            BaselineScore(kind="surrogate_without_dpd",
                          description="PA surrogate run-pa-0001 driven by x directly (no DPD), scored against the same "
                                      "linear target reference_gain * x",
                          metrics=[MetricValue(name="NMSE", value=-9.81, unit="dB", better=lower),
                                   MetricValue(name="EVM", value=-11.02, unit="dB", better=lower),
                                   MetricValue(name="ACLR_L", value=-19.40, unit="dBc", better=lower),
                                   MetricValue(name="ACLR_R", value=-20.11, unit="dBc", better=lower),
                                   MetricValue(name="ACLR_AVG", value=-19.75, unit="dBc", better=lower)]),
            BaselineScore(kind="measured_without_dpd",
                          description="measured PA output of the test split (no DPD), scored against the same linear target",
                          metrics=[MetricValue(name="NMSE", value=-9.63, unit="dB", better=lower),
                                   MetricValue(name="EVM", value=-10.88, unit="dB", better=lower),
                                   MetricValue(name="ACLR_L", value=-19.02, unit="dBc", better=lower),
                                   MetricValue(name="ACLR_R", value=-19.77, unit="dBc", better=lower),
                                   MetricValue(name="ACLR_AVG", value=-19.39, unit="dBc", better=lower)]),
        ],
        surrogate_coverage=SurrogateCoverage(
            fitted_peak_abs=0.858, u_peak_abs=0.913, fraction_above_fitted_peak=0.0031,
            note="0.31% of the pre-distorted samples exceed the largest input amplitude the surrogate was fitted on "
                 "(0.858); the surrogate extrapolates there and y is unverified for those samples. Staying inside the "
                 "range would not prove the surrogate accurate either."),
        scaling=ScalingInfo(amplitude_units="normalized", input_scaling="none: amplitudes as imported",
                            reference_gain=1.0273, physical_calibration=False),
    )


def comparison_report_mock() -> ComparisonReport:
    """Two results under different protocols: shown side by side, never ranked."""
    a = result_pa_modeling_mock()
    b = result_dpd_surrogate_mock()
    from opendpd.core.metrics import incompatibilities
    return ComparisonReport(results=[a, b], comparable=False, generated_at=T0,
                            pairs=[ComparisonPair(a=a.run_id, b=b.run_id, incompatibilities=incompatibilities(a, b))],
                            note="results differ in protocol; they are shown side by side with the differences and "
                                 "must not be ranked")


def history_points_mock() -> list:
    return [HistoryPoint(epoch=e, split=split, train_loss=0.5 / (e + 1),
                         values={"NMSE": -20.0 - 3 * e - (0.4 if split == "test" else 0.0),
                                 "ACLR_AVG": -30.0 - 2 * e})
            for e in range(3) for split in ("val", "test")]


def run_lineage_dpd() -> RunLineage:
    """The graph around a DPD run: trained through a PA surrogate, applied twice (once through another surrogate)."""
    return RunLineage(
        run_id="run-dpd-0001",
        parents=[LineageLink(run_id="run-pa-0001", relation=LineageRelation.pa_surrogate, task=TaskType.train_pa,
                             status=RunStatus.succeeded, checkpoint_sha256=SHA_A)],
        children=[LineageLink(run_id="run-apply-0001", relation=LineageRelation.dpd_model, task=TaskType.run_dpd,
                              status=RunStatus.succeeded, checkpoint_sha256=SHA_B),
                  LineageLink(run_id="run-apply-0002", relation=LineageRelation.dpd_model, task=TaskType.run_dpd,
                              status=RunStatus.running, checkpoint_sha256=SHA_B)],
    )


def result_dpd_measured_mock() -> EvaluationResult:
    lower = BetterDirection.lower
    evidence = MeasurementEvidence(
        attestation=MOCK_ATTESTATION, apply_run_id="run-apply-0001", played_sha256=SHA_A,
        conditions=measurement_conditions(),
        captures=[CaptureAlignment(role="with_dpd", artifact_id="capture-with-dpd", raw_sha256=SHA_B, n_samples_raw=15360,
                                   sample_rate_hz=800e6, delay_samples=123, correlation=0.9998, gain_abs=2.31, gain_db=7.28,
                                   gain_phase_deg=22.5, rms=0.855, peak_abs=2.18, declared_output_power_dbm=30.0),
                  CaptureAlignment(role="without_dpd", artifact_id="capture-without-dpd", raw_sha256=SHA_C, n_samples_raw=15360,
                                   sample_rate_hz=800e6, delay_samples=123, correlation=0.9996, gain_abs=2.87, gain_db=9.16,
                                   gain_phase_deg=22.6, rms=1.058, peak_abs=2.45, declared_output_power_dbm=30.0)],
        level_difference_db=-1.85, declared_power_difference_db=0.0)
    return EvaluationResult(
        result_id="res-meas-0001", run_id="run-meas-0001", generated_at=T0, source="mock", is_mock=True,
        evidence_type=EvidenceType.dpd_measured, metric_profile_id="legacy-opendpd-v1", metric_profile_version=1,
        dataset=DatasetEvidence(dataset_id="dpa-200mhz", split="test", raw_sha256=SHA_C,
                                preprocessing_version="raw-v1", split_version="contiguous-v1", n_samples=15360),
        models=[ModelEvidence(role="dpd", model=ModelSpec(key="gru", parameters={"hidden_size": 15}), run_id="run-dpd-0001",
                              weights_sha256=SHA_B, n_parameters=887, training_path="gradient_dla")],
        reference=SignalReference(kind="linear_gain_target",
                                  description="target = g * x with g the complex least-squares gain of the aligned measured "
                                              "output onto x (capture units); the PA output is measured, not simulated",
                                  gain_rule="least-squares complex gain of the aligned capture onto x, per capture", gain_value=2.31),
        n_segments=6, nperseg=2560,
        metrics=[
            MetricValue(name="NMSE", value=-18.70, unit="dB", better=lower),
            MetricValue(name="EVM", value=-19.43, unit="dB", better=lower),
            MetricValue(name="ACLR_L", value=-29.12, unit="dBc", better=lower),
            MetricValue(name="ACLR_R", value=-29.87, unit="dBc", better=lower),
            MetricValue(name="ACLR_AVG", value=-29.50, unit="dBc", better=lower),
        ],
        software=SOFTWARE, device="cpu", seed=0,
        limitations=["MOCK DATA for UI development", MOCK_ATTESTATION,
                     "no physical calibration: output power is the operator's declaration (with dpd: 30 dBm, without dpd: 30 dBm), "
                     "not measured by OpenDPD; capture units are the analyser's",
                     "output level with DPD differs from the capture without DPD by -1.85 dB (declared powers: with dpd: 30 dBm, "
                     "without dpd: 30 dBm); the difference between the two captures is not attributable to the DPD alone"],
        signal_chain=[
            SignalStage(symbol="x", role="target input: the PA output should equal g * x",
                        source="dataset dpa-200mhz version raw-v1, test split (columns I/Q of the played export)",
                        n_samples=15360, peak_abs=0.842, rms=0.301),
            SignalStage(symbol="u", role="pre-distorted PA input, u = DPD(x), as played",
                        source=f"DPD gru weights {SHA_B[:12]} from run run-dpd-0001; exported by run_dpd run-apply-0001 (sha256 {SHA_A[:12]})",
                        n_samples=15360, peak_abs=0.913, rms=0.318, artifact_id="played-signal"),
            SignalStage(symbol="y", role="measured PA output while u was played, aligned (capture units)",
                        source=f"capture capture-with-dpd sha256 {SHA_B[:12]}, delay 123 samples, correlation 0.9998; {MOCK_ATTESTATION}",
                        simulated=False, n_samples=15360, peak_abs=2.18, rms=0.855, artifact_id="capture-with-dpd"),
        ],
        baselines=[BaselineScore(kind="measured_without_dpd",
                                 description="the same PA driven by x directly (no DPD), aligned with its own delay and least-squares "
                                             "gain |g0| = 2.87 (with DPD |g| = 2.31); level difference -1.85 dB is reported, not normalised",
                                 metrics=[MetricValue(name="NMSE", value=-12.40, unit="dB", better=lower),
                                          MetricValue(name="EVM", value=-13.11, unit="dB", better=lower),
                                          MetricValue(name="ACLR_L", value=-22.05, unit="dBc", better=lower),
                                          MetricValue(name="ACLR_R", value=-22.60, unit="dBc", better=lower),
                                          MetricValue(name="ACLR_AVG", value=-22.33, unit="dBc", better=lower)])],
        scaling=ScalingInfo(amplitude_units="unknown", input_scaling="capture units as received from the analyser (SMW200A -> PA -> 30 dB pad -> FSW (I/Q analyser))",
                            reference_gain=2.31, physical_calibration=False),
        measurement=evidence,
    )


def result_legacy_import() -> EvaluationResult:
    """A row imported from an old log/<dataset>/.../best/*.csv: provenance unknown."""
    lower = BetterDirection.lower
    return EvaluationResult(
        result_id="res-legacy-0001", run_id=None, generated_at=T0, source="legacy-log-import",
        evidence_type=EvidenceType.pa_modeling, metric_profile_id="legacy-opendpd-v1", metric_profile_version=1,
        dataset=DatasetEvidence(dataset_id="dpa-200mhz", split="test", preprocessing_version="raw-v1",
                                split_version="contiguous-v1"),
        models=[ModelEvidence(role="pa", model=ModelSpec(key="gru", parameters={"hidden_size": 23}), n_parameters=1911)],
        reference=SignalReference(kind="measured_pa_output", description="measured PA output (assumed from log layout)"),
        metrics=[MetricValue(name="NMSE", value=-15.75, unit="dB", better=lower),
                 MetricValue(name="ACLR_AVG", value=-27.44, unit="dBc", better=lower)],
        selected_epoch=1,
        software=SoftwareProvenance(opendpd_version="unknown", python_version="unknown", platform="unknown"),
        device="unknown",
        limitations=["imported from a legacy CSV log: software version, git commit, device and data hashes unknown",
                     "EVM column present in the log but its exact profile version could not be verified"],
    )


def artifact_manifest_complete() -> ArtifactManifest:
    return ArtifactManifest(
        run_id="run-pa-0001", complete=True,
        artifacts=[
            Artifact(artifact_id="ckpt-best", kind=ArtifactKind.checkpoint, required=True, created_at=T0,
                     file=FileRef(path="save/DPA_200MHz/train_pa/PA_S_0_M_GRU_H_23_F_50_P_1911.pt", sha256=SHA_A, size_bytes=10973)),
            Artifact(artifact_id="log-history", kind=ArtifactKind.log_history, required=True, created_at=T0,
                     file=FileRef(path="log/DPA_200MHz/train_pa/history/PA_S_0_M_GRU_H_23_F_50_P_1911.csv", sha256=SHA_B, size_bytes=1200)),
            Artifact(artifact_id="worker-log", kind=ArtifactKind.worker_log, created_at=T0,
                     file=FileRef(path="logs/worker.log")),
        ],
    )


def all_examples() -> Dict[str, object]:
    """Name -> model instance; names double as mock fixture file names."""
    return {
        "metric_profile_legacy": legacy_metric_profile(),
        "metric_profile_general": general_metric_profile(),
        "metric_profile_ofdm_evm": ofdm_metric_profile(),
        "dataset_builtin": dataset_builtin(),
        "dataset_missing_metadata": dataset_missing_metadata(),
        "diagnostics_missing_metadata": diagnostics_missing_metadata(),
        "experiment_train_pa_smoke": experiment_train_pa_smoke(),
        "experiment_train_dpd_smoke": experiment_train_dpd_smoke(),
        "experiment_run_dpd": experiment_run_dpd(),
        "experiment_evaluate_measured": experiment_evaluate_measured(),
        "resolved_train_pa_smoke": resolved_train_pa_smoke(),
        "run_queued": run_queued(),
        "run_running": run_running(),
        "run_log_disconnected": run_log_disconnected(),
        "run_interrupted": run_interrupted(),
        "run_failed": run_failed(),
        "events_running": events_running(),
        "result_pa_modeling_mock": result_pa_modeling_mock(),
        "result_metric_not_applicable_mock": result_metric_not_applicable_mock(),
        "result_dpd_surrogate_mock": result_dpd_surrogate_mock(),
        "result_dpd_measured_mock": result_dpd_measured_mock(),
        "run_lineage_dpd": run_lineage_dpd(),
        "comparison_report_mock": comparison_report_mock(),
        "history_points_mock": history_points_mock(),
        "result_legacy_import": result_legacy_import(),
        "artifact_manifest_complete": artifact_manifest_complete(),
    }
