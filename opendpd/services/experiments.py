"""Experiment service: create, execute and inspect runs.

This is the one compute path. The CLI (``opendpd run``) executes runs
in-process; the Studio worker (``opendpd.runtime.worker``) calls the same
``execute_run`` from a subprocess. Both produce identical files in
``workspace/runs/<run_id>/``.
"""

from __future__ import annotations

import math
import os
import re
import shutil
import socket
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from pydantic import ValidationError

from opendpd.schemas import (
    ProfileValidation,
    Artifact,
    ArtifactKind,
    ArtifactManifest,
    BaselineScore,
    DatasetEvidence,
    DatasetSourceKind,
    DPDReference,
    EvaluationResult,
    EvidenceType,
    ExperimentConfig,
    FileRef,
    HistoryPoint,
    InitReference,
    LineageLink,
    LineageRelation,
    MeasurementEvidence,
    ExecutionEvidence,
    MetricProfile,
    MetricValue,
    ModelEvidence,
    ModelSpec,
    PAReference,
    ResolvedExperimentConfig,
    RunError,
    RunEventType,
    RunLineage,
    RunRecord,
    RunStatus,
    ScalingInfo,
    SignalReference,
    SignalStage,
    SurrogateCoverage,
    TaskType,
    TrainingConfig,
    WorkerInfo,
    can_transition,
)
from opendpd.core.registry import RegistryError
from opendpd.services.config import ConfigError, ConfigIssue, ValidationReport, resolve, validate

TRAINING_TASKS = (TaskType.train_pa, TaskType.train_dpd, TaskType.run_dpd)   # tasks the legacy step machinery runs
from opendpd.services.legacy_adapter import (
    RunCancelled,
    build_namespace,
    legacy_command_line,
    run_in_directory,
    run_step,
)
from opendpd.services.workspace import (
    Workspace,
    WorkspaceError,
    read_json,
    sha256_file,
    software_provenance,
    write_json_atomic,
)

Emitter = Callable[[RunEventType, Dict], None]

RUN_FILE = "run.json"
USER_CONFIG_FILE = "config.user.json"
RESOLVED_CONFIG_FILE = "config.resolved.json"
PROVENANCE_FILE = "provenance.json"
ARTIFACTS_FILE = "artifacts.json"
RESULT_FILE = "result.json"
RESULTS_DIR = "results"
PLOTS_DIR = "plots"          # one result per metric profile

METRIC_COLUMNS = ("NMSE", "EVM", "ACLR_L", "ACLR_R", "ACLR_AVG")
SMOKE_EPOCH_LIMIT = 10


def _now() -> datetime:
    return datetime.now(timezone.utc)


# --- persistence helpers -----------------------------------------------------

def load_run(ws: Workspace, run_id: str) -> RunRecord:
    path = ws.run_dir(run_id) / RUN_FILE
    if not path.exists():
        raise WorkspaceError(f"run '{run_id}' does not exist in {ws.root}")
    return RunRecord.model_validate(read_json(path))


def save_run(ws: Workspace, record: RunRecord) -> None:
    write_json_atomic(ws.run_dir(record.run_id) / RUN_FILE, record)


def load_resolved(ws: Workspace, run_id: str) -> ResolvedExperimentConfig:
    return ResolvedExperimentConfig.model_validate(read_json(ws.run_dir(run_id) / RESOLVED_CONFIG_FILE))


def load_artifacts(ws: Workspace, run_id: str) -> Optional[ArtifactManifest]:
    path = ws.run_dir(run_id) / ARTIFACTS_FILE
    return ArtifactManifest.model_validate(read_json(path)) if path.exists() else None


def load_result(ws: Workspace, run_id: str, profile_id: Optional[str] = None) -> Optional[EvaluationResult]:
    """The primary result (configured profile) or the stored result under another profile."""
    path = ws.run_dir(run_id) / (RESULT_FILE if profile_id is None else f"{RESULTS_DIR}/{profile_id}.json")
    return EvaluationResult.model_validate(read_json(path)) if path.exists() else None


def list_runs(ws: Workspace) -> List[RunRecord]:
    return [load_run(ws, rid) for rid in ws.list_run_ids()]


# --- reference binding ---------------------------------------------------------

def _checkpoint_of(ws: Workspace, run_id: str, expected_task: TaskType, field: str):
    try:
        record = load_run(ws, run_id)
    except WorkspaceError as err:
        remedy = ("train a PA model on this dataset first (for example recipe pa-gru-smoke-v1) with the seed and "
                  "frame_length this DPD run uses" if expected_task == TaskType.train_pa
                  else "train a DPD on this dataset first (a train_dpd run through a PA surrogate)")
        raise ConfigError([ConfigIssue(field, str(err), remedy)]) from None
    if record.task != expected_task:
        raise ConfigError([ConfigIssue(field, f"run '{run_id}' is a {record.task.value} run, "
                                              f"expected {expected_task.value}")])
    if record.status != RunStatus.succeeded:
        raise ConfigError([ConfigIssue(field, f"run '{run_id}' has status {record.status.value}; "
                                              "only succeeded runs provide checkpoints",
                                       "train (or re-run) the referenced model first")])
    manifest = load_artifacts(ws, run_id)
    checkpoints = manifest.by_kind(ArtifactKind.checkpoint) if manifest else []
    if not checkpoints or checkpoints[0].file.sha256 is None:
        raise ConfigError([ConfigIssue(field, f"run '{run_id}' has no registered checkpoint",
                                       "the run finished without a verified checkpoint artifact")])
    return record, load_resolved(ws, run_id), checkpoints[0]


def submission_issues(ws: Workspace, config: ExperimentConfig) -> Tuple[List[ConfigIssue], List[ConfigIssue]]:
    """Checks the pure resolver cannot make: the device exists on this machine, the
    data version exists, and the split guard covers the frame context (plan S07:
    boundary isolation >= context so adjacent overlapping frames never share
    samples across train/val/test). Devices are never switched silently (S09)."""
    from opendpd.services import capabilities

    errors: List[ConfigIssue] = []
    warnings: List[ConfigIssue] = []
    device = config.execution.device
    if not getattr(ws, "device_available", capabilities.device_available)(device):
        errors.append(ConfigIssue("execution.device", f"device '{device}' is not available on this machine",
                                  hint="choose cpu or a detected device; OpenDPD never switches devices silently"))
    manifest = ws.get_dataset(config.dataset.id)
    version = config.dataset.preprocessing_version
    dv = manifest.version(version)
    if dv is None and version != "raw-v1":
        names = ", ".join(["raw-v1", *(v.version for v in manifest.versions)])
        errors.append(ConfigIssue("dataset.preprocessing_version",
                                  f"dataset '{manifest.dataset_id}' has no version '{version}'",
                                  hint=f"available: {names}"))
        return errors, warnings
    split = dv.split if dv is not None else manifest.split
    from opendpd.core.polynomial import POLYNOMIAL_KEYS, context_samples

    # the context a model reads: its frame for the gradient trainer, its memory depth for a least-squares fit
    frame = context_samples(config.model.key, config.model.parameters) if config.model.key in POLYNOMIAL_KEYS \
        else config.training.frame_length
    if split.guard_samples < frame and manifest.source.kind != DatasetSourceKind.builtin:
        warnings.append(ConfigIssue("training.frame_length",
                                    f"frame_length {frame} exceeds the split guard of {split.guard_samples} samples, "
                                    "so frames next to a split boundary share context across train/val/test",
                                    hint=f"re-import with a guard of at least {frame} samples or use a shorter frame"))
    return errors, warnings


def _bind_pa(ws: Workspace, run_id: str, dataset_id: str, training: Optional[TrainingConfig]) -> PAReference:
    """Bind a PA run to its checkpoint. ``training`` (the DPD training block) enforces the legacy checkpoint
    convention for train_dpd; evaluation through a surrogate (run_dpd) only needs the same dataset."""
    pa_record, pa_resolved, pa_ckpt = _checkpoint_of(ws, run_id, TaskType.train_pa, "pa_reference.run_id")
    if pa_resolved.dataset.id != dataset_id:
        raise ConfigError([ConfigIssue("pa_reference.run_id",
                                       f"PA surrogate '{pa_record.run_id}' was trained on dataset "
                                       f"'{pa_resolved.dataset.id}', not '{dataset_id}'",
                                       "pick a PA run from the same dataset or train one")])
    from opendpd.core.registry import get_model
    if get_model(pa_resolved.model.key).training_method != "gradient":
        raise ConfigError([ConfigIssue("pa_reference.run_id",
                                       f"PA run '{pa_record.run_id}' is a least-squares baseline "
                                       f"({pa_resolved.model.key}): a PA-modeling reference, not a DPD surrogate",
                                       "simulate DPD through a gradient-trained PA run (for example recipe "
                                       "pa-gru-smoke-v1 or pa-gru-research-v1)")])
    if training is not None and (pa_resolved.training.frame_length != training.frame_length
                                 or pa_resolved.training.seed != training.seed):
        raise ConfigError([ConfigIssue("training",
                                       "the legacy checkpoint convention requires the DPD run to use the "
                                       "same seed and frame_length as its PA surrogate "
                                       f"(PA: seed={pa_resolved.training.seed}, "
                                       f"frame_length={pa_resolved.training.frame_length})",
                                       f"set training.seed={pa_resolved.training.seed} and "
                                       f"training.frame_length={pa_resolved.training.frame_length}, or train a PA "
                                       "surrogate with this run's seed and frame_length")])
    return PAReference(run_id=pa_record.run_id, checkpoint_artifact_id=pa_ckpt.artifact_id,
                       checkpoint_sha256=pa_ckpt.file.sha256, model=pa_resolved.model)


def bind_references(ws: Workspace, config: ExperimentConfig) -> ExperimentConfig:
    """Turn ``run_id``-only references into fully bound references."""
    if config.task == TaskType.evaluate_measured:
        from opendpd.services.measurements import bind_measurement

        return bind_measurement(ws, config)
    updates = {}
    if config.task == TaskType.run_dpd:
        assert config.dpd_reference is not None
        dpd_record, dpd_resolved, dpd_ckpt = _checkpoint_of(ws, config.dpd_reference.run_id, TaskType.train_dpd,
                                                            "dpd_reference.run_id")
        transfer = config.dpd_reference.transfer
        updates["dpd_reference"] = DPDReference(run_id=dpd_record.run_id, checkpoint_artifact_id=dpd_ckpt.artifact_id,
                                                checkpoint_sha256=dpd_ckpt.file.sha256, model=dpd_resolved.model,
                                                transfer=transfer)
        if config.dataset.id != dpd_resolved.dataset.id and not transfer:
            raise ConfigError([ConfigIssue("dataset.id", f"DPD '{dpd_record.run_id}' was trained on dataset "
                                           f"'{dpd_resolved.dataset.id}', not '{config.dataset.id}'",
                                           "apply a DPD to the dataset it was trained on, or declare "
                                           "dpd_reference.transfer for a zero-update transfer to another condition")])
        if transfer and config.pa_reference is None:
            raise ConfigError([ConfigIssue("pa_reference", "a zero-update transfer needs a PA surrogate trained on the "
                                           f"target dataset '{config.dataset.id}'",
                                           "train a PA model on the target dataset and name its run")])
        # Without an explicit surrogate a run_dpd is evaluated through the one the DPD was trained with;
        # naming another PA run of the same dataset produces a new, separately stored result.
        updates["pa_reference"] = dpd_resolved.pa_reference if config.pa_reference is None \
            else _bind_pa(ws, config.pa_reference.run_id, config.dataset.id, training=None)
        if config.model != dpd_resolved.model:
            updates["model"] = _executed_model(config.model, dpd_resolved.model)
        # The legacy run_dpd step derives checkpoint ids from seed / frame_length
        # (and quantisation) of the *current* arguments: inherit them from the DPD run.
        updates["training"] = dpd_resolved.training
        updates["quantization"] = dpd_resolved.quantization
    if config.task == TaskType.train_dpd:
        from opendpd.services.polynomial import is_least_squares

        assert config.pa_reference is not None
        # the legacy checkpoint convention (same seed / frame_length as the surrogate) binds the gradient trainer
        # only; a least-squares predistorter is fitted on the data and merely evaluated through the surrogate
        training = None if is_least_squares(config.model.key) else config.training
        updates["pa_reference"] = _bind_pa(ws, config.pa_reference.run_id, config.dataset.id, training=training)
    if config.task == TaskType.evaluate_pa:
        # the weights and the architecture come from the PA run; this run only scores them on another dataset
        pa_record, pa_resolved, pa_ckpt = _checkpoint_of(ws, config.pa_reference.run_id, TaskType.train_pa,
                                                         "pa_reference.run_id")
        updates["pa_reference"] = PAReference(run_id=pa_record.run_id, checkpoint_artifact_id=pa_ckpt.artifact_id,
                                              checkpoint_sha256=pa_ckpt.file.sha256, model=pa_resolved.model)
        updates["model"] = _executed_model(config.model, pa_resolved.model)
        updates["training"] = pa_resolved.training.model_copy(update={"train_samples": None})
        updates["quantization"] = pa_resolved.quantization
    if config.initialization is not None and config.task in (TaskType.train_pa, TaskType.train_dpd):
        init_record, init_resolved, init_ckpt = _checkpoint_of(ws, config.initialization.run_id, config.task,
                                                               "initialization.run_id")
        if init_resolved.model.key != config.model.key or any(
                init_resolved.model.parameters.get(k) != v for k, v in config.model.parameters.items()):
            raise ConfigError([ConfigIssue("initialization.run_id",
                                           f"run '{init_record.run_id}' trained {init_resolved.model.key} "
                                           f"{init_resolved.model.parameters}; its weights cannot initialise "
                                           f"{config.model.key} {config.model.parameters}",
                                           "start from a run of the same model, or leave the parameters to the run")])
        updates["model"] = init_resolved.model                  # the weights define the architecture
        if config.quantization is None:
            updates["quantization"] = init_resolved.quantization
        updates["initialization"] = InitReference(run_id=init_record.run_id, checkpoint_artifact_id=init_ckpt.artifact_id,
                                                  checkpoint_sha256=init_ckpt.file.sha256)
    return config.model_copy(update=updates) if updates else config


# --- run creation ----------------------------------------------------------------

def resolve_experiment(ws: Workspace, config: ExperimentConfig, *,
                       warnings: Optional[List[ConfigIssue]] = None) -> ResolvedExperimentConfig:
    """Bind workspace references before resolution, for both preview and submission."""
    ws.get_dataset(config.dataset.id)
    try:
        # Workspace checks read model context parameters, so validate and fill
        # them first. Resolve again below to freeze the workspace warnings too.
        bound = resolve(bind_references(ws, config))
        errors, submission_warnings = submission_issues(ws, bound)
        warnings = warnings if warnings is not None else []
        warnings.extend(submission_warnings)
        if errors:
            raise ConfigError(errors)
        return resolve(bound, warnings=warnings)
    except RegistryError as err:
        raise ConfigError([ConfigIssue(err.field, err.message, err.hint)]) from err


def validate_experiment(ws: Workspace, config_data: Any) -> ValidationReport:
    """Preview the exact submitted configuration without creating a run or artifacts."""
    try:
        config = config_data if isinstance(config_data, ExperimentConfig) \
            else ExperimentConfig.model_validate(config_data)
    except ValidationError:
        return validate(config_data)
    report = ValidationReport()
    try:
        report.resolved = resolve_experiment(ws, config, warnings=report.warnings)
    except ConfigError as err:
        report.errors.extend(err.issues)
    except WorkspaceError as err:
        report.errors.append(ConfigIssue("dataset.id", str(err), hint="import the dataset first"))
    return report


def find_by_idempotency_key(ws: Workspace, key: str) -> Optional[RunRecord]:
    for record in list_runs(ws):
        if record.idempotency_key == key:
            return record
    return None


def create_run(ws: Workspace, config: ExperimentConfig, *, name: Optional[str] = None,
               idempotency_key: Optional[str] = None, parent_run_id: Optional[str] = None) -> RunRecord:
    """Validate, resolve and persist a queued run. Raises ConfigError / WorkspaceError."""
    if idempotency_key:
        existing = find_by_idempotency_key(ws, idempotency_key)
        if existing is not None:
            return existing
    problems = ws.preflight()
    if problems:
        raise WorkspaceError("; ".join(problems))
    resolved = resolve_experiment(ws, config)

    run_id = ws.new_run_id()
    run_dir = ws.run_dir(run_id)
    (run_dir / "logs").mkdir(parents=True)
    write_json_atomic(run_dir / USER_CONFIG_FILE, config)
    write_json_atomic(run_dir / RESOLVED_CONFIG_FILE, resolved)
    legacy_command = None
    plain = resolved.initialization is None and resolved.training.train_samples is None
    if resolved.task in TRAINING_TASKS and plain and not _is_least_squares(resolved.model.key):
        ns = build_namespace(resolved, dataset_dir=ws.dataset_version_dir(resolved.dataset.id, resolved.dataset.preprocessing_version),
                             dataset_name=resolved.dataset.id)
        legacy_command = legacy_command_line(ns)
    write_json_atomic(run_dir / PROVENANCE_FILE, {
        "run_id": run_id,
        "created_at": _now().isoformat(),
        "config_sha256": resolved.resolution.config_sha256,
        "dataset_raw_sha256": ws.get_dataset(resolved.dataset.id).raw_sha256,
        "software": software_provenance().model_dump(mode="json"),
        "legacy_equivalent_command": legacy_command,
        "parent_run_id": parent_run_id,
    })
    record = RunRecord(
        run_id=run_id, task=resolved.task, name=name or config.name or config.recipe_id,
        dataset_id=resolved.dataset.id, model_key=resolved.model.key, status=RunStatus.queued,
        created_at=_now(), config_sha256=resolved.resolution.config_sha256,
        device=resolved.execution.device, idempotency_key=idempotency_key, parent_run_id=parent_run_id,
        progress_total_epochs=resolved.training.epochs if resolved.task in (TaskType.train_pa, TaskType.train_dpd) else None,
    )
    save_run(ws, record)
    return record


# --- execution --------------------------------------------------------------------

def _worker_info() -> WorkerInfo:
    pid = os.getpid()
    try:
        import psutil
        create_time = psutil.Process(pid).create_time()
    except Exception:  # psutil optional in the core install
        create_time = time.time()
    return WorkerInfo(pid=pid, create_time=create_time, host=socket.gethostname())


def _transition(record: RunRecord, new: RunStatus, **updates) -> RunRecord:
    if not can_transition(record.status, new):
        raise RuntimeError(f"illegal transition {record.status.value} -> {new.value} for {record.run_id}")
    return record.model_copy(update={"status": new, **updates})


def _prepare_inputs(ws: Workspace, run_dir: Path, resolved: ResolvedExperimentConfig, ns) -> None:
    """Materialise referenced checkpoints where the legacy step expects them."""
    from models import CoreModel
    from utils.util import count_net_params

    def _pa_model_id() -> str:
        pa = resolved.pa_reference.model
        params = pa.parameters
        net = CoreModel(input_size=2, hidden_size=int(params.get("hidden_size", 0) or ns.PA_hidden_size),
                        num_layers=int(params.get("num_layers", 1)), backbone_type=ns.PA_backbone,
                        window_size=ns.window_size, num_dvr_units=ns.num_dvr_units, thx=ns.thx, thh=ns.thh)
        return f"PA_S_{ns.seed}_M_{ns.PA_backbone.upper()}_H_{ns.PA_hidden_size:d}_F_{ns.frame_length:d}" \
               f"_P_{count_net_params(net):d}"

    def _copy_checkpoint(ref, target: Path, what: str) -> None:
        src_manifest = load_artifacts(ws, ref.run_id)
        artifact = next(a for a in src_manifest.artifacts if a.artifact_id == ref.checkpoint_artifact_id)
        src = ws.run_dir(ref.run_id) / artifact.file.path
        if not src.exists() or sha256_file(src) != ref.checkpoint_sha256:
            raise FileNotFoundError(f"{what} checkpoint of run {ref.run_id} is missing or its hash changed")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)

    # S17 budget and warm start are executor-only attributes (read by project.py / the steps with getattr);
    # they are not legacy CLI flags, so the adapter's namespace stays exactly what `main.py` parses
    ns.train_samples = resolved.training.train_samples
    ns.init_weights = None
    if resolved.initialization is not None:
        target = run_dir / "init" / "weights.pt"
        _copy_checkpoint(resolved.initialization, target, "initial weights")
        ns.init_weights = str(target)
    if resolved.task in (TaskType.train_dpd, TaskType.run_dpd):
        pa_id = _pa_model_id()
        _copy_checkpoint(resolved.pa_reference, run_dir / "save" / ns.dataset_name / "train_pa" / f"{pa_id}.pt",
                         "PA surrogate")
        if resolved.task == TaskType.run_dpd:
            src_manifest = load_artifacts(ws, resolved.dpd_reference.run_id)
            artifact = next(a for a in src_manifest.artifacts
                            if a.artifact_id == resolved.dpd_reference.checkpoint_artifact_id)
            dpd_dir = run_dir / "save" / ns.dataset_name / "train_dpd" / pa_id.split("_P_")[0]
            _copy_checkpoint(resolved.dpd_reference, dpd_dir / Path(artifact.file.path).name, "DPD")


# Messages torch/CUDA/MPS produce when the accelerator cannot give the run what it asks for,
# including the case where another process holds it. Matched case-insensitively.
_DEVICE_REFUSAL_MARKERS = ("out of memory", "busy or unavailable", "cudnn_status_alloc_failed",
                           "cudnn_status_not_initialized", "cublas_status_alloc_failed", "no kernel image is available")


def classify_failure(err: BaseException, *, stage: str) -> RunError:
    """A structured error for an exception raised by the compute core (called inside the except block)."""
    message = f"{type(err).__name__}: {err}"
    tail = "".join(traceback.format_exc().splitlines(keepends=True)[-25:]) or None
    lowered = str(err).lower()
    if any(marker in lowered for marker in _DEVICE_REFUSAL_MARKERS):
        return RunError(code="device_busy_or_out_of_memory", stage=stage, message=message, traceback_tail=tail,
                        hint="the accelerator refused the allocation: another process may hold it (check nvidia-smi or "
                             "the OS activity monitor) or the batch and model do not fit its memory; free the device or "
                             "reduce batch_size, then retry — nothing was retried or moved to another device automatically")
    return RunError(code="worker_exception", stage=stage, message=message, traceback_tail=tail)


def apply_thread_budget(resolved: ExperimentConfig) -> int:
    """Apply ``execution.num_threads`` to torch's intra-op pool and return the effective count.

    ``None`` keeps torch's own default (its physical-core count). The CLI, the Python API and the
    GUI worker all execute through :func:`execute_run`, so the same configuration gets the same
    budget on every path; the budget only governs torch (least-squares fits run in NumPy)."""
    import torch

    wanted = resolved.execution.num_threads
    if wanted is not None and wanted != torch.get_num_threads():
        torch.set_num_threads(wanted)
    return torch.get_num_threads()


def execute_run(ws: Workspace, run_id: str, *, emit: Optional[Emitter] = None,
                should_cancel: Optional[Callable[[], bool]] = None) -> RunRecord:
    """Run a queued run to a terminal state in the current process."""
    emit = emit or (lambda *_: None)
    record = load_run(ws, run_id)
    if record.status != RunStatus.queued:
        raise RuntimeError(f"run {run_id} is {record.status.value}, not queued")
    resolved = load_resolved(ws, run_id)
    dataset = ws.get_dataset(resolved.dataset.id)
    run_dir = ws.run_dir(run_id)
    measured = resolved.task == TaskType.evaluate_measured
    ns = None if measured else build_namespace(
        resolved, dataset_dir=ws.dataset_version_dir(dataset.dataset_id, resolved.dataset.preprocessing_version),
        dataset_name=dataset.dataset_id)

    record = _transition(record, RunStatus.running, started_at=_now(), worker=_worker_info(),
                         last_heartbeat_at=_now())
    save_run(ws, record)
    emit(RunEventType.status, {"from": "queued", "to": "running"})

    state = {"record": record}
    from opendpd.services.live import LiveMonitor
    monitor = LiveMonitor(ws, run_id, resolved, dataset, emit)
    monitor.stage("prepare")

    def on_epoch(row: Dict) -> None:
        epoch = int(row.get("EPOCH", 0))
        total = int(row.get("N_EPOCH", ns.n_epochs))
        emit(RunEventType.progress, {"epoch": epoch, "total_epochs": total, "phase": "epoch_end"})
        for split in ("VAL", "TEST"):
            values = {m: _as_float(row.get(f"{split}_{m}")) for m in METRIC_COLUMNS if f"{split}_{m}" in row}
            if values:
                emit(RunEventType.metric, {"epoch": epoch, "split": split.lower(), "values": values,
                                           "train_loss": _as_float(row.get("TRAIN_LOSS"))})
        state["record"] = state["record"].model_copy(update={"progress_epoch": epoch + 1,
                                                             "progress_total_epochs": total,
                                                             "last_heartbeat_at": _now()})
        save_run(ws, state["record"])

    project = None
    signals = None
    error: Optional[RunError] = None
    outcome = RunStatus.succeeded
    reason = None
    stage = {TaskType.run_dpd: "apply", TaskType.evaluate_measured: "measure",
             TaskType.evaluate_pa: "evaluate"}.get(resolved.task, "train")
    apply_thread_budget(resolved)
    with run_in_directory(run_dir):
        try:
            from opendpd.services import measurements, polynomial

            if measured:
                signals = measurements.execute_measurement(ws, run_dir, resolved)
            elif resolved.task == TaskType.evaluate_pa:
                pass                                    # nothing to train: the evaluation stage scores the weights
            else:
                _prepare_inputs(ws, run_dir, resolved, ns)
                if polynomial.is_least_squares(resolved.model.key):
                    monitor.stage("apply" if resolved.task == TaskType.run_dpd else "fit")
                    # baselines never enter the legacy trainer: a deterministic fit, or a one-pass apply for run_dpd
                    project = polynomial.apply_run(ws, run_dir, resolved, ns) if resolved.task == TaskType.run_dpd \
                        else polynomial.fit_run(ws, run_dir, resolved, ns, on_epoch=on_epoch)
                else:
                    if resolved.task == TaskType.run_dpd:
                        monitor.stage("apply")
                    project = run_step(ns, on_epoch=on_epoch, should_cancel=should_cancel, observer=monitor)
                if resolved.task == TaskType.run_dpd:
                    from opendpd.services.evaluation import write_dpd_output_metadata
                    write_dpd_output_metadata(ws, run_id, resolved, ns)
        except RunCancelled as err:
            outcome, reason = RunStatus.cancelled, str(err)
        except KeyboardInterrupt:
            outcome, reason = RunStatus.cancelled, "interrupted by the user (SIGINT)"
        except FileNotFoundError as err:
            outcome = RunStatus.failed
            error = RunError(code="input_missing", stage="prepare", message=str(err),
                             hint="the referenced run's artifacts were deleted or modified")
        except measurements.MeasurementError as err:
            outcome = RunStatus.failed
            error = RunError(code="capture_rejected", stage=stage, message=str(err),
                             hint="check that the file is the analyser capture of this run's export at the declared "
                                  "sample rate, at least one full period long")
        except Exception as err:  # noqa: BLE001 - the worker must record any failure
            outcome = RunStatus.failed
            error = classify_failure(err, stage=stage)

    record = state["record"]
    manifest = collect_artifacts(run_dir, run_id, resolved, project)
    write_json_atomic(run_dir / ARTIFACTS_FILE, manifest)
    if resolved.task in (TaskType.train_pa, TaskType.train_dpd):
        from opendpd.services.model_download import MAX_MODEL_BYTES, MODEL_FILE, publish_model
        checkpoints = manifest.by_kind(ArtifactKind.checkpoint)
        if checkpoints:
            checkpoint = run_dir / checkpoints[0].file.path
            if not (run_dir / MODEL_FILE).exists() and checkpoint.stat().st_size <= MAX_MODEL_BYTES:
                publish_model(run_dir, checkpoint.read_bytes(), epoch=record.progress_epoch or 0)
    result_id = None
    if outcome == RunStatus.succeeded:
        if not manifest.complete:
            outcome = RunStatus.failed
            error = RunError(code="artifacts_incomplete", stage="finalize",
                             message="the step finished but a required artifact is missing",
                             hint="see logs/ in the run directory")
        else:
            from opendpd.services.evaluation import evaluate_all

            try:
                with run_in_directory(run_dir):
                    results = measurements.evaluate_measured(ws, run_id, resolved, manifest, signals) if measured \
                        else evaluate_all(ws, run_id, resolved, manifest, observer=monitor)
                result_id = results[resolved.evaluation.profile_id].result_id
                monitor.complete(results[resolved.evaluation.profile_id])
                manifest = collect_artifacts(run_dir, run_id, resolved, project)     # results and plots registered
                write_json_atomic(run_dir / ARTIFACTS_FILE, manifest)
                emit(RunEventType.artifact, {"artifact_id": "result", "kind": "result", "path": RESULT_FILE,
                                             "profiles": sorted(results)})
            except (ValidationError, ValueError, KeyError) as err:
                outcome = RunStatus.failed
                error = RunError(code="result_invalid", stage="evaluate", message=f"{type(err).__name__}: {err}")
            except Exception as err:  # noqa: BLE001 - evaluation failures are recorded, never hidden
                outcome = RunStatus.failed
                error = RunError(code="evaluation_failed", stage="evaluate", message=f"{type(err).__name__}: {err}",
                                 traceback_tail="".join(traceback.format_exc().splitlines(keepends=True)[-25:]))

    if outcome == RunStatus.running:  # pragma: no cover - defensive
        outcome = RunStatus.failed
    target = outcome
    if target == RunStatus.cancelled and record.status == RunStatus.running:
        record = _transition(record, RunStatus.cancel_requested)
        emit(RunEventType.status, {"from": "running", "to": "cancel_requested", "reason": reason})
    previous = record.status
    record = _transition(record, target, finished_at=_now(), exit_code=0 if target == RunStatus.succeeded else 1,
                         error=error, status_reason=reason, result_id=result_id)
    save_run(ws, record)
    if error is not None:
        emit(RunEventType.error, error.model_dump())
    emit(RunEventType.status, {"from": previous.value, "to": target.value, "reason": reason})
    return record


def _as_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


# --- artifacts and results ----------------------------------------------------------

def _rel(run_dir: Path, path: Path) -> str:
    """Path relative to the run directory. Legacy paths are relative to the
    run directory (its CWD during the step), so anchor them there explicitly
    instead of trusting the current working directory."""
    path = Path(path)
    if not path.is_absolute():
        path = run_dir / path
    return path.resolve().relative_to(run_dir.resolve()).as_posix()


def collect_artifacts(run_dir: Path, run_id: str, resolved: ResolvedExperimentConfig, project) -> ArtifactManifest:
    artifacts: List[Artifact] = []

    def add(artifact_id: str, kind: ArtifactKind, path: Path, required: bool, description: str = "") -> None:
        path = Path(path)
        if not path.is_absolute():
            path = run_dir / path
        if not path.exists():
            return
        artifacts.append(Artifact(artifact_id=artifact_id, kind=kind, required=required, description=description,
                                  file=FileRef(path=_rel(run_dir, path), sha256=sha256_file(path),
                                               size_bytes=path.stat().st_size)))

    if project is not None and resolved.task != TaskType.run_dpd:
        add("checkpoint-best", ArtifactKind.checkpoint, Path(project.path_save_file_best), True,
            "best-validation checkpoint (state_dict)")
        add("log-history", ArtifactKind.log_history, Path(project.path_log_file_hist), True, "per-epoch metrics")
        add("log-best", ArtifactKind.log_best, Path(project.path_log_file_best), True, "best-epoch metrics")
    if resolved.task == TaskType.run_dpd:
        for csv in sorted((run_dir / "dpd_out").rglob("*.csv")):
            add("dpd-output", ArtifactKind.dpd_output, csv, True,
                "pre-distorted PA *input* u = DPD(x) for the test split; not a PA output")
            add("dpd-output-meta", ArtifactKind.other, csv.with_suffix(".meta.json"), True,
                "what the exported columns are, their order, dtype, scaling and the checkpoints that produced them")
    if resolved.task == TaskType.evaluate_measured:
        from opendpd.services.measurements import CAPTURES_DIR, MEASUREMENT_FILE, PLAYED_FILE

        add("played-signal", ArtifactKind.other, run_dir / CAPTURES_DIR / PLAYED_FILE, True,
            "the run_dpd export that was played: x (I, Q) and u = DPD(x) (I_dpd, Q_dpd)")
        for role in ("with_dpd", "without_dpd"):
            for path in sorted((run_dir / CAPTURES_DIR).glob(f"{role}.*")):
                add("capture-" + role.replace("_", "-"), ArtifactKind.other, path, role == "with_dpd",
                    f"operator-provided capture of the PA output {role.replace('_', ' ')} (raw file, as uploaded)")
        add("measurement", ArtifactKind.other, run_dir / MEASUREMENT_FILE, True,
            "conditions declared by the operator, capture hashes, alignment and level statistics")
    for name in ("worker.log", "stdout.log"):
        add(name.replace(".", "-"), ArtifactKind.worker_log, run_dir / "logs" / name, False)
    add("fit-diagnostics", ArtifactKind.other, run_dir / "fit.json", False,
        "least-squares fit record: method, rank, condition number, cutoff, residual")
    add("init-weights", ArtifactKind.other, run_dir / "init" / "weights.pt", False,
        "initial weights (warm start) copied from the initialising run; the trained checkpoint is checkpoint-best")
    add("config-resolved", ArtifactKind.config, run_dir / RESOLVED_CONFIG_FILE, True)
    add("provenance", ArtifactKind.provenance, run_dir / PROVENANCE_FILE, True)
    # written by the evaluation stage; registered on the second pass after it ran
    add("result", ArtifactKind.result, run_dir / RESULT_FILE, False, "primary result (configured metric profile)")
    for stored in sorted((run_dir / RESULTS_DIR).glob("*.json")) if (run_dir / RESULTS_DIR).exists() else []:
        add(f"result-{stored.stem}", ArtifactKind.result, stored, False, f"result under profile {stored.stem}")
    for plot in sorted((run_dir / PLOTS_DIR).glob("*.json")) if (run_dir / PLOTS_DIR).exists() else []:
        add(f"plot-{plot.stem}", ArtifactKind.plot, plot, False,
            "plots-v1 derived data computed by the worker from the evaluated arrays; display only")
    ids = {a.artifact_id for a in artifacts}
    if resolved.task == TaskType.evaluate_measured:
        present = {"played-signal", "capture-with-dpd", "measurement"} <= ids
    elif resolved.task == TaskType.run_dpd:
        present = {"dpd-output", "dpd-output-meta"} <= ids
    elif resolved.task == TaskType.evaluate_pa:
        present = {"config-resolved", "provenance"} <= ids
    else:
        present = {"checkpoint-best", "log-history", "log-best"} <= ids
    complete = present and all(a.file.sha256 for a in artifacts if a.required)
    return ArtifactManifest(run_id=run_id, artifacts=artifacts, complete=complete)


def _best_row(run_dir: Path, manifest: ArtifactManifest) -> Dict[str, str]:
    import csv

    best = manifest.by_kind(ArtifactKind.log_best)[0]
    with open(run_dir / best.file.path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("best-epoch log is empty")
    return rows[-1]


def _n_params(path: str) -> Optional[int]:
    match = re.search(r"_P_(\d+)", path)
    return int(match.group(1)) if match else None


def build_result(ws: Workspace, run_id: str, resolved: ResolvedExperimentConfig, manifest: ArtifactManifest, *,
                 profile: MetricProfile, metrics: List[MetricValue], n_valid: int,
                 target_gain: Optional[float], signal_chain: Optional[List[SignalStage]] = None,
                 baselines: Optional[List[BaselineScore]] = None, surrogate_coverage: Optional[SurrogateCoverage] = None,
                 scaling: Optional[ScalingInfo] = None, measurement: Optional[MeasurementEvidence] = None,
                 execution: Optional[ExecutionEvidence] = None, limitations: Optional[List[str]] = None) -> EvaluationResult:
    """Assemble the evidence around metrics scored by ``opendpd.core.metrics``."""
    run_dir = ws.run_dir(run_id)
    dataset = ws.get_dataset(resolved.dataset.id)
    measured = resolved.task == TaskType.evaluate_measured
    if resolved.task == TaskType.run_dpd or measured:
        # the weights come from the DPD run; this run only applied them (or played its export)
        dpd_ref = resolved.dpd_reference
        dpd_manifest = load_artifacts(ws, dpd_ref.run_id)
        dpd_artifact = next(a for a in dpd_manifest.artifacts if a.artifact_id == dpd_ref.checkpoint_artifact_id)
        dpd_run_id, dpd_sha, dpd_params = dpd_ref.run_id, dpd_ref.checkpoint_sha256, _n_params(dpd_artifact.file.path)
        selected_epoch, history = None, None
    elif resolved.task == TaskType.evaluate_pa:
        # the weights come from the PA run; this run only scored them on another dataset
        pa_ref = resolved.pa_reference
        pa_manifest = load_artifacts(ws, pa_ref.run_id)
        pa_artifact = next(a for a in pa_manifest.artifacts if a.artifact_id == pa_ref.checkpoint_artifact_id)
        dpd_run_id, dpd_sha, dpd_params = pa_ref.run_id, pa_ref.checkpoint_sha256, _n_params(pa_artifact.file.path)
        selected_epoch, history = None, None
    else:
        row = _best_row(run_dir, manifest)
        checkpoint = manifest.by_kind(ArtifactKind.checkpoint)[0]
        dpd_run_id, dpd_sha, dpd_params = run_id, checkpoint.file.sha256, _n_params(checkpoint.file.path)
        selected_epoch = int(row["EPOCH"]) if "EPOCH" in row else None
        history = FileRef(path=manifest.by_kind(ArtifactKind.log_history)[0].file.path)

    nperseg = dataset.signal.nperseg
    n_segments = math.ceil(n_valid / nperseg) if (n_valid and nperseg) else None

    from opendpd.services.polynomial import fit_limitations, is_least_squares

    least_squares = is_least_squares(resolved.model.key)
    limitations: List[str] = list(limitations or [])
    if least_squares:
        selected_epoch = None                       # a fit has no epochs to select from
        if not measured:
            limitations += fit_limitations(ws, run_id, resolved)
    elif resolved.training.epochs <= SMOKE_EPOCH_LIMIT:
        limitations.append(f"{resolved.training.epochs}-epoch smoke/demo training; not a benchmark result")
    if resolved.training.reproducibility == "soft" and not measured:
        limitations.append("soft reproducibility (non-deterministic algorithms allowed); repeated runs may differ")
    if dataset.missing_metadata():
        limitations.append(f"dataset metadata missing: {', '.join(dataset.missing_metadata())}")
    if profile.validation == ProfileValidation.pending_cross_validation:
        limitations.append(f"metric profile {profile.profile_id} is pending cross-validation against an independent "
                           "backend; its numbers are not standard-conformance results")
    if resolved.initialization is not None:
        limitations.append(f"warm start: initialised from run {resolved.initialization.run_id} weights "
                           f"{(resolved.initialization.checkpoint_sha256 or '')[:12]}, not trained from scratch")
    if resolved.training.train_samples is not None:
        limitations.append(f"adaptation budget: fitted on the first {resolved.training.train_samples} samples of the "
                           "train split only")
    if resolved.task == TaskType.evaluate_pa:
        source = load_resolved(ws, resolved.pa_reference.run_id).dataset.id
        limitations.append(f"zero-update transfer: PA model trained on dataset {source} scored on dataset "
                           f"{dataset.dataset_id} without any update")
    if resolved.task == TaskType.run_dpd and resolved.dpd_reference.transfer:
        source = load_resolved(ws, resolved.dpd_reference.run_id).dataset.id
        limitations.append(f"zero-update transfer: DPD trained on dataset {source} applied to dataset "
                           f"{dataset.dataset_id} without any update")

    if execution is not None:
        from opendpd.services.streaming import streaming_limitations
        limitations.extend(streaming_limitations(resolved.model.key, execution))
    from opendpd.core.registry import get_model
    semantics = get_model(resolved.model.key).execution_semantics

    models = []
    source, is_mock = "opendpd-studio", False
    if measured:
        assert measurement is not None
        evidence = EvidenceType.dpd_measured
        reference = SignalReference(kind="linear_gain_target",
                                    description="target = g * x with g the complex least-squares gain of the aligned "
                                                "measured output onto x (capture units); the PA output is measured, "
                                                "not simulated",
                                    gain_rule="least-squares complex gain of the aligned capture onto x, per capture",
                                    gain_value=target_gain)
        models.append(ModelEvidence(role="dpd", model=resolved.model, run_id=dpd_run_id, weights_sha256=dpd_sha,
                                    n_parameters=dpd_params, lookahead_samples=_lookahead(resolved.model),
                                    execution_semantics=semantics,
                                    training_path="ila_least_squares" if least_squares else "gradient_dla"))
        if resolved.measurement is not None and resolved.measurement.source == "mock_adapter":
            source, is_mock = "mock", True
    elif resolved.task in (TaskType.train_pa, TaskType.evaluate_pa):
        evidence = EvidenceType.pa_modeling
        reference = SignalReference(kind="measured_pa_output",
                                    description="measured PA output of the test split (dataset *_output)")
        models.append(ModelEvidence(role="pa", model=resolved.model, run_id=dpd_run_id, weights_sha256=dpd_sha,
                                    n_parameters=dpd_params, lookahead_samples=_lookahead(resolved.model),
                                    execution_semantics=semantics,
                                    training_path="least_squares" if least_squares else "gradient"))
    else:
        evidence = EvidenceType.dpd_surrogate
        reference = SignalReference(kind="linear_gain_target", description="target = gain * PA input (test split)",
                                    gain_rule="max|y_train| / max|x_train| (legacy utils.util.set_target_gain)",
                                    gain_value=target_gain)
        pa = resolved.pa_reference
        models.append(ModelEvidence(role="dpd", model=resolved.model, run_id=dpd_run_id, weights_sha256=dpd_sha,
                                    n_parameters=dpd_params, lookahead_samples=_lookahead(resolved.model),
                                    execution_semantics=semantics,
                                    training_path="ila_least_squares" if least_squares else "gradient_dla"))
        pa_manifest = load_artifacts(ws, pa.run_id)
        pa_artifact = next((a for a in (pa_manifest.artifacts if pa_manifest else [])
                            if a.artifact_id == pa.checkpoint_artifact_id), None)
        models.append(ModelEvidence(role="pa", model=pa.model, run_id=pa.run_id, weights_sha256=pa.checkpoint_sha256,
                                    n_parameters=_n_params(pa_artifact.file.path) if pa_artifact else None,
                                    lookahead_samples=_lookahead(pa.model),
                                    training_path="least_squares" if is_least_squares(pa.model.key) else "gradient"))
        limitations.append(f"simulated through the learned PA surrogate {pa.run_id}; not a measured PA output")
        if surrogate_coverage is not None and surrogate_coverage.fraction_above_fitted_peak > 0:
            limitations.append(f"{surrogate_coverage.fraction_above_fitted_peak:.2%} of the pre-distorted samples exceed "
                               "the amplitude range the surrogate was fitted on (extrapolation)")
        if scaling is not None and not scaling.physical_calibration:
            limitations.append("no physical calibration: absolute output power (dBm) and efficiency are not derived")

    return EvaluationResult(
        result_id=f"res-{run_id[4:]}" + ("" if profile.profile_id == resolved.evaluation.profile_id else f"-{profile.profile_id}"),
        run_id=run_id, source=source, is_mock=is_mock,
        evidence_type=evidence, metric_profile_id=profile.profile_id, metric_profile_version=profile.version,
        dataset=DatasetEvidence(dataset_id=dataset.dataset_id, split="test", raw_sha256=dataset.raw_sha256,
                                preprocessing_version=resolved.dataset.preprocessing_version,
                                split_version=resolved.dataset.split_version, n_samples=n_valid),
        models=models, reference=reference, execution=execution,
        valid_sample_range=(0, n_valid), n_segments=n_segments, nperseg=nperseg,
        metrics=metrics, selected_epoch=selected_epoch, history=history,
        software=software_provenance(), device=resolved.execution.device, seed=resolved.training.seed,
        numeric_mode=f"float32 / reproducibility={resolved.training.reproducibility}",
        limitations=limitations, signal_chain=signal_chain or [], baselines=baselines or [],
        surrogate_coverage=surrogate_coverage, scaling=scaling, measurement=measurement,
    )


def training_history(ws: Workspace, run_id: str) -> List[HistoryPoint]:
    """Per-epoch validation / test metrics from the run's history log, in the shape of ``metric`` events."""
    import csv

    manifest = load_artifacts(ws, run_id)
    history = manifest.by_kind(ArtifactKind.log_history) if manifest else []
    if not history:
        raise WorkspaceError(f"run '{run_id}' has no training history (not a training run, or it did not finish)")
    points: List[HistoryPoint] = []
    with open(ws.run_dir(run_id) / history[0].file.path, newline="") as f:
        for row in csv.DictReader(f):
            epoch = int(row.get("EPOCH", 0))
            for split in ("val", "test"):
                values = {m: v for m in METRIC_COLUMNS
                          if (v := _as_float(row.get(f"{split.upper()}_{m}"))) is not None}
                if values:
                    points.append(HistoryPoint(epoch=epoch, split=split, values=values,
                                               train_loss=_as_float(row.get("TRAIN_LOSS"))))
    return points


# --- lineage --------------------------------------------------------------------------

def lineage(ws: Workspace, run_id: str) -> RunLineage:
    """Parents (what this run used) and children (what used it), read from resolved configurations."""
    records = {r.run_id: r for r in list_runs(ws)}
    if run_id not in records:
        raise WorkspaceError(f"run '{run_id}' does not exist")

    def links_of(rid: str) -> List[LineageLink]:
        record = records[rid]
        out: List[LineageLink] = []
        if record.parent_run_id:
            out.append(_link(records, record.parent_run_id, LineageRelation.retry_of))
        try:
            resolved = load_resolved(ws, rid)
        except (OSError, ValidationError):
            return out
        if resolved.initialization is not None:
            out.append(_link(records, resolved.initialization.run_id, LineageRelation.initialised_from,
                             resolved.initialization.checkpoint_sha256))
        if resolved.dpd_reference is not None:
            out.append(_link(records, resolved.dpd_reference.run_id, LineageRelation.dpd_model,
                             resolved.dpd_reference.checkpoint_sha256))
        if resolved.pa_reference is not None:
            relation = LineageRelation.pa_model if record.task == TaskType.evaluate_pa else LineageRelation.pa_surrogate
            out.append(_link(records, resolved.pa_reference.run_id, relation, resolved.pa_reference.checkpoint_sha256))
        if resolved.measurement is not None:
            out.append(_link(records, resolved.measurement.apply_run_id, LineageRelation.measured_playback))
        return out

    parents = links_of(run_id)
    children: List[LineageLink] = []
    for other in sorted(records):
        if other == run_id:
            continue
        for link in links_of(other):
            if link.run_id == run_id:
                children.append(_link(records, other, link.relation, link.checkpoint_sha256))
    return RunLineage(run_id=run_id, parents=parents, children=children)


def _link(records: Dict[str, RunRecord], run_id: str, relation: LineageRelation,
          sha: Optional[str] = None) -> LineageLink:
    record = records.get(run_id)
    return LineageLink(run_id=run_id, relation=relation, task=record.task if record else None,
                       status=record.status if record else None, checkpoint_sha256=sha)


def _executed_model(requested: ModelSpec, trained: ModelSpec) -> ModelSpec:
    """The weights define the architecture: a run that only applies or scores them carries the trained model, unless
    the request names a registered streaming variant of that model (plan S18), which executes the same weights."""
    from opendpd.core.registry import get_model
    if requested.key != trained.key and get_model(requested.key).weights_from == trained.key:
        return ModelSpec(key=requested.key, parameters=dict(trained.parameters))
    return trained


def _is_least_squares(key: str) -> bool:
    from opendpd.core.registry import get_model
    return get_model(key).training_method == "least_squares"


def _lookahead(spec: ModelSpec) -> Optional[int]:
    from opendpd.core.polynomial import POLYNOMIAL_KEYS, lookahead_samples
    from opendpd.core.registry import get_model
    if spec.key in POLYNOMIAL_KEYS:
        return lookahead_samples(spec.key, spec.parameters)
    return get_model(spec.key).lookahead_samples
