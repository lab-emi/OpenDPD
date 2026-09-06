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
from typing import Callable, Dict, List, Optional, Tuple

from pydantic import ValidationError

from opendpd.schemas import (
    Artifact,
    ArtifactKind,
    ArtifactManifest,
    DatasetEvidence,
    DPDReference,
    EvaluationResult,
    EvidenceType,
    ExperimentConfig,
    FileRef,
    MetricProfile,
    MetricValue,
    ModelEvidence,
    PAReference,
    ResolvedExperimentConfig,
    RunError,
    RunEventType,
    RunRecord,
    RunStatus,
    SignalReference,
    TaskType,
    WorkerInfo,
    can_transition,
    DatasetSourceKind,
)
from opendpd.services.config import ConfigError, ConfigIssue, resolve
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
RESULTS_DIR = "results"          # one result per metric profile

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
        raise ConfigError([ConfigIssue(field, str(err))]) from None
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
    if not capabilities.device_available(device):
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
    frame = config.training.frame_length
    if split.guard_samples < frame and manifest.source.kind != DatasetSourceKind.builtin:
        warnings.append(ConfigIssue("training.frame_length",
                                    f"frame_length {frame} exceeds the split guard of {split.guard_samples} samples, "
                                    "so frames next to a split boundary share context across train/val/test",
                                    hint=f"re-import with a guard of at least {frame} samples or use a shorter frame"))
    return errors, warnings


def bind_references(ws: Workspace, config: ExperimentConfig) -> ExperimentConfig:
    """Turn ``run_id``-only references into fully bound references."""
    updates = {}
    if config.task == TaskType.run_dpd:
        assert config.dpd_reference is not None
        dpd_record, dpd_resolved, dpd_ckpt = _checkpoint_of(ws, config.dpd_reference.run_id, TaskType.train_dpd,
                                                            "dpd_reference.run_id")
        updates["dpd_reference"] = DPDReference(run_id=dpd_record.run_id, checkpoint_artifact_id=dpd_ckpt.artifact_id,
                                                checkpoint_sha256=dpd_ckpt.file.sha256, model=dpd_resolved.model)
        # A run_dpd inherits the DPD run's PA surrogate binding.
        if config.pa_reference is None or config.pa_reference.model is None:
            updates["pa_reference"] = dpd_resolved.pa_reference
        if config.model != dpd_resolved.model:
            updates["model"] = dpd_resolved.model
        # The legacy run_dpd step derives checkpoint ids from seed / frame_length
        # (and quantisation) of the *current* arguments: inherit them from the DPD run.
        updates["training"] = dpd_resolved.training
        updates["quantization"] = dpd_resolved.quantization
    if config.task == TaskType.train_dpd or (config.task == TaskType.run_dpd and "pa_reference" not in updates):
        assert config.pa_reference is not None
        pa_record, pa_resolved, pa_ckpt = _checkpoint_of(ws, config.pa_reference.run_id, TaskType.train_pa,
                                                         "pa_reference.run_id")
        if pa_resolved.dataset.id != config.dataset.id:
            raise ConfigError([ConfigIssue("pa_reference.run_id",
                                           f"PA surrogate '{pa_record.run_id}' was trained on dataset "
                                           f"'{pa_resolved.dataset.id}', not '{config.dataset.id}'",
                                           "pick a PA run from the same dataset or train one")])
        if pa_resolved.training.frame_length != config.training.frame_length \
                or pa_resolved.training.seed != config.training.seed:
            raise ConfigError([ConfigIssue("training",
                                           "the legacy checkpoint convention requires the DPD run to use the "
                                           "same seed and frame_length as its PA surrogate "
                                           f"(PA: seed={pa_resolved.training.seed}, "
                                           f"frame_length={pa_resolved.training.frame_length})")])
        updates["pa_reference"] = PAReference(run_id=pa_record.run_id, checkpoint_artifact_id=pa_ckpt.artifact_id,
                                              checkpoint_sha256=pa_ckpt.file.sha256, model=pa_resolved.model)
    return config.model_copy(update=updates) if updates else config


# --- run creation ----------------------------------------------------------------

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
    ws.get_dataset(config.dataset.id)
    bound = bind_references(ws, config)
    errors, warnings = submission_issues(ws, bound)
    if errors:
        raise ConfigError(errors)
    resolved = resolve(bound, warnings=warnings)

    run_id = ws.new_run_id()
    run_dir = ws.run_dir(run_id)
    (run_dir / "logs").mkdir(parents=True)
    write_json_atomic(run_dir / USER_CONFIG_FILE, config)
    write_json_atomic(run_dir / RESOLVED_CONFIG_FILE, resolved)
    ns = build_namespace(resolved, dataset_dir=ws.dataset_version_dir(config.dataset.id, config.dataset.preprocessing_version),
                         dataset_name=config.dataset.id)
    write_json_atomic(run_dir / PROVENANCE_FILE, {
        "run_id": run_id,
        "created_at": _now().isoformat(),
        "config_sha256": resolved.resolution.config_sha256,
        "dataset_raw_sha256": ws.get_dataset(config.dataset.id).raw_sha256,
        "software": software_provenance().model_dump(mode="json"),
        "legacy_equivalent_command": legacy_command_line(ns),
        "parent_run_id": parent_run_id,
    })
    record = RunRecord(
        run_id=run_id, task=config.task, name=name or config.name or config.recipe_id,
        dataset_id=config.dataset.id, model_key=config.model.key, status=RunStatus.queued,
        created_at=_now(), config_sha256=resolved.resolution.config_sha256,
        device=config.execution.device, idempotency_key=idempotency_key, parent_run_id=parent_run_id,
        progress_total_epochs=config.training.epochs if config.task != TaskType.run_dpd else None,
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

    if resolved.task in (TaskType.train_dpd, TaskType.run_dpd):
        pa_id = _pa_model_id()
        if resolved.task == TaskType.train_dpd:
            _copy_checkpoint(resolved.pa_reference, run_dir / "save" / ns.dataset_name / "train_pa" / f"{pa_id}.pt",
                             "PA surrogate")
        else:
            src_manifest = load_artifacts(ws, resolved.dpd_reference.run_id)
            artifact = next(a for a in src_manifest.artifacts
                            if a.artifact_id == resolved.dpd_reference.checkpoint_artifact_id)
            dpd_dir = run_dir / "save" / ns.dataset_name / "train_dpd" / pa_id.split("_P_")[0]
            _copy_checkpoint(resolved.dpd_reference, dpd_dir / Path(artifact.file.path).name, "DPD")


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
    ns = build_namespace(resolved, dataset_dir=ws.dataset_version_dir(dataset.dataset_id, resolved.dataset.preprocessing_version),
                         dataset_name=dataset.dataset_id)

    record = _transition(record, RunStatus.running, started_at=_now(), worker=_worker_info(),
                         last_heartbeat_at=_now())
    save_run(ws, record)
    emit(RunEventType.status, {"from": "queued", "to": "running"})

    state = {"record": record}

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
    error: Optional[RunError] = None
    outcome = RunStatus.succeeded
    reason = None
    with run_in_directory(run_dir):
        try:
            _prepare_inputs(ws, run_dir, resolved, ns)
            project = run_step(ns, on_epoch=on_epoch, should_cancel=should_cancel)
        except RunCancelled as err:
            outcome, reason = RunStatus.cancelled, str(err)
        except KeyboardInterrupt:
            outcome, reason = RunStatus.cancelled, "interrupted by the user (SIGINT)"
        except FileNotFoundError as err:
            outcome = RunStatus.failed
            error = RunError(code="input_missing", stage="prepare", message=str(err),
                             hint="the referenced run's artifacts were deleted or modified")
        except Exception as err:  # noqa: BLE001 - the worker must record any failure
            outcome = RunStatus.failed
            error = RunError(code="worker_exception", stage="train", message=f"{type(err).__name__}: {err}",
                             traceback_tail="".join(traceback.format_exc().splitlines(keepends=True)[-25:]))

    record = state["record"]
    manifest = collect_artifacts(run_dir, run_id, resolved, project)
    write_json_atomic(run_dir / ARTIFACTS_FILE, manifest)
    result_id = None
    if outcome == RunStatus.succeeded:
        if not manifest.complete:
            outcome = RunStatus.failed
            error = RunError(code="artifacts_incomplete", stage="finalize",
                             message="the step finished but a required artifact is missing",
                             hint="see logs/ in the run directory")
        elif resolved.task != TaskType.run_dpd:
            from opendpd.services.evaluation import evaluate_all

            try:
                with run_in_directory(run_dir):
                    results = evaluate_all(ws, run_id, resolved, manifest)
                result_id = results[resolved.evaluation.profile_id].result_id
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
    for name in ("worker.log", "stdout.log"):
        add(name.replace(".", "-"), ArtifactKind.worker_log, run_dir / "logs" / name, False)
    add("config-resolved", ArtifactKind.config, run_dir / RESOLVED_CONFIG_FILE, True)
    add("provenance", ArtifactKind.provenance, run_dir / PROVENANCE_FILE, True)
    required_kinds = {ArtifactKind.dpd_output} if resolved.task == TaskType.run_dpd \
        else {ArtifactKind.checkpoint, ArtifactKind.log_history, ArtifactKind.log_best}
    present = {a.kind for a in artifacts}
    complete = required_kinds <= present and all(a.file.sha256 for a in artifacts if a.required)
    return ArtifactManifest(run_id=run_id, artifacts=artifacts, complete=complete)


def _best_row(run_dir: Path, manifest: ArtifactManifest) -> Dict[str, str]:
    import csv

    best = manifest.by_kind(ArtifactKind.log_best)[0]
    with open(run_dir / best.file.path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("best-epoch log is empty")
    return rows[-1]


def build_result(ws: Workspace, run_id: str, resolved: ResolvedExperimentConfig, manifest: ArtifactManifest, *,
                 profile: MetricProfile, metrics: List[MetricValue], n_valid: int,
                 target_gain: Optional[float]) -> EvaluationResult:
    """Assemble the evidence around metrics scored by ``opendpd.core.metrics``."""
    run_dir = ws.run_dir(run_id)
    dataset = ws.get_dataset(resolved.dataset.id)
    row = _best_row(run_dir, manifest)
    checkpoint = manifest.by_kind(ArtifactKind.checkpoint)[0]
    n_params_match = re.search(r"_P_(\d+)", checkpoint.file.path)
    n_params = int(n_params_match.group(1)) if n_params_match else None

    nperseg = dataset.signal.nperseg
    n_segments = math.ceil(n_valid / nperseg) if (n_valid and nperseg) else None

    limitations: List[str] = []
    if resolved.training.epochs < SMOKE_EPOCH_LIMIT:
        limitations.append(f"{resolved.training.epochs}-epoch smoke/demo training; not a benchmark result")
    if resolved.training.reproducibility == "soft":
        limitations.append("soft reproducibility (non-deterministic algorithms allowed); repeated runs may differ")
    if dataset.missing_metadata():
        limitations.append(f"dataset metadata missing: {', '.join(dataset.missing_metadata())}")

    models = []
    if resolved.task == TaskType.train_pa:
        evidence = EvidenceType.pa_modeling
        reference = SignalReference(kind="measured_pa_output",
                                    description="measured PA output of the test split (dataset *_output)")
        models.append(ModelEvidence(role="pa", model=resolved.model, run_id=run_id, weights_sha256=checkpoint.file.sha256,
                                    n_parameters=n_params, lookahead_samples=_lookahead(resolved.model.key)))
    else:
        evidence = EvidenceType.dpd_surrogate
        reference = SignalReference(kind="linear_gain_target", description="target = gain * PA input (test split)",
                                    gain_rule="max|y_train| / max|x_train| (legacy utils.util.set_target_gain)",
                                    gain_value=target_gain)
        pa = resolved.pa_reference
        models.append(ModelEvidence(role="dpd", model=resolved.model, run_id=run_id, weights_sha256=checkpoint.file.sha256,
                                    n_parameters=n_params, lookahead_samples=_lookahead(resolved.model.key)))
        models.append(ModelEvidence(role="pa", model=pa.model, run_id=pa.run_id, weights_sha256=pa.checkpoint_sha256,
                                    lookahead_samples=_lookahead(pa.model.key)))
        limitations.append(f"simulated through the learned PA surrogate {pa.run_id}; not a measured PA output")

    return EvaluationResult(
        result_id=f"res-{run_id[4:]}" + ("" if profile.profile_id == resolved.evaluation.profile_id else f"-{profile.profile_id}"),
        run_id=run_id, source="opendpd-studio",
        evidence_type=evidence, metric_profile_id=profile.profile_id, metric_profile_version=profile.version,
        dataset=DatasetEvidence(dataset_id=dataset.dataset_id, split="test", raw_sha256=dataset.raw_sha256,
                                preprocessing_version=resolved.dataset.preprocessing_version,
                                split_version=resolved.dataset.split_version, n_samples=n_valid),
        models=models, reference=reference,
        valid_sample_range=(0, n_valid), n_segments=n_segments, nperseg=nperseg,
        metrics=metrics, selected_epoch=int(row["EPOCH"]) if "EPOCH" in row else None,
        history=FileRef(path=manifest.by_kind(ArtifactKind.log_history)[0].file.path),
        software=software_provenance(), device=resolved.execution.device, seed=resolved.training.seed,
        numeric_mode=f"float32 / reproducibility={resolved.training.reproducibility}",
        limitations=limitations,
    )


def _lookahead(key: str) -> Optional[int]:
    from opendpd.core.registry import get_model
    return get_model(key).lookahead_samples
