"""Run records, the task state machine and run events."""

from __future__ import annotations

from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, FrozenSet, Optional

from pydantic import Field, model_validator

from .common import SCHEMA_VERSION, Sha256, Slug, StrictModel
from .experiment import TaskType


class RunStatus(str, Enum):
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"
    cancel_requested = "cancel_requested"
    cancelled = "cancelled"
    interrupted = "interrupted"


TERMINAL_STATUSES: FrozenSet[RunStatus] = frozenset(
    {RunStatus.succeeded, RunStatus.failed, RunStatus.cancelled, RunStatus.interrupted}
)

# queued -> running -> succeeded | failed | cancel_requested -> cancelled | interrupted
# queued -> cancelled.  Terminal states never change again: a late completion
# event cannot turn a cancelled run into a success.
_TRANSITIONS: Dict[RunStatus, FrozenSet[RunStatus]] = {
    RunStatus.queued: frozenset({RunStatus.running, RunStatus.cancelled, RunStatus.interrupted}),
    RunStatus.running: frozenset({RunStatus.succeeded, RunStatus.failed, RunStatus.cancel_requested,
                                  RunStatus.interrupted}),
    RunStatus.cancel_requested: frozenset({RunStatus.cancelled, RunStatus.failed, RunStatus.interrupted}),
    RunStatus.succeeded: frozenset(),
    RunStatus.failed: frozenset(),
    RunStatus.cancelled: frozenset(),
    RunStatus.interrupted: frozenset(),
}


def can_transition(current: RunStatus, new: RunStatus) -> bool:
    return new in _TRANSITIONS[current]


class WorkerInfo(StrictModel):
    """Identity of the worker process. PID alone is never trusted: the
    process creation time must match too (PIDs are recycled)."""

    pid: int = Field(ge=1)
    create_time: float          # seconds since epoch, from the OS
    host: str = Field(min_length=1)


class RunError(StrictModel):
    code: str = Field(pattern=r"^[a-z][a-z0-9_]{1,63}$")
    message: str = Field(min_length=1)
    stage: Optional[str] = None
    traceback_tail: Optional[str] = None
    hint: Optional[str] = None


class RunRecord(StrictModel):
    schema_version: int = SCHEMA_VERSION
    run_id: Slug
    experiment_id: Optional[Slug] = None
    parent_run_id: Optional[Slug] = None      # retry lineage; never overwrite the parent
    idempotency_key: Optional[str] = Field(default=None, max_length=256)
    task: TaskType
    name: Optional[str] = None
    dataset_id: Optional[Slug] = None
    model_key: Optional[Slug] = None
    status: RunStatus
    status_reason: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    config_sha256: Optional[Sha256] = None
    device: Optional[str] = None
    worker: Optional[WorkerInfo] = None
    exit_code: Optional[int] = None
    error: Optional[RunError] = None
    last_event_seq: int = Field(default=0, ge=0)
    last_heartbeat_at: Optional[datetime] = None
    progress_epoch: Optional[int] = Field(default=None, ge=0)
    progress_total_epochs: Optional[int] = Field(default=None, ge=1)
    result_id: Optional[Slug] = None

    @model_validator(mode="after")
    def _lifecycle_consistency(self) -> "RunRecord":
        if self.status in TERMINAL_STATUSES and self.finished_at is None:
            raise ValueError(f"{self.status.value} run must have finished_at")
        if self.status in (RunStatus.running, RunStatus.cancel_requested) and self.started_at is None:
            raise ValueError(f"{self.status.value} run must have started_at")
        if self.status == RunStatus.failed and self.error is None:
            raise ValueError("failed run must carry an error")
        if self.status == RunStatus.interrupted and not self.status_reason:
            raise ValueError("interrupted run must state why (e.g. service restarted)")
        if self.status == RunStatus.queued and (self.started_at or self.worker):
            raise ValueError("queued run cannot have started or have a worker")
        return self


def heartbeat_is_stale(record: RunRecord, now: datetime, timeout: timedelta) -> bool:
    """True when a run claims to be active but has not been heard from."""
    if record.status not in (RunStatus.running, RunStatus.cancel_requested):
        return False
    last = record.last_heartbeat_at or record.started_at
    return last is None or (now - last) > timeout


class RunEventType(str, Enum):
    status = "status"          # {"from": ..., "to": ..., "reason": ...}
    progress = "progress"      # {"epoch": i, "total_epochs": n, "phase": "train"|"val"|"test"}
    metric = "metric"          # {"epoch": i, "split": "val", "values": {"NMSE": -30.1, ...}}
    log = "log"                # {"level": "info", "line": "..."}
    artifact = "artifact"      # {"artifact_id": ..., "kind": ..., "path": ...}
    checkpoint = "checkpoint"  # {"epoch": i, "metric": "NMSE", "value": -30.1, "path": ...}
    heartbeat = "heartbeat"    # {}
    error = "error"            # RunError fields


class RunEvent(StrictModel):
    """Append-only, strictly increasing ``seq`` per run. Clients resume from
    the last ``seq`` they saw; when the server can no longer replay from that
    point it answers with a snapshot requirement instead of guessing."""

    seq: int = Field(ge=1)
    run_id: Slug
    ts: datetime
    type: RunEventType
    payload: Dict[str, Any] = Field(default_factory=dict)
