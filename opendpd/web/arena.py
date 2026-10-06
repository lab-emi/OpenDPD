"""Opt-in Arena jobs on the shared hosted queue and isolated GPU bridge.

This adapter never launches the Arena runner on the API VM. It queues a typed,
server-owned request for the same pinned-image container used by ordinary jobs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
import threading
import time

from opendpd.schemas import RunStatus, TERMINAL_STATUSES
from opendpd.runtime.arena_limits import ARENA_MAX_RUNTIME_SECONDS
from opendpd.schemas.arena import ArenaProgress, ArenaRow
from opendpd.services.arena import ArenaController
from opendpd.services.workspace import Conflict, InvalidInput, read_json, write_json_atomic
from opendpd.web.gpu_archive import read_regular
from opendpd.web.gpu_broker import ARENA_OUTPUTS
from opendpd.web.policy import reject


@dataclass
class ArenaQueueRecord:
    run_id: str
    created_at: datetime
    status: RunStatus = RunStatus.queued
    ready: threading.Event = field(default_factory=threading.Event)
    error: str | None = None
    owns_slot: bool = False


class ArenaQueue:
    """The small scheduler interface already consumed by TenantManager."""
    job_kind = "arena"

    def __init__(self, controller, supervisor):
        self.controller = controller
        self.ws, self.manager = controller.ws, controller.manager
        self.ip_key, self.expires_at = supervisor.ip_key, supervisor.expires_at
        self._accepting = True
        self.records = {}
        self._active = {}
        self.store = self

    def list_runs(self, *, status=None, limit=1000):
        return [record for record in self.records.values()
                if status is None or record.status == status][:limit]

    def count_runs(self, status=None):
        return len(self.list_runs(status=status))

    def active_count(self):
        return len(self._active)

    def job_input_root(self, identifier):
        return self.controller.directory(identifier) / "remote"

    def job_root(self, identifier):
        return self.job_input_root(identifier) / "runs" / identifier

    def expired(self):
        return (not self._accepting or self.controller.stopping.is_set()
                or self.manager.now() >= self.expires_at
                or self.manager.now() >= self.manager.idle_deadline(self.ip_key))

    def worker_alive(self, identifier):
        # The broker calls this with its own lock held: never acquire the
        # admission lock here (dispatch holds admission before broker locks).
        return identifier in self._active and not self.expired()

    def add(self, submission):
        directory = self.job_root(submission.submission_id)
        if any(parent.is_symlink() for parent in [directory, *directory.parents]):
            raise ValueError("Arena transfer storage cannot follow links")
        directory.mkdir(parents=True)
        (directory / "logs").mkdir()
        canonical = self.controller.directory(submission.submission_id) / "request.json"
        write_json_atomic(directory / "request.json", read_json(canonical))
        self.records[submission.submission_id] = ArenaQueueRecord(submission.submission_id, submission.created_at)
        self.manager.scheduled.add(self)
        self.manager.scheduler_wake.set()

    def _tick(self):
        if self.expired():
            for record in self.records.values():
                if record.status == RunStatus.queued:
                    record.error = "The temporary workspace expired before Arena dispatch."
                    record.status = RunStatus.cancelled
                    record.ready.set()
            return
        candidate = self.manager.next_dispatch
        if candidate is None or candidate[0] is not self:
            return
        record = candidate[1]
        if record.status != RunStatus.queued or not self.manager.slots.acquire(blocking=False):
            return
        record.owns_slot = True
        record.status = RunStatus.running
        self._active[record.run_id] = record
        try:
            self.manager.gpu.enqueue(self, record)
        except Exception as exc:
            record.error = f"Arena broker dispatch failed: {type(exc).__name__}"
            self.finish(record.run_id, RunStatus.failed)
        finally:
            record.ready.set()

    def finish(self, identifier, status):
        with self.manager.budget_lock:
            record = self.records.get(identifier)
            if record is None:
                return
            record.status = status
            self._active.pop(identifier, None)
            if record.owns_slot:
                record.owns_slot = False
                self.manager.slots.release()
            record.ready.set()
            self.manager.scheduler_wake.set()

    def stop(self):
        with self.manager.budget_lock:
            self._accepting = False
            for record in self.records.values():
                if record.status not in TERMINAL_STATUSES:
                    record.error = "The temporary workspace stopped during Arena evaluation."
                    self.finish(record.run_id, RunStatus.cancelled)
            self.manager.scheduled.discard(self)
            self.manager.gpu.cancel_owner(self)


class HostedArenaController(ArenaController):
    def __init__(self, ws, backbones, supervisor):
        self.manager, self.supervisor = supervisor.manager, supervisor
        self.admitted_units = 0
        super().__init__(ws, backbones,
            enabled=bool(self.manager.config.arena_submissions and self.manager.config.gpu_token),
            evaluator=self._remote_evaluate)
        self.queue = ArenaQueue(self, supervisor)
        supervisor.arena_controller = self

    def pending_count(self):
        return sum(record.status not in TERMINAL_STATUSES for record in self.queue.records.values())

    def catalog(self):
        catalog = super().catalog()
        available = (self.enabled and not self.stopping.is_set()
                     and self.manager.gpu.arena_available(catalog.protocol.protocol_sha256))
        catalog.submissions_available = available
        catalog.submission_unavailable_reason = None if available else (
            "Hosted Arena evaluation requires an enabled isolated worker with this exact protocol. "
            "Official rankings remain available; local Studio can evaluate submissions.")
        return catalog

    def submit(self, request):
        with self.manager.budget_lock:
            protocol = self.protocol()
            if not self.enabled or not self.manager.gpu.arena_available(protocol.protocol_sha256):
                reject(503, "arena_worker_unavailable", "No enabled isolated Arena worker matches this protocol.")
            if self.queue.expired():
                reject(401, "session_expired", "The temporary workspace has expired.")
            if not self.manager.dispatch_healthy or not self.manager.cleanup_healthy:
                reject(503, "service_unavailable", "Hosted compute admission is temporarily unavailable.")
            board = self._board(request.board_id, protocol)
            # Refuse what can never run before it is charged against any budget.
            if request.accepted_protocol_sha256 != protocol.protocol_sha256:
                raise Conflict("The Arena protocol changed. Review the current rules before submitting.")
            descriptor = next((item for item in self.bundled_backbones() if item.key == request.backbone), None)
            if descriptor is None:
                raise InvalidInput("Choose a bundled Arena backbone or a validated user template.")
            # One quota unit per condition and seed. A unit trains that case at
            # every budget of the parameter sweep, inside the Arena runtime guard.
            units = len(board.conditions) * (1 if descriptor.deterministic else len(self.protocol().seeds))
            config = self.manager.config
            ordinary = self.supervisor.store.list_runs(limit=10000)
            if len(ordinary) + self.admitted_units + units > config.runs_per_session:
                reject(429, "run_quota", "This Arena entry exceeds the remaining workspace evaluation budget.")
            if sum(record.status not in TERMINAL_STATUSES for record in ordinary) + self.pending_count() >= config.max_pending:
                reject(429, "queue_full", "Finish or cancel an existing job before submitting another.")
            pending = sum(len(owner.store.list_runs(status=RunStatus.queued, limit=1000)) + len(owner._active)
                          for owner in self.manager.scheduled)
            if pending >= config.max_pending_global:
                reject(429, "queue_full", "The shared compute queue is full.")
            used = self.manager.ip_runs.get(self.supervisor.ip_key, 0)
            if used + units > config.runs_per_ip or self.manager.total_runs + units > config.runs_per_day:
                reject(429, "run_quota", "This Arena entry exceeds the shared evaluation budget.")
            submission = super().submit(request)
            self.queue.add(submission)
            self.admitted_units += units
            self.manager.ip_runs[self.supervisor.ip_key] = used + units
            self.manager.total_runs += units
            return submission

    def _wait_for_dispatch(self, identifier):
        with self.manager.budget_lock:
            record = self.queue.records.get(identifier)
        if record is None:
            raise RuntimeError("Arena job was not admitted to the shared queue.")
        while not record.ready.wait(0.2):
            if self.queue.expired():
                raise RuntimeError("The temporary workspace expired before Arena dispatch.")
        if record.error or self.queue.expired():
            raise RuntimeError(record.error or "The temporary workspace expired.")

    def _run(self, identifier):
        try:
            super()._run(identifier)
        finally:
            record = self.get(identifier)
            status = RunStatus.succeeded if record.status == "succeeded" else RunStatus.failed
            self.queue.finish(identifier, status)

    def _remote_evaluate(self, directory: Path):
        identifier = directory.name
        remote = self.queue.job_root(identifier)
        canonical = read_json(directory / "request.json")
        last_progress = None
        deadline = time.monotonic() + min(ARENA_MAX_RUNTIME_SECONDS, self.manager.config.arena_max_runtime_seconds)
        while True:
            if self.queue.expired() or time.monotonic() >= deadline:
                raise RuntimeError("The isolated Arena evaluation expired or was cancelled.")
            raw = read_regular(remote, "result.progress.json", limit=4097)
            if raw and raw != last_progress:
                if len(raw) > 4096:
                    raise ValueError("Arena progress exceeds its transfer limit")
                progress = ArenaProgress.model_validate_json(raw)
                write_json_atomic(directory / "result.progress.json", progress)
                last_progress = raw
            completed = read_regular(remote, ".gpu-result", limit=4096)
            if completed:
                status = json.loads(completed)
                if status.get("exit_code") != 0:
                    raise RuntimeError("The isolated Arena worker failed or its lease expired.")
                result_limit = ARENA_OUTPUTS["result.json"]
                result = read_regular(remote, "result.json", limit=result_limit + 1)
                if not result or len(result) > result_limit:
                    raise ValueError("Arena result is missing or exceeds its transfer limit")
                row = ArenaRow.model_validate_json(result)
                # The sweep derived from this definition is verified again when the
                # controller recomputes the result from its accepted request.
                if row.provenance.get("model_parameters") != canonical["model_parameters"]:
                    raise ValueError("Arena worker returned a different model definition")
                if row.status == "succeeded" and row.provenance.get("model_provenance", {}) != canonical["model_provenance"]:
                    raise ValueError("Arena worker returned different template provenance")
                return row
            self.stopping.wait(0.2)

    def stop_accepting(self):
        self.stopping.set()
        self.queue.stop()

    def stop(self):
        self.stop_accepting()
        super().stop()
