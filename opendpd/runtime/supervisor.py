"""Single in-process supervisor: queue, spawn, observe and finalise workers.

Responsibilities (plan S03):

- one worker subprocess per run, serial per device (``max_per_device``);
- events tailed from ``events.jsonl`` into the SQLite store (the authority);
- cooperative cancel via a ``CANCEL`` file, forced termination after a
  bounded grace period, process-tree cleanup;
- recovery at start-up: runs left ``running``/``queued`` by a crashed or
  restarted service become ``interrupted`` with a stated reason;
- bounded shutdown that leaves no unmarked worker behind.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from opendpd.runtime.db import RunStore
from opendpd.runtime.procs import is_same_process, kill_tree, process_identity
from opendpd.runtime.worker import CANCEL_FILE, EVENTS_FILE
from opendpd.schemas import (
    ExperimentConfig,
    RunError,
    RunEventType,
    RunRecord,
    RunStatus,
    TERMINAL_STATUSES,
    WorkerInfo,
)
from opendpd.services import experiments
from opendpd.services.workspace import Workspace

log = logging.getLogger("opendpd.supervisor")


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class _Active:
    run_id: str
    process: subprocess.Popen
    device_key: str
    log_file: object
    events_path: Path
    events_offset: int = 0
    cancel_requested_at: Optional[datetime] = None
    killed: bool = False
    partial: str = ""
    started_at: datetime = field(default_factory=_now)
    owns_dispatch_slot: bool = False


class Supervisor:
    def __init__(self, workspace: Workspace, store: RunStore, *, max_per_device: int = 1,
                 poll_interval: float = 0.25, cancel_grace: float = 30.0, heartbeat_timeout: float = 60.0,
                 worker_env: Optional[Dict[str, str]] = None, dispatch_slots=None,
                 max_runtime_seconds: Optional[float] = None, inherit_worker_env: bool = True):
        self.ws = workspace
        self.store = store
        self.max_per_device = max_per_device
        self.poll_interval = poll_interval
        self.cancel_grace = cancel_grace
        self.heartbeat_timeout = timedelta(seconds=heartbeat_timeout)
        self.worker_env = worker_env or {}
        self.dispatch_slots = dispatch_slots
        self.max_runtime_seconds = max_runtime_seconds
        self.inherit_worker_env = inherit_worker_env
        self._active: Dict[str, _Active] = {}
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._accepting = True
        self._thread: Optional[threading.Thread] = None
        self._wake = threading.Event()

    # -- lifecycle -----------------------------------------------------------
    def start(self) -> None:
        self.recover()
        self._thread = threading.Thread(target=self._loop, name="opendpd-supervisor", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 10.0) -> None:
        """Stop accepting work, ask workers to stop, wait a bounded time, then
        terminate what is left and mark those runs interrupted."""
        self._accepting = False
        with self._lock:
            active = list(self._active.values())
            for a in active:
                self._request_cancel_file(a)
        deadline = time.monotonic() + timeout
        for a in active:
            remaining = max(0.0, deadline - time.monotonic())
            try:
                a.process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                pass
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        with self._lock:
            for a in list(self._active.values()):
                self._drain_events(a)
                if a.process.poll() is None:
                    kill_tree(a.process.pid, grace_seconds=2)
                    a.killed = True
                self._finalize(a, shutdown=True)
            self._active.clear()

    def index_workspace(self, run_id: Optional[str] = None) -> int:
        """Register finished runs another entry point (CLI, Python API) wrote into this workspace.
        One workspace, three entry points: a run made by ``opendpd run`` must be visible here.
        Runs still active elsewhere are left to the process that owns them."""
        known = set(self.store.run_ids())
        candidates = [run_id] if run_id is not None else self.ws.list_run_ids()
        added = 0
        for rid in candidates:
            if rid in known:
                continue
            try:
                record = experiments.load_run(self.ws, rid)
            except Exception:  # noqa: BLE001 - a foreign or half-written run.json is skipped, never guessed
                continue
            if record.status not in TERMINAL_STATUSES:
                continue
            self.store.upsert_run(record)
            added += 1
        return added

    def recover(self) -> None:
        """Mark runs left active by a previous service instance and index runs made outside the service."""
        self.index_workspace()
        for status in (RunStatus.running, RunStatus.cancel_requested, RunStatus.queued):
            for record in self.store.list_runs(status=status, limit=10 ** 6):
                if status == RunStatus.queued:
                    reason = "service restarted before the run started; resubmit to retry"
                else:
                    reason = "service restarted while the run was in progress; the worker was not found alive"
                    w = record.worker
                    if w is not None and is_same_process(w.pid, w.create_time):
                        kill_tree(w.pid)
                        reason = ("service restarted while the run was in progress; the orphaned worker "
                                  f"(pid {w.pid}) was terminated")
                self._ingest_file_events(record.run_id)
                self._mark_interrupted(record.run_id, reason)

    # -- public operations ------------------------------------------------------
    def submit(self, config: ExperimentConfig, *, name: Optional[str] = None,
               idempotency_key: Optional[str] = None, parent_run_id: Optional[str] = None) -> RunRecord:
        if not self._accepting:
            raise RuntimeError("the service is shutting down and no longer accepts runs")
        with self._lock:
            if idempotency_key:
                existing = self.store.find_idempotent(idempotency_key)
                if existing is not None:
                    return existing
            record = experiments.create_run(self.ws, config, name=name, idempotency_key=None,
                                            parent_run_id=parent_run_id)
            record = record.model_copy(update={"idempotency_key": idempotency_key})
            experiments.save_run(self.ws, record)
            self.store.upsert_run(record)
            self.store.append_event(record.run_id, RunEventType.status, {"from": None, "to": "queued"})
        self._wake.set()
        return record

    def cancel(self, run_id: str) -> RunRecord:
        """Idempotent. Queued runs are cancelled at once; running runs enter
        ``cancel_requested`` until the worker exits."""
        with self._lock:
            record = self.store.get_run(run_id)
            if record is None:
                raise KeyError(run_id)
            if record.status == RunStatus.queued:
                record = self.store.transition(run_id, RunStatus.cancelled, reason="cancelled before start",
                                               finished_at=_now())
                experiments.save_run(self.ws, record)
                return record
            if record.status == RunStatus.running:
                record = self.store.transition(run_id, RunStatus.cancel_requested, reason="cancel requested by user")
                active = self._active.get(run_id)
                if active is not None:
                    self._request_cancel_file(active)
                return record
            return record   # cancel_requested or terminal: nothing more to do

    def retry(self, run_id: str, *, name: Optional[str] = None) -> RunRecord:
        """Create a *new* run from the parent's user config (the failed record is kept)."""
        run_dir = self.ws.run_dir(run_id)
        config = ExperimentConfig.model_validate(json.loads((run_dir / experiments.USER_CONFIG_FILE).read_text()))
        return self.submit(config, name=name, parent_run_id=run_id)

    @property
    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def active_run_ids(self) -> List[str]:
        with self._lock:
            return list(self._active)

    # -- loop --------------------------------------------------------------------
    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._tick()
            except Exception:  # noqa: BLE001 - the loop must survive
                log.exception("supervisor tick failed")
            self._wake.wait(self.poll_interval)
            self._wake.clear()

    def _tick(self) -> None:
        with self._lock:
            for a in list(self._active.values()):
                self._drain_events(a)
                code = a.process.poll()
                if code is not None:
                    self._drain_events(a)
                    self._finalize(a)
                    del self._active[a.run_id]
                    continue
                if (self.max_runtime_seconds is not None and a.cancel_requested_at is None
                        and (_now() - a.started_at).total_seconds() >= self.max_runtime_seconds):
                    self.cancel(a.run_id)
                if a.cancel_requested_at and not a.killed:
                    waited = (_now() - a.cancel_requested_at).total_seconds()
                    if waited > self.cancel_grace:
                        log.warning("run %s did not stop within %.0fs; terminating", a.run_id, self.cancel_grace)
                        kill_tree(a.process.pid, grace_seconds=3)
                        a.killed = True
            if self._accepting:
                self._dispatch()

    def _dispatch(self) -> None:
        busy: Dict[str, int] = {}
        for a in self._active.values():
            busy[a.device_key] = busy.get(a.device_key, 0) + 1
        for record in reversed(self.store.list_runs(status=RunStatus.queued, limit=10 ** 6)):  # FIFO
            key = self._device_key(record)
            if busy.get(key, 0) >= self.max_per_device:
                continue
            if self.dispatch_slots is not None and not self.dispatch_slots.acquire(blocking=False):
                continue
            try:
                self._spawn(record)
            finally:
                if self.dispatch_slots is not None:
                    if record.run_id in self._active:
                        self._active[record.run_id].owns_dispatch_slot = True
                    else:
                        self.dispatch_slots.release()
            busy[key] = busy.get(key, 0) + 1

    @staticmethod
    def _device_key(record: RunRecord) -> str:
        return record.device or "cpu"

    def _spawn(self, record: RunRecord) -> None:
        run_dir = self.ws.run_dir(record.run_id)
        (run_dir / "logs").mkdir(exist_ok=True)
        log_path = run_dir / "logs" / "worker.log"
        log_file = open(log_path, "ab", buffering=0)
        env = dict(os.environ) if self.inherit_worker_env else {"PATH": os.defpath, "LANG": "C.UTF-8"}
        env.update({"MPLBACKEND": "Agg", "TQDM_DISABLE": "1", "PYTHONUNBUFFERED": "1", "KMP_DUPLICATE_LIB_OK": "TRUE"})
        env.update(self.worker_env)
        creation = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
        try:
            process = subprocess.Popen(
                self.worker_command(record),
                cwd=str(run_dir), stdout=log_file, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env,
                **creation)
        except OSError as err:
            log_file.close()
            record = self.store.transition(record.run_id, RunStatus.failed, reason="worker could not start",
                                           finished_at=_now(),
                                           error=RunError(code="spawn_failed", stage="spawn", message=str(err)))
            experiments.save_run(self.ws, record)
            return
        ident = process_identity(process.pid)
        worker = WorkerInfo(pid=process.pid, create_time=ident[1] if ident else time.time(),
                            host=os.uname().nodename if hasattr(os, "uname") else "localhost")
        self.store.transition(record.run_id, RunStatus.running, reason=None, started_at=_now(), worker=worker,
                              last_heartbeat_at=_now())
        self._active[record.run_id] = _Active(run_id=record.run_id, process=process,
                                              device_key=self._device_key(record), log_file=log_file,
                                              events_path=run_dir / EVENTS_FILE)

    def worker_command(self, record: RunRecord) -> List[str]:
        return [sys.executable, "-m", "opendpd.runtime.worker", "--workspace", str(self.ws.root),
                "--run-id", record.run_id]

    def _request_cancel_file(self, active: _Active) -> None:
        cancel = self.ws.run_dir(active.run_id) / CANCEL_FILE
        if not cancel.exists():
            cancel.write_text(_now().isoformat())
        if active.cancel_requested_at is None:
            active.cancel_requested_at = _now()

    # -- events --------------------------------------------------------------------
    def _drain_events(self, active: _Active) -> None:
        if not active.events_path.exists():
            return
        with open(active.events_path, "rb") as f:
            f.seek(active.events_offset)
            chunk = f.read()
        if not chunk:
            return
        active.events_offset += len(chunk)
        text = active.partial + chunk.decode("utf-8", errors="replace")
        lines = text.split("\n")
        active.partial = lines.pop()   # incomplete trailing line, if any
        for line in lines:
            self._ingest_line(active.run_id, line)

    def _ingest_file_events(self, run_id: str) -> None:
        """Import any events a dead worker left behind (recovery path)."""
        path = self.ws.run_dir(run_id) / EVENTS_FILE
        if not path.exists():
            return
        already = self.store.events_after(run_id, 0, limit=10 ** 9)
        seen = {(e.ts.isoformat(), e.type.value) for e in already}
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (raw.get("ts"), raw.get("type")) in seen:
                continue
            self._ingest_line(run_id, line)

    def _ingest_line(self, run_id: str, line: str) -> None:
        line = line.strip()
        if not line:
            return
        try:
            raw = json.loads(line)
            type_ = RunEventType(raw["type"])
            payload = raw.get("payload") or {}
            ts = datetime.fromisoformat(raw["ts"])
        except (ValueError, KeyError, TypeError):
            log.warning("run %s: unreadable event line ignored", run_id)
            return
        if type_ == RunEventType.status:
            # The worker's own status bookkeeping is informational; the store
            # decides status transitions when the process is observed.
            return
        updates = {"last_heartbeat_at": _now()}
        if type_ == RunEventType.progress and payload.get("scope") != "live":
            updates["progress_epoch"] = int(payload.get("epoch", 0)) + 1
            updates["progress_total_epochs"] = int(payload.get("total_epochs") or 1)
        self.store.append_event(run_id, type_, payload, ts=ts, **updates)

    # -- finalisation ------------------------------------------------------------------
    def _finalize(self, active: _Active, shutdown: bool = False) -> None:
        run_id = active.run_id
        if active.owns_dispatch_slot:
            self.dispatch_slots.release()
            active.owns_dispatch_slot = False
        try:
            active.log_file.close()
        except OSError:
            pass
        code = active.process.poll()
        record = self.store.get_run(run_id)
        if record is None or record.status in TERMINAL_STATUSES:
            return
        file_record = self._file_record(run_id)
        manifest = experiments.load_artifacts(self.ws, run_id)
        error: Optional[RunError] = None
        result_id = file_record.result_id if file_record else None

        if shutdown and (active.killed or code is None):
            target, reason = RunStatus.interrupted, "service shut down while the run was in progress"
        elif file_record is not None and file_record.status in TERMINAL_STATUSES:
            target, reason, error = file_record.status, file_record.status_reason, file_record.error
            if target == RunStatus.succeeded and (manifest is None or not manifest.complete):
                target, reason = RunStatus.failed, "required artifacts missing after a successful exit"
                error = RunError(code="artifacts_incomplete", stage="finalize", message=reason)
        else:
            target = RunStatus.failed
            reason = f"worker exited with code {code} without recording a result"
            hint = "out-of-memory kills and forced terminations end this way; see logs/worker.log"
            if active.killed:
                target, reason = RunStatus.cancelled, "worker terminated after the cancel grace period"
            else:
                error = RunError(code="worker_died", stage="run", message=reason, hint=hint)

        if record.status == RunStatus.cancel_requested and target == RunStatus.succeeded:
            target = RunStatus.cancelled
            reason = "worker finished after cancellation was requested; artifacts and result were kept"
        if record.status == RunStatus.cancel_requested and target == RunStatus.failed and active.killed:
            target, reason, error = RunStatus.cancelled, "worker terminated after the cancel grace period", None
        if record.status == RunStatus.cancel_requested and code == 3:
            target, reason, error = RunStatus.cancelled, "worker stopped after cancellation was requested", None

        # Timestamps come from the executor's own clock when the worker recorded them: the same two
        # points `opendpd run` stamps in-process, so durations compare across paths. The supervisor's
        # clock (spawn time, exit noticed at the next tick) is the fallback for workers that died.
        stamps: Dict[str, Any] = {"finished_at": _now()}
        if file_record is not None and file_record.finished_at is not None:
            stamps["finished_at"] = file_record.finished_at
            if file_record.started_at is not None and record.started_at is not None \
                    and file_record.started_at >= record.started_at:
                stamps["started_at"] = file_record.started_at
        final = self.store.transition(run_id, target, reason=reason, exit_code=code, error=error, result_id=result_id,
                                      **stamps)
        experiments.save_run(self.ws, final)
        cancel = self.ws.run_dir(run_id) / CANCEL_FILE
        if cancel.exists():
            cancel.unlink()

    def _file_record(self, run_id: str) -> Optional[RunRecord]:
        try:
            return experiments.load_run(self.ws, run_id)
        except Exception:  # noqa: BLE001 - a half-written file is treated as absent
            return None

    def _mark_interrupted(self, run_id: str, reason: str) -> None:
        record = self.store.transition(run_id, RunStatus.interrupted, reason=reason, finished_at=_now())
        try:
            experiments.save_run(self.ws, record)
        except Exception:  # noqa: BLE001 - the run dir may be gone; the store is the authority
            log.warning("could not mirror interrupted status to run.json for %s", run_id)


def open_runtime(workspace_root: Path, **kwargs) -> tuple[Workspace, RunStore, Supervisor]:
    ws = Workspace.open_or_create(workspace_root)
    store = RunStore(ws.root / "metadata.sqlite")
    return ws, store, Supervisor(ws, store, **kwargs)
