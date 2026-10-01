"""Private, host-pull GPU bridge. The VM never initiates host/LAN connections."""
from __future__ import annotations

import base64
import json
import io
import os
import re
import secrets
import tempfile
import threading
import time
import zipfile
from dataclasses import dataclass, field

from opendpd.services.workspace import write_json_atomic
from opendpd.runtime.arena_limits import ARENA_MAX_RUNTIME_SECONDS
from opendpd.web.gpu_archive import MAX_BYTES, pack, unpack

LEASE_SECONDS = 30
ARENA_OUTPUTS = {"result.json": 2 * 1024 * 1024, "result.progress.json": 4096,
                 "logs/worker.log": 64 * 1024 * 1024}


@dataclass
class Job:
    id: str
    supervisor: object
    run_id: str
    expires_at: float
    lease: str = ""
    last_seen: float = field(default_factory=time.monotonic)
    done: bool = False
    kind: str = "run"

    @property
    def root(self):
        if self.kind == "arena":
            return self.supervisor.job_root(self.run_id)
        return self.supervisor.ws.run_dir(self.run_id)

    def cancelled(self):
        return (self.done or time.time() >= self.expires_at or (self.root / "CANCEL").exists()
                or not self.supervisor.worker_alive(self.run_id))


class GpuBroker:
    def __init__(self):
        self.lock = threading.RLock()
        self.jobs: dict[str, Job] = {}
        self.last_seen = 0.0
        self.name = None
        self.telemetry = None
        self.telemetry_seen = None
        self.arena_protocol_sha256 = None

    def record_resources(self, payload):
        from opendpd.schemas.system import MachineLoad
        # Only the private host agent can supply telemetry. Validate and copy a
        # fixed numeric schema; never forward arbitrary fields or process data.
        sample = MachineLoad.model_validate(payload)
        if sample.sampled_at is None or sample.sampled_at.tzinfo is None:
            raise ValueError('telemetry requires a timezone-aware timestamp')
        with self.lock:
            self.telemetry, self.telemetry_seen = sample, time.monotonic()

    def resource_status(self):
        from opendpd.schemas.system import ResourceStatus, MachineLoad
        from opendpd.services.server_status import STALE_SECONDS
        with self.lock:
            age = max(0., time.monotonic() - self.telemetry_seen) if self.telemetry_seen is not None else None
            return ResourceStatus(load=self.telemetry.model_copy(deep=True) if self.telemetry else MachineLoad(),
                                  age_seconds=age, stale=age is None or age > STALE_SECONDS)

    def devices(self):
        available = time.monotonic() - self.last_seen < LEASE_SECONDS
        return {"cuda": {"detected": available, "count": int(available), "name": self.name}, "mps": {"detected": False}}

    def enqueue(self, supervisor, record):
        with self.lock:
            kind = "arena" if getattr(supervisor, "job_kind", None) == "arena" else "run"
            config = supervisor.manager.config
            seconds = (min(ARENA_MAX_RUNTIME_SECONDS, getattr(config, "arena_max_runtime_seconds", ARENA_MAX_RUNTIME_SECONDS))
                       if kind == "arena" else config.max_runtime_seconds)
            job = Job(secrets.token_hex(16), supervisor, record.run_id,
                      min(supervisor.expires_at, time.time() + seconds), kind=kind)
            self.jobs[job.id] = job
            return job

    def cancel_owner(self, supervisor):
        """Stop an adapter's jobs without removing its tenant workspace."""
        with self.lock:
            for job in self.jobs.values():
                if job.supervisor is supervisor and not job.done:
                    self.finish(job, 3)

    def retire(self, supervisor, directory):
        """Synchronize expiry with result extraction, then remove the tenant atomically."""
        import shutil
        with self.lock:
            for key, job in list(self.jobs.items()):
                if job.supervisor is supervisor:
                    job.done = True
                    self.jobs.pop(key, None)
            try:
                shutil.rmtree(directory)
            except FileNotFoundError:
                pass

    def sweep(self):
        with self.lock:
            for key, job in list(self.jobs.items()):
                if job.done or not job.root.exists():
                    self.jobs.pop(key, None)
                elif time.monotonic() - job.last_seen > LEASE_SECONDS or time.time() >= job.expires_at:
                    self.finish(job, 1, "GPU worker lease expired; the experiment was stopped")

    def arena_available(self, protocol_sha256):
        with self.lock:
            return (self.arena_protocol_sha256 is not None
                    and self.arena_protocol_sha256 == protocol_sha256
                    and time.monotonic() - self.last_seen < LEASE_SECONDS)

    def poll(self, name, arena_protocol_sha256=None):
        if arena_protocol_sha256 is not None and (not isinstance(arena_protocol_sha256, str)
                or re.fullmatch(r"[a-f0-9]{64}", arena_protocol_sha256) is None):
            raise ValueError("invalid Arena capability")
        with self.lock:
            self.sweep()
            self.last_seen = time.monotonic()
            self.name = str(name)[:120]
            self.arena_protocol_sha256 = arena_protocol_sha256
            # Exactly one remote container, even if a VM proxy was killed early.
            if any(j.lease and not j.done for j in self.jobs.values()):
                return None
            for job in self.jobs.values():
                if job.cancelled():
                    self.finish(job, 3)
                    continue
                if job.kind == "arena":
                    from opendpd.core.arena import protocol
                    if not self.arena_available(protocol().protocol_sha256):
                        continue
                job.lease = secrets.token_hex(32)
                job.last_seen = time.monotonic()
                return {"id": job.id, "run_id": job.run_id, "lease": job.lease,
                        "expires_at": job.expires_at, "kind": job.kind}
            return None

    def get(self, identifier, lease):
        job = self.jobs.get(identifier)
        if not job or not job.lease or not secrets.compare_digest(job.lease, lease):
            raise ValueError("unknown GPU lease")
        return job

    def input(self, job):
        if job.kind == "arena":
            root = job.supervisor.job_input_root(job.run_id)
            return pack(root, [job.root / "request.json"])
        root = job.supervisor.ws.root
        # Only this tenant's data and experiment dependencies; no API tokens, DB or caches.
        paths = [root / "workspace.json", *(root / "datasets").rglob("*"), *(root / "runs").rglob("*")]
        return pack(root, paths)

    def update(self, job, payload):
        with self.lock:
            job.last_seen = self.last_seen = time.monotonic()
            if job.cancelled():
                return {"continue": False}
            progress = None
            if job.kind == "arena" and payload.get("arena_progress"):
                from opendpd.schemas.arena import ArenaProgress
                encoded = payload["arena_progress"]
                if not isinstance(encoded, str) or len(encoded) > (4096 + 2) // 3 * 4:
                    raise ValueError("Arena progress exceeds limit")
                raw = base64.b64decode(encoded, validate=True)
                if len(raw) > 4096:
                    raise ValueError("Arena progress exceeds limit")
                progress = ArenaProgress.model_validate(json.loads(raw))
            offsets = {}
            streams = [("log", "logs/worker.log")]
            if job.kind == "run":
                streams.append(("events", "events.jsonl"))
            for key, name in streams:
                path = job.root / name
                size = path.stat().st_size if path.exists() else 0
                data = base64.b64decode(payload.get(key, ""), validate=True)
                offset = payload.get(key + "_offset", 0)
                if not isinstance(offset, int) or offset < 0 or len(data) > 256 * 1024 or size + len(data) > 64 * 1024 * 1024:
                    raise ValueError("invalid GPU stream chunk")
                if offset == size and data:
                    path.parent.mkdir(exist_ok=True)
                    with path.open("ab") as stream:
                        stream.write(data)
                    size += len(data)
                offsets[key + "_offset"] = size
            if job.kind == "run" and payload.get("live"):
                raw = base64.b64decode(payload["live"], validate=True)
                if len(raw) > 2 * 1024 * 1024:
                    raise ValueError("GPU live snapshot exceeds limit")
                write_json_atomic(job.root / "live.json", json.loads(raw))
            if progress is not None:
                write_json_atomic(job.root / "result.progress.json", progress)
            return {"continue": True, **offsets}

    def result(self, job, data, exit_code):
        with self.lock:
            if job.done:
                return
            if not job.root.exists() or time.time() >= job.expires_at:
                self.finish(job, 3)
                return
            if job.kind == "arena":
                if job.cancelled():
                    self.finish(job, 3)
                    return
                self._arena_result(job, data, exit_code)
                if job.cancelled():
                    self.finish(job, 3)
                    return
            else:
                unpack(data, job.root)
            self.finish(job, exit_code)

    @staticmethod
    def _arena_result(job, data, exit_code):
        # Validate the whole archive before copying any files; canonical Arena
        # request/submission records never belong to this transfer directory.
        if len(data) > MAX_BYTES:
            raise ValueError("Arena result exceeds transfer limit")
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                members = archive.infolist()
                if any(item.filename not in ARENA_OUTPUTS or item.file_size > ARENA_OUTPUTS[item.filename]
                       for item in members):
                    raise ValueError("invalid Arena result file")
                if exit_code == 0 and not any(item.filename == "result.json" for item in members):
                    raise ValueError("Arena evaluation produced no result")
        except zipfile.BadZipFile as exc:
            raise ValueError("invalid Arena result archive") from exc
        with tempfile.TemporaryDirectory(prefix=".arena-import-", dir=job.root.parent) as temporary:
            from pathlib import Path
            staging = Path(temporary)
            unpack(data, staging)
            progress = staging / "result.progress.json"
            if progress.exists():
                from opendpd.schemas.arena import ArenaProgress
                ArenaProgress.model_validate(json.loads(progress.read_bytes()))
            if job.cancelled():
                return
            for name in ARENA_OUTPUTS:
                source, target = staging / name, job.root / name
                if source.exists():
                    if target.is_symlink() or target.parent.is_symlink():
                        raise ValueError("Arena result cannot follow links")
                    target.parent.mkdir(exist_ok=True)
                    os.replace(source, target)

    def checkpoint(self, job, payload):
        if job.kind == "arena":
            raise ValueError("Arena jobs do not publish checkpoints")
        from opendpd.services.model_download import MAX_MODEL_BYTES, publish_model
        with self.lock:
            job.last_seen = self.last_seen = time.monotonic()
            if job.cancelled():
                return {'continue': False}
            encoded = payload.get('data', '')
            if not isinstance(encoded, str) or len(encoded) > (MAX_MODEL_BYTES + 2) // 3 * 4:
                raise ValueError('model snapshot exceeds limit')
            data = base64.b64decode(encoded, validate=True)
            publish_model(job.root, data, epoch=payload['epoch'], sha256=payload['sha256'])
            return {'continue': True}

    @staticmethod
    def finish(job, code, reason=None):
        if job.root.exists():
            if reason:
                with (job.root / "logs" / "worker.log").open("ab") as log:
                    log.write((reason + "\n").encode())
            write_json_atomic(job.root / ".gpu-result", {"exit_code": code})
        job.done = True
