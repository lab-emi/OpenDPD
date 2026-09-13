"""Private, host-pull GPU bridge. The VM never initiates host/LAN connections."""
from __future__ import annotations

import base64
import json
import secrets
import threading
import time
from dataclasses import dataclass, field

from opendpd.services.workspace import write_json_atomic
from opendpd.web.gpu_archive import pack, unpack

LEASE_SECONDS = 30


@dataclass
class Job:
    id: str
    supervisor: object
    run_id: str
    expires_at: float
    lease: str = ""
    last_seen: float = field(default_factory=time.monotonic)
    done: bool = False

    @property
    def root(self):
        return self.supervisor.ws.run_dir(self.run_id)

    def cancelled(self):
        active = self.supervisor._active.get(self.run_id)
        return (self.done or time.time() >= self.expires_at or (self.root / "CANCEL").exists()
                or active is None or active.process.poll() is not None)


class GpuBroker:
    def __init__(self):
        self.lock = threading.RLock()
        self.jobs: dict[str, Job] = {}
        self.last_seen = 0.0
        self.name = None

    def devices(self):
        available = time.monotonic() - self.last_seen < LEASE_SECONDS
        return {"cuda": {"detected": available, "count": int(available), "name": self.name}, "mps": {"detected": False}}

    def enqueue(self, supervisor, record):
        with self.lock:
            job = Job(secrets.token_hex(16), supervisor, record.run_id,
                      min(supervisor.expires_at, time.time() + supervisor.manager.config.max_runtime_seconds))
            self.jobs[job.id] = job

    def sweep(self):
        with self.lock:
            for key, job in list(self.jobs.items()):
                if job.done or not job.root.exists():
                    self.jobs.pop(key, None)
                elif time.monotonic() - job.last_seen > LEASE_SECONDS or time.time() >= job.expires_at:
                    self.finish(job, 1, "GPU worker lease expired; the experiment was stopped")

    def poll(self, name):
        with self.lock:
            self.sweep()
            self.last_seen = time.monotonic()
            self.name = str(name)[:120]
            # Exactly one remote container, even if a VM proxy was killed early.
            if any(j.lease and not j.done for j in self.jobs.values()):
                return None
            for job in self.jobs.values():
                if job.cancelled():
                    self.finish(job, 3)
                    continue
                job.lease = secrets.token_hex(32)
                job.last_seen = time.monotonic()
                return {"id": job.id, "run_id": job.run_id, "lease": job.lease, "expires_at": job.expires_at}
            return None

    def get(self, identifier, lease):
        job = self.jobs.get(identifier)
        if not job or not job.lease or not secrets.compare_digest(job.lease, lease):
            raise ValueError("unknown GPU lease")
        return job

    def input(self, job):
        root = job.supervisor.ws.root
        # Only this tenant's data and experiment dependencies; no API tokens, DB or caches.
        paths = [root / "workspace.json", *(root / "datasets").rglob("*"), *(root / "runs").rglob("*")]
        return pack(root, paths)

    def update(self, job, payload):
        with self.lock:
            job.last_seen = self.last_seen = time.monotonic()
            if job.cancelled():
                return {"continue": False}
            offsets = {}
            for key, name in [("log", "logs/worker.log"), ("events", "events.jsonl")]:
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
            if payload.get("live"):
                raw = base64.b64decode(payload["live"], validate=True)
                if len(raw) > 2 * 1024 * 1024:
                    raise ValueError("GPU live snapshot exceeds limit")
                write_json_atomic(job.root / "live.json", json.loads(raw))
            return {"continue": True, **offsets}

    def result(self, job, data, exit_code):
        with self.lock:
            if job.done:
                return
            if not job.root.exists() or time.time() >= job.expires_at:
                self.finish(job, 3)
                return
            unpack(data, job.root)
            self.finish(job, exit_code)

    def checkpoint(self, job, payload):
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
