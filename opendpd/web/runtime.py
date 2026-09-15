"""Anonymous workspaces, bounded admission and expiry for the public VM service."""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import hmac
import ipaddress
import json
import logging
import os
import secrets
import shutil
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from opendpd.runtime.supervisor import Supervisor
from opendpd.schemas import TERMINAL_STATUSES, RunStatus
from opendpd.server.app import create_app
from opendpd.web.policy import WebConfig, check_config, reject
from opendpd.web.gpu_broker import GpuBroker

log = logging.getLogger(__name__)
MARKER = "OpenDPD disposable web workspaces v1\n"
DAY = 86400


def prepare_root(root: Path):
    """Never clear an arbitrary directory. Hold the lock until all runtimes stop."""
    root = Path(root)
    if root.is_symlink() or not root.is_absolute() or root == Path("/"):
        raise ValueError("web root must be an absolute, dedicated directory")
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    marker = root / ".opendpd-disposable"
    if not marker.exists():
        if any(root.iterdir()):
            raise ValueError("web root is not empty and has no disposable-workspace marker")
        marker.write_text(MARKER)
    if marker.is_symlink() or marker.read_text() != MARKER:
        raise ValueError("invalid disposable-workspace marker")
    os.chmod(root, 0o700)
    lock = open(root / ".lock", "a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        lock.close()
        raise RuntimeError("another public service owns this web root; use exactly one ASGI worker") from None
    return lock


def clear_sessions(root: Path):
    """Only run after the service cgroup has stopped (or before its first start)."""
    sessions = root / "sessions"
    if sessions.is_symlink():
        sessions.unlink()
    elif sessions.exists():
        shutil.rmtree(sessions)
    sessions.mkdir(mode=0o700)


def directory_bytes(root: Path) -> int:
    total = 0
    for path in root.rglob('*'):
        try:
            if not path.is_symlink():
                total += path.stat(follow_symlinks=False).st_size
        except FileNotFoundError:
            # Writers replace/remove temporary files during a quota scan.
            continue
    return total


@dataclass
class Tenant:
    identifier: str
    token_hash: str
    ip_key: str
    expires_at: float
    root: Path
    app: object
    context: object
    local_session: object
    inflight: int = 0
    closing: bool = False
    uploading: bool = False
    mutating: bool = False
    last_activity: float = 0.0
    admission_key: str | None = None


@dataclass
class Admission:
    ip_key: str
    last_seen: float
    tenant_hash: str | None = None
    # Only retained for a short, unacknowledged admission retry window.
    access_token: str | None = field(default=None, repr=False)


class WebSupervisor(Supervisor):
    def __init__(self, workspace, store, *, manager, ip_key, expires_at, **kwargs):
        self.manager, self.ip_key, self.expires_at = manager, ip_key, expires_at
        cache = workspace.root / "cache"
        tmp = cache / "tmp"
        tmp.mkdir(parents=True, exist_ok=True)
        super().__init__(workspace, store, dispatch_slots=manager.slots, cancel_grace=5,
                         max_runtime_seconds=manager.config.max_runtime_seconds, inherit_worker_env=False,
                         worker_env={"HOME": str(cache), "TMPDIR": str(tmp), "XDG_CACHE_HOME": str(cache),
                                     "MPLCONFIGDIR": str(cache / "matplotlib"), "OMP_NUM_THREADS": "4",
                                     "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4",
                                     "PYTHONDONTWRITEBYTECODE": "1"}, **kwargs)

    def start(self):
        # Idle visitors do not each need a supervisor and sweep thread. Public
        # sweeps are disabled; one shared dispatcher services submitted work.
        self.recover()

    @property
    def alive(self):
        return self._accepting and self.manager.scheduler.is_alive()

    def stop(self, timeout=0):
        with self.manager.budget_lock:
            self.manager.scheduled.discard(self)
            super().stop(timeout=timeout)

    def worker_command(self, record):
        command = super().worker_command(record)
        if record.device == "cuda" and self.manager.config.gpu_token:
            command[2] = "opendpd.web.gpu_proxy"
        return command

    def _spawn(self, record):
        super()._spawn(record)
        if record.device == "cuda" and self.manager.config.gpu_token and record.run_id in self._active:
            self.manager.gpu.enqueue(self, record)

    def _dispatch(self):
        if (not self._accepting or self.manager.now() >= self.expires_at
                or self.manager.now() >= self.manager.idle_deadline(self.ip_key)):
            return
        candidate = self.manager.next_dispatch
        if candidate is None or candidate[0] is not self:
            return
        record = candidate[1]
        if sum(a.device_key == self._device_key(record) for a in self._active.values()) >= self.max_per_device:
            return
        if not self.dispatch_slots.acquire(blocking=False):
            return
        try:
            self._spawn(record)
        finally:
            if record.run_id in self._active:
                self._active[record.run_id].owns_dispatch_slot = True
            else:
                self.dispatch_slots.release()

    def submit(self, config, **kwargs):
        check_config(config)
        # Admission, including retry and idempotency, is one transaction across
        # ALL workspaces. New anonymous sessions cannot multiply the GPU limit.
        with self.manager.budget_lock:
            if (self.manager.now() >= self.expires_at
                    or self.manager.now() >= self.manager.idle_deadline(self.ip_key)):
                reject(401, "session_expired", "the temporary workspace has expired")
            existing = self.store.find_idempotent(kwargs.get("idempotency_key")) if kwargs.get("idempotency_key") else None
            if existing is not None:
                return existing
            records = self.store.list_runs(limit=10000)
            if len(records) >= self.manager.config.runs_per_session:
                reject(429, "run_quota", "this temporary workspace has reached its run limit")
            if sum(r.status not in TERMINAL_STATUSES for r in records) >= self.manager.config.max_pending:
                reject(429, "queue_full", "finish or cancel an existing run before submitting another")
            pending = sum(len(s.store.list_runs(status=RunStatus.queued, limit=1000)) + len(s._active)
                          for s in self.manager.scheduled)
            if pending >= self.manager.config.max_pending_global:
                reject(429, 'queue_full', 'The shared compute queue is full; try again when queued jobs finish.')
            ip_runs = self.manager.ip_runs.get(self.ip_key, 0)
            if ip_runs >= self.manager.config.runs_per_ip or self.manager.total_runs >= self.manager.config.runs_per_day:
                reject(429, "run_quota", "the shared compute budget for this service interval has been reached")
            record = super().submit(config, **kwargs)
            self.manager.ip_runs[self.ip_key] = ip_runs + 1
            self.manager.total_runs += 1
            self.manager.scheduled.add(self)
            self.manager.scheduler_wake.set()
            return record


class TenantManager:
    def __init__(self, config: WebConfig, *, now=time.time):
        self.config, self.now = config, now
        self.gpu = GpuBroker()
        from opendpd.services.server_status import ResourceSampler
        self.resources = ResourceSampler()
        self.expensive_requests = 0
        self.storage_reserved = 0
        self.status_counts = (0, 0)
        self.status_updated = 0.0
        self.status_lock = threading.Lock()
        self.tenants: dict[str, Tenant] = {}
        self.slots = threading.BoundedSemaphore(config.max_parallel)
        self.budget_lock = threading.RLock()
        self.create_lock = asyncio.Lock()
        self.sweep_lock = asyncio.Lock()
        self.admissions: dict[str, Admission] = {}
        self.quota_cursor = 0
        self.scheduled: set[WebSupervisor] = set()
        self.next_dispatch = None
        self.scheduler_stop = threading.Event()
        self.scheduler_wake = threading.Event()
        self.scheduler = threading.Thread(target=self.dispatch_loop, name='opendpd-web-dispatch', daemon=True)
        self.ip_secret = secrets.token_bytes(32)
        self.ip_sessions: dict[str, int] = {}
        self.ip_activity: dict[str, float] = {}
        self.ip_runs: dict[str, int] = {}
        self.rate: dict[str, tuple[int, int]] = {}
        self.total_runs = 0
        self.publication_reservations: set[tuple[str, str]] = set()
        self.ip_publications: dict[str, int] = {}
        self.day = int(now() // DAY)
        self.inflight = 0
        self.cleanup_healthy = True
        self.lock = None

    async def start(self):
        self.lock = prepare_root(self.config.root)
        clear_sessions(self.config.root)
        (self.config.root / "service-tmp").mkdir(mode=0o700, exist_ok=True)
        self.resources.start()
        self.scheduler.start()

    def dispatch_loop(self):
        while not self.scheduler_stop.is_set():
            try:
                with self.budget_lock:
                    queued = [(s, r) for s in self.scheduled
                              for r in s.store.list_runs(status=RunStatus.queued, limit=1000)]
                    self.next_dispatch = min(queued, key=lambda item: (item[1].created_at, item[1].run_id), default=None)
                    pending = {s for s, _ in queued}
                    for supervisor in list(self.scheduled):
                        supervisor._tick()
                        if supervisor not in pending and not supervisor._active:
                            self.scheduled.discard(supervisor)
            except Exception:
                self.cleanup_healthy = False
                log.exception('shared dispatcher failed; new admission disabled')
            self.scheduler_wake.wait(.25 if self.scheduled else 1)
            self.scheduler_wake.clear()

    def ip_key(self, address: str) -> str:
        ip = ipaddress.ip_address(address)
        # IPv6 privacy addresses in one /64 share an abuse budget. No raw IPs are
        # written into folders, SQLite, application logs or browser-visible data.
        network = str(ipaddress.ip_network(f"{ip}/64", strict=False)) if ip.version == 6 else str(ip)
        return hmac.new(self.ip_secret, network.encode(), hashlib.sha256).hexdigest()

    def rate_limit(self, ip_key: str, limit: int | None = None):
        minute = int(self.now() // 60)
        if len(self.rate) >= 4096 and ip_key not in self.rate:
            self.rate = {k: v for k, v in self.rate.items() if v[0] == minute}
            if len(self.rate) >= 4096:
                reject(429, "service_busy", "the service is busy; try again later")
        stamp, count = self.rate.get(ip_key, (minute, 0))
        count = count + 1 if stamp == minute else 1
        self.rate[ip_key] = (minute, count)
        if count > (limit or self.config.requests_per_minute):
            reject(429, "rate_limited", "too many requests; try again in a minute")

    def reserve_storage(self, amount):
        with self.budget_lock:
            if shutil.disk_usage(self.config.root).free - self.storage_reserved < amount + 64 * 2**20:
                reject(507, 'storage_busy', 'Shared temporary storage is nearly full. Try again after workspaces are cleared.')
            self.storage_reserved += amount

    def release_storage(self, amount):
        with self.budget_lock:
            self.storage_reserved -= amount

    @property
    def cutoff(self):
        return self.next_reset - self.config.drain_seconds

    @property
    def next_reset(self):
        interval = self.config.cleanup_interval_seconds
        return (int(self.now() // interval) + 1) * interval

    def authenticate(self, token: str) -> Tenant:
        digest = hashlib.sha256(token.encode()).hexdigest()
        tenant = self.tenants.get(digest)
        if tenant is None or not self.is_live(tenant):
            reject(401, "session_expired", "the temporary session has expired; start a new session")
        # The first authenticated request acknowledges receipt. No plaintext
        # session capability is retained after that point.
        admission = self.admissions.get(tenant.admission_key)
        if admission is not None:
            admission.access_token = None
        tenant.admission_key = None
        return tenant

    def idle_deadline(self, ip_key: str) -> float:
        return self.ip_activity.get(ip_key, 0.) + self.config.inactivity_seconds

    def close_ip(self, ip_key: str):
        # Revocation is immediate, including while an in-flight download finishes.
        # Cleanup never deletes files underneath a request that already owns them.
        for tenant in list(self.tenants.values()):
            if tenant.ip_key == ip_key:
                tenant.closing = True
                tenant.app.state.supervisor._accepting = False

    def is_live(self, tenant: Tenant) -> bool:
        if self.now() >= self.idle_deadline(tenant.ip_key):
            self.close_ip(tenant.ip_key)
        return not tenant.closing and self.now() < tenant.expires_at

    def activity(self, tenant: Tenant):
        # Only explicit foreground interaction renews this lease. Polls, status
        # reads, queue heartbeats and running jobs never count as user activity.
        if not self.is_live(tenant):
            reject(401, 'session_expired', 'the temporary session has expired; start a new session')
        self.ip_activity[tenant.ip_key] = self.now()
        tenant.last_activity = self.now()

    def prune_admissions(self):
        now = self.now()
        self.admissions = {key: item for key, item in self.admissions.items()
                           if now - item.last_seen < self.config.queue_lease_seconds}

    def waiting_count(self):
        now = self.now()
        return sum(item.tenant_hash is None and now - item.last_seen < self.config.queue_lease_seconds
                   for item in list(self.admissions.values()))

    async def admit(self, ip_key: str, ticket: str | None = None, *, cancel=False):
        """FIFO entry with bearer-only queue capabilities and bounded heartbeat leases.

        Returns (tenant, access token, queued response). Retries with the same
        ticket replay an unacknowledged admission rather than allocating twice.
        """
        ticket = ticket or secrets.token_urlsafe(32)
        digest = hashlib.sha256(ticket.encode()).hexdigest()
        async with self.create_lock:
            self.prune_admissions()
            await self.sweep(quotas=False)
            entry = self.admissions.get(digest)
            if cancel:
                if entry and entry.tenant_hash and entry.access_token:
                    tenant = self.tenants.get(entry.tenant_hash)
                    if tenant:
                        async with self.sweep_lock:
                            await self.remove(tenant)
                self.admissions.pop(digest, None)
                return None, None, None
            if not self.cleanup_healthy:
                reject(503, "cleanup_unavailable", "temporary storage maintenance is unavailable")
            if entry and entry.tenant_hash:
                tenant = self.tenants.get(entry.tenant_hash)
                if tenant and not tenant.closing and self.now() < tenant.expires_at and entry.access_token:
                    return tenant, entry.access_token, None
                reject(410, 'queue_expired', 'This admission has already ended; start a new session.')
            owner = entry.ip_key if entry else ip_key
            if self.ip_sessions.get(owner, 0) >= self.config.sessions_per_ip:
                reject(429, "session_quota", "the daily session limit for this network has been reached")
            if len(self.ip_sessions) >= 4096:
                reject(429, "service_busy", "the service is busy; try again later")
            if entry is None:
                if len(self.admissions) >= self.config.max_waiting + self.config.max_sessions:
                    reject(429, 'queue_full', 'The waiting room is full; try again in a minute.')
                waiting = [item for item in self.admissions.values() if item.tenant_hash is None]
                if len(waiting) >= self.config.max_waiting:
                    reject(429, 'queue_full', 'The waiting room is full; try again in a minute.')
                if sum(item.ip_key == ip_key for item in waiting) >= self.config.waiting_per_ip:
                    reject(429, 'queue_quota', 'This network already has several waiting visitors; try again shortly.')
                entry = Admission(ip_key, self.now())
                self.admissions[digest] = entry
            entry.last_seen = self.now()
            waiting = [key for key, item in self.admissions.items() if item.tenant_hash is None]
            position = waiting.index(digest) + 1
            reason = 'capacity'
            storage = 2 * 2**20
            if self.now() >= self.cutoff:
                reason = 'cleanup'
            elif position == 1 and len(self.tenants) < self.config.max_sessions:
                try:
                    self.reserve_storage(storage)
                except Exception as exc:
                    from fastapi import HTTPException
                    if not isinstance(exc, HTTPException) or exc.status_code != 507:
                        raise
                    reason = 'storage'
                else:
                    try:
                        tenant, token = await self.create(owner)
                    finally:
                        self.release_storage(storage)
                    entry.tenant_hash, entry.access_token = tenant.token_hash, token
                    tenant.admission_key = digest
                    return tenant, token, None
            from datetime import datetime, timezone
            return None, None, {'authenticated': False, 'mode': 'web', 'status': 'queued', 'queue_token': ticket,
                'queue_position': position, 'waiting': len(waiting), 'reason': reason, 'retry_after_seconds': 5,
                'admission_resumes_at': datetime.fromtimestamp(self.next_reset, timezone.utc).isoformat() if reason == 'cleanup' else None}

    async def create(self, ip_key: str):
        # Caller holds create_lock and has reserved storage/admission capacity.
        # Route construction is CPU work; it must not block existing visitors.
        try:
            identifier = secrets.token_hex(24)
            token = secrets.token_urlsafe(32)
            digest = hashlib.sha256(token.encode()).hexdigest()
            root = self.config.root / "sessions" / identifier
            root.mkdir(mode=0o700)
            expires_at = self.cutoff
            (root / "lease.json").write_text(json.dumps({"created_at": self.now(), "expires_at": expires_at}))

            def factory(ws, store, **kwargs):
                return WebSupervisor(ws, store, manager=self, ip_key=ip_key, expires_at=expires_at, **kwargs)

            app = await asyncio.to_thread(create_app, root / "workspace", supervisor_factory=factory, shutdown_timeout=0,
                             allow_dataset_publications=self.config.dataset_publications, monitor_resources=False, start_sweeps=False)
            if self.config.gpu_token:
                app.state.device_detector = self.gpu.devices
            app.state.workspace_label = "Temporary workspace"
            local_session = app.state.sessions.exchange(app.state.sessions.bootstrap_token)
            context = app.router.lifespan_context(app)
            try:
                await context.__aenter__()
                if self.config.gpu_token:
                    app.state.ws.device_available = lambda device: device == "cpu" or bool(self.gpu.devices().get(device, {}).get("detected"))
            except BaseException:
                shutil.rmtree(root)
                raise
            tenant = Tenant(identifier, digest, ip_key, expires_at, root, app, context, local_session)
            tenant.last_activity = self.now()
            if self.now() >= self.idle_deadline(ip_key):
                self.close_ip(ip_key)
            self.ip_activity[ip_key] = self.now()
            self.tenants[digest] = tenant
            self.ip_sessions[ip_key] = self.ip_sessions.get(ip_key, 0) + 1
            return tenant, token
        except BaseException:
            if 'root' in locals() and root.exists():
                shutil.rmtree(root)
            raise

    async def remove(self, tenant: Tenant):
        tenant.closing = True
        tenant.app.state.supervisor._accepting = False
        if tenant.inflight:
            # No directory deletion underneath an analysis request or download.
            # The independent systemd cgroup reset bounds a wedged request.
            return
        await tenant.context.__aexit__(None, None, None)
        await asyncio.to_thread(shutil.rmtree, tenant.root)
        self.tenants.pop(tenant.token_hash, None)
        if not any(t.ip_key == tenant.ip_key for t in self.tenants.values()):
            self.ip_activity.pop(tenant.ip_key, None)

    async def sweep(self, *, quotas=True):
        async with self.sweep_lock:
            await self._sweep(quotas=quotas)

    async def _sweep(self, *, quotas=True):
        try:
            self.gpu.sweep()
        except Exception:
            # GPU finalization must not stop tenant expiry or silently kill the
            # maintenance task. The independent reset remains a final backstop.
            self.cleanup_healthy = False
            log.exception('GPU maintenance failed; session admission disabled')
        day = int(self.now() // DAY)
        tenants = list(self.tenants.values())
        checked = set()
        if quotas and tenants:
            checked = {tenants[(self.quota_cursor + i) % len(tenants)].token_hash
                       for i in range(min(len(tenants), self.config.quota_checks_per_sweep))}
            self.quota_cursor = (self.quota_cursor + len(checked)) % len(tenants)
        for tenant in tenants:
            try:
                if (not self.is_live(tenant)
                        or (tenant.token_hash in checked and await asyncio.to_thread(directory_bytes, tenant.root) > self.config.max_workspace_bytes)):
                    await self.remove(tenant)
            except Exception:
                self.cleanup_healthy = False
                log.exception("temporary workspace cleanup failed; session admission disabled")
        if day != self.day:
            with self.budget_lock:
                self.ip_sessions.clear()
                self.ip_runs.clear()
                self.rate.clear()
                self.total_runs = 0
                self.day = day

    async def maintenance(self):
        while True:
            await asyncio.sleep(self.config.sweep_seconds)
            try:
                self.prune_admissions()
                await self.sweep()
            except Exception:
                self.cleanup_healthy = False
                log.exception('temporary workspace maintenance failed; session admission disabled')

    async def stop(self):
        self.scheduler_stop.set()
        self.scheduler_wake.set()
        await asyncio.to_thread(self.scheduler.join, 5)
        try:
            for tenant in list(self.tenants.values()):
                await self.remove(tenant)
        finally:
            self.resources.stop()
            if self.lock:
                self.lock.close()

    def server_status(self):
        from datetime import datetime, timezone
        from opendpd.schemas.system import ServerStatus
        from opendpd.services.server_status import job_counts, SAMPLE_SECONDS
        now = self.now()
        tenants = [t for t in list(self.tenants.values()) if not t.closing and now < t.expires_at
                   and now < self.idle_deadline(t.ip_key)]
        # Bound SQLite work across every viewer of the status page. A closing
        # workspace may disappear between snapshots; never expose its identity.
        with self.status_lock:
            if time.monotonic() - self.status_updated >= SAMPLE_SECONDS:
                counts = [0, 0]
                # Only submitted work can have nonterminal records. The same
                # lock orders dispatch and retirement, so idle workspaces need
                # no SQLite reads and a closed store cannot enter this snapshot.
                with self.budget_lock:
                    for supervisor in self.scheduled:
                        if (not supervisor._accepting or now >= supervisor.expires_at
                                or now >= self.idle_deadline(supervisor.ip_key)):
                            continue
                        try:
                            running, queued = job_counts(supervisor.store)
                        except sqlite3.Error:
                            counts = [None, None]
                            log.warning('job-count telemetry unavailable')
                            break
                        counts[0] += running
                        counts[1] += queued
                self.status_counts = tuple(counts)
                self.status_updated = time.monotonic()
            running, queued = self.status_counts
        return ServerStatus(mode='web', sampled_at=datetime.fromtimestamp(now, timezone.utc),
            active_sessions=sum(now - t.last_activity < 300 for t in tenants), workspaces=len(tenants),
            workspace_capacity=self.config.max_sessions, running_jobs=running, queued_jobs=queued,
            waiting_sessions=self.waiting_count(),
            parallel_capacity=self.config.max_parallel, api=self.resources.snapshot(), compute=self.gpu.resource_status())
