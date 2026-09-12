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
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from opendpd.runtime.supervisor import Supervisor
from opendpd.schemas import TERMINAL_STATUSES
from opendpd.server.app import create_app
from opendpd.web.policy import WebConfig, check_config, reject

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
    return sum(p.stat(follow_symlinks=False).st_size for p in root.rglob("*") if not p.is_symlink())


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

    def submit(self, config, **kwargs):
        check_config(config)
        # Admission, including retry and idempotency, is one transaction across
        # ALL workspaces. New anonymous sessions cannot multiply the GPU limit.
        with self.manager.budget_lock:
            if self.manager.now() >= self.expires_at:
                reject(401, "session_expired", "the temporary workspace has expired")
            existing = self.store.find_idempotent(kwargs.get("idempotency_key")) if kwargs.get("idempotency_key") else None
            if existing is not None:
                return existing
            records = self.store.list_runs(limit=10000)
            if len(records) >= self.manager.config.runs_per_session:
                reject(429, "run_quota", "this temporary workspace has reached its run limit")
            if sum(r.status not in TERMINAL_STATUSES for r in records) >= self.manager.config.max_pending:
                reject(429, "queue_full", "finish or cancel an existing run before submitting another")
            ip_runs = self.manager.ip_runs.get(self.ip_key, 0)
            if ip_runs >= self.manager.config.runs_per_ip or self.manager.total_runs >= self.manager.config.runs_per_day:
                reject(429, "run_quota", "the daily compute quota has been reached")
            record = super().submit(config, **kwargs)
            self.manager.ip_runs[self.ip_key] = ip_runs + 1
            self.manager.total_runs += 1
            return record


class TenantManager:
    def __init__(self, config: WebConfig, *, now=time.time):
        self.config, self.now = config, now
        self.tenants: dict[str, Tenant] = {}
        self.slots = threading.BoundedSemaphore(config.max_parallel)
        self.budget_lock = threading.RLock()
        self.create_lock = asyncio.Lock()
        self.ip_secret = secrets.token_bytes(32)
        self.ip_sessions: dict[str, int] = {}
        self.ip_runs: dict[str, int] = {}
        self.rate: dict[str, tuple[int, int]] = {}
        self.total_runs = 0
        self.day = int(now() // DAY)
        self.inflight = 0
        self.cleanup_healthy = True
        self.lock = None

    async def start(self):
        self.lock = prepare_root(self.config.root)
        clear_sessions(self.config.root)
        (self.config.root / "service-tmp").mkdir(mode=0o700, exist_ok=True)

    def ip_key(self, address: str) -> str:
        ip = ipaddress.ip_address(address)
        # IPv6 privacy addresses in one /64 share an abuse budget. No raw IPs are
        # written into folders, SQLite, application logs or browser-visible data.
        network = str(ipaddress.ip_network(f"{ip}/64", strict=False)) if ip.version == 6 else str(ip)
        return hmac.new(self.ip_secret, network.encode(), hashlib.sha256).hexdigest()

    def rate_limit(self, ip_key: str):
        minute = int(self.now() // 60)
        if len(self.rate) >= 4096 and ip_key not in self.rate:
            self.rate = {k: v for k, v in self.rate.items() if v[0] == minute}
            if len(self.rate) >= 4096:
                reject(429, "service_busy", "the service is busy; try again later")
        stamp, count = self.rate.get(ip_key, (minute, 0))
        count = count + 1 if stamp == minute else 1
        self.rate[ip_key] = (minute, count)
        if count > self.config.requests_per_minute:
            reject(429, "rate_limited", "too many requests; try again in a minute")

    @property
    def cutoff(self):
        return (int(self.now() // DAY) + 1) * DAY - self.config.drain_seconds

    def authenticate(self, token: str) -> Tenant:
        digest = hashlib.sha256(token.encode()).hexdigest()
        tenant = self.tenants.get(digest)
        if tenant is None or tenant.closing or self.now() >= tenant.expires_at:
            reject(401, "session_expired", "the temporary session has expired; start a new session")
        return tenant

    async def create(self, ip_key: str):
        async with self.create_lock:
            await self.sweep()
            if not self.cleanup_healthy:
                reject(503, "cleanup_unavailable", "temporary storage maintenance is unavailable")
            if self.now() >= self.cutoff:
                reject(503, "daily_cleanup", "daily cleanup is in progress; try again after 00:00 UTC")
            if len(self.tenants) >= self.config.max_sessions:
                reject(503, "service_busy", "all temporary workspaces are in use; try again later")
            if self.ip_sessions.get(ip_key, 0) >= self.config.sessions_per_ip:
                reject(429, "session_quota", "the daily session limit for this network has been reached")
            if len(self.ip_sessions) >= 4096:
                reject(429, "service_busy", "the service is busy; try again later")
            identifier = secrets.token_hex(24)
            token = secrets.token_urlsafe(32)
            digest = hashlib.sha256(token.encode()).hexdigest()
            root = self.config.root / "sessions" / identifier
            root.mkdir(mode=0o700)
            expires_at = self.cutoff
            (root / "lease.json").write_text(json.dumps({"created_at": self.now(), "expires_at": expires_at}))

            def factory(ws, store, **kwargs):
                return WebSupervisor(ws, store, manager=self, ip_key=ip_key, expires_at=expires_at, **kwargs)

            app = create_app(root / "workspace", supervisor_factory=factory, shutdown_timeout=0)
            app.state.workspace_label = "Temporary workspace (deleted within 24 hours)"
            local_session = app.state.sessions.exchange(app.state.sessions.bootstrap_token)
            context = app.router.lifespan_context(app)
            try:
                await context.__aenter__()
            except BaseException:
                shutil.rmtree(root)
                raise
            tenant = Tenant(identifier, digest, ip_key, expires_at, root, app, context, local_session)
            self.tenants[digest] = tenant
            self.ip_sessions[ip_key] = self.ip_sessions.get(ip_key, 0) + 1
            return tenant, token

    async def remove(self, tenant: Tenant):
        tenant.closing = True
        if tenant.inflight:
            # No directory deletion underneath an analysis request or download.
            # The independent systemd cgroup reset bounds a wedged request.
            return
        await tenant.context.__aexit__(None, None, None)
        shutil.rmtree(tenant.root)
        self.tenants.pop(tenant.token_hash, None)

    async def sweep(self):
        day = int(self.now() // DAY)
        for tenant in list(self.tenants.values()):
            try:
                if tenant.closing or self.now() >= tenant.expires_at or directory_bytes(tenant.root) > self.config.max_workspace_bytes:
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
            await self.sweep()

    async def stop(self):
        try:
            for tenant in list(self.tenants.values()):
                await self.remove(tenant)
        finally:
            if self.lock:
                self.lock.close()
