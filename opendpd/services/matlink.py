"""Ephemeral local MATLAB bridge with bounded, retry-safe named transfers.

MATLAB owns its workspace and executes actions on its event loop. The server
only queues validated metadata. A missed heartbeat marks presence offline;
it does not cancel a transfer while MATLAB is occupied with another command.
"""
from __future__ import annotations

import hmac
import secrets
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable
from uuid import uuid4

from opendpd.schemas.matlink import (
    MatlinkCompletion, MatlinkConnect, MatlinkConnection, MatlinkHeartbeat,
    MatlinkPoll, MatlinkRequest, MatlinkSession, MatlinkState, MatlinkTransfer,
)
from opendpd.schemas.run import RunStatus, TERMINAL_STATUSES
from opendpd.services.workspace import WorkspaceError

LEASE_SECONDS = 30
MAX_SESSIONS = 16
MAX_TRANSFERS = 256
MAX_PENDING = 32
MAX_KEYS = 1024
PENDING = frozenset(("waiting", "queued"))


class MatlinkError(WorkspaceError):
    def __init__(self, code: str, message: str, status_code: int = 409):
        super().__init__(message)
        self.code, self.status_code = code, status_code


@dataclass
class _Bridge:
    view: MatlinkSession
    secret: str
    seen: float
    disconnected: bool = False


class MatlinkBroker:
    def __init__(self, ws, store, *, clock: Callable[[], float] = time.monotonic,
                 load_result=None):
        self.ws, self.store, self._clock = ws, store, clock
        if load_result is None:
            from opendpd.services.experiments import load_result
        self._load_result = load_result
        self._sessions: dict[str, _Bridge] = {}
        self._transfers: dict[str, MatlinkTransfer] = {}
        self._keys: dict[tuple[str, str], str] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _now():
        return datetime.now(timezone.utc)

    def _connected(self, bridge):
        return not bridge.disconnected and self._clock() - bridge.seen < LEASE_SECONDS

    def _pending(self, client_id):
        return [t for t in self._transfers.values() if t.client_id == client_id and t.status in PENDING]

    def connect(self, body: MatlinkConnect) -> MatlinkConnection:
        with self._lock:
            if len(self._sessions) >= MAX_SESSIONS:
                # Prefer explicit disconnections, then reclaim the oldest
                # offline bridge with nothing left to deliver. A busy bridge
                # with queued actions retains its secret across missed leases.
                retired = next((cid for cid, b in self._sessions.items() if b.disconnected), None)
                if retired is None:
                    idle = [(b.seen, cid) for cid, b in self._sessions.items()
                            if not self._connected(b) and not self._pending(cid)]
                    if idle:
                        retired = min(idle)[1]
                if retired is None:
                    raise MatlinkError("matlink_session_limit", "Disconnect an existing MATLAB bridge before connecting another.", 429)
                del self._sessions[retired]
            client_id, secret = "matlab-" + uuid4().hex, secrets.token_urlsafe(32)
            view = MatlinkSession(client_id=client_id, label=body.label, release=body.release,
                                  connected=True, last_seen=self._now(), variables=body.variables, pending_count=0,
                                  capabilities=body.capabilities)
            self._sessions[client_id] = _Bridge(view, secret, self._clock())
            return MatlinkConnection(client_id=client_id, bridge_token=secret, lease_seconds=LEASE_SECONDS)

    def _bridge(self, client_id):
        bridge = self._sessions.get(client_id)
        if bridge is None:
            raise MatlinkError("matlink_session_not_found", "This MATLAB connection is no longer registered. Reconnect from MATLAB.", 404)
        return bridge

    def _authenticate(self, client_id, secret):
        bridge = self._bridge(client_id)
        if not secret or not hmac.compare_digest(secret.encode("utf-8"), bridge.secret.encode("ascii")):
            raise MatlinkError("matlink_bridge_required", "This action requires the connected MATLAB bridge.", 403)
        return bridge

    def _result_status(self, run_id):
        record = self.store.get_run(run_id)
        if record is None:
            raise MatlinkError("run_not_found", "The selected run no longer exists.", 404)
        if record.status == RunStatus.succeeded:
            try:
                report = self._load_result(self.ws, run_id)
            except (OSError, ValueError, WorkspaceError):
                report = None
            if report is None:
                raise MatlinkError("result_not_available", "This run has no readable evaluation report to send to MATLAB.")
            return "queued"
        if record.status in TERMINAL_STATUSES:
            raise MatlinkError("result_not_available", f"This run {record.status.value}; no completed result can be sent to MATLAB.")
        return "waiting"

    def _promote(self):
        for transfer in self._transfers.values():
            if transfer.status != "waiting":
                continue
            try:
                status = self._result_status(transfer.payload["run_id"])
            except MatlinkError as exc:
                transfer.status, transfer.error, transfer.updated_at = "failed", str(exc), self._now()
            else:
                if status != transfer.status:
                    transfer.status, transfer.updated_at = status, self._now()

    def snapshot(self):
        with self._lock:
            self._promote()
            sessions = [b.view.model_copy(update={"connected": self._connected(b), "pending_count": len(self._pending(cid))}, deep=True)
                        for cid, b in self._sessions.items()]
            return MatlinkState(workspace=str(self.ws.root), sessions=sessions,
                                transfers=[t.model_copy(deep=True) for t in reversed(self._transfers.values())])

    def request(self, body: MatlinkRequest):
        with self._lock:
            key = (body.client_id, body.idempotency_key)
            existing = self._keys.get(key)
            if existing is not None:
                transfer = self._transfers[existing]
                if transfer.action != body.action or transfer.payload != body.payload:
                    raise MatlinkError("matlink_idempotency_conflict", "This transfer key was already used for a different action.")
                return transfer.model_copy(deep=True)
            bridge = self._bridge(body.client_id)
            if not self._connected(bridge):
                raise MatlinkError("matlink_offline", "MATLAB is busy or disconnected. Wait for its connection to return, then retry.")
            if len(self._pending(body.client_id)) >= MAX_PENDING:
                raise MatlinkError("matlink_queue_full", "Wait for existing MATLAB transfers to finish before adding more.", 429)
            status = "queued"
            feature = "dataset_export" if body.action == "import_dataset" else (
                "result_bundle" if body.action == "import_result" and
                (body.payload.get("bundle") or body.payload.get("variable")) else None)
            if feature and feature not in bridge.view.capabilities:
                raise MatlinkError("matlink_upgrade_required", "Reconnect using OpenDPD MATLAB toolbox 0.4.0 or later for this transfer.")
            variables = {v.name: v for v in bridge.view.variables}
            if body.action == "import_iq":
                pair = [variables.get(body.payload[k]) for k in ("input", "output")]
                if any(v is None or not v.eligible for v in pair):
                    raise MatlinkError("matlink_variable_unavailable", "Choose two available I/Q variables from MATLAB. Refresh after changing the workspace.")
                if pair[0].n_samples != pair[1].n_samples:
                    raise MatlinkError("matlink_length_mismatch", "Input and output must contain the same number of I/Q samples.")
            elif body.action == "import_dataset":
                dataset = self.ws.get_dataset(body.payload["dataset_id"])
                if not dataset.simulation:
                    raise MatlinkError("matlink_dataset_unavailable", "Select a paired dataset created by Studio's Virtual PA Library.")
            elif body.action == "open_variable":
                if body.payload["variable"] not in variables:
                    raise MatlinkError("matlink_variable_unavailable", "This variable is no longer listed in MATLAB. Refresh its workspace.")
            elif body.action == "import_result":
                status = self._result_status(body.payload["run_id"])
                # A repeated click or different browser tab cannot queue the
                # same report twice while its first delivery is pending.
                pending = next((t for t in self._pending(body.client_id)
                                if t.action == body.action and t.payload == body.payload), None)
                if pending is not None:
                    self._make_room()
                    if len(self._keys) >= MAX_KEYS:
                        raise MatlinkError("matlink_queue_full", "Wait for existing MATLAB transfers to finish before adding more.", 429)
                    self._keys[key] = pending.request_id
                    return pending.model_copy(deep=True)
            self._make_room()
            if len(self._keys) >= MAX_KEYS:
                raise MatlinkError("matlink_queue_full", "Wait for existing MATLAB transfers to finish before adding more.", 429)
            now = self._now()
            transfer = MatlinkTransfer(request_id="transfer-" + uuid4().hex, client_id=body.client_id,
                                      action=body.action, status=status, payload=body.payload,
                                      created_at=now, updated_at=now)
            self._transfers[transfer.request_id] = transfer
            self._keys[key] = transfer.request_id
            return transfer.model_copy(deep=True)

    def _make_room(self):
        while len(self._transfers) >= MAX_TRANSFERS or len(self._keys) >= MAX_KEYS:
            oldest = next((rid for rid, t in self._transfers.items() if t.status not in PENDING), None)
            if oldest is None:
                raise MatlinkError("matlink_queue_full", "Wait for existing MATLAB transfers to finish before adding more.", 429)
            del self._transfers[oldest]
            self._keys = {key: rid for key, rid in self._keys.items() if rid != oldest}

    def heartbeat(self, client_id, secret, body: MatlinkHeartbeat):
        with self._lock:
            bridge = self._authenticate(client_id, secret)
            if bridge.disconnected:
                raise MatlinkError("matlink_disconnected", "This MATLAB bridge was disconnected. Connect it again.")
            bridge.view.variables = body.variables
            bridge.view.last_seen, bridge.seen = self._now(), self._clock()
            self._promote()
            commands = [t.model_copy(deep=True) for t in self._pending(client_id) if t.status == "queued"]
            return MatlinkPoll(requests=commands, lease_seconds=LEASE_SECONDS)

    def complete(self, client_id, secret, request_id, body: MatlinkCompletion):
        with self._lock:
            self._authenticate(client_id, secret)
            transfer = self._transfers.get(request_id)
            if transfer is None or transfer.client_id != client_id:
                raise MatlinkError("matlink_transfer_not_found", "This transfer does not belong to the connected MATLAB bridge.", 404)
            if transfer.status not in PENDING:
                return transfer.model_copy(deep=True)
            if transfer.status == "waiting":
                raise MatlinkError("matlink_transfer_not_ready", "This transfer is waiting for its run to finish.")
            transfer.status, transfer.result, transfer.error = body.status, body.result, body.error
            transfer.updated_at = self._now()
            return transfer.model_copy(deep=True)

    def disconnect(self, client_id, secret):
        with self._lock:
            bridge = self._authenticate(client_id, secret)
            bridge.disconnected = True
            bridge.view.variables = []
            for transfer in self._pending(client_id):
                transfer.status, transfer.error = "failed", "MATLAB disconnected before this transfer was acknowledged."
                transfer.updated_at = self._now()
