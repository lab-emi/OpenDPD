"""SQLite run store: the authoritative record of run status and events.

One connection guarded by a lock serves the supervisor thread and the API
threads. Status changes and their event are committed in one transaction.
``events.jsonl`` in the run directory is the worker's own append-only log;
the store ingests it and assigns the per-run sequence numbers clients use
to resume.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from opendpd.schemas import RunEvent, RunEventType, RunRecord, RunStatus, can_transition

DB_SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    task TEXT NOT NULL,
    dataset_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    idempotency_key TEXT UNIQUE,
    record TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS runs_status ON runs(status, created_at);
CREATE TABLE IF NOT EXISTS events (
    run_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    ts TEXT NOT NULL,
    type TEXT NOT NULL,
    payload TEXT NOT NULL,
    PRIMARY KEY (run_id, seq)
);
"""


class RunStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False, isolation_level=None, timeout=30)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA busy_timeout=30000")
        with self._lock:
            self._conn.executescript(_SCHEMA)
            row = self._conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
            if row is None:
                self._conn.execute("INSERT INTO meta(key, value) VALUES ('schema_version', ?)",
                                   (str(DB_SCHEMA_VERSION),))
            elif int(row[0]) != DB_SCHEMA_VERSION:
                raise RuntimeError(f"metadata.sqlite schema {row[0]} is not supported (expected {DB_SCHEMA_VERSION}); "
                                   "back up the workspace before migrating")

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -- runs ----------------------------------------------------------------
    def upsert_run(self, record: RunRecord) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._conn.execute(
                "INSERT INTO runs(run_id, status, task, dataset_id, created_at, updated_at, idempotency_key, record) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(run_id) DO UPDATE SET status=excluded.status, "
                "updated_at=excluded.updated_at, record=excluded.record",
                (record.run_id, record.status.value, record.task.value, record.dataset_id,
                 record.created_at.isoformat(), now, record.idempotency_key, record.model_dump_json()))

    def run_ids(self) -> List[str]:
        with self._lock:
            return [r[0] for r in self._conn.execute("SELECT run_id FROM runs").fetchall()]

    def get_run(self, run_id: str) -> Optional[RunRecord]:
        with self._lock:
            row = self._conn.execute("SELECT record FROM runs WHERE run_id=?", (run_id,)).fetchone()
        return RunRecord.model_validate_json(row[0]) if row else None

    def list_runs(self, status: Optional[RunStatus] = None, limit: int = 100, offset: int = 0) -> List[RunRecord]:
        query = "SELECT record FROM runs"
        params: tuple = ()
        if status is not None:
            query += " WHERE status=?"
            params = (status.value,)
        query += " ORDER BY created_at DESC, run_id DESC LIMIT ? OFFSET ?"
        with self._lock:
            rows = self._conn.execute(query, params + (limit, offset)).fetchall()
        return [RunRecord.model_validate_json(r[0]) for r in rows]

    def count_runs(self, status: Optional[RunStatus] = None) -> int:
        with self._lock:
            if status is None:
                return self._conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
            return self._conn.execute("SELECT COUNT(*) FROM runs WHERE status=?", (status.value,)).fetchone()[0]

    def find_idempotent(self, key: str) -> Optional[RunRecord]:
        with self._lock:
            row = self._conn.execute("SELECT record FROM runs WHERE idempotency_key=?", (key,)).fetchone()
        return RunRecord.model_validate_json(row[0]) if row else None

    def transition(self, run_id: str, new_status: RunStatus, *, reason: Optional[str] = None,
                   **updates: Any) -> RunRecord:
        """Atomically change status (checked against the state machine) and
        record the matching status event."""
        with self._lock:
            record = self.get_run(run_id)
            if record is None:
                raise KeyError(run_id)
            if not can_transition(record.status, new_status):
                raise ValueError(f"illegal transition {record.status.value} -> {new_status.value} ({run_id})")
            previous = record.status
            record = record.model_copy(update={"status": new_status, "status_reason": reason, **updates})
            self._conn.execute("BEGIN")
            try:
                event = self._append_event_locked(run_id, RunEventType.status,
                                                  {"from": previous.value, "to": new_status.value, "reason": reason})
                record = record.model_copy(update={"last_event_seq": event.seq})
                self.upsert_run(record)
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise
        return record

    def update_run(self, run_id: str, **updates: Any) -> RunRecord:
        with self._lock:
            record = self.get_run(run_id)
            if record is None:
                raise KeyError(run_id)
            record = record.model_copy(update=updates)
            self.upsert_run(record)
        return record

    # -- events --------------------------------------------------------------
    def _append_event_locked(self, run_id: str, type_: RunEventType, payload: Dict[str, Any],
                             ts: Optional[datetime] = None) -> RunEvent:
        last = self._conn.execute("SELECT COALESCE(MAX(seq), 0) FROM events WHERE run_id=?", (run_id,)).fetchone()[0]
        event = RunEvent(seq=last + 1, run_id=run_id, ts=ts or datetime.now(timezone.utc), type=type_, payload=payload)
        self._conn.execute("INSERT INTO events(run_id, seq, ts, type, payload) VALUES (?, ?, ?, ?, ?)",
                           (run_id, event.seq, event.ts.isoformat(), event.type.value,
                            json.dumps(payload, sort_keys=True, default=str)))
        return event

    def append_event(self, run_id: str, type_: RunEventType, payload: Dict[str, Any],
                     ts: Optional[datetime] = None, **record_updates: Any) -> RunEvent:
        """Append an event and update the run's cursor (plus any record fields) atomically."""
        with self._lock:
            self._conn.execute("BEGIN")
            try:
                event = self._append_event_locked(run_id, type_, payload, ts)
                record = self.get_run(run_id)
                if record is not None:
                    record = record.model_copy(update={"last_event_seq": event.seq, **record_updates})
                    self.upsert_run(record)
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise
        return event

    def events_after(self, run_id: str, after_seq: int = 0, limit: int = 1000) -> List[RunEvent]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT seq, ts, type, payload FROM events WHERE run_id=? AND seq>? ORDER BY seq LIMIT ?",
                (run_id, after_seq, limit)).fetchall()
        return [RunEvent(seq=r[0], run_id=run_id, ts=datetime.fromisoformat(r[1]), type=RunEventType(r[2]),
                         payload=json.loads(r[3])) for r in rows]

    def export_events_jsonl(self, run_id: str, path: Path) -> int:
        events = self.events_after(run_id, 0, limit=10 ** 9)
        with open(path, "w", encoding="utf-8") as f:
            for e in events:
                f.write(e.model_dump_json() + "\n")
        return len(events)
