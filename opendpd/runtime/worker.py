"""Worker subprocess: runs exactly one queued run to a terminal state.

    python -m opendpd.runtime.worker --workspace DIR --run-id ID

Events go to ``<run_dir>/events.jsonl`` (one JSON object per line, flushed
per line); the supervisor tails that file. Legacy prints and tqdm output go
to ``<run_dir>/logs/worker.log`` (the supervisor redirects stdout/stderr
there). A ``CANCEL`` file in the run directory requests a cooperative stop
at the next epoch boundary. Exit codes: 0 succeeded, 1 failed, 3 cancelled.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from opendpd.schemas import RunEventType, RunStatus

EVENTS_FILE = "events.jsonl"
CANCEL_FILE = "CANCEL"
HEARTBEAT_SECONDS = 5.0
EXIT_CODES = {RunStatus.succeeded: 0, RunStatus.failed: 1, RunStatus.cancelled: 3}


class FileEmitter:
    """Thread-safe JSON-lines writer used as the ``emit`` callback."""

    def __init__(self, path: Path):
        self._file = open(path, "a", encoding="utf-8", buffering=1)
        self._lock = threading.Lock()

    def __call__(self, type_: RunEventType, payload: Dict[str, Any]) -> None:
        line = json.dumps({"ts": datetime.now(timezone.utc).isoformat(), "type": type_.value,
                           "payload": payload}, sort_keys=True, default=str)
        with self._lock:
            self._file.write(line + "\n")
            self._file.flush()

    def close(self) -> None:
        with self._lock:
            self._file.close()


def _heartbeat(emit: FileEmitter, stop: threading.Event) -> None:
    while not stop.wait(HEARTBEAT_SECONDS):
        emit(RunEventType.heartbeat, {"pid": os.getpid()})


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m opendpd.runtime.worker")
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("TQDM_DISABLE", "1")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    from opendpd.services.experiments import execute_run
    from opendpd.services.workspace import Workspace

    ws = Workspace.open(Path(args.workspace))
    run_dir = ws.run_dir(args.run_id)
    emit = FileEmitter(run_dir / EVENTS_FILE)
    stop = threading.Event()
    threading.Thread(target=_heartbeat, args=(emit, stop), daemon=True, name="heartbeat").start()
    cancel_path = run_dir / CANCEL_FILE
    try:
        record = execute_run(ws, args.run_id, emit=emit, should_cancel=cancel_path.exists)
    except Exception as err:  # noqa: BLE001 - last resort: make the failure visible
        emit(RunEventType.error, {"code": "worker_crash", "message": f"{type(err).__name__}: {err}"})
        stop.set()
        emit.close()
        return 1
    stop.set()
    emit.close()
    sys.stdout.flush()
    time.sleep(0.05)   # let the supervisor's tail see the final lines before the exit is observed
    return EXIT_CODES.get(record.status, 1)


if __name__ == "__main__":
    sys.exit(main())
