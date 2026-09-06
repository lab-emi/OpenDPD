"""Process identity and cleanup helpers (psutil when available)."""

from __future__ import annotations

import os
import signal
import time
from typing import List, Optional, Tuple

try:  # psutil ships in the gui extra; the core install may lack it
    import psutil
except ImportError:  # pragma: no cover - exercised only in minimal installs
    psutil = None


def process_identity(pid: int) -> Optional[Tuple[int, float]]:
    """(pid, create_time) for a live process, else None."""
    if psutil is None:
        try:
            os.kill(pid, 0)
        except OSError:
            return None
        return pid, 0.0
    try:
        return pid, psutil.Process(pid).create_time()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None


def is_same_process(pid: int, create_time: float, tolerance: float = 1.0) -> bool:
    """True when ``pid`` is alive *and* was created when we recorded it.
    A recycled PID with a different start time is not our worker."""
    ident = process_identity(pid)
    if ident is None:
        return False
    if psutil is None:
        return True   # cannot check the start time without psutil
    return abs(ident[1] - create_time) <= tolerance


def kill_tree(pid: int, grace_seconds: float = 3.0) -> List[int]:
    """Terminate ``pid`` and its descendants; returns the PIDs acted on."""
    acted: List[int] = []
    if psutil is None:
        try:
            os.kill(pid, signal.SIGTERM)
            acted.append(pid)
            time.sleep(grace_seconds)
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
        return acted
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return acted
    procs = parent.children(recursive=True) + [parent]
    for p in procs:
        try:
            p.terminate()
            acted.append(p.pid)
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(procs, timeout=grace_seconds)
    for p in alive:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(alive, timeout=grace_seconds)
    return acted


def descendants(pid: int) -> List[int]:
    if psutil is None:
        return []
    try:
        return [p.pid for p in psutil.Process(pid).children(recursive=True)]
    except psutil.NoSuchProcess:
        return []
