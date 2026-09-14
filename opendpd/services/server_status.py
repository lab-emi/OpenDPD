"""Bounded background sampling; HTTP reads never invoke a GPU command or sleep."""
from __future__ import annotations

import math
import shutil
import subprocess
import threading
import time
from datetime import datetime, timezone

import psutil

from opendpd.schemas.system import GpuLoad, MachineLoad, ResourceStatus, ServerStatus
from opendpd.schemas import RunStatus

SAMPLE_SECONDS = 5
STALE_SECONDS = 20


def _number(value, maximum=None):
    try:
        result = float(value)
    except (ValueError, TypeError):
        return None
    return result if math.isfinite(result) and result >= 0 and (maximum is None or result <= maximum) else None


def gpu_load():
    executable = shutil.which('nvidia-smi')
    if executable is None:
        return None
    try:
        result = subprocess.run([executable, '--id=0', '--query-gpu=utilization.gpu,memory.used,memory.total',
                                 '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True, timeout=2)
        values = result.stdout.strip().split(',')
        if len(values) != 3:
            return None
        utilization, used, total = _number(values[0], 100), _number(values[1]), _number(values[2])
        if total is not None and (total <= 0 or (used is not None and used > total)):
            return None
        return GpuLoad(utilization_percent=utilization,
                       memory_used_bytes=int(used * 2**20) if used is not None else None,
                       memory_total_bytes=int(total * 2**20) if total is not None else None)
    except (OSError, subprocess.SubprocessError):
        return None


class ResourceSampler:
    def __init__(self, *, include_gpu=False):
        self.include_gpu = include_gpu
        self.lock = threading.Lock()
        self.stopped = threading.Event()
        self.thread = None
        self.latest = MachineLoad()
        self.updated = None

    def start(self):
        self.thread = threading.Thread(target=self._loop, name='opendpd-resource-sampler', daemon=True)
        self.thread.start()

    def stop(self):
        self.stopped.set()
        if self.thread:
            self.thread.join(timeout=3)

    def _loop(self):
        # psutil's initial nonblocking CPU value is not a measurement. Prime it
        # on this same thread, publish memory now, and CPU after the first interval.
        psutil.cpu_percent(interval=None)
        first = True
        while not self.stopped.is_set():
            try:
                memory = psutil.virtual_memory()
                current = MachineLoad(sampled_at=datetime.now(timezone.utc),
                    cpu_percent=None if first else psutil.cpu_percent(interval=None),
                    memory_percent=memory.percent, memory_used_bytes=memory.total - memory.available,
                    memory_total_bytes=memory.total, gpu=gpu_load() if self.include_gpu else None)
                with self.lock:
                    self.latest, self.updated = current, time.monotonic()
            except (OSError, ValueError, psutil.Error):
                pass  # Keep the last sample; readers see its age and stale flag.
            first = False
            self.stopped.wait(SAMPLE_SECONDS)

    def snapshot(self):
        with self.lock:
            age = max(0., time.monotonic() - self.updated) if self.updated is not None else None
            return ResourceStatus(load=self.latest.model_copy(deep=True), age_seconds=age,
                                  stale=age is None or age > STALE_SECONDS)


def job_counts(store):
    return (store.count_runs(RunStatus.running) + store.count_runs(RunStatus.cancel_requested),
            store.count_runs(RunStatus.queued))


def local_status(app):
    running, queued = job_counts(app.state.store)
    return ServerStatus(mode='local', sampled_at=datetime.now(timezone.utc), workspaces=1,
                        running_jobs=running, queued_jobs=queued, api=app.state.resources.snapshot())
