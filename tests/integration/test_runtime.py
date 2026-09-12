"""S03 runtime tests: real worker processes, cancel, crash, restart recovery."""

import json

import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone

import pytest

from opendpd.runtime.db import RunStore
from opendpd.runtime.procs import is_same_process, process_identity
from opendpd.runtime.supervisor import Supervisor
from opendpd.schemas import RunEventType, RunStatus, TERMINAL_STATUSES, WorkerInfo
from opendpd.services import experiments
from opendpd.services.recipes import instantiate
from opendpd.services.workspace import Workspace

pytestmark = pytest.mark.integration


def wait_for(predicate, timeout=120, interval=0.2):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(interval)
    raise AssertionError("condition not met in time")


@pytest.fixture
def runtime(tmp_path):
    ws = Workspace.create(tmp_path / "ws")
    ws.register_builtin_dataset("DPA_200MHz")
    store = RunStore(ws.root / "metadata.sqlite")
    sup = Supervisor(ws, store, poll_interval=0.1, cancel_grace=20)
    sup.start()
    yield ws, store, sup
    sup.stop(timeout=5)
    store.close()


def smoke(epochs=3):
    cfg = instantiate("pa-gru-smoke-v1", "dpa-200mhz")
    return cfg.model_copy(update={"training": cfg.training.model_copy(update={"epochs": epochs})})


def test_run_completes_through_worker_with_events(runtime):
    ws, store, sup = runtime
    record = sup.submit(smoke(), idempotency_key="k1")
    assert record.status == RunStatus.queued
    assert sup.submit(smoke(), idempotency_key="k1").run_id == record.run_id   # idempotent
    final = wait_for(lambda: (r := store.get_run(record.run_id)) and r.status in TERMINAL_STATUSES and r)
    assert final.status == RunStatus.succeeded, final.error
    assert final.worker and final.exit_code == 0 and final.result_id and final.progress_epoch == 3
    types = [e.type for e in store.events_after(record.run_id, 0)]
    assert types[0] == RunEventType.status and RunEventType.metric in types and RunEventType.progress in types
    assert types[-1] == RunEventType.status
    seqs = [e.seq for e in store.events_after(record.run_id, 0)]
    assert seqs == list(range(1, len(seqs) + 1))
    # resume from a cursor: only later events come back
    later = store.events_after(record.run_id, seqs[-3])
    assert [e.seq for e in later] == seqs[-2:]
    # file-side record and artifacts agree with the store
    assert experiments.load_run(ws, record.run_id).status == RunStatus.succeeded
    assert experiments.load_artifacts(ws, record.run_id).complete
    assert (ws.run_dir(record.run_id) / "logs" / "worker.log").stat().st_size > 0
    assert (ws.run_dir(record.run_id) / "events.jsonl").exists()
    assert not is_same_process(final.worker.pid, final.worker.create_time)
    # started_at / finished_at are the executor's own clock (the worker's status events), not the
    # supervisor's spawn time and exit detection: durations compare with `opendpd run` in-process
    worker_status = [json.loads(line) for line in (ws.run_dir(record.run_id) / "events.jsonl").read_text().splitlines()
                     if '"status"' in line]
    worker_status = [e for e in worker_status if e["type"] == "status"]
    assert worker_status[0]["payload"]["to"] == "running" and worker_status[-1]["payload"]["to"] == "succeeded"
    first_ts, last_ts = (datetime.fromisoformat(worker_status[i]["ts"]) for i in (0, -1))
    # the executor stamps, saves run.json, then emits: its stamp sits a few ms *before* the event, whereas
    # the supervisor's spawn time is ≥ an interpreter start-up earlier and its exit detection is *after*
    assert 0 <= (first_ts - final.started_at).total_seconds() < 0.1
    assert 0 <= (last_ts - final.finished_at).total_seconds() < 1.0
    assert final.created_at <= final.started_at <= final.finished_at


def test_api_stays_responsive_and_queue_is_serial(runtime):
    ws, store, sup = runtime
    first = sup.submit(smoke(epochs=40))
    wait_for(lambda: store.get_run(first.run_id).status == RunStatus.running)
    t0 = time.perf_counter()
    second = sup.submit(smoke())
    listed = store.list_runs()
    assert time.perf_counter() - t0 < 1.0
    assert {r.run_id for r in listed} == {first.run_id, second.run_id}
    assert store.get_run(second.run_id).status == RunStatus.queued     # same device -> waits
    assert sup.active_run_ids() == [first.run_id]
    # cancelling the queued run is immediate; cancelling the running one is cooperative
    assert sup.cancel(second.run_id).status == RunStatus.cancelled
    wait_for(lambda: store.get_run(first.run_id).progress_epoch and store.get_run(first.run_id).progress_epoch >= 1)
    t0 = time.perf_counter()
    requested = sup.cancel(first.run_id)
    assert requested.status == RunStatus.cancel_requested and time.perf_counter() - t0 < 1.0
    assert sup.cancel(first.run_id).status == RunStatus.cancel_requested      # idempotent
    final = wait_for(lambda: (r := store.get_run(first.run_id)) and r.status in TERMINAL_STATUSES and r)
    assert final.status == RunStatus.cancelled and final.progress_epoch < 40
    assert not is_same_process(final.worker.pid, final.worker.create_time)
    assert experiments.load_run(ws, first.run_id).status == RunStatus.cancelled
    assert not (ws.run_dir(first.run_id) / "CANCEL").exists()
    # a late completion can never resurrect a cancelled run
    with pytest.raises(ValueError):
        store.transition(first.run_id, RunStatus.succeeded)


def test_killed_worker_is_reported_as_failed(runtime):
    ws, store, sup = runtime
    record = sup.submit(smoke(epochs=40))
    running = wait_for(lambda: (r := store.get_run(record.run_id)) and r.status == RunStatus.running and r.worker and r)
    os.kill(running.worker.pid, signal.SIGKILL)
    final = wait_for(lambda: (r := store.get_run(record.run_id)) and r.status in TERMINAL_STATUSES and r)
    assert final.status == RunStatus.failed
    assert final.error.code == "worker_died" and final.exit_code not in (0, None)
    assert "out-of-memory" in final.error.hint


@pytest.mark.skipif(os.name == "nt" or os.geteuid() == 0, reason="permission bits are not enforced here")
def test_unwritable_run_directory_fails_explicitly(runtime):
    ws, store, sup = runtime
    sup._accepting = False    # create the run without dispatching it yet
    record = experiments.create_run(ws, smoke())
    store.upsert_run(record)
    run_dir = ws.run_dir(record.run_id)
    os.chmod(run_dir, 0o500)
    try:
        sup._accepting = True
        final = wait_for(lambda: (r := store.get_run(record.run_id)) and r.status in TERMINAL_STATUSES and r)
    finally:
        os.chmod(run_dir, 0o700)
    assert final.status == RunStatus.failed and final.error is not None


def test_restart_recovery_marks_interrupted(tmp_path):
    ws = Workspace.create(tmp_path / "ws")
    store = RunStore(ws.root / "metadata.sqlite")
    now = datetime.now(timezone.utc)
    dead = experiments.RunRecord(run_id="run-dead", task="train_pa", status=RunStatus.running, created_at=now,
                                 started_at=now, worker=WorkerInfo(pid=2 ** 22 - 1, create_time=1.0, host="h"))
    queued = experiments.RunRecord(run_id="run-queued", task="train_pa", status=RunStatus.queued, created_at=now)
    orphan_proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    ident = process_identity(orphan_proc.pid)
    orphan = experiments.RunRecord(run_id="run-orphan", task="train_pa", status=RunStatus.running, created_at=now,
                                   started_at=now, worker=WorkerInfo(pid=orphan_proc.pid, create_time=ident[1], host="h"))
    for r in (dead, queued, orphan):
        store.upsert_run(r)
    sup = Supervisor(ws, store, poll_interval=0.1)
    sup.recover()
    assert store.get_run("run-dead").status == RunStatus.interrupted
    assert "not found alive" in store.get_run("run-dead").status_reason
    assert store.get_run("run-queued").status == RunStatus.interrupted
    assert store.get_run("run-orphan").status == RunStatus.interrupted
    assert "terminated" in store.get_run("run-orphan").status_reason
    # psutil already reaped the orphan inside kill_tree, so Popen.wait() cannot
    # see the signal; liveness (pid + start time) is the meaningful check.
    orphan_proc.wait(timeout=10)
    assert not is_same_process(orphan_proc.pid, ident[1])
    assert store.count_runs(RunStatus.running) == 0
    store.close()


def test_stop_terminates_workers_and_marks_interrupted(tmp_path):
    ws = Workspace.create(tmp_path / "ws")
    ws.register_builtin_dataset("DPA_200MHz")
    store = RunStore(ws.root / "metadata.sqlite")
    sup = Supervisor(ws, store, poll_interval=0.1, cancel_grace=60)
    sup.start()
    record = sup.submit(smoke(epochs=40))
    running = wait_for(lambda: (r := store.get_run(record.run_id)) and r.status == RunStatus.running and r.worker and r)
    t0 = time.perf_counter()
    sup.stop(timeout=0.2)
    assert time.perf_counter() - t0 < 15
    with pytest.raises(RuntimeError):
        sup.submit(smoke())
    final = store.get_run(record.run_id)
    # Either the worker honoured the cancel file at an epoch boundary within
    # the bounded wait (cancelled) or it was terminated (interrupted).
    assert final.status in (RunStatus.interrupted, RunStatus.cancelled), final
    if final.status == RunStatus.interrupted:
        assert "shut down" in final.status_reason
    assert not is_same_process(running.worker.pid, running.worker.create_time)
    store.close()


def test_store_state_machine_and_schema_version(tmp_path):
    store = RunStore(tmp_path / "m.sqlite")
    now = datetime.now(timezone.utc)
    rec = experiments.RunRecord(run_id="r1", task="train_pa", status=RunStatus.queued, created_at=now)
    store.upsert_run(rec)
    store.transition("r1", RunStatus.running, started_at=now)
    with pytest.raises(ValueError):
        store.transition("r1", RunStatus.queued)
    store.transition("r1", RunStatus.succeeded, finished_at=now)
    assert store.get_run("r1").last_event_seq == 2
    store.close()
    import sqlite3
    conn = sqlite3.connect(tmp_path / "m.sqlite")
    conn.execute("UPDATE meta SET value='99' WHERE key='schema_version'"); conn.commit(); conn.close()
    with pytest.raises(RuntimeError, match="schema 99"):
        RunStore(tmp_path / "m.sqlite")


def test_heartbeat_staleness_uses_store_timestamps(tmp_path):
    from opendpd.schemas import heartbeat_is_stale
    store = RunStore(tmp_path / "m.sqlite")
    now = datetime.now(timezone.utc)
    rec = experiments.RunRecord(run_id="r1", task="train_pa", status=RunStatus.running, created_at=now,
                                started_at=now - timedelta(minutes=5),
                                last_heartbeat_at=now - timedelta(minutes=4))
    store.upsert_run(rec)
    assert heartbeat_is_stale(store.get_run("r1"), now, timedelta(seconds=60))
    store.append_event("r1", RunEventType.heartbeat, {}, last_heartbeat_at=now)
    assert not heartbeat_is_stale(store.get_run("r1"), now, timedelta(seconds=60))
    store.close()
