"""Evaluate every bundled DPD registry entry under the frozen Arena protocol.

The parameter sweep is trained first, one isolated process per sweep unit, into
a durable cache bound to the training fingerprint. Each board/backbone job then
reloads those checkpoints, judges them and is scored. Completed rows are
published to the bundled reference JSON only with the matching protocol hash.
No push, deployment, network publication or hardware RF output is performed.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from opendpd.core import arena
from opendpd.runtime.arena_limits import ARENA_MAX_RUNTIME_SECONDS
from opendpd.schemas.arena import ArenaRow
from opendpd.services.workspace import read_json, write_json_atomic


def publish(root, protocol, output=None):
    rows = []
    allowed = {(board.board_id, model.key) for board in protocol.boards for model in arena.bundled_backbones()}
    for path in sorted((root / "jobs").glob("*/result.json")):
        row = ArenaRow.model_validate(read_json(path))
        if row.protocol_sha256 != protocol.protocol_sha256 or (row.board_id, row.backbone) not in allowed:
            continue
        row = row.model_copy(
            update={
                "entry_id": f"official-{row.board_id}-{row.backbone}",
                "origin": "official",
            }
        )
        if row.status == "succeeded":
            row = ArenaRow.model_validate({**row.model_dump(), **arena.summarize_cases(
                row.backbone, row.board_id, row.cases, [p.model_dump(mode="json") for p in row.budgets])})
        rows.append(row.model_dump(mode="json"))
    content = {
        "protocol_sha256": protocol.protocol_sha256,
        "protocol_id": protocol.protocol_id,
        "rows": rows,
        "generated_by": "benchmark/run_arena_baselines.py",
    }
    write_json_atomic(
        output or arena.ASSETS / arena.RESULTS_FILE,
        {**content, "sha256": arena.canonical_hash(content)},
    )
    return rows


# Rough seconds per training, used only to order work. One GPU shared by time slicing is a
# serial device: concurrent processes gain almost nothing, while CPU cores are truly parallel.
# The Python recurrent cells keep a GPU busy with long replayed graphs of tiny kernels, so
# CPU workers take them unless the GPU runs out of other work. The device is
# operational only; every training record states the one that produced it.
GPU_SECONDS = dict(pgjanet=230, dvrjanet=194, apnrru=123, bojanet=73, gmp=32, qgru=18, qgru_amp1=19,
                   tres_deltagru=15, user_template=15)
CPU_SECONDS = dict(deltagru=500, deltajanet=480, pgjanet=480, bojanet=300, dvrjanet=1300, apnrru=840)
CPU_ONLY = ()  # Zero-threshold Delta cells now have a reviewed dense training path.


def worker_command(root, folder, key):
    return [sys.executable, "-m", "opendpd.core.arena_runner",
        "--request", str(folder / "request.json"), "--output", str(folder / "result.json"),
        "--cache", str(root / "cache"), "--cached-only"]


def run_guarded(command, log_path, timeout, env=None):
    """One isolated process group; a timeout or crash never leaves children behind."""
    with log_path.open("w") as log:
        proc = subprocess.Popen(command, stdout=log, stderr=log, start_new_session=True, env=env)
        try:
            return proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            return None


def foreign_worker(unit):
    """True while another process is already training this sweep unit."""
    wanted = ["--prefetch", *unit]
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit() and int(entry.name) != os.getpid():
            try:
                argv = (entry / "cmdline").read_bytes().decode(errors="replace").split("\0")
            except OSError:
                continue
            if any(argv[i:i + 4] == wanted for i in range(len(argv))):
                return True
    return False


def claim(path):
    """Hold one sweep unit for this launcher, or return None while another launcher holds it.

    The process table shows a unit that is already training; this closes the gap between
    two launchers both finding it free. The hold is an advisory lock, so it ends with the
    launcher however it exits and no stale owner ever has to be guessed.
    """
    try:
        import fcntl
    except ImportError:  # no advisory locks on this platform: one launcher at a time
        return open(path, "a")
    for _ in range(3):
        handle = open(path, "a")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:  # held elsewhere; any other failure is a real error and is reported
            handle.close()
            return None
        except OSError:
            handle.close()
            raise
        try:
            if os.fstat(handle.fileno()).st_ino == os.stat(path).st_ino:
                return handle
        except FileNotFoundError:
            pass
        handle.close()  # removed by its last holder meanwhile: hold the file that exists now
    return None


def release(path, hold):
    """Give a unit back. The file goes first, so a later holder never shares a removed one."""
    if hold is not None:
        path.unlink(missing_ok=True)
        hold.close()


def prefetch(root, keys, boards, cpu_workers, gpu_workers, reverse=False):
    """Train every sweep unit into the shared cache: a GPU queue and one CPU worker per core.

    A unit is one base backbone at one budget on one condition (all its seeds).
    Units are independent; the final per-board jobs then only judge and score.
    A second launcher may help with `reverse`: it starts from the other end of
    the queue, and a unit another process is training is left to that process.
    """
    import threading
    from opendpd.core.registry import get_model
    (root / "prefetch").mkdir(exist_ok=True)
    pending = [(key, str(budget), condition)
               for key in sorted({get_model(key).weights_from or key for key in keys})
               for budget in arena.BUDGETS if arena.model_parameters(key, budget) is not None
               for condition in sorted({c for board in boards for c in board.conditions})]
    total, finished, lock, busy = len(pending), [0], threading.Lock(), []
    ratio = lambda key: CPU_SECONDS[key] / GPU_SECONDS.get(key, 5)

    def take(kind):
        with lock:
            mine = [u for u in pending if (u[0] in CPU_SECONDS if kind == "cpu" else u[0] not in CPU_ONLY)]
            free = [u for u in mine if u not in busy] or mine  # postponed units come last, never lost
            if kind == "cpu":  # CPU-only cells first, then those the GPU gains least from
                choice = (max if reverse else min)(free, default=None,
                                                  key=lambda u: (u[0] not in CPU_ONLY, ratio(u[0]), -int(u[1])))
            else:  # GPU-only work first, longest first, then the cells a CPU core handles worst
                choice = min(free, default=None,
                             key=lambda u: (u[0] in CPU_SECONDS, -ratio(u[0]) if u[0] in CPU_SECONDS
                                            else -GPU_SECONDS.get(u[0], 5), -int(u[1])))
            if choice:
                pending.remove(choice)
            return choice

    def work(kind):
        # The protocol fixes four intra-op threads per process. These models are tiny, so the
        # helpers mostly wait: sleeping instead of spinning costs a quarter of the CPU time per
        # epoch and lets one process per core run. Numerics do not depend on the wait policy.
        # OMP_NUM_THREADS only reaches NumPy's OpenBLAS here; torch keeps the protocol's four.
        env = dict(os.environ, OMP_WAIT_POLICY="PASSIVE", GOMP_SPINCOUNT="0", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   NVIDIA_TF32_OVERRIDE="0")
        while unit := take(kind):
            name = "--".join(unit)
            done = root / "prefetch" / f"{name}.done"
            start, status = time.monotonic(), "cached"
            if not (done.exists() and done.read_text() == arena.training_fingerprint()):
                reserved, hold = root / "prefetch" / f"{name}.claim", None
                try:
                    hold = claim(reserved)
                    if unit not in busy and (hold is None or foreign_worker(unit)):
                        release(reserved, hold)  # ours, but an orphaned worker still trains it: hand it back meanwhile
                        with lock:  # another launcher is training it: revisit after everything else
                            busy.append(unit)
                            pending.append(unit)
                        continue
                    while hold is None or foreign_worker(unit):  # finish verifying its checkpoints once it exits
                        time.sleep(15)
                        hold = hold or claim(reserved)
                    try:
                        code = run_guarded([sys.executable, "-m", "opendpd.core.arena_runner", "--cache", str(root / "cache"),
                                            "--prefetch", *unit, *(["--device", "cpu"] if kind == "cpu" else [])],
                                           root / "prefetch" / f"{name}.{kind}.log", ARENA_MAX_RUNTIME_SECONDS, env)
                    finally:
                        release(reserved, hold)
                except OSError as exc:  # reported like any failed unit; the row job retrains on a cache miss
                    code = type(exc).__name__
                status = "ok" if code == 0 else f"failed ({code})"
                if code == 0:
                    done.write_text(arena.training_fingerprint())
            with lock:
                finished[0] += 1
                print(json.dumps({"prefetch": name, "worker": kind, "status": status, "done": finished[0],
                                  "seconds": round(time.monotonic() - start, 1), "units": total}), flush=True)

    crew = [threading.Thread(target=work, args=(kind,))
            for kind, count in (("gpu", gpu_workers), ("cpu", cpu_workers)) for _ in range(count)]
    for thread in crew:
        thread.start()
    for thread in crew:
        thread.join()
    if pending:
        raise RuntimeError(f"{len(pending)} sweep units had no worker; start at least one CPU and one GPU worker")


def evaluate(root, protocol, board, key, retry_failed):
    folder = root / "jobs" / f"{board.board_id}--{key}"
    folder.mkdir(exist_ok=True)
    output = folder / "result.json"
    if output.exists():
        prior = read_json(output)
        if prior.get("protocol_sha256") == protocol.protocol_sha256 and (
            prior["status"] == "succeeded" or not retry_failed
        ):
            return None
    # A fresh attempt must never inherit a previous run's completion
    # or partial evidence, even if its worker crashes before writing.
    for stale in (output, folder / "completed-cases.json", output.with_suffix(".progress.json")):
        stale.unlink(missing_ok=True)
    write_json_atomic(folder / "request.json", dict(
        board_id=board.board_id, backbone=key, protocol_sha256=protocol.protocol_sha256,
        model_parameters={}, model_provenance={"source": "bundled registry"}))
    start = time.monotonic()
    if run_guarded(worker_command(root, folder, key), folder / "worker.log", ARENA_MAX_RUNTIME_SECONDS,
                   dict(os.environ, OMP_WAIT_POLICY="PASSIVE")) is None:
        output.unlink(missing_ok=True)
    if not output.exists():
        partial = read_json(folder / "completed-cases.json") if (folder / "completed-cases.json").exists() else []
        deterministic = key in arena.DETERMINISTIC
        points = arena.sweep(key)
        available = sum(point["model_parameters"] is not None for point in points)
        row = ArenaRow(
            entry_id="pending-official", board_id=board.board_id, backbone=key,
            display_name=next(m.display_name for m in arena.bundled_backbones() if m.key == key),
            origin="official", status="failed", protocol_sha256=protocol.protocol_sha256,
            evidence_type=board.evidence_type, cases=partial,
            budgets=[dict(point, available=point["model_parameters"] is not None) for point in points],
            available_budgets=available, seeds=[0] if deterministic else protocol.seeds,
            expected_cases=available * len(board.conditions) * (1 if deterministic else len(protocol.seeds)),
            completed_cases=len(partial),
            error="Worker exceeded the Arena runtime guard or exited without a result.")
        write_json_atomic(output, row)
    row = read_json(output)
    if (row.get("protocol_sha256"), row.get("board_id"), row.get("backbone")) != (
            protocol.protocol_sha256, board.board_id, key):
        raise RuntimeError("Worker returned evidence for a different Arena request")
    return dict(board=board.board_id, backbone=key, status=row["status"], eligible=row["eligible"],
                score=row["score"], seconds=round(time.monotonic() - start, 2), error=row.get("error"))


def main():
    from concurrent.futures import ThreadPoolExecutor
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", type=Path, required=True)
    p.add_argument("--keys", nargs="+")
    p.add_argument("--boards", nargs="+")
    p.add_argument("--workers", type=int, default=1, help="Parallel isolated judging processes")
    p.add_argument("--cpu-workers", type=int, default=1, help="CPU training processes, about one per core")
    p.add_argument("--gpu-workers", type=int, default=1, help="Training processes sharing the GPU")
    p.add_argument("--reverse", action="store_true", help="Helper launcher: take CPU units from the far end of the queue")
    p.add_argument("--retry-failed", action="store_true")
    p.add_argument("--prefetch-only", action="store_true", help="Train the cache; judge and publish later")
    p.add_argument("--no-publish", action="store_true", help="Judge and score, but leave the bundled reference file alone")
    args = p.parse_args()
    root = args.workspace.resolve()
    (root / "jobs").mkdir(parents=True, exist_ok=True)
    protocol = arena.protocol()
    write_json_atomic(root / "protocol.json", protocol)
    all_keys = [m.key for m in arena.bundled_backbones()]
    keys = args.keys or all_keys
    if set(keys) - set(all_keys):
        raise ValueError("Unknown backbone requested")
    boards = [b for b in protocol.boards if not args.boards or b.board_id in args.boards]
    training = protocol.training_sha256
    prefetch(root, keys, boards, max(0, args.cpu_workers), max(0, args.gpu_workers), args.reverse)
    if arena.training_fingerprint() != training:
        raise RuntimeError("Training sources changed during the official matrix; refusing mixed evidence")
    if args.prefetch_only:
        return
    # Fail closed after any worker error: no row may load test data while a
    # different row still lacks a fitted checkpoint.
    from benchmark.run_arena_distributed import freeze, units
    from opendpd.core.registry import get_model
    bases={get_model(key).weights_from or key for key in keys}
    conditions={condition for board in boards for condition in board.conditions}
    freeze(root,[unit for unit in units() if unit[0] in bases and unit[2] in conditions])
    # Derived streaming entries reuse their base's cached weights and are
    # actually re-executed with a distinct rank cohort.
    jobs = [(board, key) for key in keys for board in boards]

    def run(job):
        if arena.protocol().protocol_sha256 != protocol.protocol_sha256:
            raise RuntimeError("Protocol changed during the official matrix; refusing mixed evidence")
        return evaluate(root, protocol, *job, args.retry_failed)

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for summary in pool.map(run, jobs):
            if summary:
                print(json.dumps(summary), flush=True)
    if args.no_publish:
        return
    rows = publish(root, protocol)
    expected = {(b.board_id, k) for b in protocol.boards for k in all_keys}
    missing = expected - {(r["board_id"], r["backbone"]) for r in rows}
    print(json.dumps({"rows": len(rows), "expected": len(expected), "missing": sorted(missing)}), flush=True)


if __name__ == "__main__":
    main()
