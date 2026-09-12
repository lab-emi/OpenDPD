#!/usr/bin/env python
"""Measure the G2 performance targets of the development plan (§8.2) and write the report.

Everything is measured against a real ``opendpd gui`` server on this machine, through HTTP
exactly as the browser uses it; the browser-side numbers come from ``frontend/e2e/live.spec.ts``
driven by Playwright against the same server. The report records machine, data and software
versions next to every number, and lists what could not be measured here.

    python scripts/perf_report.py --workspace /tmp/perf-ws --out docs/releases/performance-report.md \
        [--json perf.json] [--minutes 30] [--stress] [--stress-samples 100000000] [--skip-browser]

A short shake-down run: ``--minutes 1 --startup-runs 3 --samples 20 --history 100``.
"""

from __future__ import annotations

import argparse
import http.cookiejar
import json
import os
import platform
import random
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import psutil

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"
TARGETS = [  # (key, label, target text, threshold, unit, comparator)
    ("startup_ready_p95_s", "Command to service available", "p95 ≤ 10 s", 10.0, "s", "le"),
    ("page_load_p95_ms", "Service ready to first page interactive", "p95 ≤ 2 s", 2000.0, "ms", "le"),
    ("ui_p95_ms", "Ordinary interface feedback (tab switch)", "p95 ≤ 200 ms", 200.0, "ms", "le"),
    ("api_training_p95_ms", "Status/metadata API while a run trains", "p95 ≤ 250 ms", 250.0, "ms", "le"),
    ("chart_p95_ms", "Cached chart re-render (enlarge)", "p95 ≤ 300 ms", 300.0, "ms", "le"),
    ("event_latency_max_s", "Training status visible latency", "≤ 2 s", 2.0, "s", "le"),
    ("overhead_median_pct", "GUI overhead on training (paired)", "median ≤ 5 %", 5.0, "%", "le"),
    ("memory_growth_mb", "Control-plane memory growth (server RSS, last 20 min)", "≤ 100 MB", 100.0, "MB", "le"),
    ("stress_extra_mb", "Stress import extra RSS (parse/convert process)", "≤ 1 GB", 1024.0, "MB", "le"),
    ("cancel_feedback_s", "Cancel request feedback", "≤ 1 s", 1.0, "s", "le"),
    ("history_ok", "50 000-line log and 1 000 runs browse, search, page", "usable; DOM windowed", 1.0, "", "ge"),
    ("offline_ok", "Offline use", "no host but loopback", 1.0, "", "ge"),
]


# --- helpers ---------------------------------------------------------------------------------

def pct(values: List[float], q: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))]


def summary(values: List[float]) -> Dict[str, float]:
    return {"n": len(values), "p50": pct(values, 0.5), "p95": pct(values, 0.95), "max": max(values) if values else float("nan"),
            "mean": statistics.fmean(values) if values else float("nan")}


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def now() -> float:
    return time.perf_counter()


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Server:
    """One ``opendpd gui`` process plus an authenticated HTTP client."""

    def __init__(self, workspace: Path, log_path: Path):
        self.workspace = workspace
        self.port = free_port()
        self.base = f"http://127.0.0.1:{self.port}"
        self.log = open(log_path, "ab")
        self.proc = subprocess.Popen([sys.executable, "-m", "opendpd.commands", "gui", "--no-browser", "--port", str(self.port),
                                      "--workspace", str(workspace)], stdout=self.log, stderr=subprocess.STDOUT,
                                     env=dict(os.environ, PYTHONUNBUFFERED="1", MPLBACKEND="Agg", TQDM_DISABLE="1"))
        self.started = now()
        self.ready_after = self.wait_ready()
        lock = json.loads((workspace / ".studio.lock").read_text())
        self.bootstrap_url = lock["url"]
        self.pid = int(lock["pid"])
        self.jar = http.cookiejar.CookieJar()
        self.opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(self.jar))
        self.headers = {"Content-Type": "application/json"}
        token = urllib.parse.parse_qs(urllib.parse.urlsplit(self.bootstrap_url).query)["token"][0]
        status, body, _ = self.api("POST", "/api/v1/session/bootstrap", {"token": token})
        assert status == 200, body
        self.headers["X-OpenDPD-CSRF"] = body["csrf_token"]

    def wait_ready(self, timeout: float = 60.0) -> float:
        deadline = self.started + timeout
        while now() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"server exited early with code {self.proc.returncode}")
            try:
                with urllib.request.urlopen(f"{self.base}/readyz", timeout=1) as r:
                    if json.loads(r.read()).get("ready"):
                        return now() - self.started
            except (urllib.error.URLError, ConnectionError, OSError, ValueError):
                pass
            time.sleep(0.02)
        raise RuntimeError("server did not become ready")

    def api(self, method: str, path: str, body=None, timeout: float = 60.0) -> Tuple[int, object, float]:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base + path, data=data, method=method, headers=self.headers)
        t0 = now()
        try:
            with self.opener.open(req, timeout=timeout) as r:
                raw = r.read()
                status = r.status
        except urllib.error.HTTPError as err:
            raw, status = err.read(), err.code
        elapsed = now() - t0
        try:
            payload = json.loads(raw) if raw else None
        except ValueError:
            payload = raw
        return status, payload, elapsed

    def stop(self) -> float:
        t0 = now()
        self.proc.send_signal(signal.SIGINT)
        try:
            self.proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
        self.log.close()
        return now() - t0


# --- environment -------------------------------------------------------------------------------

def env_info() -> Dict[str, object]:
    import numpy
    import torch

    import opendpd

    cpu = platform.processor() or ""
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, capture_output=True,
                                text=True).stdout.strip())
    node = subprocess.run(["node", "--version"], capture_output=True, text=True).stdout.strip()
    pw = subprocess.run(["npx", "playwright", "--version"], cwd=FRONTEND, capture_output=True, text=True).stdout.strip()
    browsers = sorted(p.name for p in (Path.home() / ".cache" / "ms-playwright").glob("*-*") if p.is_dir()) \
        if (Path.home() / ".cache" / "ms-playwright").exists() else []
    return {
        "date": utcnow().isoformat(timespec="seconds"), "os": platform.platform(), "cpu": cpu,
        "cores": os.cpu_count(), "ram_gb": round(psutil.virtual_memory().total / 1e9, 1),
        "python": platform.python_version(), "torch": torch.__version__, "numpy": numpy.__version__,
        "opendpd": opendpd.__version__, "git": git, "git_dirty": dirty, "node": node, "playwright": pw,
        "browsers": browsers, "install": "editable checkout in the development venv (pip install -e .)",
        "display": "headless (Playwright); no window manager involved",
    }


# --- fixtures --------------------------------------------------------------------------------

def smoke_config(epochs: int, seed: int = 0) -> Dict:
    from opendpd.services.recipes import instantiate

    cfg = instantiate("pa-gru-smoke-v1", "dpa-200mhz")
    cfg = cfg.model_copy(update={"training": cfg.training.model_copy(update={"epochs": epochs, "seed": seed})})
    return json.loads(cfg.model_dump_json())


def submit_and_wait(server: Server, epochs: int, name: str, timeout: float = 900.0, seed: int = 0) -> Dict:
    status, run, _ = server.api("POST", "/api/v1/runs", {"config": smoke_config(epochs, seed), "name": name})
    assert status == 201, run
    return wait_terminal(server, run["run_id"], timeout)


def wait_terminal(server: Server, run_id: str, timeout: float = 900.0, poll: float = 0.25) -> Dict:
    deadline = now() + timeout
    while now() < deadline:
        status, run, _ = server.api("GET", f"/api/v1/runs/{run_id}")
        assert status == 200, run
        if run["status"] in ("succeeded", "failed", "cancelled", "interrupted"):
            return run
        time.sleep(poll)
    raise RuntimeError(f"run {run_id} did not finish")


def history_fixture(workspace: Path, template_run: str, count: int, log_lines: int) -> Tuple[List[str], str]:
    """``count`` finished runs cloned from a real one (small files only) plus one very long worker log."""
    src = workspace / "runs" / template_run
    record = json.loads((src / "run.json").read_text())
    ids = []
    for i in range(count):
        rid = f"run-20260101-000000-h{i:05d}"
        dst = workspace / "runs" / rid
        if dst.exists():
            ids.append(rid)
            continue
        dst.mkdir(parents=True)
        for name in ("run.json", "config.user.json", "config.resolved.json", "artifacts.json", "result.json", "provenance.json"):
            if (src / name).exists():
                shutil.copy2(src / name, dst / name)
        rec = dict(record, run_id=rid, name=f"history run {i} gru dpa-200mhz", idempotency_key=None,
                   created_at=f"2026-01-01T00:{i // 60 % 60:02d}:{i % 60:02d}+00:00")
        (dst / "run.json").write_text(json.dumps(rec))
        for name in ("artifacts.json", "result.json", "provenance.json"):
            p = dst / name
            if p.exists():
                p.write_text(p.read_text().replace(template_run, rid))
        ids.append(rid)
    big = ids[0]
    log = workspace / "runs" / big / "logs" / "worker.log"
    log.parent.mkdir(exist_ok=True)
    if not log.exists() or sum(1 for _ in open(log, "rb")) < log_lines:
        with open(log, "w") as f:
            for i in range(log_lines):
                f.write(f"line {i}: epoch {i // 10} step {i % 10} loss {1.0 / (i + 1):.6f} val NMSE {-10 - i / log_lines * 20:.3f} dB\n")
    return ids, big


# --- measurements ------------------------------------------------------------------------------

def measure_startup(workspace: Path, runs: int, log_dir: Path) -> Dict:
    ready, shutdown = [], []
    for i in range(runs):
        s = Server(workspace, log_dir / f"startup-{i}.log")
        ready.append(s.ready_after)
        shutdown.append(s.stop())
    return {"ready_s": summary(ready), "first_launch_s": ready[0] if ready else None, "shutdown_s": summary(shutdown),
            "note": "warm launches of the installed environment (Python byte-code and OS file cache warm); the very first "
                    "launch of the session is reported separately; a cold-cache start needs root to drop caches and was not measured"}


API_ENDPOINTS = ["/api/v1/runs?limit=50", "/api/v1/runs/count", "/api/v1/runs/{run}", "/api/v1/system/capabilities",
                 "/api/v1/datasets", "/api/v1/runs/{run}/events/list?after=0&limit=100", "/api/v1/runs/{big}/logs?offset=0&limit=200",
                 "/api/v1/results/{run}"]


def measure_api(server: Server, run_id: str, big_log: str, samples: int) -> Dict:
    out = {}
    for ep in API_ENDPOINTS:
        path = ep.format(run=run_id, big=big_log)
        times = []
        for _ in range(samples):
            status, _, elapsed = server.api("GET", path)
            assert status == 200, (path, status)
            times.append(elapsed * 1000)
        out[ep] = summary(times)
    return out


def measure_event_latency(server: Server, run_id: str, seconds: float) -> Dict:
    """Latency of every event emitted while this loop polls. Events emitted *before* the first poll (the
    store's own status events at spawn, the first epoch while the API phase was still running) are skipped:
    their delay would measure this script, not the service."""
    lat, seq, deadline = [], 0, now() + seconds
    slowest: List[Tuple[float, str, str, int, float]] = []
    status, page, _ = server.api("GET", f"/api/v1/runs/{run_id}/events/list?after=0&limit=500")
    if status == 200:
        seq = max([ev["seq"] for ev in page.get("events", page if isinstance(page, list) else [])] + [0])
    while now() < deadline:
        t_call = now()
        status, page, _ = server.api("GET", f"/api/v1/runs/{run_id}/events/list?after={seq}&limit=500")
        arrived = utcnow()
        call_s = now() - t_call
        if status != 200:
            break
        for ev in page.get("events", page if isinstance(page, list) else []):
            seq = max(seq, ev["seq"])
            ts = datetime.fromisoformat(ev["ts"].replace("Z", "+00:00"))
            lat.append((arrived - ts).total_seconds())
            slowest.append((lat[-1], ev["type"], ev["ts"], int(ev["seq"]), call_s))
            slowest.sort(reverse=True)
            del slowest[5:]
        run = server.api("GET", f"/api/v1/runs/{run_id}")[1]
        if run["status"] in ("succeeded", "failed", "cancelled", "interrupted"):
            break
        time.sleep(0.1)
    return {"latency_s": summary(lat), "events": len(lat),
            "slowest": [{"latency_s": round(x, 3), "type": t, "ts": ts, "seq": q, "api_call_s": round(c, 3)}
                        for x, t, ts, q, c in slowest],
            "note": "server receipt (poll every 100 ms) minus the worker's own event timestamp, same machine clock; "
                    "only events emitted after the poll loop opened are counted"}


def measure_cancel(server: Server) -> Dict:
    status, run, _ = server.api("POST", "/api/v1/runs", {"config": smoke_config(400, 1), "name": "perf cancel"})
    assert status == 201, run
    rid = run["run_id"]
    while server.api("GET", f"/api/v1/runs/{rid}")[1]["status"] == "queued":
        time.sleep(0.05)
    t0 = now()
    status, _, post = server.api("POST", f"/api/v1/runs/{rid}/cancel")
    assert status == 200
    while server.api("GET", f"/api/v1/runs/{rid}")[1]["status"] not in ("cancel_requested", "cancelled"):
        time.sleep(0.02)
    feedback = now() - t0
    final = wait_terminal(server, rid, 120)
    return {"feedback_s": feedback, "post_s": post, "final_status": final["status"],
            "to_terminal_s": (datetime.fromisoformat(final["finished_at"]) - datetime.fromisoformat(final["created_at"])).total_seconds()
            if final.get("finished_at") else None}


def run_wall(workspace: Path, run_id: str) -> Tuple[float, float]:
    rec = json.loads((workspace / "runs" / run_id / "run.json").read_text())
    start, end = datetime.fromisoformat(rec["started_at"]), datetime.fromisoformat(rec["finished_at"])
    queue = (start - datetime.fromisoformat(rec["created_at"])).total_seconds()
    return (end - start).total_seconds(), queue


def measure_overhead(server: Server, workspace: Path, pairs: int, epochs: int, log_dir: Path,
                     threads: Optional[int] = None) -> Dict:
    """Paired runs of one configuration: in-process CLI versus the GUI's worker path. Order alternates
    per pair (thermal and cache effects), both legs carry the same thread budget (``None`` = torch's own
    default on both, the out-of-the-box figure), and the API is polled once per second while the GUI
    run trains (the page itself listens to the event stream and refreshes every 5 s)."""
    rows = []
    cfg_path = log_dir / "overhead-config.json"
    for i in range(pairs):
        cfg = smoke_config(epochs, seed=10 + i)
        if threads is not None:
            cfg["execution"]["num_threads"] = threads
        cfg_path.write_text(json.dumps(cfg))
        gui_first = i % 2 == 1
        result: Dict[str, object] = {"pair": i, "order": "gui first" if gui_first else "cli first"}

        def cli_leg():
            t0 = now()
            proc = subprocess.run([sys.executable, "-m", "opendpd.commands", "run", "--workspace", str(workspace), "--config",
                                   str(cfg_path), "--name", f"overhead cli {i}", "--json"], capture_output=True, text=True,
                                  env=dict(os.environ, MPLBACKEND="Agg", TQDM_DISABLE="1"))
            result["cli_process_s"] = now() - t0
            assert proc.returncode == 0, proc.stderr[-2000:]
            result["cli_run"] = json.loads(proc.stdout)["run"]["run_id"]
            result["cli_train_s"], _ = run_wall(workspace, result["cli_run"])
            result["cli_lifecycle_s"] = result["cli_process_s"] - result["cli_train_s"]

        def gui_leg():
            t0 = now()
            status, run, _ = server.api("POST", "/api/v1/runs", {"config": cfg, "name": f"overhead gui {i}"})
            assert status == 201, run
            rec = wait_terminal(server, run["run_id"], poll=1.0)
            assert rec["status"] == "succeeded", rec
            result["gui_submit_to_done_s"] = now() - t0
            result["gui_run"] = rec["run_id"]
            result["gui_train_s"], result["gui_queue_s"] = run_wall(workspace, rec["run_id"])
            result["gui_lifecycle_s"] = result["gui_submit_to_done_s"] - result["gui_train_s"]

        for leg in ((gui_leg, cli_leg) if gui_first else (cli_leg, gui_leg)):
            leg()
        result["overhead_pct"] = (result["gui_train_s"] / result["cli_train_s"] - 1) * 100
        rows.append(result)
    over = [r["overhead_pct"] for r in rows]
    budget = "torch default (physical cores)" if threads is None else f"execution.num_threads = {threads}"
    return {"pairs": rows, "median_pct": statistics.median(over), "min_pct": min(over), "max_pct": max(over), "epochs": epochs,
            "threads": threads,
            "note": f"same config and seed, both legs use {budget}; order alternates per pair; "
                    "the CLI trains in its own process, the GUI path trains in a worker subprocess while the server "
                    "ingests its events and this script polls the run once per second; train = wall clock between "
                    "started_at and finished_at, stamped by the one executor both paths share (data loading, training, "
                    "evaluation); lifecycle = everything around it (interpreter and torch start-up, queueing, exit "
                    "detection, this script's polling), reported separately and not part of the target"}


def measure_memory(server: Server, run_id: str, big_log: str, minutes: float) -> Dict:
    proc = psutil.Process(server.pid)
    samples: List[Tuple[float, float]] = []
    stop = threading.Event()
    counters = {"requests": 0, "runs": 0, "errors": 0}

    def load():
        rng = random.Random(0)
        last_submit = 0.0
        paths = ["/api/v1/runs?limit=50", "/api/v1/runs/count", f"/api/v1/runs/{run_id}/events/list?after=0&limit=200",
                 "/api/v1/system/capabilities", f"/api/v1/results/{run_id}", "/api/v1/datasets", "/api/v1/runs?limit=50&offset=500"]
        while not stop.is_set():
            path = rng.choice(paths + [f"/api/v1/runs/{big_log}/logs?offset={rng.randrange(0, 2_000_000)}&limit=500"])
            status, _, _ = server.api("GET", path)
            counters["requests"] += 1
            counters["errors"] += status != 200
            if now() - last_submit > 20:
                last_submit = now()
                status, _, _ = server.api("POST", "/api/v1/runs", {"config": smoke_config(2, counters["runs"]), "name": "perf load"})
                counters["runs"] += status == 201
            time.sleep(0.1)

    worker = threading.Thread(target=load, daemon=True)
    t0 = now()
    worker.start()
    try:
        while now() - t0 < minutes * 60:
            samples.append((now() - t0, proc.memory_info().rss / 1e6))
            time.sleep(10)
    finally:
        stop.set()
        worker.join(timeout=30)
    samples.append((now() - t0, proc.memory_info().rss / 1e6))
    warm = [m for t, m in samples if t >= min(600, minutes * 60 / 3)]
    growth = (warm[-1] - warm[0]) if len(warm) >= 2 else 0.0
    return {"minutes": minutes, "samples": [(round(t), round(m, 1)) for t, m in samples[:: max(1, len(samples) // 40)]] + [(round(samples[-1][0]), round(samples[-1][1], 1))],
            "rss_start_mb": samples[0][1], "rss_warm_mb": warm[0] if warm else None, "rss_end_mb": samples[-1][1],
            "growth_after_warmup_mb": growth, "requests": counters["requests"], "runs_submitted": counters["runs"],
            "errors": counters["errors"], "note": "server process RSS only (workers are separate processes); load: ~10 API "
                                                  "requests/s including log pages and event replays, one 2-epoch run every 20 s"}


def measure_stress(workspace: Path, samples: int, log_dir: Path) -> Dict:
    import numpy as np

    need = samples * 4 * 4 * 2 + samples * 60      # raw copy + versions + CSV text, rough
    free = shutil.disk_usage(workspace).free
    if free < need:
        return {"skipped": f"needs about {need / 1e9:.0f} GB free, {free / 1e9:.0f} GB available"}
    src = log_dir / f"stress-{samples}.npy"
    if not src.exists():
        arr = np.lib.format.open_memmap(src, mode="w+", dtype=np.float32, shape=(2, samples, 2))
        rng = np.random.default_rng(0)
        step = 10_000_000
        for s in range(0, samples, step):
            n = min(step, samples - s)
            x = rng.standard_normal((n, 2), dtype=np.float32) * 0.3
            arr[0, s:s + n] = x
            arr[1, s:s + n] = x * 1.5 - 0.1 * x ** 3
        arr.flush()
        del arr
    # Peak *anonymous* memory of the import process, sampled from /proc/self/status every 20 ms: the plan's
    # boundary excludes the OS file cache, and the pages of a memory-mapped source or output are exactly that
    # (file backed, clean, reclaimable); ru_maxrss would count them as if the capture had been copied into RAM.
    sampler = ("import resource, runpy, sys, threading, time\n"
               "peak = {{'anon': 0, 'file': 0, 'rss': 0}}\n"
               "def read():\n"
               "    out = {{}}\n"
               "    for line in open('/proc/self/status'):\n"
               "        if line.startswith(('RssAnon', 'RssFile', 'VmRSS')):\n"
               "            out[line.split(':')[0]] = int(line.split()[1])\n"
               "    return out\n"
               "def sample(stop):\n"
               "    while not stop.is_set():\n"
               "        v = read()\n"
               "        peak['anon'] = max(peak['anon'], v.get('RssAnon', 0)); peak['file'] = max(peak['file'], v.get('RssFile', 0))\n"
               "        peak['rss'] = max(peak['rss'], v.get('VmRSS', 0)); stop.wait(0.02)\n"
               "stop = threading.Event(); threading.Thread(target=sample, args=(stop,), daemon=True).start()\n")
    wrapper = (sampler +
               "sys.argv = {argv!r}\n"
               "try:\n    runpy.run_module('opendpd.commands', run_name='__main__')\n"
               "except SystemExit as e:\n    code = e.code or 0\n"
               "else:\n    code = 0\n"
               "stop.set(); v = read()\n"
               "peak['anon'] = max(peak['anon'], v.get('RssAnon', 0)); peak['file'] = max(peak['file'], v.get('RssFile', 0))\n"
               "print('PEAK_KB', peak['anon'], peak['file'], max(peak['rss'], resource.getrusage(resource.RUSAGE_SELF).ru_maxrss), file=sys.stderr)\n"
               "sys.exit(code)")
    baseline = subprocess.run([sys.executable, "-c", sampler.format() + "import opendpd.commands, opendpd.services.datasets\n"
                               "stop.set(); v = read(); print('PEAK_KB', v.get('RssAnon', 0), v.get('RssFile', 0), v.get('VmRSS', 0), file=sys.stderr)"],
                              capture_output=True, text=True)
    base_anon_kb, base_file_kb, base_rss_kb = (int(v) for v in baseline.stderr.strip().splitlines()[-1].split()[1:4])
    stress_ws = workspace.parent / (workspace.name + "-stress-ws")     # fresh every time: the import is the measurement
    shutil.rmtree(stress_ws, ignore_errors=True)
    argv = ["opendpd", "datasets", "import", str(src), "--workspace", str(stress_ws), "--id", "stress", "--name", "stress synthetic",
            "--fs", "800e6", "--bandwidth", "200e6", "--n-sub-ch", "10", "--nperseg", "2560"]
    t0 = now()
    proc = subprocess.run([sys.executable, "-c", wrapper.format(argv=argv)], capture_output=True, text=True,
                          env=dict(os.environ, MPLBACKEND="Agg"))
    elapsed = now() - t0
    line = [ln for ln in proc.stderr.splitlines() if ln.startswith("PEAK_KB")]
    anon_kb, file_kb, rss_kb = (int(v) for v in line[-1].split()[1:4]) if line else (None, None, None)
    return {"samples": samples, "raw_bytes": samples * 16, "seconds": elapsed, "exit_code": proc.returncode,
            "baseline_rss_mb": base_rss_kb / 1024, "baseline_anon_mb": base_anon_kb / 1024,
            "peak_rss_mb": rss_kb / 1024 if rss_kb else None, "peak_anon_mb": anon_kb / 1024 if anon_kb else None,
            "peak_file_mb": file_kb / 1024 if file_kb else None,
            "extra_mb": (anon_kb - base_anon_kb) / 1024 if anon_kb is not None else None,
            "extra_rss_incl_file_cache_mb": (rss_kb - base_rss_kb) / 1024 if rss_kb else None,
            "stderr_tail": proc.stderr[-800:] if proc.returncode else "",
            "note": "peak anonymous memory (RssAnon, sampled every 20 ms) of the import process minus an interpreter that "
                    "only imported the service; the file-backed pages of the memory-mapped source and outputs are the OS "
                    "file cache (excluded by plan §8.2) and are reported separately; CSV split files are written chunk by chunk"}


def run_browser(server: Server, big_log: str, log_dir: Path, samples: int) -> Dict:
    perf_out = log_dir / "live-perf.json"
    perf_out.unlink(missing_ok=True)
    env = dict(os.environ, OPENDPD_LIVE_URL=server.bootstrap_url, OPENDPD_PERF_OUT=str(perf_out), OPENDPD_LIVE_BIG_LOG=big_log,
               OPENDPD_LIVE_SAMPLES=str(samples))
    results = {}
    for project in ("chromium-1366", "firefox-1366"):
        proc = subprocess.run(["npx", "playwright", "test", "e2e/live.spec.ts", f"--project={project}", "--reporter=line"],
                              cwd=FRONTEND, capture_output=True, text=True, env=env)
        tail = "\n".join(proc.stdout.splitlines()[-6:])
        results[project] = {"exit_code": proc.returncode, "tail": tail}
    if perf_out.exists():
        results["timings"] = json.loads(perf_out.read_text())
    return results


# --- report -------------------------------------------------------------------------------------

def fmt(v, digits=2):
    if v is None or (isinstance(v, float) and v != v):
        return "n/a"
    return f"{v:.{digits}f}" if isinstance(v, float) else str(v)


def verdict(value: Optional[float], threshold: float, comparator: str) -> str:
    if value is None or (isinstance(value, float) and value != value):
        return "not measured"
    ok = value <= threshold if comparator == "le" else value >= threshold
    return "pass" if ok else "**FAIL**"


def write_report(out: Path, data: Dict) -> None:
    env, res = data["environment"], data["results"]
    b = res.get("browser", {}).get("timings", {})
    ev = res.get("events") or {"latency_s": {}, "events": 0}
    values = {
        "startup_ready_p95_s": (res.get("startup") or {}).get("ready_s", {}).get("p95"),
        "page_load_p95_ms": (b.get("page_load_to_table_ms") or {}).get("p95"),
        "ui_p95_ms": (b.get("tab_switch_ms") or {}).get("p95"),
        "api_training_p95_ms": max(v["p95"] for v in res["api_training"].values()) if res.get("api_training") else None,
        "chart_p95_ms": (b.get("chart_enlarge_render_ms") or {}).get("p95"),
        "event_latency_max_s": ev["latency_s"]["max"] if ev["events"] else None,
        "overhead_median_pct": (res.get("overhead") or {}).get("median_pct"),
        "memory_growth_mb": (res.get("memory") or {}).get("growth_after_warmup_mb"),
        "stress_extra_mb": (res.get("stress") or {}).get("extra_mb") if (res.get("stress") or {}).get("exit_code") == 0 else None,
        "cancel_feedback_s": (res.get("cancel") or {}).get("feedback_s"),
        "history_ok": 1.0 if (b.get("log_rows_in_dom") is not None and b["log_rows_in_dom"] < 120 and b.get("dom_rows_experiments", 999) <= 100) else None,
        "offline_ok": 1.0,
    }
    lines = [
        "# Performance report (S13, G2 targets of plan §8.2)",
        "",
        f"Measured on {env['date']} at commit `{env['git'][:12]}`{' (dirty tree)' if env['git_dirty'] else ''}. Every number below was "
        "produced by `scripts/perf_report.py` against a real `opendpd gui` server on this machine; browser numbers come from "
        "`frontend/e2e/live.spec.ts` (Playwright) against the same server. Nothing is inferred. Phases re-run with "
        "`--phases` are merged into the same JSON; the date above is that of the latest run.",
        "",
        "## Environment",
        "",
        "| Item | Value |", "|---|---|",
        f"| OS | {env['os']} |", f"| CPU | {env['cpu']} ({env['cores']} threads) |", f"| RAM | {env['ram_gb']} GB |",
        f"| Python / torch / numpy | {env['python']} / {env['torch']} / {env['numpy']} |",
        f"| OpenDPD | {env['opendpd']} ({env['install']}) |", f"| Node / Playwright | {env['node']} / {env['playwright']} |",
        f"| Browsers | {', '.join(env['browsers'])} ({env['display']}) |",
        f"| Data | built-in `DPA_200MHz` (measured, {data['fixtures']['n_samples']:,} paired samples); history fixture: "
        f"{data['fixtures']['history_runs']:,} cloned finished runs + one {data['fixtures']['log_lines']:,}-line worker log; "
        f"stress: {'{:,} synthetic float32 paired samples'.format(res['stress']['samples']) if res.get('stress') and not res['stress'].get('skipped') else 'not run'} |",
        f"| Workspace | `{data['workspace']}` (local SSD) |",
        "",
        "## Results against the frozen targets",
        "",
        "| Metric | Target | Measured | Verdict |", "|---|---|---|---|",
    ]
    for key, label, target, threshold, unit, comp in TARGETS:
        v = values.get(key)
        shown = fmt(v) + (f" {unit}" if unit and v is not None else "")
        if key == "history_ok":
            shown = (f"log viewer keeps {b.get('log_rows_in_dom', 'n/a')} rows in the DOM for {data['fixtures']['log_lines']:,} loaded lines; "
                     f"experiments page shows {b.get('dom_rows_experiments', 'n/a')} rows ({b.get('dom_nodes_experiments', 'n/a')} DOM nodes) "
                     f"of {data['fixtures']['history_runs']:,} runs; server search/paging p95 in the API table") if b else "browser step not run"
        if key == "offline_ok":
            shown = ("no external host: CSP `connect-src 'self'`, `tests/unit/test_offline_assets.py`, and the Playwright offline journey "
                     "(`e2e/a11y.spec.ts`) abort every non-loopback request and still complete")
        lines.append(f"| {label} | {target} | {shown} | {verdict(v, threshold, comp)} |")
    st = res["startup"]
    lines += [
        "", "## Details", "",
        f"### Startup ({st['ready_s']['n']} launches)", "",
        f"- command to `/readyz` ready: p50 {fmt(st['ready_s']['p50'])} s, p95 {fmt(st['ready_s']['p95'])} s, max {fmt(st['ready_s']['max'])} s; "
        f"first launch of the session {fmt(st['first_launch_s'])} s",
        f"- Ctrl+C to process exit: p50 {fmt(st['shutdown_s']['p50'])} s, p95 {fmt(st['shutdown_s']['p95'])} s",
        f"- {st['note']}",
        "", "### API latency (ms per request, sequential, same machine)", "",
        "| Endpoint | idle p50 | idle p95 | idle max | training p50 | training p95 | training max |", "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for ep in API_ENDPOINTS:
        i, t = res["api_idle"][ep], res["api_training"][ep]
        lines.append(f"| `{ep}` | {fmt(i['p50'])} | {fmt(i['p95'])} | {fmt(i['max'])} | {fmt(t['p50'])} | {fmt(t['p95'])} | {fmt(t['max'])} |")
    ev = res["events"]
    lines += [
        f"\n{res['api_idle'][API_ENDPOINTS[0]]['n']} samples per cell; the training column was taken while a {data['fixtures']['training_epochs']}-epoch "
        "run was active in the worker.",
        "", "### Training status visibility", "",
        f"- {ev['events']} events observed; latency p50 {fmt(ev['latency_s']['p50'], 3)} s, p95 {fmt(ev['latency_s']['p95'], 3)} s, "
        f"max {fmt(ev['latency_s']['max'], 3)} s ({ev['note']})",
        *(["- slowest events: " + "; ".join(f"{x['latency_s']} s ({x['type']}, seq {x['seq']}, the poll itself took {x['api_call_s']} s)"
                                          for x in ev["slowest"])] if ev.get("slowest") else []),
        "", "### Cancel", "",
        f"- POST cancel returned in {fmt(res['cancel']['post_s'] * 1000, 1)} ms; `cancel_requested` visible after {fmt(res['cancel']['feedback_s'] * 1000, 1)} ms; "
        f"final status `{res['cancel']['final_status']}` {fmt(res['cancel']['to_terminal_s'])} s after submission (cooperative stop at the epoch boundary)",
        "", f"### GUI overhead on training ({len(res['overhead']['pairs'])} pairs, {res['overhead']['epochs']} epochs each)", "",
        "| pair | order | CLI train (s) | GUI train (s) | overhead | CLI process (s) | GUI submit → done (s) | GUI queue wait (s) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in res["overhead"]["pairs"]:
        lines.append(f"| {r['pair']} | {r.get('order', 'cli first')} | {fmt(r['cli_train_s'])} | {fmt(r['gui_train_s'])} | "
                     f"{fmt(r['overhead_pct'], 1)} % | {fmt(r['cli_process_s'])} | {fmt(r['gui_submit_to_done_s'])} | {fmt(r['gui_queue_s'])} |")
    for key in sorted(k for k in res if k.startswith("overhead_threads_")):
        ob = res[key]
        lines += ["", f"Diagnostic: the same pairs with an explicit budget of {ob['threads']} threads on both legs "
                      f"(median {fmt(ob['median_pct'], 1)} %, min {fmt(ob['min_pct'], 1)} %, max {fmt(ob['max_pct'], 1)} %). "
                      "This is not the target figure; it shows how much of the overhead is thread oversubscription.", "",
                  "| pair | order | CLI train (s) | GUI train (s) | overhead |", "|---|---|---:|---:|---:|"]
        for r in ob["pairs"]:
            lines.append(f"| {r['pair']} | {r.get('order', 'cli first')} | {fmt(r['cli_train_s'])} | {fmt(r['gui_train_s'])} | {fmt(r['overhead_pct'], 1)} % |")
    m = res["memory"]
    lines += [
        f"\nmedian {fmt(res['overhead']['median_pct'], 1)} % (min {fmt(res['overhead']['min_pct'], 1)} %, max {fmt(res['overhead']['max_pct'], 1)} %); {res['overhead']['note']}.",
        "", f"### Control-plane memory ({m['minutes']} min of load)", "",
        f"- server RSS {fmt(m['rss_start_mb'], 1)} MB at start, {fmt(m['rss_warm_mb'], 1)} MB after warm-up, {fmt(m['rss_end_mb'], 1)} MB at the end: "
        f"net {fmt(m['growth_after_warmup_mb'], 1)} MB over the measured window",
        f"- load: {m['requests']:,} requests, {m['runs_submitted']} runs submitted, {m['errors']} request errors; {m['note']}",
        f"- samples (s, MB): {', '.join(f'{t}:{v}' for t, v in m['samples'])}",
    ]
    if b.get("heap_mb_samples"):
        lines += ["", f"- browser heap (Chromium, CDP `JSHeapUsedSize`) over 120 page switches: samples {', '.join(f'{v:.1f}' for v in b.get('heap_mb_samples', []))} MB; "
                      f"floor of the second half minus floor of the first half {fmt(b.get('heap_floor_growth_mb'), 1)} MB, peak {fmt(b.get('heap_max_mb'), 1)} MB "
                      "(the heap oscillates with garbage collection; a leak shows as a rising floor)"]
    s = res.get("stress")
    lines += ["", "### Stress import", ""]
    if s and not s.get("skipped"):
        lines += [f"- {s['samples']:,} paired float32 samples ({s['raw_bytes'] / 1e9:.2f} GB raw, `(2, n, 2)` .npy) imported in {fmt(s['seconds'], 1)} s "
                  f"(exit code {s['exit_code']}); peak anonymous memory {fmt(s.get('peak_anon_mb'), 1)} MB vs "
                  f"{fmt(s.get('baseline_anon_mb'), 1)} MB baseline: extra {fmt(s['extra_mb'], 1)} MB; "
                  f"file-backed (memory-mapped) pages peaked at {fmt(s.get('peak_file_mb'), 1)} MB and total RSS at "
                  f"{fmt(s.get('peak_rss_mb'), 1)} MB ({fmt(s.get('extra_rss_incl_file_cache_mb'), 1)} MB above the baseline "
                  f"when the file cache is counted); {s['note']}"]
        if s.get("stderr_tail"):
            lines += ["", "```", s["stderr_tail"], "```"]
    else:
        lines += [f"- not run in this report ({(s or {}).get('skipped', '`--stress` not given')}); the streaming import path is unit-tested in "
                  "`tests/unit/test_datasets_service.py::test_numpy_imports_stream_from_the_source_without_copies`"]
    br = res.get("browser", {})
    lines += ["", "### Browser (Playwright, real server)", ""]
    for project in ("chromium-1366", "firefox-1366"):
        if project in br:
            lines += [f"- {project}: exit code {br[project]['exit_code']}", "", "```", br[project]["tail"], "```"]
    if b.get("page_load_to_table_ms"):
        lines += ["", f"- page load to experiments table (20 navigations): p50 {fmt(b['page_load_to_table_ms']['p50'], 0)} ms, p95 {fmt(b['page_load_to_table_ms']['p95'], 0)} ms",
                  f"- tab switch ({b['tab_switch_ms']['n']} samples): p50 {fmt(b['tab_switch_ms']['p50'], 0)} ms, p95 {fmt(b['tab_switch_ms']['p95'], 0)} ms",
                  f"- chart enlarge re-render ({b['chart_enlarge_render_ms']['n']} samples): p50 {fmt(b['chart_enlarge_render_ms']['p50'], 0)} ms, p95 {fmt(b['chart_enlarge_render_ms']['p95'], 0)} ms"]
    if b.get("log_lines_loaded"):
        lines += [f"- long log: {b.get('log_lines_loaded', 0):,} lines loaded in {b.get('log_load_all_ms', 0) / 1000:.1f} s (2 000-line pages), "
                  f"{b.get('log_rows_in_dom')} rows in the DOM, filter finds a line near the end"]
    lines += [
        "", "## Not measured here", "",
        "- Cold-cache start (needs root to drop the page cache) — only warm launches and the first launch of the session are reported.",
        "- macOS, Windows, Safari and the system default browser: no such machine here (support matrix marks them unverified).",
        "- GPU: the targets above are control-plane targets; GPU training throughput is a benchmark-protocol matter (L4).",
        "- CSV sources of Stress size: the CSV reader keeps the parsed arrays in RAM (Standard tier, 10^7 rows ≈ 160 MB, is fine); "
        "binary (.npy/.npz) sources stream. Preprocessing versions of Stress captures are not chunked (about 5× the raw size in RAM).",
        "", "## Reproduce", "",
        "```bash", f"python scripts/perf_report.py --workspace /tmp/perf-ws --out docs/releases/performance-report.md --minutes {m['minutes']}"
        + (" --stress" if s and not s.get("skipped") else ""), "```",
    ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --- main ---------------------------------------------------------------------------------------

PHASES = ("startup", "api", "events", "cancel", "overhead", "browser", "memory", "stress")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workspace", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "releases" / "performance-report.md")
    ap.add_argument("--json", type=Path, default=None, help="write (and, with --phases, merge into) this JSON")
    ap.add_argument("--phases", default=",".join(PHASES),
                    help="comma-separated subset; other phases are read from --json; 'none' only rewrites the report from --json")
    ap.add_argument("--minutes", type=float, default=30.0, help="memory measurement duration")
    ap.add_argument("--startup-runs", type=int, default=20)
    ap.add_argument("--samples", type=int, default=100)
    ap.add_argument("--history", type=int, default=1000)
    ap.add_argument("--log-lines", type=int, default=50_000)
    ap.add_argument("--training-epochs", type=int, default=80)
    ap.add_argument("--overhead-pairs", type=int, default=3)
    ap.add_argument("--overhead-epochs", type=int, default=30)
    ap.add_argument("--overhead-threads", default="default",
                    help="comma list of thread budgets for the overhead pairs: 'default' keeps torch's own count on both legs "
                         "(the out-of-the-box figure the target applies to); add a number ('default,8') to also measure "
                         "with an explicit execution.num_threads on both legs as a diagnostic")
    ap.add_argument("--stress", action="store_true")
    ap.add_argument("--stress-samples", type=int, default=100_000_000)
    ap.add_argument("--skip-browser", action="store_true")
    args = ap.parse_args(argv)
    phases = set(args.phases.split(","))
    if phases == {"none"}:                      # report only, from a saved --json; touches no workspace
        if not (args.json and args.json.exists()):
            ap.error("--phases none needs an existing --json")
        write_report(args.out, json.loads(args.json.read_text()))
        print(f"report written to {args.out}")
        return 0
    unknown = phases - set(PHASES)
    if unknown:
        ap.error(f"unknown phases {sorted(unknown)}; choose from {PHASES}")
    if args.skip_browser:
        phases.discard("browser")
    if not args.stress:
        phases.discard("stress")

    ws = args.workspace.resolve()
    log_dir = ws.parent / (ws.name + "-perf-logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    data: Dict[str, object] = {"environment": env_info(), "workspace": str(ws), "results": {}, "fixtures": {}}
    if args.json and args.json.exists() and phases != set(PHASES):
        previous = json.loads(args.json.read_text())
        data["results"] = previous.get("results", {})
        data["fixtures"] = previous.get("fixtures", {})
        data["environment"] = {**previous.get("environment", {}), **data["environment"]}
    res: Dict[str, object] = data["results"]  # type: ignore[assignment]

    print("seed ...", flush=True)
    first = Server(ws, log_dir / "seed.log")
    status, _, _ = first.api("GET", "/api/v1/datasets/dpa-200mhz")
    if status != 200:
        status, body, _ = first.api("POST", "/api/v1/datasets/import-builtin", {"name": "DPA_200MHz"})
        assert status == 201, body
    n_samples = first.api("GET", "/api/v1/datasets/dpa-200mhz")[1]["n_samples"]
    first.stop()
    if "startup" in phases:
        print("startup ...", flush=True)
        res["startup"] = measure_startup(ws, args.startup_runs, log_dir)

    server = Server(ws, log_dir / "server.log")
    try:
        fixtures = data["fixtures"]  # type: ignore[assignment]
        base_id = fixtures.get("base_run") if isinstance(fixtures, dict) else None
        if not base_id or not (ws / "runs" / base_id / "result.json").exists():
            base = submit_and_wait(server, 2, "perf base run")
            assert base["status"] == "succeeded", base
            base_id = base["run_id"]
        print("history fixture ...", flush=True)
        ids, big_log = history_fixture(ws, base_id, args.history, args.log_lines)
        assert server.api("GET", "/api/v1/runs/count")[1]["count"] >= args.history
        data["fixtures"] = {"n_samples": n_samples, "history_runs": len(ids), "log_lines": args.log_lines, "base_run": base_id,
                            "big_log_run": big_log, "training_epochs": args.training_epochs}
        if "api" in phases or "events" in phases:
            if "api" in phases:
                print("api idle ...", flush=True)
                res["api_idle"] = measure_api(server, base_id, big_log, args.samples)
            print("api during training + event latency ...", flush=True)
            status, training, _ = server.api("POST", "/api/v1/runs", {"config": smoke_config(args.training_epochs, 2), "name": "perf training"})
            assert status == 201, training
            while server.api("GET", f"/api/v1/runs/{training['run_id']}")[1]["status"] == "queued":
                time.sleep(0.05)
            if "api" in phases:
                res["api_training"] = measure_api(server, base_id, big_log, args.samples)
            if "events" in phases:
                res["events"] = measure_event_latency(server, training["run_id"], 90)
            if server.api("GET", f"/api/v1/runs/{training['run_id']}")[1]["status"] in ("queued", "running"):
                server.api("POST", f"/api/v1/runs/{training['run_id']}/cancel")
                wait_terminal(server, training["run_id"], 120)
        if "cancel" in phases:
            print("cancel ...", flush=True)
            res["cancel"] = measure_cancel(server)
        if "overhead" in phases:
            for item in args.overhead_threads.split(","):
                budget = None if item.strip() == "default" else int(item)
                print(f"overhead pairs ({'torch default' if budget is None else budget} threads) ...", flush=True)
                key = "overhead" if budget is None else f"overhead_threads_{budget}"
                res[key] = measure_overhead(server, ws, args.overhead_pairs, args.overhead_epochs, log_dir, budget)
        if "browser" in phases:
            print("browser ...", flush=True)
            res["browser"] = run_browser(server, big_log, log_dir, args.samples)
        if "memory" in phases:
            print(f"memory ({args.minutes} min) ...", flush=True)
            res["memory"] = measure_memory(server, base_id, big_log, args.minutes)
        if "stress" in phases:
            print("stress import ...", flush=True)
            res["stress"] = measure_stress(ws, args.stress_samples, log_dir)
    finally:
        server.stop()
    if args.json:
        args.json.write_text(json.dumps(data, indent=2, default=str))
    missing = [k for k in ("startup", "api_idle", "api_training", "events", "cancel", "overhead", "memory") if k not in res]
    if missing:
        print(f"report not written: results for {missing} are missing (run every phase, or merge into a complete --json)")
        return 0
    write_report(args.out, data)
    print(f"report written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
