"""A workspace has one Studio writer throughout startup, service and recovery."""

import json
import subprocess
import sys
import time
import urllib.request

import pytest

from opendpd.studio.launcher import GUARD_FILE, LOCK_FILE

pytestmark = pytest.mark.integration


def _wait_for(check, timeout=20):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if check():
            return
        time.sleep(0.05)
    pytest.fail("Studio did not reach the expected lifecycle state")


def _healthy(port):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=0.5) as response:
            return response.status == 200
    except OSError:
        return False


def test_startup_is_exclusive_and_a_crash_releases_ownership(tmp_path):
    workspace = tmp_path / "Studio workspace 工作区"
    entered, release = tmp_path / "entered", tmp_path / "release"
    script = """
import sys, time
from pathlib import Path
from opendpd.studio.launcher import launch
def serve(app, host, port):
    Path(sys.argv[2]).touch()
    while not Path(sys.argv[3]).exists():
        time.sleep(0.05)
    import uvicorn
    uvicorn.run(app, host=host, port=port, log_level="warning")
sys.exit(launch(Path(sys.argv[1]), open_in_browser=False, serve=serve))
"""
    command = [sys.executable, "-m", "opendpd.commands", "gui", "--workspace", str(workspace), "--no-browser"]
    processes = []
    with (tmp_path / "server.log").open("w") as log:
        try:
            first = subprocess.Popen([sys.executable, "-c", script, str(workspace), str(entered), str(release)],
                                     stdout=log, stderr=log)
            processes.append(first)
            _wait_for(entered.exists)
            metadata = (workspace / LOCK_FILE).read_text()
            port = json.loads(metadata)["port"]

            # The first owner has not started listening yet. A second launcher
            # must refuse without replacing metadata or opening another writer.
            second = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            processes.append(second)
            _, error = second.communicate(timeout=10)
            assert second.returncode == 2
            assert "already starting, running, or shutting down" in error
            assert (workspace / LOCK_FILE).read_text() == metadata

            release.touch()
            _wait_for(lambda: _healthy(port))
            reused = subprocess.run(command, capture_output=True, text=True, timeout=10)
            assert reused.returncode == 0, reused.stderr
            assert "already running" in reused.stdout
            assert (workspace / LOCK_FILE).read_text() == metadata

            # Simulate a crash that cannot run Python's finally blocks.
            first.kill()
            first.wait(timeout=10)
            assert (workspace / GUARD_FILE).exists()
            restarted = subprocess.Popen(command, stdout=log, stderr=log)
            processes.append(restarted)
            _wait_for(lambda: (workspace / LOCK_FILE).exists()
                      and json.loads((workspace / LOCK_FILE).read_text())["pid"] == restarted.pid)
            new_port = json.loads((workspace / LOCK_FILE).read_text())["port"]
            _wait_for(lambda: _healthy(new_port))
            restarted.terminate()
            restarted.wait(timeout=20)
            assert restarted.returncode == 0
            assert not (workspace / LOCK_FILE).exists()
        finally:
            for process in processes:
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=10)
