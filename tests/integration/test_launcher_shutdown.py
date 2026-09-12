"""Native launcher's server shutdown with a real worker and an open SSE client.

The window callback supplies a headless close request; HTTP, the event stream,
the supervisor and CPU training are real. Native UI evidence is recorded
separately in docs/baseline/.
"""

import json
import time
from urllib.parse import urlsplit

import httpx
import pytest

from opendpd.runtime.procs import descendants, is_same_process, kill_tree, process_identity
from opendpd.schemas import TERMINAL_STATUSES
from opendpd.server.security import CSRF_HEADER
from opendpd.services.recipes import instantiate
from opendpd.studio.launcher import LOCK_FILE, launch, port_is_free, workspace_guard
from opendpd.studio.window import Availability

pytestmark = pytest.mark.integration


def test_window_shutdown_drains_live_events_and_stops_worker_before_unlocking(tmp_path):
    workspace = tmp_path / "workspace"
    worker = None
    children = []
    run_id = None
    port = None
    stream = None
    event_lines = None
    client = httpx.Client(follow_redirects=True, timeout=10)

    def close_window(url, active_runs):
        nonlocal worker, children, run_id, port, stream, event_lines
        parts = urlsplit(url)
        origin, port = f"{parts.scheme}://{parts.netloc}", parts.port
        response = client.get(url)
        assert response.status_code == 200
        session = client.get(origin + "/api/v1/session").json()
        client.headers[CSRF_HEADER] = session["csrf_token"]
        response = client.post(origin + "/api/v1/datasets/import-builtin", json={"name": "DPA_200MHz"})
        assert response.status_code == 201, response.text

        config = instantiate("pa-gru-smoke-v1", "dpa-200mhz")
        # Leave time to close during computation without mutating recipe defaults.
        config = config.model_copy(update={"training": config.training.model_copy(update={"epochs": 3000})})
        response = client.post(origin + "/api/v1/runs", json={"config": json.loads(config.model_dump_json())})
        assert response.status_code == 201, response.text
        run_id = response.json()["run_id"]
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            record = client.get(origin + f"/api/v1/runs/{run_id}").json()
            if record.get("worker"):
                worker = record["worker"]
            if record["status"] == "running" and (record.get("progress_epoch") or 0) >= 1:
                break
            time.sleep(0.1)
        else:
            pytest.fail("worker did not begin training")
        assert worker is not None and is_same_process(worker["pid"], worker["create_time"])
        children = [identity for pid in descendants(worker["pid"]) if (identity := process_identity(pid))]
        assert active_runs() == 1

        # Deliberately retain the live stream beyond the window callback. The
        # old unbounded HTTP drain prevented lifespan shutdown from ever running.
        stream = client.stream("GET", origin + f"/api/v1/runs/{run_id}/events")
        response = stream.__enter__()
        assert response.status_code == 200
        event_lines = response.iter_lines()  # keep the reader alive until launch() has returned
        assert next(event_lines).startswith("id:")
        assert not response.is_closed

    try:
        assert launch(workspace, mode="window", availability=lambda: Availability("headless test shell"),
                      window_runner=close_window) == 0
        assert worker is not None
        assert not is_same_process(worker["pid"], worker["create_time"])
        assert all(not is_same_process(pid, created) for pid, created in children)
        record = json.loads((workspace / "runs" / run_id / "run.json").read_text())
        assert record["status"] in {status.value for status in TERMINAL_STATUSES}
        assert not (workspace / LOCK_FILE).exists()
        assert port_is_free(port)
        with workspace_guard(workspace):
            pass
    finally:
        if stream is not None:
            stream.__exit__(None, None, None)
        client.close()
        if worker and is_same_process(worker["pid"], worker["create_time"]):
            kill_tree(worker["pid"])
