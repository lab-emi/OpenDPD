"""S09: one configuration through the HTTP API (worker subprocess) and the CLI (in-process)
resolves to the same config hash and, on the CPU fixture with hard reproducibility,
the same numbers within the frozen tolerance."""

import json
import time

import pytest
from fastapi.testclient import TestClient

from opendpd.cli import studio_main
from opendpd.schemas import TERMINAL_STATUSES
from opendpd.server.app import create_app
from opendpd.server.security import CSRF_HEADER
from opendpd.services.experiments import load_resolved, load_result
from opendpd.services.recipes import instantiate
from opendpd.services.workspace import Workspace

pytestmark = pytest.mark.integration
TOLERANCE_DB = 1e-3     # docs/protocols/acceptance-thresholds.md: GUI vs CLI, CPU float32, hard reproducibility


def hard_config():
    cfg = instantiate("pa-gru-smoke-v1", "dpa-200mhz")
    return json.loads(cfg.model_copy(update={
        "name": "entry-consistency",
        "training": cfg.training.model_copy(update={"epochs": 2, "reproducibility": "hard"}),
    }).model_dump_json())


def test_gui_and_cli_entry_points_agree_on_the_deterministic_fixture(tmp_path, capsys):
    root = tmp_path / "ws"
    payload = hard_config()
    app = create_app(root, bootstrap_token="t", supervisor_kwargs={"poll_interval": 0.1}, shutdown_timeout=5)
    with TestClient(app, base_url="http://127.0.0.1:8765") as client:
        r = client.post("/api/v1/session/bootstrap", json={"token": "t"})
        client.headers[CSRF_HEADER] = r.json()["csrf_token"]
        assert client.post("/api/v1/datasets/import-builtin", json={"name": "DPA_200MHz"}).status_code == 201
        api_run = client.post("/api/v1/runs", json={"config": payload, "idempotency_key": "api"}).json()
        deadline = time.monotonic() + 240
        while time.monotonic() < deadline:
            view = client.get(f"/api/v1/runs/{api_run['run_id']}").json()
            if view["status"] in {s.value for s in TERMINAL_STATUSES}:
                break
            time.sleep(0.3)
        assert view["status"] == "succeeded", view

    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(payload))
    assert studio_main(["run", "--config", str(cfg_path), "--workspace", str(root), "--idempotency-key", "cli", "--json"]) == 0
    out = capsys.readouterr().out
    cli_run = json.loads(out[out.index("{"):])["run"]
    assert cli_run["status"] == "succeeded"

    ws = Workspace.open(root)
    api_resolved, cli_resolved = load_resolved(ws, api_run["run_id"]), load_resolved(ws, cli_run["run_id"])
    assert api_resolved.resolution.config_sha256 == cli_resolved.resolution.config_sha256
    assert api_resolved.model_dump(exclude={"resolution"}) == cli_resolved.model_dump(exclude={"resolution"})
    a, b = load_result(ws, api_run["run_id"]), load_result(ws, cli_run["run_id"])
    assert a.selected_epoch == b.selected_epoch
    for m in a.metrics:
        assert m.status.value == "ok" and b.metric(m.name).value == pytest.approx(m.value, abs=TOLERANCE_DB), m.name
