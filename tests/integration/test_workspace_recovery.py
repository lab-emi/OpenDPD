"""S13 migration and recovery: a workspace survives being moved, copied and reopened, paths with
spaces and non-ASCII characters work end to end, and a busy port is a clear refusal."""

import json
import shutil
import socket

import pytest
from fastapi.testclient import TestClient

from opendpd.schemas import RunStatus
from opendpd.server.app import create_app
from opendpd.server.security import CSRF_HEADER
from opendpd.services import packages
from opendpd.services.evaluation import evaluate_run
from opendpd.services.experiments import create_run, execute_run, load_artifacts, load_run
from opendpd.services.recipes import instantiate
from opendpd.services.workspace import Workspace, sha256_file

pytestmark = pytest.mark.integration


def _smoke_run(ws: Workspace):
    ws.register_builtin_dataset("DPA_200MHz", "dpa-200mhz")
    cfg = instantiate("pa-gru-smoke-v1", "dpa-200mhz", name="before the move")
    cfg = cfg.model_copy(update={"training": cfg.training.model_copy(update={"epochs": 2})})
    record = execute_run(ws, create_run(ws, cfg).run_id)
    assert record.status == RunStatus.succeeded, record.error
    return record


def _serve(root, token="recovery-token"):
    app = create_app(root, bootstrap_token=token, supervisor_kwargs={"poll_interval": 0.1}, shutdown_timeout=3)
    client = TestClient(app, base_url="http://127.0.0.1:8765")
    client.__enter__()
    r = client.post("/api/v1/session/bootstrap", json={"token": token})
    assert r.status_code == 200, r.text
    client.headers[CSRF_HEADER] = r.json()["csrf_token"]
    return client


def test_a_moved_workspace_keeps_its_runs_results_and_artifacts(tmp_path):
    old_root = tmp_path / "lab workspace ω" / "工作区"
    ws = Workspace.create(old_root)
    record = _smoke_run(ws)
    stored = json.loads((ws.run_dir(record.run_id) / "result.json").read_text())
    ckpt = load_artifacts(ws, record.run_id).by_kind("checkpoint")[0]
    ckpt_sha = sha256_file(ws.run_dir(record.run_id) / ckpt.file.path)

    new_root = tmp_path / "moved" / "my lab (copy)"
    new_root.parent.mkdir()
    shutil.move(str(old_root), str(new_root))
    assert not old_root.exists()

    moved = Workspace.open(new_root)
    assert moved.list_run_ids() == [record.run_id]
    assert load_run(moved, record.run_id).status == RunStatus.succeeded
    assert sha256_file(moved.run_dir(record.run_id) / ckpt.file.path) == ckpt_sha
    # nothing the run needs refers to the old absolute path (provenance.json keeps the command that was
    # executed, absolute dataset path included: it is a historical record, not an input)
    for name in ("run.json", "config.resolved.json", "artifacts.json", "result.json"):
        assert str(old_root) not in (moved.run_dir(record.run_id) / name).read_text(), name
    # re-evaluation from the moved checkpoint reproduces the stored metrics
    again = evaluate_run(moved, record.run_id, stored["metric_profile_id"])
    for item in again.metrics:
        before = next(m for m in stored["metrics"] if m["name"] == item.name)
        if item.value is not None:
            assert abs(item.value - before["value"]) <= 1e-4 + 1e-4 * abs(before["value"]), item.name
    # the service indexes and serves the moved run, downloads its artifacts, exports it
    client = _serve(new_root)
    try:
        listing = client.get("/api/v1/runs").json()
        assert [r["run_id"] for r in listing] == [record.run_id]
        assert client.get(f"/api/v1/artifacts/{record.run_id}/{ckpt.artifact_id}").status_code == 200
        r = client.post("/api/v1/exports", json={"run_id": record.run_id, "kind": "full"})
        assert r.status_code == 201, r.text
        assert client.get(r.json()["download_url"]).status_code == 200
    finally:
        client.__exit__(None, None, None)


def test_a_backup_copy_restores_into_a_fresh_location(tmp_path):
    ws = Workspace.create(tmp_path / "live")
    record = _smoke_run(ws)
    backup = tmp_path / "backup" / "live-2026-09-06"
    shutil.copytree(ws.root, backup)
    (ws.run_dir(record.run_id) / "result.json").unlink()          # the live copy gets damaged
    restored = Workspace.open(backup)
    assert (restored.run_dir(record.run_id) / "result.json").exists()
    assert load_run(restored, record.run_id).status == RunStatus.succeeded
    manifest = packages.export_run(restored, record.run_id, tmp_path / "from-backup.zip", kind="full")
    assert manifest.run_id == record.run_id
    fresh = Workspace.create(tmp_path / "fresh")
    report = packages.import_package(fresh, tmp_path / "from-backup.zip")
    assert report.run_id == record.run_id and report.dataset_status == "registered_builtin"
    assert evaluate_run(fresh, record.run_id, "legacy-opendpd-v1").metrics


def test_a_busy_port_is_refused_with_a_clear_message(tmp_path, capsys):
    from opendpd.cli import studio_main

    ws = tmp_path / "ws"
    with socket.socket() as blocker:
        blocker.bind(("127.0.0.1", 0))
        blocker.listen(1)
        port = blocker.getsockname()[1]
        code = studio_main(["gui", "--workspace", str(ws), "--port", str(port), "--no-browser"])
    assert code != 0
    err = capsys.readouterr().err
    assert str(port) in err and ("in use" in err or "busy" in err or "not free" in err), err
    assert not (ws / ".studio.lock").exists()
