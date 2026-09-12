"""Private GPU protocol and public isolation; GPU hardware is not needed in CI."""
import base64
import io
import json
import time
import zipfile

import pytest
from fastapi.testclient import TestClient

from opendpd.services.recipes import instantiate
from opendpd.web.app import create_web_app
from opendpd.web.gpu_archive import pack, unpack, read_regular
from opendpd.web.policy import WebConfig

pytestmark = pytest.mark.integration
TOKEN = "private-test-gpu-token-" * 4
TUNNEL = "a" * 48 + ".internal"


def eventually(predicate):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(0.05)
    raise AssertionError("condition did not become true")


def test_private_gpu_lease_stream_replay_and_cross_tenant_cancel(tmp_path):
    config = WebConfig(tmp_path / "web", "https://opendpd.com", "api.opendpd.com", TUNNEL, gpu_token=TOKEN)
    app = create_web_app(config)
    with TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 10)) as client:
        private = {"X-OpenDPD-GPU": TOKEN}
        public = {"Host": TUNNEL, "Origin": "https://opendpd.com", "X-Forwarded-Proto": "https", "CF-Connecting-IP": "203.0.113.10"}
        assert client.post("/_gpu/poll", json={"name": "CUDA test device"}).status_code == 403
        assert client.post("/_gpu/poll", json={"name": "CUDA test device"}, headers={**private, **public}).status_code == 403
        assert client.post("/_gpu/poll", json={"name": "CUDA test device"}, headers=private).json() == {"job": None}
        auth = []
        for _ in range(2):
            token = client.post("/api/v1/web/sessions", json={}, headers=public).json()["access_token"]
            auth.append({**public, "Authorization": "Bearer " + token})
        capability = client.get("/api/v1/system/capabilities", headers=auth[0]).json()
        assert next(d for d in capability["devices"] if d["device"] == "cuda")["detected"]
        assert client.post("/api/v1/datasets/import-builtin", json={"name": "MyCustomPA"}, headers=auth[0]).status_code == 201
        cfg = instantiate("pa-gru-smoke-v1", "mycustompa").model_dump(mode="json")
        cfg["execution"]["device"] = "cuda"
        response = client.post("/api/v1/runs", json={"config": cfg}, headers=auth[0])
        assert response.status_code == 201, response.text
        rid = response.json()["run_id"]
        job = eventually(lambda: client.post("/_gpu/poll", json={"name": "CUDA test device"}, headers=private).json()["job"])
        lease = {**private, "X-OpenDPD-Lease": job["lease"]}
        assert client.post("/_gpu/poll", json={"name": "CUDA test device"}, headers=private).json()["job"] is None
        assert client.get(f"/_gpu/jobs/{job['id']}/input", headers=private).status_code == 409
        archive = client.get(f"/_gpu/jobs/{job['id']}/input", headers=lease)
        assert archive.status_code == 200
        names = zipfile.ZipFile(io.BytesIO(archive.content)).namelist()
        assert f"runs/{rid}/config.resolved.json" in names
        assert all(not any(part in name for part in ["lease.json", ".sqlite", "service-tmp"]) for name in names)
        event = json.dumps({"ts": "2026-09-13T00:00:00+00:00", "type": "metric", "payload": {"epoch": 1, "values": {"NMSE": -30}}}) + "\n"
        payload = {"log_offset": 0, "log": base64.b64encode(b"CUDA training\n").decode(), "events_offset": 0, "events": base64.b64encode(event.encode()).decode(),
                   "live": base64.b64encode(b'{"preview":{"revision":1,"metrics":{"NMSE":-30}}}').decode()}
        for _ in range(2):
            assert client.post(f"/_gpu/jobs/{job['id']}/update", headers=lease, json=payload).json()["continue"]
        events = eventually(lambda: [e for e in client.get(f"/api/v1/runs/{rid}/events/list", headers=auth[0]).json()["events"] if e["type"] == "metric"])
        assert len(events) == 1
        assert client.get(f"/api/v1/runs/{rid}/live", headers=auth[0]).json()["preview"]["metrics"]["NMSE"] == -30
        assert client.post(f"/api/v1/runs/{rid}/cancel", json={}, headers=auth[1]).status_code == 404
        assert client.post(f"/api/v1/runs/{rid}/cancel", json={}, headers=auth[0]).status_code == 200
        assert not client.post(f"/_gpu/jobs/{job['id']}/update", headers=lease, json=payload).json()["continue"]
        empty = tmp_path / "empty"
        empty.mkdir()
        assert client.post(f"/_gpu/jobs/{job['id']}/result", headers={**lease, "X-OpenDPD-Exit": "3"}, content=pack(empty, [])).status_code == 200
        eventually(lambda: client.get(f"/api/v1/runs/{rid}", headers=auth[0]).json()["status"] == "cancelled")

        # A lost host lease must terminate the VM proxy, freeing the shared slot.
        response = client.post("/api/v1/runs", json={"config": cfg}, headers=auth[0])
        assert response.status_code == 201
        lost_id = response.json()["run_id"]
        lost = eventually(lambda: client.post("/_gpu/poll", json={"name": "CUDA test device"}, headers=private).json()["job"])
        broker = app.state.manager.gpu
        with broker.lock:
            broker.jobs[lost["id"]].last_seen -= 31
            broker.sweep()
        eventually(lambda: client.get(f"/api/v1/runs/{lost_id}", headers=auth[0]).json()["status"] == "failed")
        broker.last_seen -= 31
        assert not broker.devices()["cuda"]["detected"]


@pytest.mark.parametrize("name", ["../escape", "/absolute", "runs/../../escape", "a\\escape", "./alias", ".gpu-result"])
def test_transfer_rejects_unsafe_paths(tmp_path, name):
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as archive:
        archive.writestr(name, "unsafe")
    with pytest.raises(ValueError):
        unpack(data.getvalue(), tmp_path)
    assert not list(tmp_path.iterdir())


def test_transfer_rejects_symlinks_in_archive_and_destination(tmp_path):
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as archive:
        entry = zipfile.ZipInfo("link")
        entry.external_attr = 0o120777 << 16
        archive.writestr(entry, "/etc/passwd")
    with pytest.raises(ValueError):
        unpack(data.getvalue(), tmp_path)

    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as archive:
        archive.writestr("link/file", "unsafe")
    (tmp_path / "link").symlink_to(tmp_path.parent, target_is_directory=True)
    with pytest.raises(ValueError):
        unpack(data.getvalue(), tmp_path)


def test_live_output_reader_cannot_follow_container_links_or_fifos(tmp_path):
    import os
    outside = tmp_path / "secret"
    outside.write_text("host secret")
    root = tmp_path / "job"
    root.mkdir()
    (root / "live.json").symlink_to(outside)
    with pytest.raises(OSError):
        read_regular(root, "live.json")
    (root / "logs").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(OSError):
        read_regular(root, "logs/secret")
    os.mkfifo(root / "events.jsonl")
    with pytest.raises(ValueError):
        read_regular(root, "events.jsonl")
