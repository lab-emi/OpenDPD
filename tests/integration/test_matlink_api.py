"""The browser and MATLAB bridge use the same Studio authentication boundary."""
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from opendpd.server.app import create_app
from opendpd.server.security import CSRF_HEADER
from opendpd.schemas import RunRecord, RunStatus, TaskType

pytestmark = pytest.mark.integration


class QuietSupervisor:
    def __init__(self, ws, store):
        self.alive = True

    def start(self):
        pass

    def stop(self, timeout):
        self.alive = False


@pytest.fixture
def client(tmp_path):
    app = create_app(tmp_path, bootstrap_token="matlink-test-token", monitor_resources=False,
                     start_sweeps=False, allow_arena_submissions=False, supervisor_factory=QuietSupervisor)
    with TestClient(app, base_url="http://127.0.0.1:8765") as c:
        response = c.post("/api/v1/session/bootstrap", json={"token": "matlink-test-token"})
        c.headers[CSRF_HEADER] = response.json()["csrf_token"]
        yield c


def connect(client):
    response = client.post("/api/v1/matlink/connect", json={"label": "MATLAB test", "release": "R2026a", "variables": []})
    assert response.status_code == 201, response.text
    return response.json()


def test_session_csrf_origin_and_bridge_secret_required(client):
    anonymous = TestClient(client.app, base_url="http://127.0.0.1:8765")
    assert anonymous.get("/api/v1/matlink").status_code == 401
    anonymous.cookies = client.cookies
    assert anonymous.post("/api/v1/matlink/connect", json={}).status_code == 403
    assert client.post("/api/v1/matlink/connect", json={}, headers={"Origin": "https://foreign.example"}).status_code == 403
    bridge = connect(client)
    heartbeat = f"/api/v1/matlink/{bridge['client_id']}/heartbeat"
    assert client.post(heartbeat, json={"variables": []}).status_code == 403
    assert client.post(heartbeat, json={}, headers={"X-OpenDPD-Matlink": "wrong"}).status_code == 403
    headers = {"X-OpenDPD-Matlink": bridge["bridge_token"]}
    assert client.post(heartbeat, json={}, headers=headers).status_code == 200
    duplicate = [("X-OpenDPD-Matlink", bridge["bridge_token"])] * 2
    assert client.post(heartbeat, json={}, headers=duplicate).status_code == 403
    assert bridge["bridge_token"] not in client.get("/api/v1/matlink").text


def test_browser_request_and_retry_safe_bridge_round_trip(client):
    bridge = connect(client)
    response = client.post("/api/v1/matlink/requests", json={
        "client_id": bridge["client_id"], "action": "create_demo", "payload": {}, "idempotency_key": "demo-click"})
    assert response.status_code == 201, response.text
    transfer = response.json()
    assert transfer["status"] == "queued"
    headers = {"X-OpenDPD-Matlink": bridge["bridge_token"]}
    route = f"/api/v1/matlink/{bridge['client_id']}"
    for _ in range(2):
        response = client.post(route + "/heartbeat", json={}, headers=headers)
        assert response.json()["requests"][0]["request_id"] == transfer["request_id"]
    complete = route + f"/requests/{transfer['request_id']}/complete"
    body = {"status": "succeeded", "result": {"variables": ["opendpdX", "opendpdY"]}, "error": None}
    for _ in range(2):
        response = client.post(complete, json=body, headers=headers)
        assert response.status_code == 200 and response.json()["status"] == "succeeded"
    assert not client.post(route + "/heartbeat", json={}, headers=headers).json()["requests"]
    state = client.get("/api/v1/matlink").json()
    assert state["available"] and state["workspace"]
    assert state["sessions"][0]["pending_count"] == 0
    assert client.post(route + "/disconnect", json={}, headers=headers).json() == {"disconnected": True}


def test_payload_rejects_executable_and_unknown_fields(client):
    bridge = connect(client)
    base = {"client_id": bridge["client_id"], "idempotency_key": "bad"}
    for action, payload in [("eval", {"code": "quit"}), ("create_demo", {"code": "quit"}),
                            ("open_variable", {"variable": "x);quit"})]:
        response = client.post("/api/v1/matlink/requests", json={**base, "action": action, "payload": payload})
        assert response.status_code == 422 and response.json()["error"]["code"] == "invalid_request"
    response = client.post("/api/v1/matlink/requests", json={**base, "action": "import_result", "payload": {"run_id": "missing"}})
    assert response.status_code == 404 and response.json()["error"]["code"] == "run_not_found"


def test_matlink_bootstrap_remains_same_origin(client):
    sessions = client.app.state.sessions  # the fixture already spent its single-use token
    token = sessions.mint(sessions.launcher_secret)
    response = client.get("/bootstrap", params={"token": token, "next": "/matlink"}, follow_redirects=False)
    assert response.status_code == 303 and response.headers["location"] == "/matlink"
    assert client.get("/readyz").json()["matlink_protocol_version"] == 1


def test_success_requires_a_readable_formal_result_on_disk(client):
    bridge = connect(client)
    now = datetime.now(timezone.utc)
    record = RunRecord(run_id="run-a", task=TaskType.train_pa, status=RunStatus.succeeded,
                       created_at=now, finished_at=now)
    client.app.state.store.upsert_run(record)
    request = {"client_id": bridge["client_id"], "idempotency_key": "missing-report",
               "action": "import_result", "payload": {"run_id": "run-a"}}
    response = client.post("/api/v1/matlink/requests", json=request)
    assert response.status_code == 409 and response.json()["error"]["code"] == "result_not_available"
    directory = client.app.state.ws.run_dir("run-a")
    directory.mkdir()
    (directory / "result.json").write_text('{"run_id":"run-a","metrics":[]}')
    response = client.post("/api/v1/matlink/requests", json=request)
    assert response.status_code == 409 and response.json()["error"]["code"] == "result_not_available"
    assert not client.get("/api/v1/matlink").json()["transfers"]
