"""S04 API tests through the real ASGI app (real supervisor, real workers)."""

import json
import time

import pytest
from fastapi.testclient import TestClient

from opendpd.schemas import TERMINAL_STATUSES
from opendpd.server.app import create_app
from opendpd.server.security import CSRF_HEADER
from opendpd.services.recipes import instantiate

pytestmark = pytest.mark.integration
TOKEN = "test-bootstrap-token"


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    app = create_app(tmp_path_factory.mktemp("ws"), bootstrap_token=TOKEN,
                     supervisor_kwargs={"poll_interval": 0.1, "cancel_grace": 20}, shutdown_timeout=3)
    with TestClient(app, base_url="http://127.0.0.1:8765") as c:
        yield c


@pytest.fixture(scope="module")
def session(client):
    r = client.post("/api/v1/session/bootstrap", json={"token": TOKEN})
    assert r.status_code == 200, r.text
    csrf = r.json()["csrf_token"]
    client.headers[CSRF_HEADER] = csrf
    r = client.post("/api/v1/datasets/import-builtin", json={"name": "DPA_200MHz"})
    assert r.status_code == 201, r.text
    return csrf


def smoke_config(epochs=3):
    cfg = instantiate("pa-gru-smoke-v1", "dpa-200mhz")
    cfg = cfg.model_copy(update={"training": cfg.training.model_copy(update={"epochs": epochs})})
    return json.loads(cfg.model_dump_json())


def wait_terminal(client, run_id, timeout=120):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        r = client.get(f"/api/v1/runs/{run_id}")
        assert r.status_code == 200
        if r.json()["status"] in {s.value for s in TERMINAL_STATUSES}:
            return r.json()
        time.sleep(0.2)
    raise AssertionError("run did not finish")


# --- liveness, sessions, boundary ------------------------------------------------

def test_health_and_ready(client):
    assert client.get("/healthz").json() == {"status": "ok"}
    r = client.get("/readyz")
    assert r.status_code in (200, 503)
    body = r.json()
    assert "frontend" in body and "problems" in body
    if not body["frontend"]["present"]:
        assert any("frontend" in p for p in body["problems"])


def test_unauthenticated_requests_are_rejected_with_error_shape(client):
    anonymous = TestClient(client.app, base_url="http://127.0.0.1:8765")
    r = anonymous.get("/api/v1/runs")
    assert r.status_code == 401
    err = r.json()["error"]
    assert err["code"] == "unauthorized" and "hint" in err and err["details"] == []
    assert anonymous.post("/api/v1/session/bootstrap", json={"token": "wrong"}).status_code == 401
    assert anonymous.get("/api/v1/session").json()["authenticated"] is False


def test_bootstrap_redirect_strips_token(client):
    fresh = TestClient(client.app, base_url="http://127.0.0.1:8765")
    r = fresh.get("/bootstrap", params={"token": TOKEN}, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/"
    assert "opendpd_session" in r.cookies
    assert "token" not in r.headers["location"]
    assert fresh.get("/api/v1/session").json()["authenticated"] is True
    assert fresh.get("/bootstrap", params={"token": "nope"}).status_code == 401


def test_cross_origin_and_csrf(client, session):
    # write without CSRF header
    bare = TestClient(client.app, base_url="http://127.0.0.1:8765")
    bare.cookies = client.cookies
    r = bare.post("/api/v1/runs", json={"config": smoke_config()})
    assert r.status_code == 403 and r.json()["error"]["code"] == "csrf_required"
    # foreign origin, even with a valid CSRF header
    r = client.post("/api/v1/runs", json={"config": smoke_config()}, headers={"Origin": "http://evil.example"})
    assert r.status_code == 403 and r.json()["error"]["code"] == "cross_origin_write"
    # preflight is refused: no CORS support at all
    r = client.options("/api/v1/runs", headers={"Origin": "http://evil.example",
                                                "Access-Control-Request-Method": "POST"})
    assert r.status_code == 403 and "access-control-allow-origin" not in {k.lower() for k in r.headers}
    # same origin + CSRF header works (validated later in the run tests)
    r = client.post("/api/v1/experiments/validate", json={"config": smoke_config()},
                    headers={"Origin": "http://127.0.0.1:8765"})
    assert r.status_code == 200


def test_host_header_enforced(client, session):
    r = client.get("/api/v1/runs", headers={"Host": "evil.example"})
    assert r.status_code == 400 and r.json()["error"]["code"] == "host_not_allowed"
    assert client.get("/api/v1/runs", headers={"Host": "localhost:8765"}).status_code == 200


def test_payload_too_large(client, session):
    big = {"config": smoke_config(), "name": "x" * (3 * 1024 * 1024)}
    r = client.post("/api/v1/runs", json=big)
    assert r.status_code == 413


def test_unknown_api_route_is_json_not_index(client, session):
    r = client.get("/api/v1/does-not-exist")
    assert r.status_code == 404 and r.json()["error"]["code"] == "not_found"
    r = client.get("/some/spa/route")
    assert r.status_code in (200, 503) and r.headers["content-type"].startswith("text/html")


# --- capabilities and catalogues ----------------------------------------------------

def test_capabilities_separate_detected_from_tested(client, session):
    body = client.get("/api/v1/system/capabilities").json()
    devices = {d["device"]: d for d in body["devices"]}
    assert devices["cpu"]["detected"] is True and "gru" in devices["cpu"]["tested_models"]
    assert "tres_deltagru" in devices["cuda"]["tested_models"]
    assert "does not imply" in body["note"]
    assert any(m["key"] == "gru" for m in client.get("/api/v1/models").json())
    assert any(r["recipe_id"] == "pa-gru-smoke-v1" for r in client.get("/api/v1/recipes").json())
    assert client.get("/api/v1/datasets/dpa-200mhz").json()["n_samples"] == 38400


# --- validation, runs, events ----------------------------------------------------------

def test_validate_reports_structured_errors_without_starting(client, session):
    bad = smoke_config()
    bad["model"]["key"] = "transformer"
    r = client.post("/api/v1/experiments/validate", json={"config": bad})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False and body["errors"][0]["field"] == "model.key" and "gru" in body["errors"][0]["hint"]
    assert client.get("/api/v1/runs").json() == []
    bad = smoke_config()
    bad["dataset"]["id"] = "missing-dataset"
    body = client.post("/api/v1/experiments/validate", json={"config": bad}).json()
    assert not body["ok"] and body["errors"][0]["field"] == "dataset.id"
    # submitting an invalid config is a 422 with the same detail shape
    r = client.post("/api/v1/runs", json={"config": {"task": "train_pa"}})
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_request"
    bad = smoke_config()
    bad["model"]["key"] = "transformer"
    r = client.post("/api/v1/runs", json={"config": bad})
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_config"
    assert r.json()["error"]["details"][0]["field"] == "model.key"
    assert client.get("/api/v1/runs").json() == []


def test_run_lifecycle_events_logs_artifacts_result(client, session):
    r = client.post("/api/v1/runs", json={"config": smoke_config(), "idempotency_key": "api-1", "name": "api smoke"})
    assert r.status_code == 201, r.text
    run_id = r.json()["run_id"]
    assert r.json()["status"] == "queued"
    again = client.post("/api/v1/runs", json={"config": smoke_config(), "idempotency_key": "api-1"})
    assert again.status_code == 200 and again.json()["run_id"] == run_id

    final = wait_terminal(client, run_id)
    assert final["status"] == "succeeded", final
    assert final["heartbeat_stale"] is False and final["result_id"]

    page = client.get(f"/api/v1/runs/{run_id}/events/list", params={"after": 0}).json()
    seqs = [e["seq"] for e in page["events"]]
    assert seqs == list(range(1, len(seqs) + 1)) and page["terminal"] is True
    types = {e["type"] for e in page["events"]}
    assert {"status", "progress", "metric"} <= types
    resumed = client.get(f"/api/v1/runs/{run_id}/events/list", params={"after": seqs[-2]}).json()
    assert [e["seq"] for e in resumed["events"]] == [seqs[-1]]
    r = client.get(f"/api/v1/runs/{run_id}/events/list", params={"after": seqs[-1] + 100})
    assert r.status_code == 409 and r.json()["error"]["code"] == "cursor_out_of_range"

    # SSE replay of a finished run ends with an `end` event
    with client.stream("GET", f"/api/v1/runs/{run_id}/events", params={"after": seqs[-3]}) as stream:
        assert stream.headers["content-type"].startswith("text/event-stream")
        text = "".join(stream.iter_text())
    assert f"id: {seqs[-1]}" in text and "event: end" in text and '"status": "succeeded"' in text

    logs = client.get(f"/api/v1/runs/{run_id}/logs", params={"limit": 5}).json()
    assert len(logs["lines"]) == 5 and logs["next_offset"] > 0
    rest = client.get(f"/api/v1/runs/{run_id}/logs", params={"offset": logs["next_offset"], "limit": 2000}).json()
    assert rest["eof"] is True and any("Training Completed" in line for line in rest["lines"])

    manifest = client.get(f"/api/v1/runs/{run_id}/artifacts").json()
    assert manifest["complete"] is True
    ckpt = next(a for a in manifest["artifacts"] if a["kind"] == "checkpoint")
    download = client.get(f"/api/v1/artifacts/{run_id}/{ckpt['artifact_id']}")
    assert download.status_code == 200 and len(download.content) == ckpt["file"]["size_bytes"]

    result = client.get(f"/api/v1/results/{run_id}").json()
    assert result["evidence_type"] == "pa_modeling" and result["source"] == "opendpd-studio"
    assert all(m["value"] is not None for m in result["metrics"])
    cfg = client.get(f"/api/v1/runs/{run_id}/config").json()
    assert cfg["resolution"]["config_sha256"] == final["config_sha256"]


def test_artifact_download_by_id_only(client, session):
    run_id = client.get("/api/v1/runs").json()[-1]["run_id"]
    assert client.get(f"/api/v1/artifacts/{run_id}/..%2F..%2Fworkspace.json").status_code == 404
    assert client.get(f"/api/v1/artifacts/{run_id}/nope").status_code == 404
    assert client.get("/api/v1/artifacts/run-none/x").status_code == 404
    assert client.get("/api/v1/results/run-none").status_code == 404


def test_cancel_through_api(client, session):
    r = client.post("/api/v1/runs", json={"config": smoke_config(epochs=40)})
    run_id = r.json()["run_id"]
    deadline = time.monotonic() + 60
    while client.get(f"/api/v1/runs/{run_id}").json().get("progress_epoch") is None and time.monotonic() < deadline:
        time.sleep(0.2)
    t0 = time.perf_counter()
    r = client.post(f"/api/v1/runs/{run_id}/cancel")
    assert r.status_code == 200 and r.json()["status"] == "cancel_requested"
    assert time.perf_counter() - t0 < 1.0
    assert client.post(f"/api/v1/runs/{run_id}/cancel").json()["status"] in ("cancel_requested", "cancelled")
    final = wait_terminal(client, run_id)
    assert final["status"] == "cancelled" and final["progress_epoch"] < 40
    assert client.get("/api/v1/results/" + run_id).status_code == 404
    retry = client.post(f"/api/v1/runs/{run_id}/retry")
    assert retry.status_code == 201 and retry.json()["parent_run_id"] == run_id
    assert client.post(f"/api/v1/runs/{retry.json()['run_id']}/cancel").json()["status"] in ("cancelled", "cancel_requested")
    wait_terminal(client, retry.json()["run_id"])


def test_openapi_is_exportable_and_committed(client):
    schema = client.app.openapi()
    assert "/api/v1/runs/{run_id}/events" in schema["paths"]
    from pathlib import Path
    target = Path(__file__).resolve().parents[2] / "docs" / "contracts" / "openapi.json"
    committed = json.loads(target.read_text())
    assert committed == json.loads(json.dumps(schema)), "docs/contracts/openapi.json is stale; run scripts/export_openapi.py"
