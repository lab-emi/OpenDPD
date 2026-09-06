"""S04 API tests through the real ASGI app (real supervisor, real workers)."""

import io
import json
import time
import zipfile

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
    # S08: the definition behind every score is served, and every registered profile has a stored result
    profiles = {p["profile_id"]: p for p in client.get("/api/v1/metrics/profiles").json()}
    assert profiles["legacy-opendpd-v1"]["frozen"] is True and "ACPR_L" in [m["name"] for m in profiles["general-spectral-v1"]["metrics"]]
    assert client.get("/api/v1/metrics/profiles/nope").status_code == 404
    assert client.get(f"/api/v1/results/{run_id}/profiles").json() == ["legacy-opendpd-v1", "general-spectral-v1"]
    general = client.get(f"/api/v1/results/{run_id}", params={"profile": "general-spectral-v1"}).json()
    assert general["metric_profile_id"] == "general-spectral-v1" and general["metrics"][0]["name"] == "NMSE"
    missing = client.get(f"/api/v1/results/{run_id}", params={"profile": "nope-v1"})
    assert missing.status_code == 404 and "stored profiles" in missing.json()["error"]["hint"]
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


def test_minimal_frontend_payload_validates_and_submits(client, session):
    """The form sends only what the user chose; every default comes from the server."""
    minimal = {"task": "train_pa", "recipe_id": "pa-gru-smoke-v1", "dataset": {"id": "dpa-200mhz"},
               "model": {"key": "gru", "parameters": {"hidden_size": 23, "num_layers": 1}},
               "training": {"epochs": 1, "frame_length": 50, "frame_stride": 16, "batch_size_eval": 256},
               "execution": {"device": "cpu"}}
    report = client.post("/api/v1/experiments/validate", json={"config": minimal}).json()
    assert report["ok"], report["errors"]
    resolved = report["resolved"]
    assert resolved["evaluation"]["evidence_type"] == "pa_modeling"
    assert resolved["evaluation"]["checkpoint_selection_metric"] == "NMSE"
    assert resolved["training"]["learning_rate"] == 0.005 and resolved["dataset"]["split_version"]
    wrong = dict(minimal, evaluation={"evidence_type": "dpd_measured"})
    assert client.post("/api/v1/experiments/validate", json={"config": wrong}).json()["ok"] is False
    r = client.post("/api/v1/runs", json={"config": minimal, "name": "minimal payload"})
    assert r.status_code == 201, r.text
    final = wait_terminal(client, r.json()["run_id"])
    assert final["status"] == "succeeded", final


def test_runs_are_paged_and_searchable_on_the_server(client, session):
    runs = client.get("/api/v1/runs?limit=500").json()
    assert runs, "earlier tests created runs"
    total = client.get("/api/v1/runs/count").json()["count"]
    assert total == len(runs)
    first = client.get("/api/v1/runs?limit=1&offset=0").json()
    second = client.get("/api/v1/runs?limit=1&offset=1").json()
    assert first[0]["run_id"] == runs[0]["run_id"] and second[0]["run_id"] == runs[1]["run_id"]
    needle = runs[0]["run_id"][-6:]
    hits = client.get(f"/api/v1/runs?q={needle}").json()
    assert [r["run_id"] for r in hits] == [r["run_id"] for r in runs if needle in r["run_id"]]
    assert client.get(f"/api/v1/runs/count?q={needle}").json()["count"] == len(hits)
    assert client.get("/api/v1/runs?q=%25").json() == []          # LIKE wildcards are literal characters
    assert client.get("/api/v1/runs?q=" + "x" * 201).status_code == 422


def test_runs_made_by_the_cli_are_visible_to_the_service(client, session):
    """One workspace, three entry points: a run written by `opendpd run` shows up in the API and in lineage."""
    from opendpd.services.experiments import create_run, execute_run
    from opendpd.services.workspace import Workspace

    ws = Workspace.open(client.app.state.ws.root)
    record = execute_run(ws, create_run(ws, instantiate("pa-gru-smoke-v1", "dpa-200mhz", name="from the CLI")).run_id)
    assert record.status.value == "succeeded"
    r = client.get(f"/api/v1/runs/{record.run_id}")
    assert r.status_code == 200 and r.json()["name"] == "from the CLI"
    assert record.run_id in {x["run_id"] for x in client.get("/api/v1/runs").json()}
    assert client.get(f"/api/v1/results/{record.run_id}").status_code == 200
    # a run another process still owns is not adopted
    queued = create_run(ws, instantiate("pa-gru-smoke-v1", "dpa-200mhz", name="cli queued"))
    assert client.get(f"/api/v1/runs/{queued.run_id}").status_code == 404


def test_history_plots_and_comparison_routes(client, session):
    runs = [r for r in client.get("/api/v1/runs", params={"status": "succeeded"}).json() if r["task"] == "train_pa"]
    assert len(runs) >= 2
    a, b = runs[0]["run_id"], runs[1]["run_id"]
    history = client.get(f"/api/v1/runs/{a}/history").json()
    assert history and history[0]["split"] == "val" and "NMSE" in history[0]["values"]
    r = client.get(f"/api/v1/artifacts/{a}/plot-spectrum")
    assert r.status_code == 200 and r.json()["version"] == "plots-v1"
    report = client.get("/api/v1/results/compare", params={"runs": [a, b]}).json()
    assert report["comparable"] is True and [x["run_id"] for x in report["results"]] == [a, b]
    csv_text = client.get("/api/v1/results/compare", params={"runs": [a, b], "format": "csv"}).text
    assert csv_text.startswith(f"field,{a},{b}")
    r = client.get("/api/v1/results/compare", params={"runs": [a]})
    assert r.status_code == 422
    r = client.get("/api/v1/results/compare", params={"runs": [a, "run-does-not-exist"]})
    assert r.status_code == 404 and r.json()["error"]["code"] == "run_not_found"


def test_export_download_import_and_report_routes(client, session):
    # a run made by the worker (it has a worker log to leave out), not the one the CLI made in-process
    runs = [r for r in client.get("/api/v1/runs", params={"status": "succeeded"}).json() if r["task"] == "train_pa"]
    run_id = next(r["run_id"] for r in runs
                  if any(a["kind"] == "worker_log" for a in client.get(f"/api/v1/runs/{r['run_id']}/artifacts").json()["artifacts"]))
    r = client.post("/api/v1/exports", json={"run_id": run_id, "kind": "share"})
    assert r.status_code == 201, r.text
    info = r.json()
    assert info["manifest"]["kind"] == "share" and info["manifest"]["run_id"] == run_id and info["size_bytes"] > 0
    assert not info["manifest"]["dataset"]["included"] and info["manifest"]["redaction"]
    download = client.get(info["download_url"])
    assert download.status_code == 200 and download.headers["content-type"] == "application/zip"
    # the worker log stays behind and the packaged artifact manifest no longer lists it
    assert any(line.startswith("artifacts.json:") and "worker-log" in line for line in info["manifest"]["redaction"])
    with zipfile.ZipFile(io.BytesIO(download.content)) as zf:
        packaged = json.loads(zf.read(f"run/{run_id}/artifacts.json"))
        assert packaged["complete"] and not any(a["kind"] == "worker_log" for a in packaged["artifacts"])
        assert not any(n.startswith(f"run/{run_id}/logs/") for n in zf.namelist())
    assert client.get("/api/v1/exports/does-not-exist").status_code == 404
    # importing into the same workspace is refused with the specific code, nothing is written
    r = client.post("/api/v1/imports", files={"file": (info["filename"], download.content, "application/zip")})
    assert r.status_code == 422 and r.json()["error"]["code"] == "run_exists", r.text
    r = client.post("/api/v1/imports", files={"file": ("x.zip", b"not a zip", "application/zip")})
    assert r.status_code == 422 and r.json()["error"]["code"] == "not_a_package"
    report = client.get(f"/api/v1/results/{run_id}/report", params={"format": "md"})
    assert report.status_code == 200 and report.text.startswith("# OpenDPD Studio report")
    assert "text/markdown" in report.headers["content-type"]
    assert client.get(f"/api/v1/results/{run_id}/report").headers["content-type"].startswith("text/html")
    assert client.post("/api/v1/exports", json={"run_id": "run-does-not-exist", "kind": "full"}).status_code == 404


def test_lineage_route_reads_the_graph_from_resolved_configs(client, session):
    runs = client.get("/api/v1/runs", params={"status": "succeeded"}).json()
    pa = next(r for r in runs if r["task"] == "train_pa")
    graph = client.get(f"/api/v1/runs/{pa['run_id']}/lineage").json()
    assert graph == {"run_id": pa["run_id"], "parents": [], "children": []}
    r = client.get("/api/v1/runs/run-does-not-exist/lineage")
    assert r.status_code == 404 and r.json()["error"]["code"] == "run_not_found"


def test_undetected_device_is_refused_never_switched(client, session, monkeypatch):
    from opendpd.services import capabilities

    monkeypatch.setattr(capabilities, "detect_devices",
                        lambda: {"cuda": {"detected": False, "count": 0, "name": None}, "mps": {"detected": False}})
    cfg = smoke_config()
    cfg["execution"] = {"device": "cuda"}
    report = client.post("/api/v1/experiments/validate", json={"config": cfg}).json()
    assert not report["ok"]
    issue = next(e for e in report["errors"] if e["field"] == "execution.device")
    assert "not available" in issue["message"] and "never switches" in issue["hint"]
    r = client.post("/api/v1/runs", json={"config": cfg, "idempotency_key": "cuda-refused"})
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_config"
    assert r.json()["error"]["details"][0]["field"] == "execution.device"
