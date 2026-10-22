"""The NR test-model routes through the real public ASGI boundary: sessions, body limits, rates, no storage, 1,000,000-sample captures."""
import json

import pytest
from fastapi.testclient import TestClient

from opendpd.core.waveforms import nr_tm
from opendpd.schemas.nr_test_model import NrTestModelConfig
from opendpd.web.app import create_web_app
from opendpd.web.policy import WebConfig
from opendpd.web.runtime import DAY, directory_bytes

pytestmark = pytest.mark.integration
ORIGIN = "https://opendpd.com"
TUNNEL_HOST = "a" * 48 + ".internal"
TM = "/api/v1/signal-generator/nr-test-models"
RECIPE = {"model": "NR-FR1-TM3.1a", "duplex": "fdd", "carriers": [{"scs_khz": 30, "bandwidth_mhz": 20}, {"scs_khz": 30, "bandwidth_mhz": 20}]}


@pytest.fixture
def public(tmp_path):
    now = [DAY * 20000 + 3600]
    config = WebConfig(root=tmp_path / "web", origin=ORIGIN, api_host="api.opendpd.com", tunnel_host=TUNNEL_HOST, requests_per_session_minute=1000)
    app = create_web_app(config, now=lambda: now[0])
    with TestClient(app, base_url="http://" + TUNNEL_HOST, client=("127.0.0.1", 10000)) as client:
        client.headers.update({"Origin": ORIGIN, "X-Forwarded-Proto": "https", "CF-Connecting-IP": "203.0.113.10"})
        yield client, app.state.manager, now


def new_session(client):
    response = client.post("/api/v1/web/sessions", json={})
    assert response.status_code == 201, response.text
    return {"Authorization": "Bearer " + response.json()["access_token"]}


def test_catalog_needs_a_session_and_is_the_local_catalog(public):
    client, _, _ = public
    assert client.get(TM).status_code == 401
    auth = new_session(client)
    response = client.get(TM, headers=auth)
    assert response.status_code == 200
    assert response.json() == json.loads(nr_tm.catalog().model_dump_json())
    assert client.get(TM, headers={**auth, "Origin": "https://evil.example"}).status_code == 403
    assert client.post(TM, json={}, headers=auth).status_code == 403
    assert client.get(TM + "/resolve", headers=auth).status_code == 403
    assert client.get(TM + "/extra", headers=auth).status_code == 403


def test_resolve_matches_the_local_resolution_and_stores_nothing(public):
    client, manager, _ = public
    auth = new_session(client)
    tenant = next(iter(manager.tenants.values()))
    before = directory_bytes(tenant.root)
    response = client.post(TM + "/resolve", json=RECIPE, headers=auth)
    assert response.status_code == 200, response.text
    assert response.json() == json.loads(nr_tm.resolve(NrTestModelConfig.model_validate(RECIPE)).model_dump_json())
    assert directory_bytes(tenant.root) == before
    assert client.get("/api/v1/signal-generator/signals", headers=auth).json() == []
    assert manager.inflight == 0 and not manager.ip_inflight and not manager.ip_expensive and not manager.storage_reserved
    assert client.post(TM + "/resolve", json=RECIPE).status_code == 401


def test_resolve_needs_none_of_the_shared_capacity_that_other_generator_posts_need(public, monkeypatch):
    client, manager, _ = public
    auth = new_session(client)
    tenant = next(iter(manager.tenants.values()))
    monkeypatch.setattr(manager, "reserve_storage", lambda *_: pytest.fail("resolve must not reserve storage"))
    try:
        manager.expensive_requests = manager.config.max_expensive_requests          # every shared compute slot is taken
        assert client.post(TM + "/resolve", json=RECIPE, headers=auth).status_code == 200
        assert client.get(TM, headers=auth).status_code == 200
        busy = client.post("/api/v1/signal-generator/validate", json={}, headers=auth)
        assert busy.status_code == 429 and busy.json()["error"]["code"] == "compute_busy"
        manager.expensive_requests = 0
        tenant.mutating = True                                                      # a workspace change is in progress
        assert client.post(TM + "/resolve", json=RECIPE, headers=auth).status_code == 200
        waiting = client.post("/api/v1/signal-generator/validate", json={}, headers=auth)
        assert waiting.status_code == 429 and waiting.json()["error"]["code"] == "workspace_busy"
    finally:
        manager.expensive_requests, tenant.mutating = 0, False
    assert client.post(TM + "/resolve", json=RECIPE, headers=auth).status_code == 200
    assert tenant.mutating is False and manager.expensive_requests == 0 and not manager.storage_reserved   # resolve never sets the flag


def test_resolve_refuses_oversized_wrong_typed_and_malformed_bodies_and_invalid_recipes_before_computing(public, monkeypatch):
    client, manager, _ = public
    auth = new_session(client)
    monkeypatch.setattr(nr_tm, "resolve", lambda *_a, **_k: pytest.fail("nothing may be derived for a refused request"))
    headers = {**auth, "Content-Type": "application/json"}
    assert client.post(TM + "/resolve", content="x" * 65537, headers=headers).status_code == 413
    assert client.post(TM + "/resolve", content=iter([b"x" * 32000, b"x" * 34000]), headers=headers).status_code == 413
    assert client.post(TM + "/resolve", content=b"{}", headers=auth).status_code == 415
    assert client.post(TM + "/resolve", content=b"{nope", headers=headers).status_code == 422
    assert client.post(TM + "/resolve", json=[1, 2], headers=auth).status_code == 422
    invalid = client.post(TM + "/resolve", json={"model": "NR-FR1-TM1.1", "carriers": [{"scs_khz": 60, "bandwidth_mhz": 5}]}, headers=auth)
    assert invalid.status_code == 422
    error = invalid.json()["error"]
    assert error["code"] == "invalid_request" and "5 MHz channel is not defined for 60 kHz" in error["details"][0]["message"]
    many = {"model": "NR-FR1-TM1.1", "carriers": [{"scs_khz": 30, "bandwidth_mhz": 20}] * 17}
    assert client.post(TM + "/resolve", json=many, headers=auth).status_code == 422
    assert client.post(TM + "/resolve", json={"version": "../../x"}, headers=auth).status_code == 422
    assert manager.inflight == 0 and not manager.ip_inflight and not manager.storage_reserved


def test_resolve_is_rate_limited_per_address(public):
    client, _, _ = public
    auth = new_session(client)
    statuses = [client.post(TM + "/resolve", json={}, headers=auth).status_code for _ in range(124)]
    assert statuses[:120] == [200] * 120 and set(statuses[120:]) == {429}
    assert client.post(TM + "/resolve", json={}, headers=auth).json()["error"]["code"] == "rate_limited"


def test_a_test_model_signal_is_generated_downloaded_and_revalidated_in_the_public_studio(public):
    client, manager, _ = public
    auth, other = new_session(client), new_session(client)
    config = client.post(TM + "/resolve", json={**RECIPE, "subframes": 2}, headers=auth).json()["config"]
    generated = client.post("/api/v1/signal-generator/batches", json={"configs": [config]}, headers=auth)
    assert generated.status_code == 201, generated.text
    signal = generated.json()[0]
    assert client.get(f"/api/v1/signal-generator/signals/{signal['signal_id']}", headers=other).status_code != 200
    detail = client.get(f"/api/v1/signal-generator/signals/{signal['signal_id']}", headers=auth).json()
    assert detail["coverage"] == "standard_test_model" and detail["config"]["nr_test_model"]["model"] == "NR-FR1-TM3.1a"
    assert client.get(detail["download_url"], headers=auth).headers["content-type"] == "application/zip"
    assert client.post("/api/v1/signal-generator/validate", json=detail["config"], headers=auth).status_code == 200
    stale = client.post("/api/v1/signal-generator/validate", json={**detail["config"], "fft_size": 4096}, headers=auth)
    assert stale.status_code == 422 and "Edit the recipe" in json.dumps(stale.json())
    assert client.post("/api/v1/signal-generator/batches", json={"configs": [{**detail["config"], "n_samples": 5000}]}, headers=auth).status_code == 422
    pair = client.post("/api/v1/pa-library/simulations", json={"input_signal_id": signal["signal_id"], "model_id": "rapp-am-pm"}, headers=auth)
    assert pair.status_code == 201, pair.text


def test_public_limits_accept_the_largest_test_model_captures_and_stop_at_the_workspace_budget(public):
    client, manager, _ = public
    auth = new_session(client)
    resolved = {model: client.post(TM + "/resolve", json={"model": model, "carriers": [{"scs_khz": 30, "bandwidth_mhz": 100}]}, headers=auth).json()["config"]
                for model in ("NR-FR1-TM1.1", "NR-FR1-TM3.1", "NR-FR1-TM3.1a")}
    assert {c["n_samples"] for c in resolved.values()} == {983040}                           # the standard 10 ms at 122.88 MS/s: below 1,000,000
    # one full-length capture, its CSV and its archive fit the 256 MiB temporary workspace
    generated = client.post("/api/v1/signal-generator/batches", json={"configs": [resolved["NR-FR1-TM1.1"]]}, headers=auth)
    assert generated.status_code == 201, generated.text
    signal = generated.json()[0]
    assert signal["n_samples"] == 983040
    assert client.get(signal["csv_url"], headers=auth).status_code == 200
    assert client.get(f"/api/v1/signal-generator/signals/{signal['signal_id']}/download", headers=auth).status_code == 200
    tenant = next(iter(manager.tenants.values()))
    assert directory_bytes(tenant.root) < manager.config.max_workspace_bytes
    # three full-length captures in one request are refused by the storage estimate, before anything is generated
    refused = client.post("/api/v1/signal-generator/batches", json={"configs": list(resolved.values())}, headers=auth)
    assert refused.status_code == 413 and refused.json()["error"]["code"] == "workspace_limit"
    assert len(client.get("/api/v1/signal-generator/signals", headers=auth).json()) == 1
