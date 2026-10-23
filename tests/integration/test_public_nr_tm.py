"""The NR test-model routes through the real public ASGI boundary: sessions, body limits, rates, no storage, 1,000,000-sample captures."""
import json
import logging

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from opendpd.core.waveforms import nr_tm
from opendpd.schemas.nr_test_model import NrTestModelConfig
from opendpd.schemas.signal_generator import GeneratorBatchRequest
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


def test_invalid_recipes_count_against_the_resolve_budget(public):
    """An attempt that fails late (a 3 MHz carrier cannot take 12 dB for two NB-IoT carriers) costs the budget like a valid one."""
    client, manager, _ = public
    auth = new_session(client)
    late = {"model": "NR-N-TM", "nb_iot": {"position": "both", "boost_db": 12}, "carriers": [{"scs_khz": 15, "bandwidth_mhz": 3}]}
    statuses = []
    for cell in range(122):
        late["carriers"][0]["n_cell_id"] = cell + 1                                    # a different recipe every time: no cache helps
        statuses.append(client.post(TM + "/resolve", json=late, headers=auth).status_code)
    assert statuses[:120] == [422] * 120 and set(statuses[120:]) == {429}
    assert client.post(TM + "/resolve", json=late, headers=auth).json()["error"]["code"] == "rate_limited"
    assert manager.inflight == 0 and not manager.ip_inflight and not manager.storage_reserved


def recipe_text(boost="6", carriers='[{"scs_khz":15,"bandwidth_mhz":10}]'):
    """A recipe as JSON text, so that numbers a JSON library will not write (1e999, Infinity, a 400-digit integer) reach the server as typed."""
    return '{"model":"NR-N-TM","nb_iot":{"position":"lower","boost_db":%s},"carriers":%s}' % (boost, carriers)


NESTED = lambda depth: "[" * depth + "]" * depth       # noqa: E731
# (what the client sent, the error code of the answer). 4,400 digits is more than Python converts to an integer; 30,000 levels is more than JSON reads.
HOSTILE_RECIPES = {
    "a 400-digit boost": (recipe_text("1" + "0" * 400), "invalid_request"),
    "a negative 400-digit boost": (recipe_text("-1" + "0" * 400), "invalid_request"),
    "a 4,299-digit boost": (recipe_text("1" * 4299), "invalid_request"),
    "a 4,400-digit boost": (recipe_text("1" * 4400), "invalid_json"),
    "a boost of '1e999'": (recipe_text('"1e999"'), "invalid_request"),
    "a boost of 1e999": (recipe_text("1e999"), "invalid_request"),
    "a boost of -1e999": (recipe_text("-1e999"), "invalid_request"),
    "a boost of Infinity": (recipe_text("Infinity"), "invalid_request"),
    "a boost of NaN": (recipe_text("NaN"), "invalid_request"),
    "a boost of null": (recipe_text("null"), "invalid_request"),
    "a gap of 400 digits": (recipe_text(carriers='[{"scs_khz":15,"bandwidth_mhz":10},{"scs_khz":15,"bandwidth_mhz":10,"gap_before_mhz":%s}]' % ("1" + "0" * 400)), "invalid_request"),
    "carriers nested 500 deep": (recipe_text(carriers=NESTED(500)), "invalid_request"),
    "carriers nested 3,000 deep": (recipe_text(carriers=NESTED(3000)), None),
    "carriers nested 30,000 deep": (recipe_text(carriers=NESTED(30000)), "invalid_json"),
}
RECIPE_BODIES = {                  # the four public routes whose bodies can carry a recipe
    TM + "/resolve": lambda recipe: recipe,
    "/api/v1/signal-generator/signals": lambda recipe: '{"nr_test_model":%s}' % recipe,
    "/api/v1/signal-generator/batches": lambda recipe: '{"configs":[{"nr_test_model":%s}]}' % recipe,
    "/api/v1/signal-generator/validate": lambda recipe: '{"nr_test_model":%s}' % recipe,
}


def test_hostile_numbers_in_a_recipe_are_a_422_on_every_route_that_takes_one_and_never_a_500(public, caplog):
    """A 400-digit integer used to escape the boost validator as an OverflowError: a 500 and a traceback per request."""
    client, manager, now = public
    auth = {**new_session(client), "Content-Type": "application/json"}
    with caplog.at_level(logging.DEBUG):
        for name, (recipe, code) in HOSTILE_RECIPES.items():
            for path, wrap in RECIPE_BODIES.items():
                body = wrap(recipe)
                expected_code = code
                if expected_code is None:
                    # Older CPython JSON decoders hit their recursion limit
                    # before schema validation; both paths must remain a 422.
                    try:
                        json.loads(body)
                    except RecursionError:
                        expected_code = "invalid_json"
                    else:
                        expected_code = "invalid_request"
                response = client.post(path, content=body.encode(), headers=auth)
                assert response.status_code == 422, (name, path, response.status_code, response.text[:200])
                assert response.json()["error"]["code"] == expected_code, (name, path, response.text[:200])
            now[0] += 60                                                       # a new minute: the per-address budgets start again
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING or r.exc_info], [r.getMessage() for r in caplog.records][:3]
    assert manager.inflight == 0 and not manager.ip_inflight and not manager.ip_expensive and not manager.storage_reserved
    assert client.post(TM + "/resolve", json=RECIPE, headers=auth).status_code == 200      # the session and the service are fine


def test_a_huge_boost_is_explained_like_any_other_invalid_number(public):
    client, _, _ = public
    auth = {**new_session(client), "Content-Type": "application/json"}
    refused = client.post(TM + "/resolve", content=recipe_text("1" + "0" * 400).encode(), headers=auth)
    assert refused.status_code == 422 and refused.json()["error"]["message"] == "request validation failed"
    assert [d["field"] for d in refused.json()["error"]["details"]] == ["nb_iot.boost_db"]
    above = client.post(TM + "/resolve", content=recipe_text("12.0000001").encode(), headers=auth).json()["error"]["details"]
    assert "12.0000001 dB is above the largest NB-IoT boost Studio offers (12 dB" in above[0]["message"]


def test_a_validator_that_fails_unexpectedly_is_still_the_clients_422_and_costs_one_log_line(public, monkeypatch, caplog):
    """Defence in depth: whatever the schemas raise about a body, the public boundary answers 422 and logs no traceback per request."""
    from opendpd.web import mutation_policy
    client, manager, now = public
    auth = {**new_session(client), "Content-Type": "application/json"}
    monkeypatch.setattr(mutation_policy, "_REPORTED", set())

    def explode(*_args, **_kwargs):
        raise OverflowError("int too large to convert to float")
    monkeypatch.setattr(NrTestModelConfig, "model_validate", explode)
    with caplog.at_level(logging.DEBUG):
        statuses = [client.post(TM + "/resolve", json=RECIPE, headers=auth).status_code for _ in range(5)]
    assert statuses == [422] * 5
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1 and warnings[0].exc_info is None and "OverflowError" in warnings[0].getMessage()
    assert manager.inflight == 0 and not manager.ip_inflight and not manager.storage_reserved


def wide_ntm_configs(count, first_cell):
    """`count` valid configurations of 16-carrier NR-N-TM (50 MHz hosts) whose cell identities differ from any other body."""
    configs = []
    for i in range(count):
        cells = [{"scs_khz": 15, "bandwidth_mhz": 50, "n_cell_id": first_cell + i}] + [{"scs_khz": 15, "bandwidth_mhz": 50}] * 14
        cells.append({"scs_khz": 15, "bandwidth_mhz": 50, "n_cell_id": first_cell + i + 150})
        recipe = NrTestModelConfig.model_validate({"model": "NR-N-TM", "nb_iot": {"position": "both", "boost_db": 6}, "subframes": 1, "carriers": cells})
        configs.append(nr_tm.resolve(recipe).config.model_dump(mode="json"))
    return configs


def test_invalid_batches_are_rate_limited_before_they_are_judged(public):
    """The worst legal-but-invalid body: 16 configurations of 16 carriers (about 43 KB) that is refused for its total size (16 x 983,040 samples)."""
    client, manager, _ = public
    auth = new_session(client)
    bodies = [{"configs": wide_ntm_configs(16, 1 + 17 * attempt)} for attempt in range(13)]
    assert 30_000 < len(json.dumps(bodies[0])) < 65_536
    with pytest.raises(ValidationError, match="at most 4,000,000 samples"):          # every configuration is legal; the batch is not
        GeneratorBatchRequest.model_validate(bodies[0])
    responses = [client.post("/api/v1/signal-generator/batches", json=body, headers=auth) for body in bodies]
    assert [r.status_code for r in responses] == [422] * 12 + [429]
    assert responses[0].json()["error"]["code"] == "invalid_request" and responses[-1].json()["error"]["code"] == "rate_limited"
    assert client.post("/api/v1/signal-generator/batches", json=bodies[0], headers=auth).status_code == 429
    assert not manager.storage_reserved and manager.inflight == 0 and not manager.ip_inflight
    assert next(iter(manager.tenants.values())).mutating is False


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
