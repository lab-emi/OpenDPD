"""Arena endpoint authentication and strict executable-submission contracts."""

import json
import os

import pytest
from fastapi.testclient import TestClient

from opendpd.core import arena as scoring
from opendpd.server.app import create_app
from opendpd.server.security import CSRF_HEADER
from opendpd.schemas.arena import ArenaProtocol, ArenaRow
from opendpd.services.arena import ArenaController
from tests.unit.test_arena import CONDITIONS, EARLIER_VERSION, as_arena_v1_wrote_it, store, values

pytestmark = pytest.mark.integration


def arena_protocol():
    return ArenaProtocol(protocol_id="test-v2", protocol_sha256="a" * 64, training_sha256="c" * 64, title="Arena",
        description="Test", boards=[dict(board_id="dpa-160mhz", title="Synthetic",
        description="Test", evidence_type="synthetic_simulation", evidence_label="Simulated",
        dataset="synthetic", conditions=CONDITIONS)], seeds=[1, 2, 3], budgets=scoring.BUDGETS, rules=[],
        rankings=scoring.RANKINGS, score_formula="test", training={}, scoring={}, cost_model={"operations": "ops = mul + add"})


def test_arena_routes_authentication_csrf_and_no_client_metrics(tmp_path, monkeypatch):
    protocol = arena_protocol()
    shipped = [values(protocol, entry_id="official-gru", score=4.),
               values(protocol, entry_id="official-lstm", backbone="lstm", score=6.,
                      rankings={**{ranking.ranking_id: {"score": 6.} for ranking in protocol.rankings},
                                "linearization": {"score": 2.}, "budget-250": {"score": None},
                                "house-special": {"score": 50., "rank": 1}})]      # not a ranking of this protocol
    monkeypatch.setattr(ArenaController, "protocol", lambda _: protocol)
    monkeypatch.setattr(ArenaController, "bundled_backbones", lambda _: [])
    monkeypatch.setattr(ArenaController, "official_rows", lambda _: shipped)
    app = create_app(tmp_path / "ws", bootstrap_token="arena", monitor_resources=False, start_sweeps=False)
    with TestClient(app, base_url="http://127.0.0.1:8877") as client:
        for path in ("/arena", "/arena/boards/dpa-160mhz", "/arena/submissions"):
            assert client.get("/api/v1" + path).status_code == 401
        auth = client.post("/api/v1/session/bootstrap", json={"token": "arena"}).json()
        client.headers["origin"] = "http://127.0.0.1:8877"
        body = dict(board_id="dpa-160mhz", backbone="gru", display_name="test", accepted_protocol_sha256="a" * 64)
        assert client.post("/api/v1/arena/submissions", json=body).status_code == 403
        client.headers[CSRF_HEADER] = auth["csrf_token"]
        catalog = client.get("/api/v1/arena").json()["protocol"]
        assert catalog["protocol_sha256"] == "a" * 64 and catalog["training_sha256"] == "c" * 64
        assert catalog["budgets"] == [250, 500, 1000, 2000] and catalog["cost_model"] == {"operations": "ops = mul + add"}
        assert [ranking["ranking_id"] for ranking in catalog["rankings"]][:2] == ["overall", "linearization"]
        # Every ranking arrives ranked; a row's position is its Overall FoM position.
        board = client.get("/api/v1/arena/boards/dpa-160mhz").json()
        rows = {row["entry_id"]: row for row in board["rows"]}
        assert [row["entry_id"] for row in board["rows"]] == ["official-lstm", "official-gru"]
        assert rows["official-lstm"]["rank"] == rows["official-lstm"]["rankings"]["overall"]["rank"] == 1
        assert rows["official-gru"]["rankings"]["linearization"] == {"score": 4., "rank": 1}
        assert rows["official-lstm"]["rankings"]["linearization"] == {"score": 2., "rank": 2}
        assert rows["official-lstm"]["rankings"]["budget-250"] == {"score": None, "rank": None}
        assert rows["official-gru"]["rankings"]["budget-250"]["rank"] == 1
        for row in rows.values():       # Exactly the published rankings: a row cannot bring one of its own.
            assert set(row["rankings"]) == {ranking["ranking_id"] for ranking in catalog["rankings"]}
        assert "house-special" not in json.dumps(board)
        assert [point["budget"] for point in rows["official-gru"]["budgets"]] == catalog["budgets"]
        assert all(point["ops"] == point["mul"] + point["add"] for point in rows["official-gru"]["budgets"])
        assert not any(word in json.dumps(board) for word in ("latency", "timing_cohort", "macs_per_sample", "score_std"))
        assert client.get("/api/v1/arena/submissions").json() == []
        assert client.get("/api/v1/arena/boards/not-a-board").status_code == 404
        for forged in ({"score": 999}, {"budgets": [2000]}, {"rankings": {"overall": {"rank": 1}}}, {"model_parameters": {}}):
            assert client.post("/api/v1/arena/submissions", json={**body, **forged}).status_code == 422
        assert client.post("/api/v1/arena/submissions", json=body).status_code == 422
        # A record that lost its accepted request is shown as failed; it cannot take the workspace's API down.
        lost = store(app.state.arena, "7", protocol, ArenaRow(**values(protocol, origin="workspace")))
        (app.state.arena.directory(lost.submission_id) / "request.json").unlink()
        listed = client.get("/api/v1/arena/submissions")
        assert listed.status_code == 200, listed.text
        assert [(item["submission_id"], item["status"], item["result"]) for item in listed.json()] == [
            (lost.submission_id, "failed", None)]
        assert "accepted Arena request is missing or damaged" in listed.json()[0]["error"]
        assert client.get(f"/api/v1/arena/submissions/{lost.submission_id}").json()["status"] == "failed"
        board = client.get("/api/v1/arena/boards/dpa-160mhz")
        assert board.status_code == 200, board.text
        assert [(row["entry_id"], row["rank"]) for row in board.json()["rows"]] == [
            ("official-lstm", 1), ("official-gru", 2), (lost.submission_id, None)]


def test_damaged_and_earlier_version_records_never_take_the_arena_api_down_or_serve_a_server_path(tmp_path, monkeypatch):
    protocol = arena_protocol()
    monkeypatch.setattr(ArenaController, "protocol", lambda _: protocol)
    monkeypatch.setattr(ArenaController, "bundled_backbones", lambda _: [])
    monkeypatch.setattr(ArenaController, "official_rows", lambda _: [])

    def studio():
        app = create_app(tmp_path / "ws", bootstrap_token="arena", monitor_resources=False, start_sweeps=False)
        client = TestClient(app, base_url="http://127.0.0.1:8877")
        return app, client

    def sign_in(client):
        auth = client.post("/api/v1/session/bootstrap", json={"token": "arena"}).json()
        client.headers.update({"origin": "http://127.0.0.1:8877", CSRF_HEADER: auth["csrf_token"]})

    app, client = studio()
    with client:
        sign_in(client)
        arena = app.state.arena
        row = lambda **updates: ArenaRow(**values(protocol, origin="workspace", **updates))
        earlier = store(arena, "5", protocol, row(protocol_sha256="b" * 64), accepted="b" * 64)
        as_arena_v1_wrote_it(arena.directory(earlier.submission_id) / "submission.json")
        unreadable = store(arena, "6", protocol, row())
        (arena.directory(unreadable.submission_id) / "submission.json").write_text('{"submission_id": "arena-')
        locked = store(arena, "8", protocol, row())
        accepted = arena.directory(locked.submission_id) / "request.json"
        accepted.chmod(0)
        sealed = store(arena, "9", protocol, row())
        arena.directory(sealed.submission_id).chmod(0)   # A record folder the server may not even enter.
        try:
            responses = {name: client.get("/api/v1/arena" + route) for name, route in dict(
                listed="/submissions", earlier=f"/submissions/{earlier.submission_id}",
                locked=f"/submissions/{locked.submission_id}", unreadable=f"/submissions/{unreadable.submission_id}",
                sealed=f"/submissions/{sealed.submission_id}", board="/boards/dpa-160mhz", catalog="").items()}
        finally:
            arena.directory(sealed.submission_id).chmod(0o700)
        enterable = os.access(accepted, os.R_OK)          # A superuser reads everything: those records are healthy.
        assert {name: response.status_code for name, response in responses.items()} == dict(
            listed=200, earlier=200, locked=200, unreadable=409, sealed=200 if enterable else 409, board=200, catalog=200)
        if not enterable:
            assert responses["sealed"].json()["error"]["message"] == "A stored Arena submission is damaged."
        # The earlier version's record is shown without its evidence; the unreadable one is left out of lists.
        listed = {item["submission_id"]: item for item in responses["listed"].json()}
        assert set(listed) == {earlier.submission_id, locked.submission_id} | ({sealed.submission_id} if enterable else set())
        assert listed[earlier.submission_id] == responses["earlier"].json()
        assert (listed[earlier.submission_id]["status"], listed[earlier.submission_id]["result"],
                listed[earlier.submission_id]["error"]) == ("failed", None, EARLIER_VERSION)
        refused = responses["unreadable"].json()["error"]          # A workspace problem (409), not a server error.
        assert (refused["code"], refused["message"]) == ("workspace_error", "A stored Arena submission is damaged.")
        if not os.access(accepted, os.R_OK):              # Evidence the server cannot open: said without its location.
            assert listed[locked.submission_id]["error"] == (
                "Stored Arena result failed validation: its stored evidence could not be read")
        assert listed[locked.submission_id]["status"] == "failed" and listed[locked.submission_id]["result"] is None
        # A board lists its own protocol: the earlier record is absent, the current failed one is unranked.
        assert [(entry["entry_id"], entry["status"], entry["rank"]) for entry in responses["board"].json()["rows"]
                if entry["entry_id"] != sealed.submission_id] == [(locked.submission_id, "failed", None)]
        for response in responses.values():
            assert str(tmp_path) not in response.text and "Errno" not in response.text
    # Studio starts again on that workspace and serves the same records.
    app, client = studio()
    with client:
        sign_in(client)
        again = client.get("/api/v1/arena/submissions")           # The folder is enterable again: its record is back.
        assert again.status_code == 200
        assert {item["submission_id"] for item in again.json()} == set(listed) | {sealed.submission_id}
        assert client.get("/api/v1/arena/boards/dpa-160mhz").status_code == 200


def test_hosted_arena_exposes_rankings_but_cannot_start_unbrokered_compute(tmp_path):
    from opendpd.web.app import create_web_app
    from opendpd.web.policy import WebConfig
    tunnel = "a" * 48 + ".internal"
    app = create_web_app(WebConfig(root=tmp_path / "public", origin="https://opendpd.com",
                                  api_host="api.opendpd.com", tunnel_host=tunnel))
    with TestClient(app, base_url="http://" + tunnel, client=("127.0.0.1", 10000)) as client:
        client.headers.update({"Origin": "https://opendpd.com", "X-Forwarded-Proto": "https",
                               "CF-Connecting-IP": "203.0.113.10"})
        auth = client.post("/api/v1/web/sessions", json={}).json()
        client.headers["Authorization"] = "Bearer " + auth["access_token"]
        catalog = client.get("/api/v1/arena")
        assert catalog.status_code == 200, catalog.text
        data = catalog.json()
        assert data["submissions_available"] is False
        assert data["submission_unavailable_reason"]
        assert len(data["protocol"]["boards"]) == 1
        assert data["protocol"]["protocol_id"] == "dpd-arena-v6-apa-b" and data["protocol"]["budgets"] == [250, 500, 1000, 2000]
        assert data["protocol"]["training_sha256"] == scoring.training_fingerprint()
        assert {"overall", "linearization", "parameter_efficiency", "arithmetic_efficiency", "budget-2000"} <= {
            ranking["ranking_id"] for ranking in data["protocol"]["rankings"]}
        assert data["protocol"]["cost_model"]["operations"] == "ops = mul + add"
        assert not any("latency" in key or "timing" in key for section in ("training", "scoring")
                       for key in data["protocol"][section])
        assert len(data["backbones"]) == 23
        assert {board["dataset"] for board in data["protocol"]["boards"]} == {
            "APA_200MHz_b"}
        assert client.get("/api/v1/arena/boards/synthetic-suite").status_code == 403
        # The shipped reference matrix: absent while the benchmark runs, 23 entries per board afterwards.
        for board_id in ("apa-200mhz-b",):
            response = client.get(f"/api/v1/arena/boards/{board_id}")
            assert response.status_code == 200, response.text
            board = response.json()
            assert board["protocol_sha256"] == data["protocol"]["protocol_sha256"] and board["coverage"]["expected"] == 23
            assert len(board["rows"]) == board["coverage"]["evaluated"] in (0, 23)
            for row in board["rows"]:
                assert row["origin"] == "official" and row["rank"] == row["rankings"]["overall"]["rank"]
                assert [point["budget"] for point in row["budgets"]] == data["protocol"]["budgets"]
        response = client.post("/api/v1/arena/submissions", json={"board_id": "dpa-160mhz",
            "backbone": "gru", "display_name": "cannot run here",
            "accepted_protocol_sha256": data["protocol"]["protocol_sha256"]})
        assert response.status_code == 403
        assert all(tenant.app.state.arena.thread is None for tenant in app.state.manager.tenants.values())
