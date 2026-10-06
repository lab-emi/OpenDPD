"""Hosted Arena shares admission, FIFO slots and retirement; no VM evaluator."""
import base64
import json
import time

import pytest
from fastapi.testclient import TestClient

from opendpd.core import arena as scoring
from opendpd.core.arena import protocol
from opendpd.schemas.arena import ArenaRow
from opendpd.services.arena import ArenaController
from opendpd.services.recipes import instantiate
from opendpd.services.workspace import write_json_atomic
from opendpd.web.app import create_web_app
from opendpd.web.gpu_archive import pack
from opendpd.web.policy import WebConfig
from tests.unit.test_arena_scoring import adjusted, cases

pytestmark = pytest.mark.integration

# One quota unit per condition and seed (deterministic fits: per condition). A unit covers that case at
# every budget of the parameter sweep, so the evidence of an entry is larger than what it is charged.
DPA_UNITS, DPA160_UNITS = 1 * 3, 1 * 3
DPA_SWEEP_CASES = 4 * 1 * 3


@pytest.fixture
def hosted(tmp_path, monkeypatch):
    def forbidden(*args):
        raise AssertionError("The public VM must never run the local Arena evaluator")

    monkeypatch.setattr(ArenaController, "_evaluate", forbidden)
    tunnel = "a" * 48 + ".internal"
    # Every quota is the production default: a parameter sweep has to fit the ordinary budgets.
    config = WebConfig(root=tmp_path / "web", origin="https://opendpd.com", api_host="api.opendpd.com",
        tunnel_host=tunnel, gpu_token="a" * 64, arena_submissions=True)
    app = create_web_app(config)
    with TestClient(app, base_url="http://" + tunnel, client=("127.0.0.1", 10000)) as client:
        client.headers.update({"Origin": "https://opendpd.com", "X-Forwarded-Proto": "https",
                               "CF-Connecting-IP": "203.0.113.10"})
        yield client, app.state.manager


def session(client):
    response = client.post("/api/v1/web/sessions", json={})
    assert response.status_code == 201, response.text
    return {"Authorization": "Bearer " + response.json()["access_token"]}


def announce(manager, digest=None):
    manager.gpu.poll("Test isolated GPU", arena_protocol_sha256=digest or protocol().protocol_sha256)


def body(board="apa-200mhz-b", **updates):
    return {**dict(board_id=board, backbone="gru", display_name="Hosted test",
                   accepted_protocol_sha256=protocol().protocol_sha256), **updates}


def worker_row(board_id="apa-200mhz-b", backbone="gru", quality=6.):
    """What the isolated runner returns: raw cases of the registered sweep, and a summary the VM never trusts."""
    current = protocol()
    raw = cases(scoring.calibration(), board_id, qualities=(quality,) * 3, backbone=backbone)
    derived = scoring.summarize_cases(backbone, board_id, raw, scoring.sweep(backbone))
    claimed = {ranking.ranking_id: {"score": 999., "rank": 1} for ranking in current.rankings}
    return ArenaRow(entry_id="arena-local-result", board_id=board_id, backbone=backbone, display_name="Worker",
        origin="workspace", status="succeeded", protocol_sha256=current.protocol_sha256, cases=raw,
        evidence_type=scoring.board(board_id).evidence_type, execution_semantics="offline_overlap_200_100",
        provenance={"training_sha256": current.training_sha256, "model_parameters": {}, "model_provenance": {}},
        **{**derived, "score": 999., "rankings": claimed})


def wait_for(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(.01)
    raise AssertionError("Timed out waiting for hosted Arena state")


def ordinary(client, auth):
    response = client.post("/api/v1/datasets/import-builtin", json={"name": "MyCustomPA"}, headers=auth)
    assert response.status_code == 201, response.text
    config = instantiate("pa-gru-smoke-v1", "mycustompa").model_dump(mode="json")
    config["training"]["epochs"] = 1
    config["execution"]["device"] = "cpu"
    return client.post("/api/v1/runs", json={"config": config}, headers=auth)


def test_opt_in_without_matching_image_capability_is_unavailable(hosted):
    client, manager = hosted
    auth = session(client)
    assert client.get("/api/v1/arena", headers=auth).json()["submissions_available"] is False
    assert client.post("/api/v1/arena/submissions", json=body(), headers=auth).status_code == 503
    announce(manager, "b" * 64)
    assert client.post("/api/v1/arena/submissions", json=body(), headers=auth).status_code == 503
    assert manager.total_runs == 0
    announce(manager)
    assert client.get("/api/v1/arena", headers=auth).json()["submissions_available"] is True


def test_arena_consumes_existing_session_ip_and_global_run_budgets(hosted):
    client, manager = hosted
    first, second = session(client), session(client)
    announce(manager)
    assert manager.slots.acquire(blocking=False)
    try:
        response = client.post("/api/v1/arena/submissions", json=body("apa-200mhz-b"), headers=first)
        assert response.status_code == 202, response.text
        assert manager.total_runs == DPA160_UNITS == 3 and sum(manager.ip_runs.values()) == 3
        tenant = manager.authenticate(first["Authorization"][7:])
        assert tenant.app.state.arena.admitted_units == 3
        # An Arena evaluation and an ordinary run count toward one workspace queue.
        assert ordinary(client, first).status_code == 201
        blocked = client.post("/api/v1/arena/submissions", json=body(), headers=first)
        assert blocked.status_code == 429
        assert manager.total_runs == 4
        object.__setattr__(manager.config, "runs_per_ip", 4 + DPA_UNITS - 1)
        blocked = client.post("/api/v1/arena/submissions", json=body(), headers=second)
        assert blocked.status_code == 429 and blocked.json()["error"]["code"] == "run_quota"
        assert manager.total_runs == 4
        # Ordinary submissions cannot bypass the Arena reservation either.
        object.__setattr__(manager.config, "runs_per_session", 4)
        blocked = ordinary(client, first)
        assert blocked.status_code == 429 and blocked.json()["error"]["code"] == "run_quota"
        # An entry is admitted whole or not at all: one unit short of the session budget is refused.
        object.__setattr__(manager.config, "runs_per_ip", 96)
        object.__setattr__(manager.config, "runs_per_session", DPA_UNITS - 1)
        blocked = client.post("/api/v1/arena/submissions", json=body(), headers=second)
        assert blocked.status_code == 429 and blocked.json()["error"]["code"] == "run_quota"
        assert manager.total_runs == 4 and manager.authenticate(second["Authorization"][7:]).app.state.arena.admitted_units == 0
        object.__setattr__(manager.config, "runs_per_session", DPA_UNITS)
        assert client.post("/api/v1/arena/submissions", json=body(), headers=second).status_code == 202
        assert manager.total_runs == 4 + DPA_UNITS == sum(manager.ip_runs.values())
    finally:
        # Retire pending work before releasing the test-held global slot.
        for tenant in list(manager.tenants.values()):
            tenant.app.state.arena.stop_accepting()
            tenant.app.state.supervisor.stop_accepting()
        manager.slots.release()


@pytest.mark.parametrize("board,backbone,units,budgets", [
    ("apa-200mhz-b", "gru", 1 * 3, 4), ("apa-200mhz-b", "gru", 1 * 3, 4), ("apa-200mhz-b", "gru", 1 * 3, 4), ("apa-200mhz-b", "gru", 1 * 3, 4),
    ("apa-200mhz-b", "gru_stream", 3, 4), ("apa-200mhz-b", "user_template", 3, 4),
    ("apa-200mhz-b", "mp_ls", 1 * 1, 4), ("apa-200mhz-b", "gmp_ls", 1 * 1, 4),       # deterministic: one fit per condition
    # The number of budgets a backbone has a configuration for changes its evidence, never its charge.
    ("apa-200mhz-b", "mcldnn", 3, 2), ("apa-200mhz-b", "gmp", 3, 1),
])
def test_quota_units_are_one_per_condition_and_seed_whatever_the_sweep_covers(hosted, board, backbone, units, budgets):
    client, manager = hosted
    auth = session(client)
    announce(manager)
    chosen = next(item for item in protocol().boards if item.board_id == board)
    assert units == len(chosen.conditions) * (1 if backbone in scoring.DETERMINISTIC else len(protocol().seeds))
    assert budgets == sum(point["model_parameters"] is not None for point in scoring.sweep(backbone))
    # Every entry fits the production session budget, the largest with room for five more units.
    assert units <= DPA160_UNITS < manager.config.runs_per_session == 8
    assert manager.slots.acquire(blocking=False)
    try:
        response = client.post("/api/v1/arena/submissions", json=body(board, backbone=backbone), headers=auth)
        assert response.status_code == 202, response.text
        tenant = manager.authenticate(auth["Authorization"][7:])
        assert tenant.app.state.arena.admitted_units == manager.total_runs == sum(manager.ip_runs.values()) == units
    finally:
        for tenant in list(manager.tenants.values()):
            tenant.app.state.arena.stop_accepting()
        manager.slots.release()


def finish_as_failed(manager, controller, folder):
    """The isolated worker reports a failure: the evaluation ends, and what it was charged stays spent."""
    polled = wait_for(lambda: manager.gpu.poll("Test isolated GPU", arena_protocol_sha256=protocol().protocol_sha256), 30)
    assert polled["kind"] == "arena"
    manager.gpu.result(manager.gpu.get(polled["id"], polled["lease"]), pack(folder, []), 1)
    controller.thread.join(30)
    assert not controller.thread.is_alive()


def test_production_quotas_admit_a_gradient_sweep_and_end_exactly_at_the_session_budget(hosted, tmp_path):
    client, manager = hosted
    config = manager.config
    assert (config.runs_per_session, config.runs_per_ip, config.runs_per_day, config.max_pending) == (8, 96, 512, 2)
    auth = session(client)
    announce(manager)
    controller = manager.authenticate(auth["Authorization"][7:]).app.state.arena
    # Each measured-source gradient entry uses three units, regardless of its four budgets.
    for board in ("apa-200mhz-b", "apa-200mhz-b"):
        admitted = client.post("/api/v1/arena/submissions", json=body(board), headers=auth)
        assert admitted.status_code == 202, admitted.text
        finish_as_failed(manager, controller, tmp_path)
        assert controller.get(admitted.json()["submission_id"]).status == "failed"
    assert controller.admitted_units == manager.total_runs == 6
    # A third gradient entry would require nine of the eight units and is not charged.
    for board in ("apa-200mhz-b", "apa-200mhz-b"):
        refused = client.post("/api/v1/arena/submissions", json=body(board, backbone="lstm"), headers=auth)
        assert refused.status_code == 429 and refused.json()["error"]["code"] == "run_quota", refused.text
        assert "remaining workspace evaluation budget" in refused.json()["error"]["message"]
    assert controller.admitted_units == manager.total_runs == sum(manager.ip_runs.values()) == 6
    assert len(controller.list()) == 2
    # Failed evaluations retain their charge; two one-unit deterministic entries fill the budget.
    for board in ("apa-200mhz-b", "apa-200mhz-b"):
        exact = client.post("/api/v1/arena/submissions", json=body(board, backbone="mp_ls"), headers=auth)
        assert exact.status_code == 202, exact.text
        finish_as_failed(manager, controller, tmp_path)
    spent = client.post("/api/v1/arena/submissions", json=body(backbone="mp_ls"), headers=auth)
    assert spent.status_code == 429 and spent.json()["error"]["code"] == "run_quota"
    assert ordinary(client, auth).status_code == 429
    assert controller.admitted_units == manager.total_runs == 8 and len(controller.list()) == 4
    # Another workspace has a session budget of its own (and the same address its larger one).
    other = session(client)
    assert client.post("/api/v1/arena/submissions", json=body(), headers=other).status_code == 202
    assert manager.total_runs == 8 + DPA_UNITS == sum(manager.ip_runs.values())
    finish_as_failed(manager, manager.authenticate(other["Authorization"][7:]).app.state.arena, tmp_path)


def test_request_that_can_never_run_is_answered_as_such_and_charged_to_no_budget(hosted):
    client, manager = hosted
    auth = session(client)
    announce(manager)
    controller = manager.authenticate(auth["Authorization"][7:]).app.state.arena

    def refusals():
        unknown = client.post("/api/v1/arena/submissions", json=body(backbone="not-a-model"), headers=auth)
        assert unknown.status_code == 422, unknown.text                    # Not a 500 from the sweep registry.
        assert "bundled Arena backbone" in unknown.text
        stale = client.post("/api/v1/arena/submissions", json=body(accepted_protocol_sha256="b" * 64), headers=auth)
        assert stale.status_code == 409 and "protocol changed" in stale.text, stale.text
        both = client.post("/api/v1/arena/submissions", headers=auth,
                           json=body(backbone="not-a-model", accepted_protocol_sha256="b" * 64))
        assert both.status_code == 409                                      # Review the rules first.
        assert client.post("/api/v1/arena/submissions", json=body("not-a-board"), headers=auth).status_code == 404
        assert client.post("/api/v1/arena/submissions", json=body("synthetic-suite"), headers=auth).status_code == 404

    refusals()
    assert manager.total_runs == 0 and manager.ip_runs == {} and controller.admitted_units == 0
    assert controller.list() == [] and not manager.gpu.jobs and controller.pending_count() == 0
    # The same answers with every budget exhausted: what can never run is not reported as a quota problem.
    assert manager.slots.acquire(blocking=False)
    try:
        assert client.post("/api/v1/arena/submissions", json=body("apa-200mhz-b"), headers=auth).status_code == 202
        object.__setattr__(manager.config, "runs_per_session", DPA160_UNITS)
        control = client.post("/api/v1/arena/submissions", json=body(backbone="lstm"), headers=auth)
        assert control.status_code == 429 and control.json()["error"]["code"] == "run_quota"
        for quota in ("runs_per_session", "runs_per_ip", "runs_per_day"):
            object.__setattr__(manager.config, quota, DPA160_UNITS)
        refusals()
        assert manager.total_runs == sum(manager.ip_runs.values()) == controller.admitted_units == DPA160_UNITS
        assert len(controller.list()) == 1 and controller.pending_count() == 1
    finally:
        for tenant in list(manager.tenants.values()):
            tenant.app.state.arena.stop_accepting()
        manager.slots.release()


def test_arena_and_ordinary_jobs_share_global_pending_limit_and_fifo(hosted):
    client, manager = hosted
    first, second = session(client), session(client)
    announce(manager)
    assert manager.slots.acquire(blocking=False)
    try:
        arena = client.post("/api/v1/arena/submissions", json=body(), headers=first)
        assert arena.status_code == 202, arena.text
        object.__setattr__(manager.config, "max_pending_global", 1)
        denied = ordinary(client, second)
        assert denied.status_code == 429 and denied.json()["error"]["code"] == "queue_full"
        object.__setattr__(manager.config, "max_pending_global", 128)
        run = ordinary(client, second)
        assert run.status_code == 201, run.text
        wait_for(lambda: manager.next_dispatch)
        assert manager.next_dispatch[1].run_id == arena.json()["submission_id"]
        status = manager.server_status()
        assert status.running_jobs == 0 and status.queued_jobs == 2
    finally:
        for tenant in list(manager.tenants.values()):
            tenant.app.state.arena.stop_accepting()
            tenant.app.state.supervisor.stop_accepting()
        manager.slots.release()


def _another_definition(row):
    row.provenance["model_parameters"] = {"hidden_size": 1}


def _another_sweep(row):
    row.budgets[1].model_parameters = {**row.budgets[1].model_parameters, "hidden_size": 8}


def _another_template(row):
    row.provenance["model_provenance"] = {"backbone_id": "ub-" + "d" * 64}


@pytest.mark.parametrize("tamper,refusal", [(None, None), (_another_definition, "different model definition"),
    (_another_template, "different template provenance"), (_another_sweep, "sweep registered for this request"),
    (lambda row: row.cases.pop(), "exactly once")])
def test_worker_progress_result_and_canonical_request_use_isolated_transfer(hosted, tmp_path, tamper, refusal):
    client, manager = hosted
    auth = session(client)
    announce(manager)
    tenant = manager.authenticate(auth["Authorization"][7:])
    controller = tenant.app.state.arena
    submitted = client.post("/api/v1/arena/submissions", json=body(), headers=auth).json()
    identifier = submitted["submission_id"]
    canonical = controller.directory(identifier) / "request.json"
    original = canonical.read_bytes()
    assert json.loads(original) == {"board_id": "apa-200mhz-b", "backbone": "gru", "model_parameters": {},
                                    "model_provenance": {}, "protocol_sha256": protocol().protocol_sha256}
    wait_for(lambda: manager.gpu.jobs)
    polled = manager.gpu.poll("Test isolated GPU", arena_protocol_sha256=protocol().protocol_sha256)
    assert polled and polled["kind"] == "arena"
    job = manager.gpu.get(polled["id"], polled["lease"])
    assert not manager.slots.acquire(blocking=False)
    progress = dict(phase="training", epoch=100, epochs=150, completed_cases=0, expected_cases=DPA_SWEEP_CASES,
                    message="apa-200mhz-b · ≤250 parameters · seed 0 · epoch 100/150")
    manager.gpu.update(job, {"arena_progress": base64.b64encode(json.dumps(progress).encode()).decode()})
    observed = wait_for(lambda: controller.get(identifier).progress)
    assert observed.epoch == 100 and observed.expected_cases == 12
    # The bridge transports a complete sweep result; the VM derives every score again from its raw
    # cases and the sweep registered for its own canonical request (scorer tests cover the numbers).
    row = worker_row()
    if tamper:
        tamper(row)
    output = tmp_path / "result.json"
    write_json_atomic(output, row)
    manager.gpu.result(job, pack(tmp_path, [output]), 0)
    controller.thread.join(5)
    complete = controller.get(identifier)
    if refusal:
        assert complete.status == "failed" and complete.result is None
        assert refusal in complete.error
    else:
        assert complete.status == "succeeded" and complete.result.display_name == "Hosted test"
        assert complete.result.entry_id == identifier and complete.result.origin == "workspace"
        expected = max(adjusted("gru", budget, 6.) for budget in protocol().budgets)
        assert complete.result.score == complete.result.rankings["overall"].score == pytest.approx(expected)
        assert complete.result.qualified_budgets == 4 and len(complete.result.cases) == DPA_SWEEP_CASES
        assert controller.admitted_units == manager.total_runs == DPA_UNITS      # Twelve cases, three units.
        board = client.get("/api/v1/arena/boards/apa-200mhz-b", headers=auth).json()
        mine = next(entry for entry in board["rows"] if entry["entry_id"] == identifier)
        assert mine["rank"] == 1 and {entry["rank"] for entry in mine["rankings"].values()} == {1}
    assert controller.process is None and canonical.read_bytes() == original
    assert manager.slots.acquire(blocking=False)
    manager.slots.release()


def test_session_end_cancels_lease_and_blocks_late_results(hosted):
    client, manager = hosted
    auth = session(client)
    announce(manager)
    tenant = manager.authenticate(auth["Authorization"][7:])
    response = client.post("/api/v1/arena/submissions", json=body(), headers=auth)
    assert response.status_code == 202, response.text
    wait_for(lambda: manager.gpu.jobs)
    controller = tenant.app.state.arena
    assert client.post("/api/v1/web/sessions/end", json={}, headers=auth).status_code == 204
    controller.thread.join(5)
    assert all(job.done for job in manager.gpu.jobs.values())
    assert not controller.queue.worker_alive(response.json()["submission_id"])
    for job in list(manager.gpu.jobs.values()):
        manager.gpu.result(job, b"a late result must be ignored", 0)
        assert not (job.root / "result.json").exists()
    manager.gpu.sweep()
    assert not manager.gpu.jobs
    assert controller.get(response.json()["submission_id"]).status == "interrupted"
    assert manager.slots.acquire(blocking=False)
    manager.slots.release()


def test_invalid_protocol_does_not_charge_compute(hosted):
    client, manager = hosted
    auth = session(client)
    announce(manager)
    payload = body()
    payload["accepted_protocol_sha256"] = "b" * 64
    response = client.post("/api/v1/arena/submissions", json=payload, headers=auth)
    assert response.status_code == 409
    assert manager.total_runs == 0 and manager.ip_runs == {} and not manager.gpu.jobs


def test_global_day_budget_cannot_be_multiplied_by_new_ip_sessions(hosted):
    client, manager = hosted
    first = session(client)
    client.headers["CF-Connecting-IP"] = "203.0.113.11"
    second = session(client)
    announce(manager)
    object.__setattr__(manager.config, "runs_per_day", DPA160_UNITS + DPA_UNITS - 1)
    assert manager.slots.acquire(blocking=False)
    try:
        response = client.post("/api/v1/arena/submissions", json=body("apa-200mhz-b"), headers=first)
        assert response.status_code == 202, response.text
        denied = client.post("/api/v1/arena/submissions", json=body(), headers=second)
        assert denied.status_code == 429 and denied.json()["error"]["code"] == "run_quota"
        assert manager.total_runs == DPA160_UNITS == 3 and len(manager.ip_runs) == 1
    finally:
        for tenant in list(manager.tenants.values()):
            tenant.app.state.arena.stop_accepting()
        manager.slots.release()


def test_ordinary_pending_work_blocks_arena_and_queued_expiry_never_dispatches(hosted):
    client, manager = hosted
    first, second = session(client), session(client)
    announce(manager)
    assert manager.slots.acquire(blocking=False)
    try:
        assert ordinary(client, first).status_code == 201
        object.__setattr__(manager.config, "max_pending_global", 1)
        denied = client.post("/api/v1/arena/submissions", json=body(), headers=second)
        assert denied.status_code == 429 and denied.json()["error"]["code"] == "queue_full"
        object.__setattr__(manager.config, "max_pending_global", 128)
        queued = client.post("/api/v1/arena/submissions", json=body(), headers=second)
        assert queued.status_code == 202, queued.text
        tenant = manager.authenticate(second["Authorization"][7:])
        assert client.post("/api/v1/web/sessions/end", json={}, headers=second).status_code == 204
        tenant.app.state.arena.thread.join(5)
        assert tenant.app.state.arena.get(queued.json()["submission_id"]).status == "interrupted"
        assert not manager.gpu.jobs
    finally:
        for tenant in list(manager.tenants.values()):
            tenant.app.state.arena.stop_accepting()
            tenant.app.state.supervisor.stop_accepting()
        manager.slots.release()
