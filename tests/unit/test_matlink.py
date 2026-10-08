"""MATLINK queues named, authenticated actions without exposing MATLAB code."""
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from opendpd.schemas.matlink import MatlinkCompletion, MatlinkConnect, MatlinkHeartbeat, MatlinkRequest, MatlabVariable
from opendpd.schemas.run import RunStatus
from opendpd.services.matlink import MatlinkBroker, MatlinkError, MAX_PENDING


def variable(name="x", n=4096):
    return dict(name=name, class_name="double", size=[n, 1], complex=True, eligible=True, n_samples=n)


@pytest.fixture
def broker(tmp_path):
    runs, results, now = {}, {}, [100.]
    value = MatlinkBroker(SimpleNamespace(root=tmp_path), SimpleNamespace(get_run=runs.get),
                          clock=lambda: now[0], load_result=lambda _ws, rid: results.get(rid))
    value._test_runs, value._test_results, value._test_now = runs, results, now
    return value


def connect(broker, variables=None):
    return broker.connect(MatlinkConnect(label="MATLAB R2026a", release="R2026a",
        variables=variables if variables is not None else [variable("x"), variable("y")]))


def request(connection, action="create_demo", payload=None, key="demo"):
    return MatlinkRequest(client_id=connection.client_id, action=action, payload=payload or {}, idempotency_key=key)


def test_old_bridge_cannot_silently_ignore_new_transfer_options(broker):
    connection = connect(broker)
    for action, payload in [("import_dataset", {"dataset_id": "signals"}),
                            ("import_result", {"run_id": "run-a", "bundle": True}),
                            ("import_result", {"run_id": "run-a", "variable": "myResult"})]:
        with pytest.raises(MatlinkError, match="0.4.0"):
            broker.request(request(connection, action, payload))
    for variable_name in ["for", "x.y", "x);quit", "1result"]:
        with pytest.raises(ValidationError):
            request(connection, "import_result", {"run_id": "run-a", "variable": variable_name})


def test_lease_revival_retains_pending_and_never_exposes_secret(broker):
    connection = connect(broker)
    transfer = broker.request(request(connection))
    assert connection.bridge_token not in broker.snapshot().model_dump_json()
    broker._test_now[0] += 31
    state = broker.snapshot()
    assert not state.sessions[0].connected and state.sessions[0].pending_count == 1
    with pytest.raises(MatlinkError, match="busy or disconnected"):
        broker.request(request(connection, key="new-action"))
    first = broker.heartbeat(connection.client_id, connection.bridge_token, MatlinkHeartbeat())
    second = broker.heartbeat(connection.client_id, connection.bridge_token, MatlinkHeartbeat())
    assert first.requests[0].request_id == second.requests[0].request_id == transfer.request_id
    assert broker.snapshot().sessions[0].connected


def test_ack_retry_cannot_execute_or_overwrite_a_transfer_twice(broker):
    connection = connect(broker)
    body = request(connection)
    transfer = broker.request(body)
    assert broker.request(body).request_id == transfer.request_id
    with pytest.raises(MatlinkError, match="different action"):
        broker.request(request(connection, action="open_variable", payload={"variable": "x"}))
    success = MatlinkCompletion(status="succeeded", result={"variable": "opendpdDemo"})
    actual = broker.complete(connection.client_id, connection.bridge_token, transfer.request_id, success)
    replay = broker.complete(connection.client_id, connection.bridge_token, transfer.request_id,
                             MatlinkCompletion(status="failed", error="late retry"))
    assert replay == actual and replay.result == {"variable": "opendpdDemo"}
    assert not broker.heartbeat(connection.client_id, connection.bridge_token, MatlinkHeartbeat()).requests
    assert broker.request(body).status == "succeeded"


def test_bridge_secret_and_transfer_ownership(broker):
    one, two = connect(broker), connect(broker)
    transfer = broker.request(request(one))
    with pytest.raises(MatlinkError) as exc:
        broker.heartbeat(one.client_id, two.bridge_token, MatlinkHeartbeat())
    assert exc.value.status_code == 403
    with pytest.raises(MatlinkError) as exc:
        broker.complete(two.client_id, two.bridge_token, transfer.request_id, MatlinkCompletion(status="succeeded"))
    assert exc.value.status_code == 404


def test_result_waits_for_success_and_verified_report(broker):
    connection = connect(broker)
    broker._test_runs["train-123"] = SimpleNamespace(status=RunStatus.running)
    body = request(connection, "import_result", {"run_id": "train-123"}, "result")
    transfer = broker.request(body)
    assert transfer.status == "waiting"
    assert not broker.heartbeat(connection.client_id, connection.bridge_token, MatlinkHeartbeat()).requests
    with pytest.raises(MatlinkError, match="waiting"):
        broker.complete(connection.client_id, connection.bridge_token, transfer.request_id, MatlinkCompletion(status="succeeded"))
    broker._test_runs["train-123"].status = RunStatus.succeeded
    broker._test_results["train-123"] = object()
    commands = broker.heartbeat(connection.client_id, connection.bridge_token, MatlinkHeartbeat()).requests
    assert commands[0].request_id == transfer.request_id and commands[0].status == "queued"
    assert broker.request(request(connection, "import_result", {"run_id": "train-123"}, "other-click")).request_id == transfer.request_id


@pytest.mark.parametrize("terminal", [RunStatus.failed, RunStatus.cancelled, RunStatus.interrupted, RunStatus.succeeded])
def test_failed_or_missing_results_do_not_deliver(broker, terminal):
    connection = connect(broker)
    broker._test_runs["run-a"] = SimpleNamespace(status=RunStatus.queued)
    body = request(connection, "import_result", {"run_id": "run-a"})
    broker.request(body)
    broker._test_runs["run-a"].status = terminal
    state = broker.snapshot()
    assert state.transfers[0].status == "failed" and state.transfers[0].error
    with pytest.raises(MatlinkError):
        broker.request(request(connection, "import_result", {"run_id": "run-a"}, "retry-after-run"))
    with pytest.raises(MatlinkError) as exc:
        broker.request(request(connection, "import_result", {"run_id": "missing"}, "missing"))
    assert exc.value.status_code == 404


def test_explicit_disconnect_fails_pending_and_cannot_revive(broker):
    connection = connect(broker)
    broker.request(request(connection))
    broker.disconnect(connection.client_id, connection.bridge_token)
    broker.disconnect(connection.client_id, connection.bridge_token)
    state = broker.snapshot()
    assert not state.sessions[0].connected and not state.sessions[0].variables
    assert state.transfers[0].status == "failed"
    with pytest.raises(MatlinkError, match="disconnected"):
        broker.heartbeat(connection.client_id, connection.bridge_token, MatlinkHeartbeat())


def test_import_requires_available_equal_length_pair(broker):
    connection = connect(broker, [variable("x"), variable("y", 2048)])
    payload = dict(input="x", output="y", name="capture-one", sample_rate_mhz=80, bandwidth_mhz=20, segment_samples=256)
    with pytest.raises(MatlinkError, match="same number"):
        broker.request(request(connection, "import_iq", payload))
    broker.heartbeat(connection.client_id, connection.bridge_token, MatlinkHeartbeat(variables=[variable("x"), variable("y")]))
    transfer = broker.request(request(connection, "import_iq", payload))
    assert transfer.payload["origin"] == "unknown" and transfer.payload["segment_samples"] == 256
    with pytest.raises(MatlinkError, match="available I/Q"):
        broker.request(request(connection, "import_iq", {**payload, "input": "missing"}, "missing"))


def test_pending_queue_and_history_are_bounded(broker, monkeypatch):
    connection = connect(broker)
    for i in range(MAX_PENDING):
        broker.request(request(connection, key=str(i)))
    with pytest.raises(MatlinkError) as exc:
        broker.request(request(connection, key="overflow"))
    assert exc.value.status_code == 429
    for item in broker.snapshot().transfers:
        broker.complete(connection.client_id, connection.bridge_token, item.request_id, MatlinkCompletion(status="succeeded"))
    monkeypatch.setattr("opendpd.services.matlink.MAX_TRANSFERS", MAX_PENDING)
    newest = broker.request(request(connection, key="newest"))
    assert len(broker.snapshot().transfers) == MAX_PENDING
    assert broker.snapshot().transfers[0].request_id == newest.request_id
    assert len(broker._keys) == MAX_PENDING


@pytest.mark.parametrize("action,payload", [
    ("eval", {"code": "quit"}), ("create_demo", {"code": "quit"}),
    ("open_variable", {"variable": "x);quit"}), ("open_variable", {"variable": "end"}),
    ("import_result", {"run_id": "../secret"}),
    ("import_iq", dict(input="x", output="x", name="a", sample_rate_mhz=80, bandwidth_mhz=20, segment_samples=256)),
    ("import_iq", dict(input="x", output="y", name="a", sample_rate_mhz=20, bandwidth_mhz=80, segment_samples=256)),
    ("import_iq", dict(input="x", output="y", name="a", sample_rate_mhz=float("nan"), bandwidth_mhz=20, segment_samples=256)),
    ("import_iq", dict(input="x", output="y", name="spaces not allowed", sample_rate_mhz=80, bandwidth_mhz=20,
                       segment_samples=256)),
    ("import_iq", dict(input="x", output="y", origin="MATLAB", sample_rate_mhz=80, bandwidth_mhz=20, segment_samples=256)),
    # the segment length is the PSD segment and the evaluation reset interval: it has no default and a bounded range
    ("import_iq", dict(input="x", output="y", name="a", sample_rate_mhz=80, bandwidth_mhz=20)),
    ("import_iq", dict(input="x", output="y", name="a", sample_rate_mhz=80, bandwidth_mhz=20, segment_samples=1)),
    ("import_iq", dict(input="x", output="y", name="a", sample_rate_mhz=80, bandwidth_mhz=20, segment_samples=2_000_000)),
    ("import_iq", dict(input="x", output="y", name="a", sample_rate_mhz=80, bandwidth_mhz=20, segment_samples=256.5)),
])
def test_no_arbitrary_code_or_invalid_rf_payload(action, payload):
    with pytest.raises(ValidationError):
        MatlinkRequest(client_id="matlab-a", action=action, payload=payload, idempotency_key="a")


def test_metadata_and_completion_bounds():
    with pytest.raises(ValidationError):
        MatlinkHeartbeat(variables=[variable("x"), variable("x")])
    with pytest.raises(ValidationError):
        MatlabVariable(**{**variable(), "n_samples": 123})
    with pytest.raises(ValidationError):
        MatlabVariable(**{**variable(), "complex": False, "size": [2048, 2]})
    with pytest.raises(ValidationError):
        MatlinkCompletion(status="succeeded", result={"large": "a" * 32769})
    with pytest.raises(ValidationError):
        MatlinkCompletion(status="succeeded", result={"number": float("inf")})
    with pytest.raises(ValidationError):
        MatlinkCompletion(status="failed")


def test_retired_idempotency_keys_cannot_permanently_fill_the_queue(broker, monkeypatch):
    monkeypatch.setattr("opendpd.services.matlink.MAX_KEYS", 3)
    connection = connect(broker)
    broker._test_runs["run-a"] = SimpleNamespace(status=RunStatus.succeeded)
    broker._test_results["run-a"] = object()
    for i in range(3):
        transfer = broker.request(request(connection, "import_result", {"run_id": "run-a"}, str(i)))
    assert len(broker._transfers) == 1 and len(broker._keys) == 3
    with pytest.raises(MatlinkError):
        broker.request(request(connection, "import_result", {"run_id": "run-a"}, "too-many"))
    broker.complete(connection.client_id, connection.bridge_token, transfer.request_id, MatlinkCompletion(status="succeeded"))
    assert broker.request(request(connection, key="next-action")).status == "queued"
    assert len(broker._keys) == 1


def test_connect_at_capacity_reclaims_oldest_idle_bridge(broker, monkeypatch):
    monkeypatch.setattr("opendpd.services.matlink.MAX_SESSIONS", 2)
    oldest = connect(broker)
    broker._test_now[0] += 1
    newer = connect(broker)
    broker._test_now[0] += 31
    replacement = connect(broker)
    state = broker.snapshot()
    assert {s.client_id for s in state.sessions} == {newer.client_id, replacement.client_id}
    with pytest.raises(MatlinkError) as exc:
        broker.heartbeat(oldest.client_id, oldest.bridge_token, MatlinkHeartbeat())
    assert exc.value.code == "matlink_session_not_found"
    # An offline bridge not reclaimed at capacity can still revive normally.
    broker.heartbeat(newer.client_id, newer.bridge_token, MatlinkHeartbeat())
    assert all(s.connected for s in broker.snapshot().sessions)


def test_connect_never_reclaims_live_or_pending_bridges(broker, monkeypatch):
    monkeypatch.setattr("opendpd.services.matlink.MAX_SESSIONS", 2)
    pending, idle = connect(broker), connect(broker)
    transfer = broker.request(request(pending))
    broker._test_now[0] += 31
    live = connect(broker)
    assert {s.client_id for s in broker.snapshot().sessions} == {pending.client_id, live.client_id}
    with pytest.raises(MatlinkError) as exc:
        connect(broker)
    assert exc.value.code == "matlink_session_limit"
    queued = broker.heartbeat(pending.client_id, pending.bridge_token, MatlinkHeartbeat()).requests
    assert queued[0].request_id == transfer.request_id
    assert broker.snapshot().transfers[0].status == "queued"


def test_explicit_disconnect_is_reclaimed_before_offline_idle_bridge(broker, monkeypatch):
    monkeypatch.setattr("opendpd.services.matlink.MAX_SESSIONS", 2)
    oldest, disconnected = connect(broker), connect(broker)
    broker.disconnect(disconnected.client_id, disconnected.bridge_token)
    broker._test_now[0] += 31
    replacement = connect(broker)
    assert {s.client_id for s in broker.snapshot().sessions} == {oldest.client_id, replacement.client_id}
