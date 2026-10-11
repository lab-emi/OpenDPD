"""Real Studio service tests for the Python role used by the MATLAB bridge."""
import json
from types import SimpleNamespace

import pytest

from opendpd.sdk import SDKError, open_project
from opendpd.sdk.matlink import MatlinkClient

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    value = open_project(tmp_path_factory.mktemp("matlink-sdk") / "workspace 中文")
    yield value
    value.close(stop_service=True)


@pytest.fixture
def bridge(project):
    value = MatlinkClient(project, "MATLAB test", "R2026a", "[]")
    yield value
    value.disconnect()


def enqueue(project, bridge, key="demo"):
    return project._request("POST", "/matlink/requests", {
        "client_id": bridge.client_id, "action": "create_demo", "payload": {}, "idempotency_key": key})


def test_bridge_role_connects_publishes_metadata_and_keeps_secret_private(project, bridge):
    assert project.studio_info()["matlink_protocol_version"] == 1
    variables = [{"name": "x", "class_name": "single", "size": [128, 1], "complex": True,
                  "eligible": True, "n_samples": 128}]
    response = json.loads(bridge.heartbeat(json.dumps(variables)))
    assert response["lease_seconds"] == 30 and response["requests"] == []
    state = project._request("GET", "/matlink")
    session = next(s for s in state["sessions"] if s["client_id"] == bridge.client_id)
    assert session["variables"] == variables and session["connected"]
    assert state["workspace"] == str(project.workspace)
    assert bridge._bridge_token not in repr(bridge)
    assert bridge._bridge_token not in json.dumps(state)


def test_second_authenticated_browser_session_can_queue_and_bridge_can_ack(project, bridge):
    browser = open_project(project.workspace, start=False)
    try:
        transfer = enqueue(browser, bridge)
        duplicate = enqueue(browser, bridge)
        assert transfer["request_id"] == duplicate["request_id"]
        first = json.loads(bridge.heartbeat("[]"))["requests"]
        repeated = json.loads(bridge.heartbeat("[]"))["requests"]
        assert first == repeated and first[0]["request_id"] == transfer["request_id"]
        bridge.complete(transfer["request_id"], json.dumps({"status": "succeeded", "result": {"input": "x", "output": "y"}}))
        assert json.loads(bridge.heartbeat("[]"))["requests"] == []
        completed = next(t for t in browser._request("GET", "/matlink")["transfers"] if t["request_id"] == transfer["request_id"])
        assert completed["status"] == "succeeded" and completed["result"] == {"input": "x", "output": "y"}
    finally:
        browser.close()
    assert project.studio_info()["matlink_protocol_version"] == 1


def test_lost_acknowledgment_can_be_retried_without_changing_outcome(project, bridge, monkeypatch):
    transfer = enqueue(project, bridge, "ack-loss")
    original = project._request
    lost = [False]

    def uncertain(method, path, data=None, **kwargs):
        result = original(method, path, data, **kwargs)
        if path.endswith("/complete") and not lost[0]:
            lost[0] = True
            raise SDKError("connection_lost", "Response lost after the service stored the completion")
        return result

    monkeypatch.setattr(project, "_request", uncertain)
    outcome = json.dumps({"status": "succeeded", "result": {"variable": "pa_gru_capture"}})
    with pytest.raises(SDKError, match="connection_lost"):
        bridge.complete(transfer["request_id"], outcome)
    bridge.complete(transfer["request_id"], outcome)
    assert json.loads(bridge.heartbeat("[]"))["requests"] == []
    stored = next(t for t in project._request("GET", "/matlink")["transfers"] if t["request_id"] == transfer["request_id"])
    assert stored["status"] == "succeeded" and stored["result"]["variable"] == "pa_gru_capture"


def test_disconnect_fails_pending_and_new_bridge_can_reconnect(project):
    old = MatlinkClient(project, "MATLAB old", "R2026a", "[]")
    transfer = enqueue(project, old, "disconnect")
    old.disconnect()
    old.disconnect()
    with pytest.raises(SDKError, match="matlink_closed"):
        old.heartbeat("[]")
    state = project._request("GET", "/matlink")
    assert next(t for t in state["transfers"] if t["request_id"] == transfer["request_id"])["status"] == "failed"
    replacement = MatlinkClient(project, "MATLAB reconnected", "R2026a", "[]")
    try:
        assert replacement.client_id != old.client_id
        assert json.loads(replacement.heartbeat("[]"))["requests"] == []
        assert project.studio_info()["matlink_protocol_version"] == 1
    finally:
        replacement.disconnect()


@pytest.mark.parametrize("version", [None, 0, 2])
def test_unsupported_service_never_attempts_to_connect(version):
    calls = []
    project = SimpleNamespace(studio_info=lambda: {"matlink_protocol_version": version},
                              _request=lambda *args, **kwargs: calls.append((args, kwargs)))
    with pytest.raises(SDKError, match="matlink_unavailable"):
        MatlinkClient(project, "MATLAB", "R2026a", "[]")
    assert calls == []


def test_invalid_metadata_is_rejected_without_losing_the_live_bridge(project, bridge):
    invalid = [{"name": "x);quit", "class_name": "double", "size": [128, 1], "complex": True,
                "eligible": True, "n_samples": 128}]
    with pytest.raises(SDKError, match="invalid_request"):
        bridge.heartbeat(json.dumps(invalid))
    assert json.loads(bridge.heartbeat("[]"))["requests"] == []


def test_current_bridge_detects_disconnect_and_project_close(project):
    borrowed = open_project(project.workspace, start=False)
    link = MatlinkClient(borrowed, "MATLAB borrowed", "R2026a", "[]")
    try:
        assert link.is_current()
        link.disconnect()
        assert not link.is_current()
        link = MatlinkClient(borrowed, "MATLAB borrowed again", "R2026a", "[]")
        assert link.is_current()
        borrowed.close()
        assert not link.is_current()
    finally:
        try:
            link.disconnect()
        except SDKError:
            pass
        borrowed.close()


def test_current_bridge_detects_service_restart(tmp_path):
    original = open_project(tmp_path / "restart")
    # Keep a separate MATLAB-side Project open: detection must use service
    # identity, not merely the closed flag of the launcher that stopped it.
    old_connection = open_project(tmp_path / "restart", start=False)
    old_link = MatlinkClient(old_connection, "Before restart", "R2026a", "[]")
    replacement = None
    new_link = None
    try:
        assert old_link.is_current()
        original.close(stop_service=True)
        assert not old_connection._closed
        assert not old_link.is_current()
        replacement = open_project(tmp_path / "restart")
        new_link = MatlinkClient(replacement, "After restart", "R2026a", "[]")
        assert new_link.is_current() and not old_link.is_current()
        assert new_link.client_id != old_link.client_id
        assert json.loads(new_link.heartbeat("[]"))["requests"] == []
    finally:
        old_connection.close()
        if new_link is not None:
            new_link.disconnect()
        if replacement is not None:
            replacement.close(stop_service=True)
        elif not original._closed:
            original.close(stop_service=True)


def test_generated_collection_preserves_all_iq_metadata_and_rejects_tampering(project, bridge):
    import numpy as np
    from opendpd.services.workspace import Workspace
    presets = project._request('GET', '/signal-generator/presets')
    configs = [dict(p['config'], n_samples=8192, length_mode='samples') for p in presets[:2]]
    signals = project._request('POST', '/signal-generator/batches', {'configs': configs})
    paired = project._request('POST', '/pa-library/datasets', {
        'input_signal_ids': [s['signal_id'] for s in signals], 'model_id': 'rapp-am-pm', 'parameters': {}})
    dataset_id = paired['dataset']['dataset_id']
    metadata, captures = bridge.dataset(dataset_id)
    assert len(captures) == len(json.loads(metadata)['captures']) == 2
    ws = Workspace.open(project.workspace)
    for serialized, x, y in captures:
        capture = json.loads(serialized)
        directory = ws.dataset_version_dir(capture['dataset_id'], 'raw-v1')
        np.testing.assert_array_equal(x, np.load(directory / 'input_iq.npy'))
        np.testing.assert_array_equal(y, np.load(directory / 'output_iq.npy'))
        assert x.shape == y.shape == (8192, 2)
        assert capture['signal']['sample_rate_hz'] > 0
        assert capture['split']['guard_samples'] == 256
    path = ws.dataset_version_dir(dataset_id, 'raw-v1') / 'input_iq.npy'
    original = path.read_bytes()
    try:
        path.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
        with pytest.raises(SDKError, match='dataset_changed'):
            bridge.dataset(dataset_id)
    finally:
        path.write_bytes(original)


def test_result_bundle_uses_saved_report_settings_and_verified_plots(project, bridge):
    import numpy as np
    n = 4096
    x = (.2 * np.exp(2j*np.pi*.123*np.arange(n))).astype(np.complex64)
    dataset = project.import_iq(x, .8*x, sample_rate_hz=80e6, bandwidth_hz=20e6, nperseg=128)
    job = project.train_pa(dataset['dataset_id'], parameters={'hidden_size': 4},
        training={'epochs': 1, 'frame_length': 32, 'frame_stride': 32, 'batch_size': 16}, device='cpu')
    job.wait(timeout=120)
    bundle = json.loads(bridge.result_bundle(job.run_id))
    assert bundle['metrics'] == job.result()['metrics']
    assert bundle['configuration'] == job.config()
    assert bundle['plots']['spectrum']['traces']
    assert bundle['plots']['time']['traces']
    assert bundle['plots']['am_am_pm']['traces']
