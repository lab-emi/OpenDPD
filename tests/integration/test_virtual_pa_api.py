"""Input-only export, explicit simulation, paired dataset integrity and real training."""
import io

import numpy as np
import pytest
from fastapi.testclient import TestClient

from opendpd.server.app import create_app
from opendpd.server.security import CSRF_HEADER
from opendpd.services import experiments
from opendpd.services.recipes import instantiate, run_dpd_config
from opendpd.services.workspace import Workspace

pytestmark = pytest.mark.integration


@pytest.fixture
def client(tmp_path):
    root = tmp_path / "ws"
    with TestClient(create_app(root, bootstrap_token="virtual-pa"), base_url="http://127.0.0.1:8877") as client:
        assert client.get("/api/v1/pa-library/models").status_code == 401
        auth = client.post("/api/v1/session/bootstrap", json={"token": "virtual-pa"}).json()
        client.headers[CSRF_HEADER] = auth["csrf_token"]
        client.headers["origin"] = "http://127.0.0.1:8877"
        yield client, Workspace.open_or_create(root)


def test_input_to_virtual_output_to_paired_data_and_real_pa_dpd_training(client):
    client, ws = client
    preset = client.get("/api/v1/signal-generator/presets").json()[0]["config"]
    signal = client.post("/api/v1/signal-generator/signals", json={**preset, "n_samples": 32768}).json()
    assert signal["kind"] == "pa_input"
    assert client.get("/api/v1/datasets").json() == []
    inputs = client.get("/api/v1/signal-generator/signals").json()
    assert len(inputs) == 1 and inputs[0]["kind"] == "pa_input"
    csv = client.get(inputs[0]["csv_url"])
    metadata = client.get(inputs[0]["metadata_url"]).json()
    raw = np.loadtxt(io.StringIO(csv.text), delimiter=",", skiprows=1, dtype=np.float32)
    assert raw.shape == (32768, 2) and csv.text.startswith("I,Q\n")
    assert metadata["has_pa_output"] is False and metadata["signal_role"] == "pa_input"
    assert metadata["sample_rate_hz"] == preset["sample_rate_hz"]
    models = client.get("/api/v1/pa-library/models").json()
    assert len(models) == 9 and len({m["category"] for m in models}) == 5
    request = {"input_signal_id": signal["signal_id"], "model_id": "rapp-am-pm", "parameters": {"gain": 1.6}}
    preview = client.post("/api/v1/pa-library/simulations", json=request)
    assert preview.status_code == 201, preview.text
    output = preview.json()
    assert output["analysis"]["n_samples"] == 32768
    assert client.get("/api/v1/datasets").json() == []
    assert client.post("/api/v1/pa-library/simulations", json=request).json()["simulation_id"] == output["simulation_id"]
    alternate = client.post("/api/v1/pa-library/simulations", json={**request, "parameters": {"gain": 1.7}}).json()
    assert alternate["simulation_id"] != output["simulation_id"]
    pair = np.loadtxt(io.StringIO(client.get(output["paired_csv_url"]).text), delimiter=",", skiprows=1, dtype=np.float32)
    np.testing.assert_array_equal(pair[:, :2], raw)
    assert not np.allclose(pair[:, :2], pair[:, 2:])
    simulation_metadata = client.get(output["metadata_url"]).json()
    assert simulation_metadata["has_measured_output"] is False
    assert simulation_metadata["simulation"]["model"]["equations"]
    url = f"/api/v1/pa-library/simulations/{output['simulation_id']}/dataset"
    response = client.post(url, json={"dataset_id": "explicit-pair"})
    assert response.status_code == 201, response.text
    data = response.json()
    assert data["test_samples"] == 6452
    assert data["dataset"]["origin"] == "synthetic"
    assert data["dataset"]["simulation"]["output_iq_sha256"] == output["output_iq_sha256"]
    assert client.post(url, json={"dataset_id": "explicit-pair"}).status_code == 201
    assert client.post(url, json={"dataset_id": "explicit-pair", "guard_samples": 128}).status_code == 409
    pa = instantiate("pa-gru-smoke-v1", "explicit-pair")
    pa.training.epochs, pa.training.train_samples = 1, 4096
    pa.execution.num_threads = 2
    trained_pa = experiments.execute_run(ws, experiments.create_run(ws, pa).run_id)
    assert trained_pa.status.value == "succeeded", trained_pa.error
    dpd = instantiate("dpd-gru-smoke-v1", "explicit-pair", pa_run_id=trained_pa.run_id)
    dpd.training.epochs, dpd.training.train_samples = 1, 4096
    dpd.execution.num_threads = 2
    trained_dpd = experiments.execute_run(ws, experiments.create_run(ws, dpd).run_id)
    assert trained_dpd.status.value == "succeeded", trained_dpd.error
    tested = experiments.execute_run(ws, experiments.create_run(ws, run_dpd_config("explicit-pair", trained_dpd.run_id)).run_id)
    assert tested.status.value == "succeeded", tested.error
    # Editing stored output cannot silently feed a differently labelled training dataset.
    (ws.root / "pa_simulations" / output["simulation_id"] / "output.npy").write_bytes(b"changed")
    assert client.post(url, json={"dataset_id": "tampered"}).status_code == 409


def test_custom_import_boundary_also_covers_virtual_pairs(tmp_path):
    with TestClient(create_app(tmp_path / "ws", bootstrap_token="x", allow_custom_datasets=False),
                    base_url="http://127.0.0.1:8877") as client:
        auth = client.post("/api/v1/session/bootstrap", json={"token": "x"}).json()
        client.headers[CSRF_HEADER] = auth["csrf_token"]
        client.headers["origin"] = "http://127.0.0.1:8877"
        assert client.get("/api/v1/pa-library/models").status_code == 200
        response = client.post("/api/v1/pa-library/simulations/vpa-" + "a"*64 + "/dataset",
            json={"dataset_id": "blocked"})
        assert response.status_code == 403
        assert client.post("/api/v1/pa-library/datasets", json={"input_signal_ids": ["sg-" + "a"*64], "model_id": "rapp-am-pm"}).status_code == 403
