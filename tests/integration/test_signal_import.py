"""Custom waveform upload → immutable PA input → simulation → real training."""
import io
import zipfile

import numpy as np
import pytest
from fastapi.testclient import TestClient

from opendpd.server.app import create_app
from opendpd.server.security import CSRF_HEADER
from opendpd.services import experiments
from opendpd.services.datasets import load_version_arrays
from opendpd.services.recipes import instantiate, run_dpd_config

pytestmark = pytest.mark.integration


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(tmp_path / "ws", bootstrap_token="import", monitor_resources=False,
                    start_sweeps=False), base_url="http://127.0.0.1:8877") as c:
        c.headers["origin"] = "http://127.0.0.1:8877"
        assert c.post("/api/v1/signal-generator/import", json={}).status_code == 401
        auth = c.post("/api/v1/session/bootstrap", json={"token": "import"}).json()
        assert c.post("/api/v1/signal-generator/import", json={}).status_code == 403
        c.headers[CSRF_HEADER] = auth["csrf_token"]
        yield c


def upload(client, data):
    response = client.post("/api/v1/signal-analyzer/upload", files={"file": ("my-waveform.csv", data, "text/csv")})
    assert response.status_code == 201, response.text
    return {"upload_id": response.json()["source"]["source_id"], "dataset_name": "usr_pa_in_custom_n1",
            "sample_rate_hz": 20e6, "bandwidth_hz": 5e6}


@pytest.mark.parametrize("header,row,fmt,expected", [
    ("I,Q", "0.123456789012345,0.3", "iq", [0.123456789012345, .3]),
    ("signal", "0.2+0.3j", "complex", [.2, .3]),
    ("signal", "-0.25", "real", [-.25, 0]),
])
def test_import_formats_preserve_samples_metadata_and_export(client, header, row, fmt, expected):
    request = {**upload(client, header + "\n" + (row + "\n") * 512), "sample_format": fmt}
    response = client.post("/api/v1/signal-generator/import", json=request)
    assert response.status_code == 201, response.text
    info = response.json()
    assert info["origin"] == "uploaded" and info["n_samples"] == 512
    assert client.post("/api/v1/signal-generator/import", json=request).json() == info
    assert client.get("/api/v1/datasets").json() == []
    signal = client.get("/api/v1/signal-generator/signals/" + info["signal_id"]).json()
    assert signal["config"]["waveform"] == "imported"
    assert signal["analysis"]["evm_percent"] is None
    raw = np.loadtxt(io.StringIO(client.get(info["csv_url"]).text), delimiter=",", skiprows=1)
    np.testing.assert_array_equal(raw, np.tile(np.asarray(expected, dtype=np.float32), (512, 1)))
    metadata = client.get(info["metadata_url"]).json()
    assert metadata["generator_config"] is None
    assert metadata["import_config"]["sample_rate_hz"] == 20e6
    assert metadata["origin"] == "uploaded" and metadata["has_pa_output"] is False
    assert metadata["provenance"]["csv_sha256"] == request["upload_id"][3:]
    collection = next(d for d in client.get("/api/v1/signal-analyzer/datasets").json() if d["dataset_id"] == info["dataset_id"])
    assert collection["kind"] == "pa_input" and collection["signals"][0]["origin"] == "uploaded"
    assert client.get(collection["download_url"]).content.startswith(b"PK")
    archive = zipfile.ZipFile(io.BytesIO(client.get(signal["download_url"]).content))
    assert "cannot be regenerated" in archive.read("README.txt").decode()
    legacy = client.post("/api/v1/signal-generator/signals/" + info["signal_id"] + "/dataset", json={"dataset_id": "legacy"})
    assert legacy.status_code == 409 and "PA Library" in legacy.text


def test_import_keeps_full_capture_and_selected_columns(client):
    # Longer than the analyzer's default 262,144-sample window, with a distinct tail.
    n = 262145
    data = "time,Q,I\n" + "0,-0.2,0.1\n" * (n - 1) + "0,-0.4,0.3\n"
    request = {**upload(client, data), "i_column": 2, "q_column": 1}
    result = client.post("/api/v1/signal-generator/import", json=request)
    assert result.status_code == 201, result.text
    info = result.json()
    assert info["n_samples"] == n
    raw = np.load(client.app.state.ws.root / "signals" / info["signal_id"] / "iq.npy", allow_pickle=False)
    np.testing.assert_array_equal(raw[0], np.asarray([.1, -.2], dtype=np.float32))
    np.testing.assert_array_equal(raw[-1], np.asarray([.3, -.4], dtype=np.float32))
    assert raw.shape == (n, 2)


def test_import_validation_and_tamper_detection(client):
    request = upload(client, "I,Q\n" + "0.2,0.3\n" * 512)
    for invalid in ({"sample_rate_hz": 0}, {"bandwidth_hz": 21e6}, {"sample_rate_hz": 3e9},
                    {"i_column": 1, "q_column": 1}, {"upload_id": "../outside"},
                    {"dataset_name": "../outside"}, {"sample_format": "auto"}):
        assert client.post("/api/v1/signal-generator/import", json={**request, **invalid}).status_code == 422
    assert client.post("/api/v1/signal-generator/import", json={**request, "q_column": 7}).status_code == 409
    assert client.post("/api/v1/signal-generator/import", json={**request, "upload_id": "sa-" + "0" * 64}).status_code == 409
    complex_request = upload(client, "signal\n" + "0.1+0.2j\n" * 512)
    assert client.post("/api/v1/signal-generator/import", json={**complex_request, "sample_format": "real"}).status_code == 409
    zero = upload(client, "signal\n" + "0\n" * 512)
    assert client.post("/api/v1/signal-generator/import", json={**zero, "sample_format": "real"}).status_code == 409
    info = client.post("/api/v1/signal-generator/import", json=request).json()
    ws = client.app.state.ws
    (ws.root / "signals" / info["signal_id"] / "iq.npy").write_bytes(b"changed")
    assert client.post("/api/v1/pa-library/simulations", json={"input_signal_id": info["signal_id"], "model_id": "rapp-am-pm"}).status_code == 409
    (ws.root / "signal_uploads" / request["upload_id"] / "samples.npy").write_bytes(b"changed")
    assert client.post("/api/v1/signal-generator/import", json=request).status_code == 409


def test_imported_waveform_runs_through_pa_and_training(client):
    rng = np.random.default_rng(59)
    x = rng.normal(0, .15, (32768, 2))
    csv = io.StringIO()
    np.savetxt(csv, x, delimiter=",", header="I,Q", comments="", fmt="%.17g")
    request = upload(client, csv.getvalue())
    info = client.post("/api/v1/signal-generator/import", json=request).json()
    response = client.post("/api/v1/pa-library/datasets", json={"input_signal_ids": [info["signal_id"]],
        "model_id": "rapp-am-pm", "dataset_name": "syn_pa_inout_custom_rapp_n1"})
    assert response.status_code == 201, response.text
    manifest = response.json()["dataset"]
    assert manifest["origin"] == "synthetic" and manifest["simulation"]["input_origin"] == "uploaded"
    assert manifest["simulation"]["input_provenance"]["csv_sha256"] == request["upload_id"][3:]
    assert manifest["signal"]["sample_rate_hz"] == 20e6
    dataset_id, ws = manifest["dataset_id"], client.app.state.ws
    raw, output, _ = load_version_arrays(ws, dataset_id, "raw-v1")
    np.testing.assert_array_equal(raw, x.astype(np.float32))
    assert np.isfinite(output).all() and not np.allclose(output, raw)
    pa = instantiate("pa-gru-smoke-v1", dataset_id)
    pa.training.epochs, pa.training.train_samples, pa.execution.num_threads = 1, 4096, 2
    trained_pa = experiments.execute_run(ws, experiments.create_run(ws, pa).run_id)
    assert trained_pa.status.value == "succeeded", trained_pa.error
    dpd = instantiate("dpd-gru-smoke-v1", dataset_id, pa_run_id=trained_pa.run_id)
    dpd.training.epochs, dpd.training.train_samples, dpd.execution.num_threads = 1, 4096, 2
    trained_dpd = experiments.execute_run(ws, experiments.create_run(ws, dpd).run_id)
    assert trained_dpd.status.value == "succeeded", trained_dpd.error
    tested = experiments.execute_run(ws, experiments.create_run(ws, run_dpd_config(dataset_id, trained_dpd.run_id)).run_id)
    assert tested.status.value == "succeeded", tested.error


def test_disabled_custom_imports_block_promotion(tmp_path):
    with TestClient(create_app(tmp_path / "ws", allow_custom_datasets=False), base_url="http://127.0.0.1:8877") as c:
        assert c.post("/api/v1/signal-generator/import", json={}).status_code == 403
