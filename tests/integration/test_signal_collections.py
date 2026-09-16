"""Multi-rate captures, one-step simulation, portable exports and admission boundaries."""
import io
import json
import os
import subprocess
import sys
import zipfile

import numpy as np
import pytest
from fastapi.testclient import TestClient

from opendpd.core.virtual_pa import catalog
from opendpd.core.waveforms.generator_presets import presets
from opendpd.server.app import create_app
from opendpd.server.security import CSRF_HEADER
from opendpd.services.workspace import Workspace

pytestmark = pytest.mark.integration


@pytest.fixture
def client(tmp_path):
    ws = Workspace.create(tmp_path / "ws")
    with TestClient(create_app(ws.root, bootstrap_token="collection"), base_url="http://127.0.0.1:8877") as c:
        auth = c.post("/api/v1/session/bootstrap", json={"token": "collection"}).json()
        c.headers[CSRF_HEADER] = auth["csrf_token"]
        c.headers["origin"] = "http://127.0.0.1:8877"
        yield c, ws


def inputs(c):
    configs = [next(p.config for p in presets() if p.preset_id == key).model_dump(mode="json")
               for key in ("nr-20", "wifi7-20")]
    configs[0]["n_samples"], configs[1]["n_samples"] = 8192, 12288
    response = c.post("/api/v1/signal-generator/batches", json={"configs": configs})
    assert response.status_code == 201, response.text
    return [p["signal_id"] for p in response.json()]


@pytest.mark.parametrize("model", catalog(), ids=lambda m: m.model_id)
def test_collection_zip_replays_each_capture_exactly(client, tmp_path, model):
    c, ws = client
    ids = inputs(c)
    request = {"input_signal_ids": ids, "model_id": model.model_id}
    response = c.post("/api/v1/pa-library/datasets", json=request)
    assert response.status_code == 201, response.text
    ds = response.json()["dataset"]
    assert [capture["n_samples"] for capture in ds["captures"]] == [8192, 12288]
    assert [capture["sample_rate_hz"] for capture in ds["captures"]] == [122880000, 80000000]
    assert c.post("/api/v1/pa-library/datasets", json=request).json()["dataset"] == ds
    response = c.get(f"/api/v1/datasets/{ds['dataset_id']}/download")
    assert response.status_code == 200, response.text[:200] if response.status_code != 200 else ""
    assert response.headers["content-type"] == "application/zip"
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        assert "simulate_pa.py" in z.namelist()
        csvs = [name for name in z.namelist() if name.endswith(".csv")]
        assert len(csvs) == 2 and all('/' not in name for name in z.namelist())
        z.extractall(tmp_path / "export")
    for filename, capture in zip(csvs, ds["captures"]):
        target = tmp_path / "export"
        subprocess.run([sys.executable, str(target / "simulate_pa.py"), str(target / filename), "--output", str(target / "output.csv")],
                       check=True, capture_output=True, env={**os.environ, "PYTHONPATH": ""})
        pair = np.loadtxt(target / filename, delimiter=",", skiprows=1, dtype=np.float32)
        output = np.loadtxt(target / "output.csv", delimiter=",", skiprows=1, dtype=np.float32)
        np.testing.assert_array_equal(pair[:, 2:], output)
        single = c.get(f"/api/v1/datasets/{capture['dataset_id']}/download?collection=false")
        assert single.status_code == 200
        np.testing.assert_array_equal(np.loadtxt(io.StringIO(single.text), delimiter=",", skiprows=1, dtype=np.float32), pair)
        metadata = json.loads((target / filename.replace('.csv', '.json')).read_text())
        assert metadata["dataset"]["signal"]["sample_rate_hz"] == capture["sample_rate_hz"]
    assert not list(ws.exports_dir.glob('dataset-download-*'))


def test_invalid_batch_and_capture_fail_before_dataset_writes(client):
    c, ws = client
    assert c.post('/api/v1/signal-generator/batches', json={'configs': [{}]*17}).status_code == 422
    assert c.post('/api/v1/signal-generator/batches', json={'configs': [{}]*2}).status_code == 422
    ids = inputs(c)
    assert c.post('/api/v1/pa-library/datasets', json={'input_signal_ids': ids*2, 'model_id': 'rapp-am-pm'}).status_code == 422
    response = c.post('/api/v1/pa-library/datasets', json={'input_signal_ids': [ids[0], 'sg-'+'0'*64], 'model_id': 'rapp-am-pm'})
    assert response.status_code == 409
    assert not ws.list_datasets()
    short = c.post('/api/v1/signal-generator/signals', json={'n_samples': 512}).json()
    response = c.post('/api/v1/pa-library/datasets', json={'input_signal_ids': [ids[0], short['signal_id']], 'model_id': 'rapp-am-pm'})
    assert response.status_code == 409 and not ws.list_datasets()


def test_failed_second_import_rolls_back_only_owned_datasets(client, monkeypatch):
    from opendpd.services import virtual_pa
    c, ws = client
    ids = inputs(c)
    original = virtual_pa.import_arrays
    calls = []
    def failing(*args, **kwargs):
        calls.append(True)
        if len(calls) == 2:
            raise ValueError('injected disk failure')
        return original(*args, **kwargs)
    monkeypatch.setattr(virtual_pa, 'import_arrays', failing)
    with pytest.raises(ValueError, match='disk failure'):
        c.post('/api/v1/pa-library/datasets', json={'input_signal_ids': ids, 'model_id': 'rapp-am-pm'})
    assert not ws.list_datasets()
    assert not list(ws.datasets_dir.iterdir())
