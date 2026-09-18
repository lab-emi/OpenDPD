"""Dataset names/membership survive reopening and retain each capture's metadata."""
import io
import json
import zipfile

import numpy as np
import pytest
from fastapi.testclient import TestClient

from opendpd.core.waveforms.dataset_names import dataset_name
from opendpd.core.waveforms.generator_presets import presets
from opendpd.schemas.dataset import DatasetOrigin, SignalSpec
from opendpd.server.app import create_app
from opendpd.server.security import CSRF_HEADER
from opendpd.services import signal_analyzer, signal_datasets
from opendpd.services.datasets import import_arrays
from opendpd.services.workspace import Workspace

pytestmark = pytest.mark.integration


@pytest.fixture
def client(tmp_path):
    ws = Workspace.create(tmp_path / "ws")
    with TestClient(create_app(ws.root, bootstrap_token="names"), base_url="http://127.0.0.1:8877") as c:
        assert c.get("/api/v1/signal-analyzer/datasets").status_code == 401
        auth = c.post("/api/v1/session/bootstrap", json={"token": "names"}).json()
        c.headers[CSRF_HEADER] = auth["csrf_token"]
        c.headers["origin"] = "http://127.0.0.1:8877"
        yield c, ws


def configs():
    return [next(p.config for p in presets() if p.preset_id == key).model_copy(update={"n_samples": count})
            for key, count in [("nr-20", 8192), ("wifi7-80-q256-c4", 12288)]]


def generate(c, name=None, signals=None):
    body = {"configs": [s.model_dump(mode="json") for s in (signals or configs())]}
    if name:
        body["dataset_name"] = name
    response = c.post("/api/v1/signal-generator/batches", json=body)
    assert response.status_code == 201, response.text
    return response.json()


def test_names_membership_analysis_and_downloads_survive_reopen(client):
    c, ws = client
    generated = generate(c)
    expected = "syn_pa_in_nr-w7_bw20-80M_q64-256_c1-4_s30-78p125k_n2"
    assert {s["dataset_name"] for s in generated} == {expected}
    assert len({s["dataset_id"] for s in generated}) == 1
    assert generate(c) == generated
    pair_response = c.post("/api/v1/pa-library/datasets", json={
        "input_signal_ids": [s["signal_id"] for s in generated], "model_id": "rapp-am-pm"})
    assert pair_response.status_code == 201, pair_response.text
    pair = pair_response.json()["dataset"]
    assert pair["display_name"] == expected.replace("syn_pa_in_", "syn_pa_inout_") + "_rapp-am-pm"
    groups = signal_analyzer.list_datasets(Workspace.open(ws.root))
    assert len(groups) == 3
    assert {g.kind for g in groups} == {"pa_input", "pa_output", "paired"}
    assert len(next(g for g in groups if g.kind == "paired").signals) == 4
    for group in groups:
        if group.kind == "paired":
            continue
        assert [s.sample_rate_hz for s in group.signals] == [122880000, 320000000]
        assert [s.bandwidth_hz for s in group.signals] == [20000000, 80000000]
        assert [s.sample_count for s in group.signals] == [8192, 12288]
        response = c.get(group.download_url)
        assert response.status_code == 200
        assert group.name + ".zip" in response.headers["content-disposition"]
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            manifest = json.loads(z.read("manifest.json"))
            assert manifest["name"] == group.name
            assert len([n for n in z.namelist() if n.endswith(".csv")]) == 2
            for i, member in enumerate(group.signals, 1):
                csv = np.loadtxt(io.BytesIO(z.read(f"{i:02d}.csv")), delimiter=",", skiprows=1)
                assert csv.shape == (member.sample_count, 2)
                response = c.post("/api/v1/signal-analyzer/analyze", json={"source": member.source.model_dump(mode="json"),
                    "config": {"sample_rate_hz": member.sample_rate_hz, "bandwidth_hz": member.bandwidth_hz, "n_samples": 4096}})
                assert response.status_code == 200, response.text
                result = response.json()
                assert result["source"]["sample_rate_hz"] == member.sample_rate_hz
                assert result["source"]["bandwidth_hz"] == member.bandwidth_hz
                assert result["sample_count"] == 4096
    download = c.get(f"/api/v1/datasets/{pair['dataset_id']}/download")
    assert pair["display_name"] + ".zip" in download.headers["content-disposition"]
    assert not list(ws.exports_dir.iterdir())


def test_custom_names_collisions_and_aliases_never_replace_signals(client):
    c, ws = client
    name = "syn_pa_in_lab_bw20-80M_n2"
    original = generate(c, name)
    alias = generate(c, "syn_pa_in_alternate_n2")
    assert [s["signal_id"] for s in original] == [s["signal_id"] for s in alias]
    different = generate(c, name, [s.model_copy(update={"seed": 123}) for s in configs()])
    assert {s["dataset_name"] for s in different} == {name + "_2"}
    assert generate(c, name) == original
    assert signal_datasets.read_dataset(ws, original[0]["dataset_id"]).name == name
    assert len(signal_datasets.list_datasets(Workspace.open(ws.root))) == 3
    request = {"input_signal_ids": [s["signal_id"] for s in original], "model_id": "rapp-am-pm",
               "dataset_name": "syn_pa_inout_lab_n2"}
    first = c.post("/api/v1/pa-library/datasets", json=request).json()["dataset"]
    assert first["display_name"] == "syn_pa_inout_lab_n2"
    assert c.post("/api/v1/pa-library/datasets", json=request).json()["dataset"] == first
    second = c.post("/api/v1/pa-library/datasets", json={**request, "parameters": {"gain": 2.5}}).json()["dataset"]
    assert second["display_name"] == "syn_pa_inout_lab_n2_2"
    names = {d.name for d in signal_datasets.list_datasets(ws)}
    assert {"syn_pa_out_lab_n2", "syn_pa_out_lab_n2_2"} <= names
    assert c.post("/api/v1/signal-generator/batches", json={"configs": [{}], "dataset_name": "../escape"}).status_code == 422
    assert c.post("/api/v1/pa-library/datasets", json={**request, "dataset_name": "wrong_prefix"}).status_code == 422


def test_measured_dataset_and_legacy_signal_remain_selectable(client):
    c, ws = client
    x = np.random.default_rng(1).normal(size=(8192, 2)).astype(np.float32)
    import_arrays(ws, x, x * .9, dataset_id="measured-bench", display_name="Bench capture", origin=DatasetOrigin.measured,
                  signal=SignalSpec(sample_rate_hz=100e6, bandwidth_hz=25e6))
    legacy = c.post("/api/v1/signal-generator/signals", json={"n_samples": 1024}).json()
    groups = c.get("/api/v1/signal-analyzer/datasets").json()
    measured = next(d for d in groups if d["dataset_id"] == "measured-bench")
    assert measured["name"] == "Bench capture"
    assert [s["source"]["role"] for s in measured["signals"]] == ["input", "output"]
    assert all(s["sample_rate_hz"] == 100e6 and s["bandwidth_hz"] == 25e6 for s in measured["signals"])
    assert any(d["signals"][0]["source"]["source_id"] == legacy["signal_id"] for d in groups)


def test_default_name_is_order_independent_and_presets_fit_name_limit():
    assert dataset_name(configs()) == dataset_name(list(reversed(configs())))
    for preset in presets():
        name = dataset_name([preset.config], "inout", "generalized-memory")
        assert len(name) <= 96
        assert "_n1_" in name
    assert "bw1e-6M" in dataset_name([configs()[0].model_copy(update={"bandwidth_hz": 1})])


def test_named_dataset_routes_follow_public_workspace_policy():
    from opendpd.web.policy import allowed, expensive_request
    assert allowed("GET", "/signal-analyzer/datasets")
    assert expensive_request("GET", "/signal-analyzer/datasets")
    download = "/signal-generator/datasets/sds-" + "a" * 64 + "/download"
    assert allowed("GET", download)
    assert expensive_request("GET", download)


def test_removing_one_dataset_preserves_aliases_waveforms_and_restore(client):
    c, ws = client
    original = generate(c, 'syn_pa_in_original_n2')
    alias = generate(c, 'syn_pa_in_alias_n2')
    identifier = original[0]['dataset_id']
    assert c.post(f'/api/v1/signal-generator/datasets/{identifier}/archive').status_code == 200
    groups = c.get('/api/v1/signal-analyzer/datasets').json()
    assert [d['dataset_id'] for d in groups] == [alias[0]['dataset_id']]
    assert c.get(f"/api/v1/signal-generator/signals/{original[0]['signal_id']}").status_code == 200
    assert c.post(f'/api/v1/signal-generator/datasets/{identifier}/restore').status_code == 200
    assert len(c.get('/api/v1/signal-analyzer/datasets').json()) == 2
    c.post(f'/api/v1/signal-generator/datasets/{identifier}/archive')
    assert generate(c, 'syn_pa_in_original_n2') == original
    assert len(c.get('/api/v1/signal-analyzer/datasets').json()) == 2
