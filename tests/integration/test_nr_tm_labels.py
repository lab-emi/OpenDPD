"""Pickers and dataset captures show the recipe of a test-model signal; every other signal keeps the label of earlier releases."""
import pytest
from fastapi.testclient import TestClient

from opendpd.server.app import create_app
from opendpd.server.security import CSRF_HEADER

pytestmark = pytest.mark.integration
API = "/api/v1"
SG = f"{API}/signal-generator"


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(tmp_path / "ws", bootstrap_token="labels"), base_url="http://127.0.0.1:8877") as c:
        auth = c.post(f"{API}/session/bootstrap", json={"token": "labels"}).json()
        c.headers[CSRF_HEADER] = auth["csrf_token"]
        c.headers["origin"] = "http://127.0.0.1:8877"
        yield c


def test_a_test_model_is_named_by_its_recipe_and_a_preset_by_its_catalogue_entry(client):
    body = {"model": "NR-FR1-TM3.1a", "carriers": [{"scs_khz": 15, "bandwidth_mhz": 10}, {"scs_khz": 15, "bandwidth_mhz": 10}], "subframes": 5}
    tm = client.post(f"{SG}/nr-test-models/resolve", json=body).json()["config"]
    presets = {p["preset_id"]: p for p in client.get(f"{SG}/presets").json()}
    preset = {**presets["nr-20"]["config"], "n_samples": 12288}
    members = client.post(f"{SG}/batches", json={"configs": [tm, preset]}).json()
    assert {m["name"] for m in members} == {tm["preset_id"], "nr-20"}            # the identifier is unchanged

    label = "NR-FR1-TM3.1a · 2 × 10 MHz · 15 kHz · FDD"
    sources = {s["source"]["source_id"]: s["label"] for s in client.get(f"{API}/signal-analyzer/sources").json()}
    assert sorted(sources.values()) == sorted([label + " · PA input", "nr-20 · PA input"])
    collection = next(d for d in client.get(f"{API}/signal-analyzer/datasets").json() if len(d["signals"]) == 2)
    assert sorted(s["label"] for s in collection["signals"]) == sorted([label, "nr-20"])

    created = client.post(f"{API}/pa-library/datasets", json={"input_signal_ids": [m["signal_id"] for m in members], "model_id": "rapp-am-pm"})
    assert created.status_code == 201, created.text
    captures = {c["preset_id"]: c["label"] for c in created.json()["dataset"]["captures"]}
    assert captures == {tm["preset_id"]: label, "nr-20": presets["nr-20"]["label"]}
    outputs = [d for d in client.get(f"{API}/signal-analyzer/datasets").json() if d["kind"] == "pa_output" and len(d["signals"]) == 2]
    assert sorted(s["label"] for s in outputs[0]["signals"]) == sorted([label, "nr-20"])
