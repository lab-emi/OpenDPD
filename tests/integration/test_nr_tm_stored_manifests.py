"""Stored test-model manifests must stay readable after a later release changes what a recipe derives.

Every generated signal is validated again whenever it is read (one signal, the signal list, the analyzer picker, the PA library).
A manifest written by 2.4.1 holds the recipe and the derived structural fields. If a later release derived other values (or tightened a
table rule) and the stored record were judged by the new rules, the signal and every list containing it would stop working.
Records are therefore validated with the "stored" context; input from users (validate, signals, batches, resolve) keeps every rule.
"""
import io
import json
import zipfile

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from opendpd.schemas.signal_generator import GeneratedSignal, GeneratorConfig
from opendpd.server.app import create_app
from opendpd.server.security import CSRF_HEADER

pytestmark = pytest.mark.integration
API = "/api/v1"
SG = f"{API}/signal-generator"


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(tmp_path / "ws", bootstrap_token="stored"), base_url="http://127.0.0.1:8877") as c:
        auth = c.post(f"{API}/session/bootstrap", json={"token": "stored"}).json()
        c.headers[CSRF_HEADER] = auth["csrf_token"]
        c.headers["origin"] = "http://127.0.0.1:8877"
        c.workspace = tmp_path / "ws"
        yield c


def manifest_path(client, signal_id):
    return client.workspace / "signals" / signal_id / "manifest.json"


def generate_test_model(client, model="NR-FR1-TM1.1", subframes=5):
    body = {"model": model, "carriers": [{"scs_khz": 15, "bandwidth_mhz": 10}], "subframes": subframes}
    resolution = client.post(f"{SG}/nr-test-models/resolve", json=body)
    assert resolution.status_code == 200, resolution.text
    config = resolution.json()["config"]
    created = client.post(f"{SG}/signals", json=config)
    assert created.status_code == 201, created.text
    return config, created.json()["signal_id"]


def rewrite(client, signal_id, edit):
    """Edit the stored config the way a later release's derive() would have left it."""
    path = manifest_path(client, signal_id)
    manifest = json.loads(path.read_text())
    edit(manifest["config"])
    path.write_text(json.dumps(manifest))
    return manifest["config"]


def future_release(config):
    config["oversampling"] = 2 if config["oversampling"] != 2 else 4             # derived by a recipe -> rate rule that no longer exists
    config["gaussian_bt"] = 0.35                                                  # a generic default that a later release moved
    config["nr_test_model"]["carriers"][0]["bandwidth_mhz"] = 100                 # a recipe cell that the table rules of this release refuse


def test_a_stored_manifest_stays_readable_listed_downloadable_and_usable_when_a_later_release_derives_other_values(client):
    config, signal_id = generate_test_model(client)
    ordinary = client.post(f"{SG}/signals", json={**client.get(f"{SG}/presets").json()[0]["config"], "n_samples": 8192})
    assert ordinary.status_code == 201
    stored = rewrite(client, signal_id, future_release)

    detail = client.get(f"{SG}/signals/{signal_id}")
    assert detail.status_code == 200, detail.text
    assert detail.json()["config"] == stored and detail.json()["coverage"] == "standard_test_model"

    listed = client.get(f"{SG}/signals")
    assert listed.status_code == 200 and {s["signal_id"] for s in listed.json()} == {signal_id, ordinary.json()["signal_id"]}
    sources = client.get(f"{API}/signal-analyzer/sources")
    assert sources.status_code == 200 and signal_id in {s["source"]["source_id"] for s in sources.json()}
    datasets = client.get(f"{API}/signal-analyzer/datasets")
    assert datasets.status_code == 200 and signal_id in {s["dataset_id"] for s in datasets.json()}

    download = client.get(f"{SG}/signals/{signal_id}/download")
    assert download.status_code == 200
    with zipfile.ZipFile(io.BytesIO(download.content)) as archive:
        assert json.loads(archive.read("config.json")) == stored
    assert client.get(f"{SG}/signals/{signal_id}/input.csv").status_code == 200
    assert client.get(f"{SG}/signals/{signal_id}/metadata.json").json()["generator_config"] == stored

    simulation = client.post(f"{API}/pa-library/simulations", json={"input_signal_id": signal_id, "model_id": "rapp-am-pm", "parameters": {"gain": 1.6}})
    assert simulation.status_code == 201, simulation.text
    collection = client.post(f"{API}/pa-library/datasets", json={"input_signal_ids": [signal_id], "model_id": "rapp-am-pm"})
    assert collection.status_code == 201, collection.text
    assert client.post(f"{SG}/signals/{signal_id}/archive").status_code == 200
    assert client.post(f"{SG}/signals/{signal_id}/restore").json()["signal_id"] == signal_id

    # the same stored configuration sent as new input is judged by the rules of this release
    for url, body in ((f"{SG}/validate", stored), (f"{SG}/signals", stored), (f"{SG}/batches", {"configs": [stored]})):
        response = client.post(url, json=body)
        assert response.status_code == 422, (url, response.text)
        assert response.json()["error"]["code"] == "invalid_request" and "valid bandwidths" in response.text   # the recipe table rule of this release
    assert client.post(f"{SG}/nr-test-models/resolve", json=stored["nr_test_model"]).status_code == 422


def test_the_derived_values_of_a_stored_record_are_still_refused_as_user_input(client):
    config, signal_id = generate_test_model(client, subframes=2)
    stored = rewrite(client, signal_id, lambda c: c.update(oversampling=2 if c["oversampling"] != 2 else 4))
    response = client.post(f"{SG}/validate", json=stored)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request" and "derived from its nr_test_model recipe" in response.text
    assert client.post(f"{SG}/validate", json=config).status_code == 200            # the values of this release remain valid input


def test_the_stored_context_relaxes_only_what_a_release_may_change(client):
    config, signal_id = generate_test_model(client, subframes=2)
    manifest = json.loads(manifest_path(client, signal_id).read_text())
    stored = GeneratedSignal.from_stored(manifest)
    assert stored == GeneratedSignal.model_validate(manifest)                       # a consistent record is read identically either way
    broken = json.loads(json.dumps(manifest))
    broken["config"]["oversampling"] = 2 if broken["config"]["oversampling"] != 2 else 4
    with pytest.raises(ValidationError, match="derived from its nr_test_model recipe"):
        GeneratedSignal.model_validate(broken)
    assert GeneratedSignal.from_stored(broken).config.oversampling == broken["config"]["oversampling"]
    for edit in ({"sample_rate_hz": 10}, {"n_samples": 10}, {"fft_size": 1000}, {"waveform": "noise"}, {"channel_subcarriers": []}):
        invalid = json.loads(json.dumps(manifest))
        invalid["config"].update(edit)
        with pytest.raises(ValidationError):
            GeneratedSignal.from_stored(invalid)                                    # ordinary rules and the schema still apply to a stored record
    unknown = json.loads(json.dumps(manifest))
    unknown["config"]["nr_test_model"]["carriers"][0]["bandwidth_mhz"] = 1000
    with pytest.raises(ValidationError):
        GeneratedSignal.from_stored(unknown)                                        # field ranges of the recipe still apply


def test_manifests_of_signals_without_a_recipe_are_unchanged(client):
    """A 2.4.0 manifest has no nr_test_model key in config or analysis and is read, listed and rewritten exactly as before."""
    preset = client.get(f"{SG}/presets").json()[0]["config"]
    created = client.post(f"{SG}/signals", json={**preset, "n_samples": 8192})
    assert created.status_code == 201
    signal_id = created.json()["signal_id"]
    path = manifest_path(client, signal_id)
    on_disk = json.loads(path.read_text())
    assert "nr_test_model" not in on_disk["config"] and "nr_test_model" not in on_disk["analysis"]
    assert GeneratedSignal.from_stored(on_disk).model_dump(mode="json") == on_disk              # nothing is added when it is read back
    assert GeneratedSignal.model_validate(on_disk) == GeneratedSignal.from_stored(on_disk)
    assert client.get(f"{SG}/signals/{signal_id}").json() == on_disk
    assert signal_id in {s["signal_id"] for s in client.get(f"{SG}/signals").json()}
    config = GeneratorConfig.model_validate(on_disk["config"])
    assert config.nr_test_model is None and "nr_test_model" not in config.model_dump(mode="json")
