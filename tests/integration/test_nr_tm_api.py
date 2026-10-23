"""NR standard test models through the real local API: catalogue, resolve, generate, export, re-validate, PA library, datasets."""
import io
import json
import zipfile

import numpy as np
import pytest
from fastapi.testclient import TestClient

from opendpd.core.waveforms import nr_tm
from opendpd.schemas.nr_test_model import NrTestModelConfig, NrTmCarrier
from opendpd.schemas.signal_generator import GeneratorConfig
from opendpd.server.app import create_app
from opendpd.server.security import CSRF_HEADER

pytestmark = pytest.mark.integration
API = "/api/v1"
TM = f"{API}/signal-generator/nr-test-models"


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(tmp_path / "ws", bootstrap_token="nrtm"), base_url="http://127.0.0.1:8877") as c:
        assert c.get(TM).status_code == 401
        auth = c.post(f"{API}/session/bootstrap", json={"token": "nrtm"}).json()
        c.headers[CSRF_HEADER] = auth["csrf_token"]
        c.headers["origin"] = "http://127.0.0.1:8877"
        yield c


def recipe(model="NR-FR1-TM1.1", scs=15, bw=5, carriers=1, **kw):
    return {"model": model, "carriers": [{"scs_khz": scs, "bandwidth_mhz": bw} for _ in range(carriers)], **kw}


def resolve(client, body):
    response = client.post(f"{TM}/resolve", json=body)
    assert response.status_code == 200, response.text
    return response.json()


def batch(client, configs, **extra):
    response = client.post(f"{API}/signal-generator/batches", json={"configs": configs, **extra})
    assert response.status_code == 201, response.text
    return response.json()


# ---- catalogue and resolve --------------------------------------------------------------------------------------

def test_catalogue_lists_every_model_and_a_default_that_generates(client):
    response = client.get(TM)
    assert response.status_code == 200
    catalog = response.json()
    assert [m["model"] for m in catalog["models"]] == ["NR-FR1-TM1.1", "NR-FR1-TM1.2", "NR-FR1-TM2", "NR-FR1-TM2a", "NR-FR1-TM2b", "NR-FR1-TM3.1",
                                                    "NR-FR1-TM3.1a", "NR-FR1-TM3.1b", "NR-FR1-TM3.2", "NR-FR1-TM3.3", "NR-N-TM"]
    assert catalog == json.loads(nr_tm.catalog().model_dump_json())
    config = GeneratorConfig.model_validate(catalog["default_config"])
    assert config.nr_test_model == NrTestModelConfig.model_validate(catalog["default_recipe"])
    assert client.post(f"{API}/signal-generator/validate", json=catalog["default_config"]).json() == catalog["default_config"]
    assert client.get(TM).json() == catalog                                                      # static for a server version


def test_resolve_returns_the_complete_configuration_and_the_derived_summary(client):
    body = recipe("NR-FR1-TM3.1a", 30, 20, 2)
    resolution = resolve(client, body)
    local = nr_tm.resolve(NrTestModelConfig.model_validate(body))
    assert resolution == json.loads(local.model_dump_json())
    assert resolution["recipe"]["version"] == "nr-test-model-v1" and resolution["recipe"]["carriers"][1]["n_cell_id"] is None
    config, summary = resolution["config"], resolution["summary"]
    assert config["nr_test_model"] == resolution["recipe"] and config["preset_id"].startswith("nr-tm3p1a-fdd-30k-20m-c2-")
    assert [c["n_cell_id"] for c in summary["carriers"]] == [1, 2] and summary["n_samples"] == config["n_samples"]
    assert summary["carriers"][0]["nrb"] == 51 and summary["carriers"][0]["subcarriers"] == config["channel_subcarriers"][0] == 612
    assert resolve(client, body) == resolution                                                    # a pure function of the recipe
    assert client.get(f"{API}/signal-generator/signals").json() == []                            # stores nothing


@pytest.mark.parametrize("body,message", [
    (recipe("NR-FR1-TM1.1", 60, 5), "5 MHz channel is not defined for 60 kHz"),
    (recipe("NR-FR1-TM3.1", 15, 100), "100 MHz channel is not defined for 15 kHz"),
    (recipe("NR-FR1-TM3.1", 30, 100, 16), "samples"),
    (recipe("NR-FR1-TM1.2", 15, 5, layers=2), "Two layers"),
    (recipe("NR-N-TM", 30, 20), "NB-IoT"),
    (recipe(subframes=0), "greater than or equal to 1"),
    (recipe(duplex="half"), "Input should be 'fdd' or 'tdd'"),
    ({"model": "NR-FR1-TM9"}, "Input should be"),
])
def test_invalid_recipes_are_refused_with_a_readable_422_before_any_compute(client, body, message):
    response = client.post(f"{TM}/resolve", json=body)
    assert response.status_code == 422, response.text
    error = response.json()["error"]
    assert error["code"] in {"invalid_request", "invalid_signal"}
    assert message in json.dumps(error)


def _recipe_text(boost="6", carriers='[{"scs_khz":15,"bandwidth_mhz":10}]'):
    return '{"model":"NR-N-TM","nb_iot":{"position":"lower","boost_db":%s},"carriers":%s}' % (boost, carriers)


@pytest.mark.parametrize("recipe,status", [
    (_recipe_text("1" + "0" * 400), 422), (_recipe_text("-1" + "0" * 400), 422), (_recipe_text("1" * 4299), 422), (_recipe_text("1" * 4400), 400),
    (_recipe_text('"1e999"'), 422), (_recipe_text("1e999"), 422), (_recipe_text("-1e999"), 422), (_recipe_text("Infinity"), 422), (_recipe_text("NaN"), 422),
    (_recipe_text(carriers='[{"scs_khz":15,"bandwidth_mhz":10},{"scs_khz":15,"bandwidth_mhz":10,"gap_before_mhz":%s}]' % ("1" + "0" * 400)), 422),
    (_recipe_text(carriers="[" * 64 + "]" * 64), 422),
    (_recipe_text(carriers="[" * 3000 + "]" * 3000), None), (_recipe_text(carriers="[" * 30000 + "]" * 30000), 400),
], ids=["boost 400 digits", "boost -400 digits", "boost 4299 digits", "boost 4400 digits", "boost '1e999'", "boost 1e999", "boost -1e999", "boost Infinity",
        "boost NaN", "gap 400 digits", "carriers 64 deep", "carriers 3000 deep", "carriers 30000 deep"])
@pytest.mark.parametrize("route,wrap", [("nr-test-models/resolve", lambda r: r), ("signals", lambda r: '{"nr_test_model":%s}' % r),
                                        ("batches", lambda r: '{"configs":[{"nr_test_model":%s}]}' % r), ("validate", lambda r: '{"nr_test_model":%s}' % r)])
def test_hostile_numbers_in_a_recipe_are_refused_not_a_server_error(client, route, wrap, recipe, status):
    """A 400-digit integer used to escape the NB-IoT boost validator as an OverflowError (a 500 for every route that takes a recipe)."""
    body = wrap(recipe)
    if status is None:
        # CPython 3.10/3.11 reject this depth in JSON parsing; newer decoders
        # reach schema validation. Assert the exact rejection at either boundary.
        try:
            json.loads(body)
        except RecursionError:
            status = 400
        else:
            status = 422
    response = client.post(f"{API}/signal-generator/{route}", content=body.encode(), headers={"content-type": "application/json"})
    assert response.status_code == status, response.text[:300]
    if status == 422:
        assert response.json()["error"]["code"] in {"invalid_request", "invalid_signal"}


def test_resolve_needs_the_csrf_token_and_a_session(client):
    assert client.post(f"{TM}/resolve", json=recipe(), headers={CSRF_HEADER: "wrong"}).status_code == 403
    with TestClient(client.app, base_url="http://127.0.0.1:8877") as anonymous:
        assert anonymous.post(f"{TM}/resolve", json=recipe()).status_code in {401, 403}
        assert anonymous.get(TM).status_code == 401


def test_resolve_reports_a_value_error_of_the_resolver_as_invalid_signal(client, monkeypatch):
    def refuse(recipe_, **_):
        raise ValueError("cannot be derived")
    monkeypatch.setattr(nr_tm, "resolve", refuse)
    response = client.post(f"{TM}/resolve", json=recipe())
    assert response.status_code == 422 and response.json()["error"] == {"code": "invalid_signal", "message": "cannot be derived", "details": [], "hint": None}


# ---- generate, read, download, re-validate -------------------------------------------------------------------------

def test_generate_read_download_and_revalidate_a_test_model_signal(client):
    resolution = resolve(client, recipe("NR-FR1-TM3.1", 30, 20, subframes=2))
    config = {**resolution["config"], "rms": .25, "seed": 11, "snr_db": 60.0}
    summary = batch(client, [config])[0]
    assert summary["name"] == config["preset_id"] and summary["n_samples"] == config["n_samples"] and summary["dataset_name"].startswith("syn_pa_in_nrtm3p1_")
    signal = client.get(f"{API}/signal-generator/signals/{summary['signal_id']}").json()
    assert signal["coverage"] == "standard_test_model" and signal["config"]["nr_test_model"] == resolution["recipe"]
    assert signal["analysis"]["nr_test_model"]["model"] == "NR-FR1-TM3.1" and signal["analysis"]["sample_count"] == config["n_samples"]
    assert signal["analysis"]["rms"] == pytest.approx(.25, rel=1e-3)
    assert {"scs_khz", "center_offset_hz", "n_rb"} <= set(signal["analysis"]["allocation"][0])
    download = client.get(signal["download_url"])
    assert download.status_code == 200
    with zipfile.ZipFile(io.BytesIO(download.content)) as archive:
        exported = json.loads(archive.read("config.json"))
        csv = np.loadtxt(io.StringIO(archive.read("iq.csv").decode()), delimiter=",", skiprows=1)
    assert exported == signal["config"] and csv.shape == (config["n_samples"], 2)
    validated = client.post(f"{API}/signal-generator/validate", json=exported)
    assert validated.status_code == 200 and validated.json() == exported
    assert batch(client, [exported])[0]["signal_id"] == summary["signal_id"]                    # the exported configuration regenerates the same signal
    assert client.post(f"{API}/signal-generator/signals", json=exported).json()["signal_id"] == summary["signal_id"]


@pytest.mark.parametrize("edit", [{"fft_size": 2048}, {"n_samples": 4096}, {"preset_id": "nr-20"}, {"sample_rate_hz": 245.76e6}, {"cp_mode": "fixed"},
                                  {"pilot_mode": "comb"}, {"channel_subcarriers": [100]}, {"dft_spreading": True}, {"waveform": "noise"}])
def test_structural_edits_of_a_test_model_configuration_are_refused(client, edit):
    config = resolve(client, recipe(subframes=2))["config"]
    response = client.post(f"{API}/signal-generator/validate", json={**config, **edit})
    assert response.status_code == 422
    assert "recipe" in json.dumps(response.json()) or "not used by standard test models" in json.dumps(response.json()) or "OFDM signal" in json.dumps(response.json())
    assert client.post(f"{API}/signal-generator/signals", json={**config, **edit}).status_code == 422
    assert client.post(f"{API}/signal-generator/batches", json={"configs": [{**config, **edit}]}).status_code == 422


def test_editable_impairments_are_accepted_and_change_the_signal(client):
    config = resolve(client, recipe(subframes=2))["config"]
    plain = batch(client, [config])[0]
    edited = batch(client, [{**config, "burst_on_samples": 4096, "burst_off_samples": 1024, "frequency_offset_hz": 2e5, "iq_gain_db": 1.0,
                             "phase_noise_rms_deg": 1.0, "dc_i": .01, "clip_db": 8.0, "filter_enabled": True}])[0]
    assert edited["signal_id"] != plain["signal_id"] and edited["iq_sha256"] != plain["iq_sha256"]


def test_a_batch_of_models_gets_distinct_ids_one_dataset_and_a_family_name(client):
    configs = [resolve(client, recipe(model, 15, 5, subframes=2))["config"] for model in ("NR-FR1-TM1.1", "NR-FR1-TM2", "NR-FR1-TM3.3")]
    assert len({c["preset_id"] for c in configs}) == 3
    generated = batch(client, configs)
    assert len({s["signal_id"] for s in generated}) == 3 and len({s["dataset_id"] for s in generated}) == 1
    assert {s["dataset_name"] for s in generated} == {"syn_pa_in_mix_bw4p515M_q4-64_c1_n3"}
    assert batch(client, configs) == generated                                                   # idempotent
    duplicate = client.post(f"{API}/signal-generator/batches", json={"configs": [configs[0], configs[0]]})
    assert duplicate.status_code == 422
    named = batch(client, configs[:1], dataset_name="syn_pa_in_my_tm")
    assert named[0]["dataset_name"] == "syn_pa_in_my_tm"


def test_a_test_model_and_a_preset_share_a_dataset_with_both_family_tokens(client):
    preset = next(p for p in client.get(f"{API}/signal-generator/presets").json() if p["preset_id"] == "nr-20")["config"]
    tm = resolve(client, recipe("NR-FR1-TM3.1a", 30, 20, subframes=2))["config"]
    generated = batch(client, [{**preset, "n_samples": 8192}, tm])
    assert {s["dataset_name"] for s in generated} == {"syn_pa_in_nr-nrtm3p1a_bw18p39-20M_q64-256_c1_n2"}
    listing = client.get(f"{API}/signal-analyzer/datasets").json()
    assert [g["name"] for g in listing] == ["syn_pa_in_nr-nrtm3p1a_bw18p39-20M_q64-256_c1_n2"]


# ---- PA library and datasets ------------------------------------------------------------------------------------

def test_virtual_pa_simulation_and_paired_dataset_on_a_test_model_signal(client):
    config = resolve(client, recipe("NR-FR1-TM3.1", 15, 10, subframes=5))["config"]
    signal = batch(client, [config])[0]
    assert signal["n_samples"] >= 8192
    request = {"input_signal_id": signal["signal_id"], "model_id": "rapp-am-pm", "parameters": {"gain": 1.6}}
    simulation = client.post(f"{API}/pa-library/simulations", json=request)
    assert simulation.status_code == 201, simulation.text
    output = simulation.json()
    assert output["analysis"]["n_samples"] == signal["n_samples"]
    pair = np.loadtxt(io.StringIO(client.get(output["paired_csv_url"]).text), delimiter=",", skiprows=1)
    assert pair.shape == (signal["n_samples"], 4) and not np.allclose(pair[:, :2], pair[:, 2:])
    created = client.post(f"{API}/pa-library/simulations/{output['simulation_id']}/dataset", json={"dataset_id": "tm-pair"})
    assert created.status_code == 201, created.text
    dataset = created.json()["dataset"]
    assert dataset["origin"] == "synthetic"
    assert dataset["simulation"]["waveform"]["nr_test_model"]["model"] == "NR-FR1-TM3.1"
    assert client.get("/api/v1/datasets/tm-pair/sample-counts").json()["counts"]["test"] > 0


def test_paired_collection_of_a_test_model_and_a_preset_uses_the_family_names_and_labels(client):
    tm = resolve(client, recipe("NR-FR1-TM1.1", 15, 10, subframes=5))["config"]
    preset = next(p for p in client.get(f"{API}/signal-generator/presets").json() if p["preset_id"] == "nr-20")["config"]
    generated = batch(client, [tm, {**preset, "n_samples": 12288}])
    created = client.post(f"{API}/pa-library/datasets", json={"input_signal_ids": [s["signal_id"] for s in generated], "model_id": "rapp-am-pm"})
    assert created.status_code == 201, created.text
    dataset = created.json()["dataset"]
    assert dataset["display_name"] == "syn_pa_inout_nr-nrtm1p1_bw9p375-20M_q4-64_c1_s15-30k_n2_rapp-am-pm"
    assert [c["preset_id"] for c in dataset["captures"]] == [tm["preset_id"], "nr-20"]
    repeat = client.post(f"{API}/pa-library/datasets", json={"input_signal_ids": [s["signal_id"] for s in generated], "model_id": "rapp-am-pm"})
    assert repeat.status_code == 201 and repeat.json()["dataset"]["dataset_id"] == dataset["dataset_id"]


def test_the_deprecated_signal_dataset_endpoint_works_for_a_test_model(client):
    config = resolve(client, recipe("NR-FR1-TM3.1", 15, 10, subframes=5))["config"]
    signal = batch(client, [config])[0]
    response = client.post(f"{API}/signal-generator/signals/{signal['signal_id']}/dataset", json={"dataset_id": "tm-legacy-pa"})
    assert response.status_code == 201, response.text
    manifest = response.json()["dataset"]
    assert manifest["signal"]["modulation"] == "NR-FR1-TM3.1 test model, OFDM synthetic stimulus"
    assert "conformance certification" in manifest["notes"]


def test_input_exports_carry_the_recipe(client):
    config = resolve(client, recipe(subframes=2))["config"]
    signal = batch(client, [config])[0]
    metadata = client.get(signal["metadata_url"]).json()
    assert metadata["generator_config"]["nr_test_model"]["model"] == "NR-FR1-TM1.1"
    assert metadata["provenance"]["nr_test_model_source_sha256"] and metadata["provenance"]["coverage"] == "standard_test_model"
    csv = client.get(signal["csv_url"])
    assert csv.text.startswith("I,Q\n") and csv.text.count("\n") == config["n_samples"] + 1
    archived = client.post(f"{API}/signal-generator/signals/{signal['signal_id']}/archive")
    assert archived.status_code == 200
    assert client.post(f"{API}/signal-generator/signals/{signal['signal_id']}/restore").json()["signal_id"] == signal["signal_id"]


# ---- legacy behaviour unchanged ----------------------------------------------------------------------------------

def test_presets_and_ordinary_validation_are_untouched(client):
    presets = client.get(f"{API}/signal-generator/presets").json()
    assert len(presets) == 1186 and all("nr_test_model" not in p["config"] for p in presets)
    config = {**presets[0]["config"], "n_samples": 8192}
    assert client.post(f"{API}/signal-generator/validate", json=config).json() == config
    assert client.post(f"{API}/signal-generator/validate", json={**config, "fft_size": 999}).status_code == 422
    assert client.post(f"{API}/signal-generator/validate", json={**config, "pilot_mode": "explicit", "pilot_indices": [999999]}).status_code == 422
    signal = client.post(f"{API}/signal-generator/signals", json=config).json()
    assert "nr_test_model" not in signal["config"] and "nr_test_model" not in signal["analysis"] and signal["coverage"] in {"numerology", "custom"}


def test_a_default_recipe_object_round_trips_through_the_generator_config():
    r = NrTestModelConfig(carriers=[NrTmCarrier()])
    config = nr_tm.resolve(r).config
    assert GeneratorConfig.model_validate_json(config.model_dump_json()) == config


# ---- every parameter family of the recipe reaches a generated signal ---------------------------------------------------

FEATURES = {
    "frontend-shaped body with explicit nulls": {"version": "nr-test-model-v1", "model": "NR-FR1-TM1.1", "duplex": "fdd", "layers": 1, "output_port": 0,
        "carriers": [{"scs_khz": 30, "bandwidth_mhz": 10, "n_cell_id": None, "gap_before_mhz": 0}], "data_convention": "standard",
        "pn_scope": "dl_symbols", "nb_iot": None, "oversampling": None, "subframes": 2},
    "two layers, second port": recipe("NR-FR1-TM1.1", 30, 10, layers=2, output_port=1, subframes=2),
    "TDD, all-symbol PN scope": recipe("NR-FR1-TM3.2", 30, 10, duplex="tdd", pn_scope="all_symbols", subframes=5),
    "MATLAB data convention": recipe("NR-FR1-TM2", 15, 10, data_convention="matlab", subframes=3),
    "mixed numerologies": {"model": "NR-FR1-TM3.1b", "carriers": [{"scs_khz": 15, "bandwidth_mhz": 10}, {"scs_khz": 30, "bandwidth_mhz": 20}], "subframes": 2},
    "non-contiguous sub-block gap and explicit cell ids": {"model": "NR-FR1-TM3.3", "subframes": 2, "carriers": [
        {"scs_khz": 15, "bandwidth_mhz": 10, "n_cell_id": 7}, {"scs_khz": 15, "bandwidth_mhz": 10, "n_cell_id": 9, "gap_before_mhz": 3.0}]},
    "oversampling 8 and a single subframe": recipe("NR-FR1-TM1.2", 15, 5, oversampling=8, subframes=1),
    "NB-IoT in band at both edges": {"model": "NR-N-TM", "nb_iot": {"position": "both", "boost_db": 9}, "subframes": 2,
        "carriers": [{"scs_khz": 15, "bandwidth_mhz": 10}, {"scs_khz": 15, "bandwidth_mhz": 10}]},
}


@pytest.mark.parametrize("name", list(FEATURES))
def test_each_recipe_option_resolves_generates_and_round_trips(client, name):
    resolution = resolve(client, FEATURES[name])
    config, summary = resolution["config"], resolution["summary"]
    assert len(config["preset_id"]) <= 64 and config["preset_id"].startswith("nr-")
    signal = batch(client, [config])[0]
    detail = client.get(f"{API}/signal-generator/signals/{signal['signal_id']}").json()
    assert detail["coverage"] == "standard_test_model" and detail["analysis"]["sample_count"] == summary["n_samples"] == config["n_samples"]
    assert detail["analysis"]["nr_test_model"] == summary and detail["config"] == config
    assert len(detail["analysis"]["allocation"]) == len(FEATURES[name]["carriers"])
    # ideal demodulation recovers the grid; carriers of different numerologies leak into each other (the summary warns about it)
    assert detail["analysis"]["evm_percent"] < (5 if name == "mixed numerologies" else 1e-2)
    assert client.post(f"{API}/signal-generator/validate", json=detail["config"]).json() == detail["config"]
    assert resolve(client, detail["config"]["nr_test_model"]) == resolution                     # the stored recipe resolves to the same configuration


@pytest.mark.parametrize("model", ["NR-FR1-TM1.1", "NR-FR1-TM1.2", "NR-FR1-TM2", "NR-FR1-TM2a", "NR-FR1-TM2b", "NR-FR1-TM3.1", "NR-FR1-TM3.1a",
                                   "NR-FR1-TM3.1b", "NR-FR1-TM3.2", "NR-FR1-TM3.3", "NR-N-TM"])
@pytest.mark.parametrize("duplex", ["fdd", "tdd"])
def test_every_model_in_both_duplex_modes_is_generated_with_a_distinct_id(tmp_path, model, duplex):
    from opendpd.services import signal_generator as service
    from opendpd.services.workspace import Workspace
    if model == "NR-N-TM" and duplex == "tdd":
        pytest.skip("NR-N-TM is FDD only")
    body = recipe(model, 15 if model == "NR-N-TM" else 30, 10, duplex=duplex, subframes=2, **({"nb_iot": {"position": "lower", "boost_db": 6}} if model == "NR-N-TM" else {}))
    config = nr_tm.resolve(NrTestModelConfig.model_validate(body)).config
    ws = Workspace.create(tmp_path / "ws")
    result = service.generate(ws, config)
    assert result.coverage == "standard_test_model" and result.analysis.nr_test_model.model == model and result.analysis.evm_percent < 1e-2
    assert result.signal_id != service.generate(ws, config.model_copy(update={"seed": 5})).signal_id
