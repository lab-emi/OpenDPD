"""Service integration of the NR standard test models: ids, coverage, provenance, exports, pre-checks, dataset names."""
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
import scipy

import opendpd.core.waveforms.generator as generator_module
from opendpd.core.waveforms import nr_tm
from opendpd.core.waveforms.dataset_names import dataset_name, standard_model_family
from opendpd.core.waveforms.generator_presets import presets
from opendpd.schemas.nr_test_model import NrTestModelConfig, NrTmCarrier, NrTmNbIot
from opendpd.schemas.signal_generator import GeneratedSignal, GeneratorAnalysis, GeneratorBatchRequest, GeneratorConfig
from opendpd.services import signal_generator as service
from opendpd.services.workspace import Workspace, WorkspaceError

# GeneratorConfig / GeneratorAnalysis as released in 2.4.0 (pinned here, not read from the code under test).
CONFIG_KEYS_2_4_0 = {
    "version", "preset_id", "waveform", "sample_rate_hz", "bandwidth_hz", "carrier_frequency_hz", "length_mode", "n_samples", "duration_ms", "seed",
    "rms", "fft_size", "oversampling", "channel_subcarriers", "channel_modulations", "channel_power_db", "shared_channel_settings",
    "channel_gap_bins", "dc_null", "pilot_mode", "pilot_spacing", "pilot_indices", "pilot_boost_db", "cp_mode", "cp_samples", "modulation_order",
    "samples_per_symbol", "rrc_rolloff", "rrc_span_symbols", "psk_order", "payload_mode", "payload_bits", "dft_spreading", "fsk_deviation_hz",
    "gaussian_bt", "tone_frequency_hz", "tone_count", "multitone_phase", "phase_offset_deg", "phase_noise_rms_deg", "burst_on_samples",
    "burst_off_samples", "burst_ramp_samples", "frequency_offset_hz", "iq_gain_db", "iq_phase_deg", "dc_i", "dc_q", "snr_db", "filter_enabled",
    "clip_db"}
ANALYSIS_KEYS_2_4_0 = {
    "sample_count", "duration_ms", "sample_rate_hz", "subcarrier_spacing_hz", "useful_symbol_us", "cp_lengths_samples", "complete_symbols",
    "trailing_samples", "active_carriers", "data_carriers", "pilot_carriers", "rms", "peak", "papr_db", "mean_power_dbfs",
    "occupied_bandwidth_99_hz", "dc_magnitude", "evm_percent", "evm_symbols", "evm_per_symbol_percent", "evm_subcarrier_indices",
    "evm_per_subcarrier_percent", "time_us", "time_i", "time_q", "time_envelope", "frequency_mhz", "psd_dbfs_hz", "constellation_i",
    "constellation_q", "reference_i", "reference_q", "ccdf_db", "ccdf_probability", "allocation", "notes"}
PROVENANCE_KEYS_2_4_0 = {"generator_source_sha256", "numpy_version", "scipy_version", "physical_measurement", "sample_format", "coverage",
                         "full_protocol_frames", "config"}


@pytest.fixture
def ws(tmp_path):
    return Workspace.create(tmp_path / "ws")


def recipe(model="NR-FR1-TM1.1", scs=15, bw=5, carriers=1, **kw):
    return NrTestModelConfig(model=model, carriers=[NrTmCarrier(scs_khz=scs, bandwidth_mhz=bw) for _ in range(carriers)], **kw)


def tm_config(r=None, **edits):
    return nr_tm.resolve(r or recipe(subframes=2), **edits).config


def preset(key, samples=8192):
    return next(p.config for p in presets() if p.preset_id == key).model_copy(update={"n_samples": samples})


def legacy_id(config):
    """The identifier formula of every release before 2.4.1, written out independently of the service."""
    base = Path(generator_module.__file__).parent
    source = hashlib.sha256(b"".join((base / name).read_bytes() for name in ("generator.py", "modulation.py", "generator_presets.py"))).hexdigest()
    text = json.dumps(config.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return "sg-" + hashlib.sha256((source + np.__version__ + scipy.__version__ + text).encode()).hexdigest()


# ---- identity of ordinary signals is unchanged --------------------------------------------------------------------

@pytest.mark.parametrize("config", [preset("nr-20"), GeneratorConfig(n_samples=8192), preset("wifi7-80-q256-c4", 12288),
                                    GeneratorConfig(waveform="tone", n_samples=4096, filter_enabled=False)],
                         ids=["nr-20@8192", "default@8192", "wifi7", "tone"])
def test_ids_of_configurations_without_a_test_model_follow_the_legacy_formula(ws, config):
    assert set(config.model_dump(mode="json")) == CONFIG_KEYS_2_4_0
    assert service.signal_identifier(config, service._generator_source_hash()) == legacy_id(config)
    result = service.generate(ws, config)
    assert result.signal_id == legacy_id(config)
    assert service.generate(ws, config).signal_id == result.signal_id                       # idempotent
    assert result.coverage in {"numerology", "custom"}


def test_manifests_and_provenance_of_ordinary_signals_have_the_keys_of_2_4_0(ws):
    result = service.generate(ws, preset("nr-20"))
    target = service.directory(ws, result.signal_id)
    manifest = json.loads((target / "manifest.json").read_text())
    assert set(manifest["config"]) == CONFIG_KEYS_2_4_0 and set(manifest["analysis"]) == ANALYSIS_KEYS_2_4_0
    assert set(json.loads((target / "provenance.json").read_text())) == PROVENANCE_KEYS_2_4_0
    assert GeneratedSignal.model_validate(manifest) == result                                   # old manifests still load unchanged
    assert set(json.loads(result.model_dump_json())["analysis"]) == ANALYSIS_KEYS_2_4_0
    assert "nr_test_model" not in json.loads(GeneratorAnalysis.model_validate(manifest["analysis"]).model_dump_json())


# ---- test-model signals -------------------------------------------------------------------------------------------

def test_a_test_model_id_adds_the_hash_of_the_test_model_sources(ws):
    config = tm_config()
    sources = hashlib.sha256(b"".join(path.read_bytes() for path in nr_tm.SOURCE_FILES)).hexdigest()
    assert len(nr_tm.SOURCE_FILES) >= 8 and all(path.is_file() for path in nr_tm.SOURCE_FILES)
    base = Path(generator_module.__file__).parent
    source = hashlib.sha256(b"".join((base / n).read_bytes() for n in ("generator.py", "modulation.py", "generator_presets.py"))).hexdigest()
    text = json.dumps(config.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    expected = "sg-" + hashlib.sha256((source + np.__version__ + scipy.__version__ + text + sources).encode()).hexdigest()
    result = service.generate(ws, config)
    assert result.signal_id == expected != legacy_id(config)
    assert service.generate(ws, config) == result


def test_every_editable_field_and_every_recipe_field_changes_the_id(ws):
    base = tm_config()
    ids = {service.signal_identifier(base, "s", "t")}
    for edit in ({"rms": .3}, {"seed": 1}, {"snr_db": 30.0}, {"filter_enabled": True}, {"carrier_frequency_hz": 3.7e9}, {"burst_on_samples": 4096},
                 {"frequency_offset_hz": 1e5}, {"clip_db": 6.0}):
        ids.add(service.signal_identifier(base.model_copy(update=edit), "s", "t"))
    for other in (recipe(subframes=3), recipe("NR-FR1-TM3.1", subframes=2), recipe(subframes=2, data_convention="matlab"), recipe(bw=10, subframes=2),
                  recipe(scs=30, bw=10, subframes=2), recipe(carriers=2, subframes=2)):
        ids.add(service.signal_identifier(tm_config(other), "s", "t"))
    assert len(ids) == 15


def test_a_test_model_signal_is_synthesised_by_the_test_model_package(ws):
    result = service.generate(ws, tm_config())
    assert result.coverage == "standard_test_model" and result.config.nr_test_model.model == "NR-FR1-TM1.1"
    summary = result.analysis.nr_test_model
    assert summary.model == "NR-FR1-TM1.1" and summary.n_samples == result.analysis.sample_count == result.config.n_samples
    assert result.analysis.evm_percent < 1e-3 and result.analysis.evm_symbols > 0
    entry = result.analysis.allocation[0]
    assert {"scs_khz", "bandwidth_mhz", "n_rb", "n_cell_id", "center_offset_hz", "evm_percent", "subcarrier_step"} <= set(entry)
    assert all(isinstance(entry[key], list) for key in ("subcarriers", "pilots"))
    assert service.read_signal(ws, result.signal_id) == result
    assert result.config.preset_id.startswith("nr-tm1p1-fdd-15k-5m-c1-")
    assert service.list_inputs(ws)[0].name == result.config.preset_id
    notes = " ".join(result.analysis.notes)
    assert "TS 38.141-1" in notes and "Not a conformance waveform or standard test model" not in notes


def test_provenance_records_the_recipe_and_the_source_hash(ws):
    result = service.generate(ws, tm_config())
    provenance = json.loads((service.directory(ws, result.signal_id) / "provenance.json").read_text())
    assert set(provenance) == PROVENANCE_KEYS_2_4_0 | {"nr_test_model_source_sha256", "nr_test_model", "nr_test_model_definition"}
    assert provenance["nr_test_model_source_sha256"] == hashlib.sha256(b"".join(p.read_bytes() for p in nr_tm.SOURCE_FILES)).hexdigest()
    assert NrTestModelConfig.model_validate(provenance["nr_test_model"]) == result.config.nr_test_model
    assert provenance["coverage"] == "standard_test_model" and provenance["physical_measurement"] is False
    assert provenance["full_protocol_frames"] is False and "not a conformance certification" in provenance["nr_test_model_definition"]


def test_exports_describe_the_test_model_and_round_trip_the_configuration(ws):
    result = service.generate(ws, tm_config(recipe(carriers=2, subframes=2)))
    with zipfile.ZipFile(service.export_signal(ws, result.signal_id)) as archive:
        assert set(archive.namelist()) == {"iq.npy", "manifest.json", "provenance.json", "iq.csv", "config.json", "README.txt"}
        readme = archive.read("README.txt").decode()
        config = GeneratorConfig.model_validate_json(archive.read("config.json"))
    assert config == result.config and config.nr_test_model == result.config.nr_test_model
    assert "3GPP TS 38.141-1 standard test model NR-FR1-TM1.1 (FDD; 2 carriers: 5 MHz at 15 kHz, 5 MHz at 15 kHz)" in readme
    assert "Not a conformance waveform or standard test model" not in readme and "stimulus only" in readme
    plain = service.generate(ws, preset("nr-20"))
    with zipfile.ZipFile(service.export_signal(ws, plain.signal_id)) as archive:
        legacy = archive.read("README.txt").decode()
    assert "standard test model" in legacy and "3GPP TS 38.141-1" not in legacy                  # ordinary signals keep their wording
    metadata = json.loads(service.export_input(ws, result.signal_id, "metadata").read_text())
    assert metadata["generator_config"]["nr_test_model"]["model"] == "NR-FR1-TM1.1" and metadata["sample_rate_hz"] == result.config.sample_rate_hz


def test_dataset_creation_describes_the_stimulus_as_a_test_model_implementation(ws):
    from opendpd.schemas.signal_generator import GeneratorDatasetRequest
    result = service.generate(ws, tm_config(recipe("NR-FR1-TM3.1", 15, 10, subframes=5)))
    assert result.analysis.sample_count >= 8192
    response = service.create_dataset(ws, result.signal_id, GeneratorDatasetRequest(dataset_id="tm-pa"))
    dataset = response.dataset
    assert dataset.signal.modulation == "NR-FR1-TM3.1 test model, OFDM synthetic stimulus"
    assert "baseband implementation of the 3GPP TS 38.141-1 test model definition" in dataset.notes
    assert "standard conformance claim" not in dataset.notes
    assert dataset.simulation["waveform"]["nr_test_model"]["model"] == "NR-FR1-TM3.1"
    again = service.create_dataset(ws, result.signal_id, GeneratorDatasetRequest(dataset_id="tm-pa"))
    assert again.dataset.dataset_id == dataset.dataset_id                                          # idempotent re-creation


def test_ordinary_dataset_wording_and_provenance_are_unchanged(ws):
    from opendpd.schemas.signal_generator import GeneratorDatasetRequest
    result = service.generate(ws, preset("nr-20"))
    dataset = service.create_dataset(ws, result.signal_id, GeneratorDatasetRequest(dataset_id="plain-pa")).dataset
    assert dataset.signal.modulation == "OFDM synthetic stimulus"
    assert dataset.notes.startswith("SYNTHETIC stimulus and illustrative nonlinear memory PA. No physical capture, calibrated RF power or standard conformance claim.")
    assert set(dataset.simulation["waveform"]) == CONFIG_KEYS_2_4_0


# ---- pre-checks and errors ---------------------------------------------------------------------------------------

def test_batch_precheck_is_test_model_aware_and_nothing_is_generated_when_one_config_fails(ws, monkeypatch):
    good, other = tm_config(), preset("nr-20")
    batch = service.generate_batch(ws, GeneratorBatchRequest(configs=[good, other]))
    assert [s.name for s in batch] == [good.preset_id, "nr-20"] and batch[0].dataset_name.startswith("syn_pa_in_nr-nrtm1p1_")
    broken = other.model_copy(update={"pilot_mode": "explicit", "pilot_indices": [999999], "preset_id": "broken"})
    before = len(service.list_inputs(ws))
    with pytest.raises(WorkspaceError, match="Explicit pilot indices"):
        service.generate_batch(ws, GeneratorBatchRequest(configs=[tm_config(recipe("NR-FR1-TM3.1", subframes=3)), broken]))
    assert len(service.list_inputs(ws)) == before
    # the recipe is re-derived by the pre-check: a recipe that stops resolving fails before any synthesis
    request = GeneratorBatchRequest(configs=[tm_config(recipe("NR-FR1-TM2", subframes=3))])

    def refuse(_):
        raise ValueError("recipe no longer resolves")
    monkeypatch.setattr(nr_tm, "derive", refuse)
    with pytest.raises(WorkspaceError, match="no longer resolves"):
        service.generate_batch(ws, request)


def test_unfinished_features_surface_as_workspace_errors(ws, monkeypatch):
    def unfinished(_):
        raise NotImplementedError("NR-N-TM: not available")
    monkeypatch.setattr(nr_tm, "synthesize_nr_tm", unfinished)
    with pytest.raises(WorkspaceError, match="not available"):
        service.generate(ws, tm_config())
    assert service.list_inputs(ws) == []


def test_precheck_accepts_ordinary_and_test_model_configurations():
    service.precheck(preset("nr-20"))
    service.precheck(GeneratorConfig(waveform="noise", n_samples=1024))
    service.precheck(tm_config())
    with pytest.raises(ValueError, match="Explicit pilot indices"):
        service.precheck(preset("nr-20").model_copy(update={"pilot_mode": "explicit", "pilot_indices": [999999]}))


# ---- coverage ----------------------------------------------------------------------------------------------------

def test_coverage_literal_for_test_models_only():
    assert service.signal_coverage(tm_config()) == "standard_test_model"
    assert service.signal_coverage(preset("nr-20")) == "numerology"
    assert service.signal_coverage(GeneratorConfig()) == "custom"


# ---- dataset names: shared with frontend/src/utils/datasetNames.test.ts -------------------------------------------

@pytest.mark.parametrize("model,token", [("NR-FR1-TM1.1", "nrtm1p1"), ("NR-FR1-TM1.2", "nrtm1p2"), ("NR-FR1-TM2", "nrtm2"), ("NR-FR1-TM2a", "nrtm2a"),
                                         ("NR-FR1-TM2b", "nrtm2b"), ("NR-FR1-TM3.1", "nrtm3p1"), ("NR-FR1-TM3.1a", "nrtm3p1a"),
                                         ("NR-FR1-TM3.1b", "nrtm3p1b"), ("NR-FR1-TM3.2", "nrtm3p2"), ("NR-FR1-TM3.3", "nrtm3p3"), ("NR-N-TM", "nrntm")])
def test_family_token_of_every_model(model, token):
    assert standard_model_family(model) == token


def test_dataset_names_of_test_models():
    tm31a = tm_config(recipe("NR-FR1-TM3.1a", 30, 100))
    two = tm_config(recipe("NR-FR1-TM1.1", 30, 20, 2))
    assert dataset_name([tm31a]) == "syn_pa_in_nrtm3p1a_bw98p31M_q256_c1_n1"
    assert dataset_name([two]) == "syn_pa_in_nrtm1p1_bw38p19M_q4_c2_n1"
    assert dataset_name([two], "inout", "rapp-am-pm") == "syn_pa_inout_nrtm1p1_bw38p19M_q4_c2_n1_rapp-am-pm"
    assert dataset_name([tm31a, preset("nr-20")]) == "syn_pa_in_nr-nrtm3p1a_bw20-98p31M_q64-256_c1_n2"
    assert dataset_name([preset("nr-20"), tm31a]) == dataset_name([tm31a, preset("nr-20")])
    assert dataset_name([preset("nr-20"), preset("wifi7-80-q256-c4", 12288)]) == "syn_pa_in_nr-w7_bw20-80M_q64-256_c1-4_s30-78p125k_n2"
    assert dataset_name([tm31a, two, preset("nr-20")]).startswith("syn_pa_in_mix_")
    assert dataset_name([preset("nr-20")]) == "syn_pa_in_nr_bw20M_q64_c1_n1"
    ntm = nr_tm.resolve(recipe("NR-N-TM", 15, 10, nb_iot=NrTmNbIot())).config
    assert dataset_name([ntm]) == "syn_pa_in_nrntm_bw9p375M_q4_c1_n1"
    assert dataset_name([tm31a, ntm]) == "syn_pa_in_nrntm-nrtm3p1a_bw9p375-98p31M_q4-256_c1_s15-30k_n2"


def test_no_legacy_ofdm_allocation_is_computed_for_a_test_model(monkeypatch):
    config = tm_config()
    monkeypatch.setattr(generator_module, "allocation", lambda *_a, **_k: pytest.fail("the legacy OFDM allocation must not run for a test model"))
    service.precheck(config)
    with pytest.raises(pytest.fail.Exception):
        service.precheck(preset("nr-20"))                                                          # ordinary configurations still use it


def test_a_cell_identity_override_is_warned_about_but_keeps_the_coverage_label():
    """A free cell identity is a Studio extension of TS 38.141-1 4.9.2.3: the summary says so; the signal stays a standard test model."""
    r = NrTestModelConfig(model="NR-FR1-TM1.1", subframes=2, carriers=[NrTmCarrier(scs_khz=15, bandwidth_mhz=5, n_cell_id=7),
                                                                           NrTmCarrier(scs_khz=15, bandwidth_mhz=5, n_cell_id=1)])
    config = nr_tm.resolve(r).config
    assert service.signal_coverage(config) == "standard_test_model"
    x, analysis = nr_tm.synthesize_nr_tm(config)
    flagged = [w for w in analysis.nr_test_model.warnings if "Studio extension" in w]
    assert [w.split(":")[0] for w in flagged] == ["Carrier 1", "Carrier 2"] and all("TS 38.141-1 4.9.2.3" in w for w in flagged)
    assert any("Studio extension" in note for note in analysis.notes)                       # the notes carry the warnings
