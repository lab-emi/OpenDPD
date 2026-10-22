"""GeneratorConfig with an NR test model recipe: structural fields are derived, the serialised form of ordinary configs is unchanged."""
import json

import pytest
from pydantic import ValidationError

from opendpd.core.waveforms.nr_tm import STRUCTURAL_FIELDS, derive
from opendpd.schemas.nr_test_model import NrTestModelConfig, NrTmCarrier
from opendpd.schemas.signal_generator import GeneratorConfig


def recipe(model="NR-FR1-TM3.1a", scs=30, bw=100, **kw):
    return NrTestModelConfig(model=model, carriers=[NrTmCarrier(scs_khz=scs, bandwidth_mhz=bw)], **kw)


def config(r, **edits):
    return GeneratorConfig(nr_test_model=r, **{**derive(r).fields, **edits})


def test_a_derived_configuration_is_accepted_and_keeps_the_recipe():
    r = recipe()
    c = config(r)
    assert c.nr_test_model == r and c.preset_id.startswith("nr-tm3p1a-fdd-30k-100m-c1-")
    assert GeneratorConfig.model_validate_json(c.model_dump_json()) == c
    assert json.loads(c.model_dump_json())["nr_test_model"]["model"] == "NR-FR1-TM3.1a"


def test_the_serialised_form_of_an_ordinary_configuration_has_no_recipe_key():
    plain = GeneratorConfig()
    assert "nr_test_model" not in plain.model_dump() and "nr_test_model" not in plain.model_dump(mode="json")
    assert "nr_test_model" not in plain.model_dump_json() and "nr_test_model" not in json.loads(plain.model_dump_json())
    assert GeneratorConfig.model_validate_json(plain.model_dump_json()) == plain          # releases before 2.4.1 round-trip


@pytest.mark.parametrize("field,value", [("fft_size", 2048), ("preset_id", "nr-20"), ("n_samples", 4096), ("sample_rate_hz", 245.76e6),
                                         ("channel_subcarriers", [100]), ("channel_modulations", [64]), ("oversampling", 2),
                                         ("bandwidth_hz", 50e6), ("pilot_mode", "comb"), ("cp_mode", "fixed"), ("dc_null", True),
                                         ("channel_gap_bins", 4), ("length_mode", "duration"), ("channel_power_db", [3.0])])
def test_every_structural_field_must_equal_the_derivation(field, value):
    with pytest.raises(ValidationError, match=f"{field} of a standard test model is derived.*Edit the recipe"):
        config(recipe(), **{field: value})


def test_waveform_must_stay_ofdm():
    with pytest.raises(ValidationError, match="OFDM signal"):
        config(recipe(), waveform="noise")


@pytest.mark.parametrize("field,value", [("dft_spreading", True), ("cp_samples", 0), ("payload_mode", "prbs9"), ("pilot_indices", [3]),
                                         ("duration_ms", 2.0), ("modulation_order", 16)])
def test_generic_waveform_fields_that_the_model_ignores_keep_their_defaults(field, value):
    with pytest.raises(ValidationError, match=f"{field} is not used by standard test models"):
        config(recipe(), **{field: value})


def test_editable_fields_may_change_and_the_generic_validators_still_apply():
    r = recipe()
    c = config(r, rms=.5, seed=7, carrier_frequency_hz=3.7e9, filter_enabled=True, burst_on_samples=4096, phase_offset_deg=10,
               phase_noise_rms_deg=1, frequency_offset_hz=2e6, iq_gain_db=.5, iq_phase_deg=2, dc_i=.01, dc_q=-.01, snr_db=40, clip_db=6)
    assert c.rms == .5 and c.filter_enabled
    assert config(r, filter_enabled=False).filter_enabled is False
    for edits, message in [({"rms": 2}, "less than or equal to 1"), ({"burst_on_samples": 4096, "burst_ramp_samples": 4096}, "Burst ramps"),
                           ({"frequency_offset_hz": 80e6}, "inside Nyquist"), ({"snr_db": -1}, "greater than or equal to 0")]:
        with pytest.raises(ValidationError, match=message):
            config(r, **edits)


def test_the_extent_is_checked_against_nyquist_through_the_symmetric_bandwidth():
    r = recipe("NR-FR1-TM1.1", 15, 5)
    fields = derive(r).fields
    limit = fields["sample_rate_hz"] / 2 - fields["bandwidth_hz"] / 2
    assert config(r, frequency_offset_hz=limit * .999).frequency_offset_hz > 0
    with pytest.raises(ValidationError, match="inside Nyquist"):
        config(r, frequency_offset_hz=limit * 1.001)


def test_the_legacy_allocation_rules_do_not_apply_to_test_models(monkeypatch):
    import opendpd.core.waveforms.nr_tm as nr_tm

    r = recipe()
    fields = {**derive(r).fields, "fft_size": 64, "channel_subcarriers": [3276], "channel_gap_bins": 1024, "bandwidth_hz": 2.0e6,
              "sample_rate_hz": 3.0e6, "n_samples": 1000}
    with pytest.raises(ValidationError, match="do not fit the FFT"):
        GeneratorConfig(**fields)                                                           # an ordinary configuration is checked
    monkeypatch.setattr(nr_tm, "derive", lambda recipe: type("Derived", (), {"fields": fields})())
    assert GeneratorConfig(nr_test_model=r, **fields).fft_size == 64                         # a test model is described by its recipe


def test_structural_fields_are_the_documented_set():
    assert set(derive(recipe()).fields) == set(STRUCTURAL_FIELDS)
