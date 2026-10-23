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
    monkeypatch.setattr(nr_tm, "derived_fields", lambda recipe: dict(fields))
    assert GeneratorConfig(nr_test_model=r, **fields).fft_size == 64                         # a test model is described by its recipe


def test_structural_fields_are_the_documented_set():
    assert set(derive(recipe()).fields) == set(STRUCTURAL_FIELDS)


# ---------------------------------------------------------------------------------------------------------------------
# validate_assignment and the "stored" context
# ---------------------------------------------------------------------------------------------------------------------

STORED = {"stored": True}


def stale_record(r=None, **edits):
    """What a later release finds on disk: a config whose derived fields are no longer what this release derives."""
    r = r or recipe("NR-FR1-TM1.1", 15, 10)
    record = config(r).model_dump(mode="json")
    record["oversampling"] = 2 if record["oversampling"] != 2 else 4
    record.update(edits)
    return record


def test_editing_a_field_of_a_record_read_from_storage_does_not_judge_it_again():
    stale = GeneratorConfig.model_validate(stale_record(), context=STORED)
    stale.seed = 5
    stale.rms = 0.3
    stale.snr_db = 40.0
    stale.clip_db = 6.0
    stale.carrier_frequency_hz = 3.7e9
    stale.burst_on_samples = 4096
    assert (stale.seed, stale.rms, stale.snr_db) == (5, 0.3, 40.0)
    copy = stale.model_copy(update={"seed": 6})
    assert copy.seed == 6 and copy.oversampling == stale.oversampling
    assert GeneratorConfig.model_validate_json(stale.model_dump_json(), context=STORED) == stale      # and it still round-trips as stored


def test_assigning_a_field_bound_to_the_recipe_is_still_judged():
    def fresh():            # a failed assignment leaves the value in place (pydantic does not roll it back), so every case gets its own
        return config(recipe("NR-FR1-TM1.1", 15, 10))
    for field, value in (("fft_size", 4096), ("n_samples", 5000), ("preset_id", "nr-20"), ("channel_subcarriers", [100])):
        with pytest.raises(ValidationError, match=f"{field} of a standard test model is derived"):
            setattr(fresh(), field, value)
    with pytest.raises(ValidationError, match="dft_spreading is not used by standard test models"):
        fresh().dft_spreading = True
    with pytest.raises(ValidationError, match="OFDM signal"):
        fresh().waveform = "noise"
    ok = fresh()
    ok.nr_test_model = recipe("NR-FR1-TM1.1", 15, 10)                      # assigning the very same recipe back is fine
    with pytest.raises(ValidationError, match="derived from its nr_test_model recipe"):
        ok.nr_test_model = recipe("NR-FR1-TM1.1", 30, 20)                  # another recipe no longer matches the fields
    ok.nr_test_model = None                                                # without a recipe the ordinary rules apply
    assert ok.nr_test_model is None


def test_a_stored_record_without_the_derived_default_of_shared_channel_settings_is_read_and_stays_editable():
    for edits in ({"shared_channel_settings": None}, {"shared_channel_settings": None, "oversampling": 8}):
        record = stale_record(**edits)
        record["shared_channel_settings"] = None
        stored = GeneratorConfig.model_validate(record, context=STORED)         # the default is derived while the context is in force
        assert stored.shared_channel_settings is True
        assert "shared_channel_settings" in stored.model_fields_set
        stored.seed = 9
        assert stored.seed == 9 and stored.shared_channel_settings is True
    ordinary = GeneratorConfig.model_validate({**GeneratorConfig().model_dump(), "shared_channel_settings": None})
    assert ordinary.shared_channel_settings is True
    fresh = config(recipe("NR-FR1-TM1.1", 15, 10)).model_dump(mode="json")
    fresh["shared_channel_settings"] = None
    assert GeneratorConfig.model_validate(fresh).shared_channel_settings is True            # user input derives the same default


def test_the_rules_of_a_nested_configuration_are_not_mistaken_for_an_assignment():
    """Inside a batch pydantic names the parent's field ('configs') for the nested model; that must not skip the recipe check."""
    from opendpd.schemas.signal_generator import GeneratorBatchRequest
    good = config(recipe("NR-FR1-TM1.1", 15, 10)).model_dump(mode="json")
    assert len(GeneratorBatchRequest.model_validate({"configs": [good]}).configs) == 1
    for edit in ({"n_samples": 5000}, {"fft_size": 4096}, {"preset_id": "nr-20"}):
        with pytest.raises(ValidationError, match="derived from its nr_test_model recipe"):
            GeneratorBatchRequest.model_validate({"configs": [{**good, **edit}]})
    stale = stale_record()
    with pytest.raises(ValidationError, match="derived from its nr_test_model recipe"):
        GeneratorBatchRequest.model_validate({"configs": [stale]})
    assert GeneratorBatchRequest.model_validate({"configs": [stale]}, context=STORED).configs[0].oversampling == stale["oversampling"]
