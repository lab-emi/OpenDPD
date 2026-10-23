"""Carrier placement (TS 38.104 5.4.1.2), cell identities, errors, truncation and oversampling, and the derived GeneratorConfig fields."""
import math
from fractions import Fraction

import numpy as np
import pytest
from pydantic import ValidationError

from opendpd.core.waveforms import nr_tm
from opendpd.core.waveforms.nr_tm import plan as plan_module
from opendpd.core.waveforms.nr_tm.plan import make_plan, nominal_spacing_khz, preset_id_for
from opendpd.core.waveforms.nr_tm.synth import baseband
from opendpd.core.waveforms.nr_tm.tables import GUARD_BAND_KHZ, MODELS, NRB, TDD_PATTERNS
from opendpd.schemas.nr_test_model import NrTestModelConfig, NrTmCarrier, NrTmNbIot
from opendpd.schemas.signal_generator import GeneratorConfig


def carriers(*cells, gaps=None, ids=None):
    return [NrTmCarrier(scs_khz=scs, bandwidth_mhz=bw, gap_before_mhz=(gaps[i] if gaps else 0), n_cell_id=(ids[i] if ids else None))
            for i, (scs, bw) in enumerate(cells)]


def recipe(*cells, model="NR-FR1-TM1.1", duplex="fdd", **kwargs):
    return NrTestModelConfig(model=model, duplex=duplex, carriers=carriers(*cells, gaps=kwargs.pop("gaps", None), ids=kwargs.pop("ids", None)), **kwargs)


def gap_formula_khz(scs1, bw1, scs2, bw2, spacing_khz):
    """Edge-to-edge distance of the occupied subcarriers with F_REF at subcarrier 6 N_RB (TS 38.104 5.3A.2 / 5.4.2.2)."""
    n1, n2 = NRB[scs1][bw1], NRB[scs2][bw2]
    return Fraction(spacing_khz) - Fraction((12 * n1 - 1) * scs1, 2) - Fraction((12 * n2 + 1) * scs2, 2)


# (case, S [kHz], gap [kHz]) from the worked examples in the research notes (5.4.1.2, 100 kHz raster)
WORKED = [
    (((15, 20), (15, 20)), 19800, 720),
    (((30, 100), (30, 100)), 99900, 1620),
    (((30, 5), (30, 100)), 51900, 780),
    (((30, 40), (30, 20)), 29700, 1440),
    (((60, 10), (60, 10)), 9900, 1980),
    (((15, 5), (15, 5)), 4800, 300),
    (((15, 3), (15, 3)), 3000, 300),
    (((15, 20), (30, 100)), 59700, Fraction(2025, 2)),
]


@pytest.mark.parametrize("cells,spacing,gap", WORKED)
def test_nominal_spacing_and_gap_match_the_worked_examples(cells, spacing, gap):
    (scs1, bw1), (scs2, bw2) = cells
    assert nominal_spacing_khz(scs1, bw1, scs2, bw2) == spacing
    assert gap_formula_khz(scs1, bw1, scs2, bw2, spacing) == gap
    plan = make_plan(recipe(*cells))
    a, b = plan.carriers
    assert b.offset_hz - a.offset_hz == spacing * 1000
    assert b.low_edge_hz - a.high_edge_hz == gap * 1000
    assert a.low_edge_hz == -b.high_edge_hz                                         # the aggregate is centred by its edges
    if cells[0] == cells[1]:
        assert a.offset_hz + b.offset_hz == a.scs_khz * 1000             # equal carriers are symmetric about DC; F_REF is half a subcarrier above the centre


def test_equal_bandwidth_pairs_follow_floor_2bw_over_0_6():
    for scs, table in NRB.items():
        for bw in table:
            expected = math.floor(2 * bw / 0.6) * 0.3 * 1000
            assert nominal_spacing_khz(scs, bw, scs, bw) == pytest.approx(expected, abs=1e-6), (scs, bw)
    assert [nominal_spacing_khz(30, bw, 30, bw) for bw in (5, 10, 20, 40, 100)] == [4800, 9900, 19800, 39900, 99900]
    assert [nominal_spacing_khz(15, bw, 15, bw) for bw in (3, 15, 25, 35, 45, 50)] == [3000, 15000, 24900, 34800, 45000, 49800]


def test_the_guard_band_is_read_at_the_larger_numerology():
    # 10 MHz@15 + 10 MHz@30: GB of 10 MHz at 30 kHz (665 kHz) is used for both; at 15 kHz it would be 312.5 kHz
    assert GUARD_BAND_KHZ[30][10] != GUARD_BAND_KHZ[15][10]
    assert nominal_spacing_khz(15, 10, 30, 10) == nominal_spacing_khz(30, 10, 15, 10) == 9900
    # 3 MHz exists at 15 kHz only: the 15 kHz guard band is the one left for both
    assert nominal_spacing_khz(15, 3, 15, 50) == nominal_spacing_khz(15, 50, 15, 3)


def test_no_pair_of_valid_cells_overlaps_and_the_tightest_gap_is_known():
    cells = [(scs, bw) for scs, table in NRB.items() for bw in table]
    assert len(cells) == 40
    smallest = None
    for scs1, bw1 in cells:
        for scs2, bw2 in cells:
            gap = gap_formula_khz(scs1, bw1, scs2, bw2, nominal_spacing_khz(scs1, bw1, scs2, bw2))
            assert gap >= 0, (scs1, bw1, scs2, bw2)
            smallest = gap if smallest is None else min(smallest, gap)
    assert smallest == Fraction(15, 2)                                             # kHz: 3 MHz@15 kHz next to 80 MHz@60 kHz


def test_nominal_spacing_is_symmetric_and_a_multiple_of_300_khz():
    for cells in ((15, 20, 30, 100), (15, 5, 60, 90), (30, 10, 60, 40), (60, 100, 15, 3), (15, 3, 30, 5)):
        a = nominal_spacing_khz(*cells)
        b = nominal_spacing_khz(cells[2], cells[3], cells[0], cells[1])
        assert a == b and a % 300 == 0, cells


def test_cell_identities_default_to_1_2_3_in_ascending_frequency_and_can_be_overridden():
    plan = make_plan(recipe((30, 10), (30, 10), (30, 10)))
    assert [c.n_cell_id for c in plan.carriers] == [1, 2, 3]
    assert [float(c.offset_hz) for c in plan.carriers] == sorted(float(c.offset_hz) for c in plan.carriers)
    plan = make_plan(recipe((30, 10), (30, 10), (30, 10), ids=[None, 500, None]))
    assert [c.n_cell_id for c in plan.carriers] == [1, 500, 3]
    plan = make_plan(recipe((30, 10), (30, 10), ids=[7, 7]))
    assert any("share the cell identity 7" in w for w in plan.warnings)
    _, grids = baseband(make_plan(recipe((30, 10), (30, 10), ids=[7, 7], subframes=2)))
    assert np.array_equal(grids[0][1].values, grids[1][1].values)                          # same identity and numerology: identical data
    assert not any("share" in w for w in make_plan(recipe((30, 10), (30, 10), ids=[7, 8])).warnings)
    for bad in (-1, 1008):
        with pytest.raises(ValidationError):
            NrTmCarrier(n_cell_id=bad)


def test_gap_before_is_a_multiple_of_300_khz_and_moves_the_carrier():
    base = make_plan(recipe((30, 10), (30, 10)))
    wide = make_plan(recipe((30, 10), (30, 10), gaps=[0, 9.3]))
    assert wide.carriers[1].offset_hz - wide.carriers[0].offset_hz == base.carriers[1].offset_hz - base.carriers[0].offset_hz + 9_300_000
    assert wide.carriers[0].low_edge_hz == -wide.carriers[1].high_edge_hz
    for ok in (0.3, 0.6, 1.5, 15.0, 60.3):
        make_plan(recipe((30, 10), (30, 10), gaps=[0, ok]))
    for bad in (0.1, 0.4, 0.31, 10.0):
        with pytest.raises(ValidationError, match="multiple of 0.3 MHz"):
            recipe((30, 10), (30, 10), gaps=[0, bad])
    with pytest.raises(ValidationError, match="first carrier"):
        recipe((30, 10), (30, 10), gaps=[0.3, 0])


def test_overlapping_carriers_are_refused(monkeypatch):
    monkeypatch.setattr(plan_module, "nominal_spacing_khz", lambda *args: 8000)          # less than the occupied width of 8.64 MHz (24 RB at 30 kHz)
    with pytest.raises(ValueError, match="overlap"):
        make_plan(recipe((30, 10), (30, 10)))


def test_mixed_numerologies_are_placed_and_flagged():
    plan = make_plan(recipe((15, 20), (30, 100), (60, 40)))
    assert [c.scs_khz for c in plan.carriers] == [15, 30, 60]
    assert {c.n_fft * c.scs_khz * 1000 for c in plan.carriers} == {plan.sample_rate_hz}      # one common sample rate
    assert [c.slots for c in plan.carriers] == [plan.subframes, 2 * plan.subframes, 4 * plan.subframes]
    assert any("different subcarrier spacings" in w and "leakage" in w for w in plan.warnings)
    for a, b in zip(plan.carriers, plan.carriers[1:]):
        assert b.low_edge_hz > a.high_edge_hz


def test_carriers_have_equal_power_spectral_density():
    for cells in (((15, 10), (15, 10)), ((30, 20), (30, 5)), ((15, 10), (60, 20))):
        r = recipe(*cells, subframes=4, oversampling=1)
        config = nr_tm.resolve(r).config
        x, _ = nr_tm.synthesize_nr_tm(config)
        plan = make_plan(r)
        spectrum = np.abs(np.fft.fftshift(np.fft.fft(x.astype(np.complex128)))) ** 2
        freq = np.fft.fftshift(np.fft.fftfreq(len(x), 1 / plan.sample_rate_hz))
        density = []
        for carrier in plan.carriers:
            lo, hi = float(carrier.low_edge_hz), float(carrier.high_edge_hz)
            margin = 0.1 * (hi - lo)
            selected = (freq > lo + margin) & (freq < hi - margin)
            density.append(spectrum[selected].mean())
        assert 10 * math.log10(max(density) / min(density)) < 0.3, cells


def test_errors_for_values_the_standard_does_not_define():
    with pytest.raises(ValidationError, match="5 MHz"):
        recipe((60, 5))
    with pytest.raises(ValidationError):
        recipe((30, 3))
    with pytest.raises(ValidationError):
        recipe((15, 60))
    with pytest.raises(ValidationError, match="two layers|Two layers"):
        recipe((30, 20), model="NR-FR1-TM3.1", layers=2)
    with pytest.raises(ValidationError, match="Output port 1"):
        recipe((30, 20), output_port=1)
    recipe((30, 20), layers=2, output_port=1)
    with pytest.raises(ValidationError, match="NB-IoT block belongs"):
        recipe((15, 20), nb_iot=NrTmNbIot())
    with pytest.raises(ValidationError, match="needs the NB-IoT block"):
        recipe((15, 20), model="NR-N-TM")
    with pytest.raises(ValidationError, match="FDD"):
        recipe((15, 20), model="NR-N-TM", duplex="tdd", nb_iot=NrTmNbIot())
    with pytest.raises(ValidationError, match="15 kHz"):
        recipe((30, 20), model="NR-N-TM", nb_iot=NrTmNbIot())
    with pytest.raises(ValidationError):
        recipe((15, 20), model="NR-N-TM", nb_iot=NrTmNbIot(), gaps=None, subframes=0)
    make_plan(recipe((15, 20), model="NR-N-TM", nb_iot=NrTmNbIot()))


def test_errors_for_captures_that_exceed_the_sample_limit():
    with pytest.raises(ValidationError, match="at most 1,000,000"):
        recipe(*([(30, 100)] * 16))                                                         # one subframe of 16 x 100 MHz is too long
    with pytest.raises(ValidationError, match="Oversampling 8"):
        recipe((30, 100), (30, 100), oversampling=8)                                       # 1.97 M samples per subframe
    recipe((30, 100), oversampling=8, subframes=1)                                         # 983,040 samples fit
    with pytest.raises(ValidationError, match="at most 1,000,000 fit, i.e. 1 subframe"):
        recipe((30, 100), oversampling=8, subframes=2)
    with pytest.raises(ValidationError, match="at most 1,000,000 fit, i.e. 8 subframes"):
        recipe((30, 100), oversampling=1, subframes=9)
    with pytest.raises(ValidationError, match="subframe"):
        recipe((30, 100), oversampling=1, subframes=20)


@pytest.mark.parametrize("cells,duplex,oversampling,subframes,truncated", [
    (((15, 5),), "fdd", 4, 10, False),            # 7.68 MS/s native: 4x keeps the 10 ms (307,200 samples)
    (((15, 20),), "fdd", 2, 10, False),           # 30.72 MS/s: 4x would be 1,228,800 samples
    (((15, 10),), "tdd", 2, 20, False),
    (((30, 20),), "fdd", 2, 10, False),
    (((30, 100),), "fdd", 1, 8, True),            # 122.88 MS/s: 10 ms are 1,228,800 samples -> 8 subframes, 983,040 samples
    (((30, 100), (30, 100)), "fdd", 1, 4, True),
    (((30, 100),), "tdd", 1, 8, True),
    (((60, 100),), "fdd", 1, 8, True),
])
def test_oversampling_and_truncation_rule(cells, duplex, oversampling, subframes, truncated):
    plan = make_plan(recipe(*cells, duplex=duplex))
    assert (plan.oversampling, plan.subframes, plan.truncated) == (oversampling, subframes, truncated)
    assert plan.n_samples == subframes * plan.sample_rate_hz // 1000 <= 1_000_000
    assert plan.sample_rate_hz == plan.native_sample_rate_hz * oversampling
    assert any("shortened" in w for w in plan.warnings) == truncated
    if truncated:
        assert (subframes + 1) * plan.sample_rate_hz // 1000 > 1_000_000                  # the largest whole number of subframes


def test_explicit_choices_are_respected_and_the_automatic_factor_follows_a_longer_capture():
    plan = make_plan(recipe((15, 10), oversampling=8, subframes=3))
    assert (plan.oversampling, plan.subframes, plan.truncated) == (8, 3, True) and plan.n_samples == 3 * 15_360 * 8
    assert any("shorter than the standard" in w for w in plan.warnings)
    auto = make_plan(recipe((15, 10), subframes=20))
    assert auto.oversampling == 2 and auto.subframes == 20 and auto.n_samples == 20 * 30_720    # 4x would not fit 20 ms
    assert make_plan(recipe((15, 10), subframes=5)).oversampling == 4
    tdd_short = make_plan(recipe((15, 10), duplex="tdd", subframes=3))
    assert any("5 ms TDD pattern" in w for w in tdd_short.warnings)
    assert make_plan(recipe((15, 5))).standard_subframes == 10 and make_plan(recipe((15, 5), duplex="tdd")).standard_subframes == 20


def all_recipes():
    cells = [(15, 5), (15, 20), (30, 10), (30, 100), (60, 10), (60, 100)]
    for model in MODELS:
        if model.nb_iot:
            yield NrTestModelConfig(model=model.id, carriers=carriers((15, 20)), nb_iot=NrTmNbIot())
            continue
        for duplex in ("fdd", "tdd"):
            for scs, bw in cells:
                yield NrTestModelConfig(model=model.id, duplex=duplex, carriers=carriers((scs, bw)))
    yield recipe((15, 10), (15, 10), (15, 10), (15, 10))
    yield recipe((15, 20), (30, 100))
    yield recipe((30, 5), (30, 100), duplex="tdd")
    yield recipe(*([(15, 10)] * 16))
    yield recipe((15, 3), (30, 5), (60, 10), gaps=[0, 1.2, 0.3])
    yield recipe((30, 20), layers=2, output_port=1)


def test_derived_fields_construct_a_plain_generator_config_and_the_complete_one():
    count = 0
    for r in all_recipes():
        derived = nr_tm.derive(r)
        fields = derived.fields
        assert set(fields) == set(nr_tm.STRUCTURAL_FIELDS)
        plain = GeneratorConfig(**fields)                                                 # the legacy validators accept the description
        full = nr_tm.resolve(r).config
        assert full.nr_test_model == r
        for name in nr_tm.STRUCTURAL_FIELDS:
            assert getattr(full, name) == getattr(plain, name), (r.model, name)
        assert plain.sample_count == derived.summary.n_samples == derived.plan.n_samples
        assert plain.sample_rate_hz == derived.summary.sample_rate_hz == derived.plan.sample_rate_hz
        assert 256 <= plain.sample_count <= 1_000_000 and plain.sample_rate_hz <= 2e9
        assert plain.fft_size <= 16384 and plain.fft_size & (plain.fft_size - 1) == 0 and plain.oversampling in (1, 2, 4, 8)
        assert plain.fft_size * plain.oversampling * min(c.scs_khz for c in r.carriers) * 1000 == plain.sample_rate_hz
        assert fields["preset_id"] == preset_id_for(r) and len(fields["preset_id"]) <= 64 and fields["preset_id"].startswith("nr-")
        count += 1
    assert count >= 125


def test_summary_describes_the_derived_signal():
    r = recipe((15, 20), (30, 100), model="NR-FR1-TM3.2", duplex="tdd")
    summary = nr_tm.derive(r).summary
    assert summary.model == "NR-FR1-TM3.2" and len(summary.carriers) == 2 and summary.oversampling == 1
    first, second = summary.carriers
    assert (first.nrb, first.subcarriers, first.n_fft, first.n_cell_id) == (106, 1272, 16384, 1)
    assert (second.nrb, second.subcarriers, second.n_fft, second.n_cell_id) == (273, 3276, 8192, 2)
    assert first.rbg_size == 8 and first.n_rbg == 7 and second.rbg_size == 16 and second.n_rbg == 9      # min(floor(0.6 N / P), (A - A % 2) / 2 + 1)
    assert second.center_offset_hz - first.center_offset_hz == 59_700_000
    assert summary.tdd_pattern is not None and len(summary.tdd_pattern) == TDD_PATTERNS[15].slots
    assert summary.duration_ms == pytest.approx(summary.subframes) and summary.n_samples == summary.subframes * 245_760
    assert [u.rnti for u in first.users] == [0, 1, 2]
    assert sum(1 for u in first.users if u.evm_measured) == 1


def test_preset_ids_are_deterministic_and_distinguish_recipes():
    a = recipe((30, 20), model="NR-FR1-TM3.1a")
    assert preset_id_for(a) == preset_id_for(recipe((30, 20), model="NR-FR1-TM3.1a"))
    ids = {preset_id_for(r) for r in all_recipes()}
    assert len(ids) == sum(1 for _ in all_recipes())
    assert preset_id_for(a).startswith("nr-tm3p1a-fdd-30k-20m-c1-")
    assert preset_id_for(recipe((15, 5), (30, 20))).startswith("nr-tm1p1-fdd-mixk-mixm-c2-")
    assert preset_id_for(recipe((30, 20), data_convention="matlab")) != preset_id_for(recipe((30, 20)))


def test_derivation_is_deterministic_and_does_not_mutate_the_recipe():
    r = recipe((15, 10), (30, 20), duplex="tdd")
    before = r.model_dump()
    one, two = nr_tm.derive(r), nr_tm.derive(r)
    assert one.fields == two.fields and one.summary == two.summary and r.model_dump() == before
    assert nr_tm.resolve(r).config.model_dump() == nr_tm.resolve(r).config.model_dump()


def test_one_gap_has_one_recipe_one_preset_id_and_one_exact_value():
    """0.9 MHz written as 3 * 0.3, as the JSON of another float library, or off by float noise is the same recipe; -0.0 is 0.0; 6 is 6.0."""
    spellings = [0.9, 3 * 0.3, 0.8999999999999999, 0.9000000000000001, 0.9 + 5e-10, "0.9", 0.9]
    recipes = [recipe((30, 10), (30, 10), gaps=[0, g]) for g in spellings]
    assert {r.carriers[1].gap_before_mhz for r in recipes} == {0.9} and all(repr(r.carriers[1].gap_before_mhz) == "0.9" for r in recipes)
    assert len({preset_id_for(r) for r in recipes}) == 1 and len({plan_module.canonical_json(r) for r in recipes}) == 1
    assert len({nr_tm.resolve(r).config.preset_id for r in recipes}) == 1
    offsets = {make_plan(r).carriers[1].offset_hz - make_plan(r).carriers[0].offset_hz for r in recipes}
    assert offsets == {make_plan(recipe((30, 10), (30, 10))).carriers[1].offset_hz - make_plan(recipe((30, 10), (30, 10))).carriers[0].offset_hz + 900_000}
    zeros = [recipe((30, 10), (30, 10), gaps=[0, g]) for g in (0, 0.0, -0.0, "0", 4e-10)]
    assert len({preset_id_for(r) for r in zeros}) == 1 and all(repr(r.carriers[1].gap_before_mhz) == "0.0" for r in zeros)
    plain = NrTestModelConfig(model="NR-FR1-TM1.1", carriers=[{"scs_khz": 30, "bandwidth_mhz": 10}, {"scs_khz": 30, "bandwidth_mhz": 10}])
    assert preset_id_for(plain) == preset_id_for(zeros[0])                                # an omitted gap is the default 0
    assert json_of(plain) == json_of(zeros[0])


def json_of(r):
    import json
    return json.dumps(r.model_dump(mode="json"), sort_keys=True)


def test_a_gap_off_the_grid_is_refused_even_by_a_hair():
    for bad in (0.30000004, 0.3000004, 0.29999996, 4e-07, 1e-7, 0.3 + 5e-9, 0.25, 9.31):
        with pytest.raises(ValidationError, match="multiple of 0.3 MHz"):
            recipe((30, 10), (30, 10), gaps=[0, bad])
    with pytest.raises(ValidationError, match="first carrier"):
        recipe((30, 10), (30, 10), gaps=[0.3, 0])
    # a record stored by an earlier release only has to be well-formed
    stored = NrTmCarrier.model_validate({"gap_before_mhz": 0.30000004}, context={"stored": True})
    assert stored.gap_before_mhz == 0.30000004


def test_the_float_fields_of_the_nb_iot_block_have_one_spelling_per_value():
    from opendpd.schemas.nr_test_model import NrTmNbIot
    spellings = [6, 6.0, "6", "6.0", 6.0000000000]
    blocks = [NrTmNbIot(boost_db=b) for b in spellings]
    assert {repr(b.boost_db) for b in blocks} == {"6.0"}
    ids = {preset_id_for(NrTestModelConfig(model="NR-N-TM", carriers=[NrTmCarrier(scs_khz=15, bandwidth_mhz=10)], nb_iot=b)) for b in blocks}
    ids.add(preset_id_for(NrTestModelConfig(model="NR-N-TM", carriers=[NrTmCarrier(scs_khz=15, bandwidth_mhz=10)], nb_iot=NrTmNbIot())))   # the default
    assert len(ids) == 1


def test_a_cell_identity_that_differs_from_the_standard_rule_is_flagged_by_carrier_as_a_studio_extension():
    rule = [w for w in make_plan(recipe((30, 10), (30, 10), (30, 10), ids=[None, 2, 3])).warnings if "Studio extension" in w]
    assert rule == []                                                                      # explicit ids that equal the rule are not deviations
    plan = make_plan(recipe((30, 10), (30, 10), (30, 10), ids=[7, 2, 1]))
    flagged = [w for w in plan.warnings if "Studio extension" in w]
    assert len(flagged) == 2
    assert flagged[0] == ("Carrier 1: cell identity 7 differs from the rule of TS 38.141-1 4.9.2.3 (N_ID^cell = n for the n-th lowest carrier, "
                          "here 1); this is a Studio extension.")
    assert flagged[1].startswith("Carrier 3: cell identity 1 differs from the rule") and "here 3" in flagged[1]
    resolution = nr_tm.resolve(recipe((30, 10), (30, 10), (30, 10), ids=[7, 2, 1], subframes=2))
    assert [w for w in resolution.summary.warnings if "Studio extension" in w] == flagged
    assert resolution.config.nr_test_model.carriers[0].n_cell_id == 7
    ordinary = make_plan(recipe((30, 10), (30, 10), ids=[8, 8]))                          # duplicates keep their own warning besides the deviation ones
    assert sum("Studio extension" in w for w in ordinary.warnings) == 2 and any("share the cell identity 8" in w for w in ordinary.warnings)


def test_tdd_with_different_subcarrier_spacings_says_whose_pattern_tdd_pattern_lists():
    plan = make_plan(recipe((15, 10), (30, 10), duplex="tdd"))
    warning, = [w for w in plan.warnings if w.startswith("TDD with different subcarrier spacings")]
    assert "tdd_pattern lists the pattern of carrier 1 (15 kHz)" in warning
    assert f"30 kHz: {' '.join(TDD_PATTERNS[30].labels())}" in warning and "15 kHz:" not in warning.split("(15 kHz)")[1]
    summary = nr_tm.resolve(recipe((15, 10), (30, 10), duplex="tdd", subframes=5)).summary
    assert summary.tdd_pattern == TDD_PATTERNS[15].labels() and warning in summary.warnings
    three, = [w for w in make_plan(recipe((60, 10), (15, 10), (30, 10), duplex="tdd")).warnings if w.startswith("TDD with different")]
    assert "carrier 1 (60 kHz)" in three and "15 kHz: " in three and "30 kHz: " in three
    for quiet in (recipe((15, 10), (30, 10)), recipe((30, 10), (30, 10), duplex="tdd"), recipe((15, 10), duplex="tdd")):
        assert not any("TDD with different" in w for w in make_plan(quiet).warnings)
