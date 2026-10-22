"""NR-N-TM (TS 38.141-1 4.9.2.2.9) end to end: NR-FR1-TM1.1 with NB-IoT carriers punctured into the host carrier(s).

The reference is built at grid level from the two validated building blocks (``grid.build_carrier_grid`` for NR-FR1-TM1.1 and
``nb_iot.nb_iot_frame`` for the TS 36.141 N-TM frame, MATLAB-checked in test_nr_tm_nbiot.py / test_nr_tm_grid_parity.py) and the
power formulas of the standard written out here; the synthesized waveform is demodulated and compared with it. Gains, RE counts,
N_ID values and the L/R tables are written out independently of the code under test.
"""
import functools
import math
import os
import subprocess
import sys
import time

import numpy as np
import pytest
from pydantic import ValidationError

from opendpd.core.waveforms import nr_tm
from opendpd.core.waveforms.nr_tm import nb_iot
from opendpd.core.waveforms.nr_tm.grid import build_carrier_grid
from opendpd.core.waveforms.nr_tm.modulate import ofdm_demodulate, phasor
from opendpd.core.waveforms.nr_tm.plan import make_plan, validate_recipe
from opendpd.core.waveforms.nr_tm.synth import baseband
from opendpd.core.waveforms.nr_tm.tables import MODEL_BY_ID
from opendpd.schemas.nr_test_model import NrTestModelConfig, NrTmCarrier, NrTmNbIot

def peak_rss_mb(snippet: str) -> float:
    """Peak resident size of a fresh interpreter that runs ``snippet``.

    ``ru_maxrss`` of the pytest process is the maximum over every earlier test of the session (torch, training, ...), so it cannot
    measure the memory one synthesis needs; a child process starts from nothing.
    """
    code = "import resource\n" + snippet + "\nprint(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)"
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(path for path in sys.path if path)}
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=300)
    assert done.returncode == 0, done.stderr[-2000:]
    return int(done.stdout.split()[-1]) / 1024                       # ru_maxrss is in kB on Linux


# TS 38.104 table 5.3.2-1 (15 kHz) and the positions of research/nbiot_ntm.md 5.2: BW (MHz) -> N_RB, L_NB-IoT, R_NB-IoT
NRB = {3: 15, 5: 25, 10: 52, 15: 79, 20: 106, 25: 133, 30: 160, 35: 188, 40: 216, 45: 242, 50: 270}
LOWER = {3: 2, 5: 2, 10: 0, 15: 4, 20: 2, 25: 1, 30: 4, 35: 3, 40: 2, 45: 0, 50: 4}
UPPER = {3: 12, 5: 22, 10: 50, 15: 74, 20: 102, 25: 131, 30: 159, 35: 183, 40: 212, 45: 240, 50: 269}
SUBFRAME_RES = [108, 108, 108, 108, 108, 105, 108, 108, 108, 116]       # NB-IoT signal REs per subframe, host v_shift 0, 1, 3 or 4
TOL = 1e-6
TM11 = MODEL_BY_ID["NR-FR1-TM1.1"]


def recipe(*bandwidths, position="lower", boost=6.0, ids=None, **kwargs):
    carriers = [NrTmCarrier(scs_khz=15, bandwidth_mhz=bw, n_cell_id=ids[i] if ids else None) for i, bw in enumerate(bandwidths)]
    return NrTestModelConfig(model="NR-N-TM", carriers=carriers, nb_iot=NrTmNbIot(position=position, boost_db=boost), **kwargs)


@functools.lru_cache(maxsize=None)
def _synthesize(bandwidths, position, boost, ids, subframes):
    r = recipe(*bandwidths, position=position, boost=boost, ids=ids, subframes=subframes)
    config = nr_tm.resolve(r).config
    x, analysis = nr_tm.synthesize_nr_tm(config)
    return r, make_plan(r), config, x, analysis


def synthesize(bandwidths, position="lower", boost=6.0, ids=None, subframes=None):
    """(recipe, plan, config, samples, analysis); results are cached per argument set."""
    return _synthesize(tuple(bandwidths), position, float(boost), None if ids is None else tuple(ids), subframes)


def demodulate(plan, x, carrier):
    z = x.astype(np.complex128)
    if carrier.offset_hz:
        z = z * np.conj(phasor(carrier.offset_hz, plan.sample_rate_hz, plan.n_samples))
    return ofdm_demodulate(z, carrier.mu, carrier.n_fft, carrier.subcarriers, carrier.slots, gain=math.sqrt(carrier.n_fft))


def reference(nrb, cell_id, subframes, n_id=None, rbs=(), boost=6.0):
    """(grid, NB-IoT RE mask, EVM reference mask, nr_gain, nb_gain) of one carrier built from the validated pieces.

    Standard power formulas: P_avg = 1 per RE, P_NB = 10^(X/10), P_NR = (N_tot - N_NB P_NB) / (N_tot - N_NB) over the REs of a 10 ms frame.
    """
    base = build_carrier_grid(TM11, nrb, 15, "fdd", cell_id, subframes)
    grid, nb_mask = base.values.copy(), np.zeros(base.values.shape, dtype=bool)
    if n_id is None:
        return grid, nb_mask, base.data_mask.copy(), 1.0, 1.0
    values, mask = nb_iot.nb_iot_frame(n_id, host_cell_id=cell_id)
    n_total, n_nb = nrb * 12 * 14 * 10, len(rbs) * int(mask.sum())
    p_nb = 10 ** (boost / 10)
    nr_gain, nb_gain = math.sqrt((n_total - n_nb * p_nb) / (n_total - n_nb)), math.sqrt(p_nb)
    grid *= nr_gain
    for slot in range(subframes):                               # slot s of a 15 kHz FDD carrier is NB-IoT subframe s mod 10
        for rb in rbs:
            sel = mask[slot % 10].T
            block = grid[12 * rb:12 * rb + 12, 14 * slot:14 * slot + 14]
            block[sel] = nb_gain * values[slot % 10].T[sel]
            nb_mask[12 * rb:12 * rb + 12, 14 * slot:14 * slot + 14] = sel
    return grid, nb_mask, base.data_mask & ~nb_mask, nr_gain, nb_gain


def assert_matches(spectrum, expected, tol=TOL):
    """The demodulated spectrum equals scale x the expected grid for one real scale (the RMS normalisation); returns the scale."""
    scale = float(np.vdot(expected, spectrum).real / np.vdot(expected, expected).real)
    assert np.max(np.abs(spectrum - scale * expected)) < tol * scale
    return scale


def test_every_bandwidth_and_position_equals_tm11_times_gain_plus_boosted_nb_iot_frame():
    for bw, nrb in NRB.items():
        for position, rbs in (("lower", [LOWER[bw]]), ("upper", [UPPER[bw]]), ("both", [LOWER[bw], UPPER[bw]])):
            r, plan, config, x, analysis = synthesize((bw,), position)
            assert len(x) == config.sample_count == plan.n_samples and plan.subframes == 10 and not plan.truncated
            assert analysis.nr_test_model.nb_iot_rbs == rbs == [rb for _, rb in plan.nb_iot_placements], (bw, position)
            carrier = plan.carriers[0]
            assert (carrier.nrb, carrier.n_cell_id) == (nrb, 1)
            grid, nb_mask, _, _, _ = reference(nrb, 1, 10, n_id=103, rbs=rbs)               # lowest host, cell identity 1
            assert_matches(demodulate(plan, x, carrier), grid)
            assert nb_mask.sum() == 1085 * len(rbs)
            assert analysis.evm_percent < 1e-4 and np.isclose(analysis.rms, config.rms, rtol=1e-5)


def test_re_budget_and_power_ratios_measured_from_the_waveform():
    """NB-IoT REs are X dB above the average RE and 10 log10 g (N_tot - N_NB) / (N_tot - N_NB g) dB above the NR REs."""
    for bw, nrb in NRB.items():
        for position, n_carriers in (("lower", 1), ("both", 2)):
            for boost in ((6.0, 3.0) if bw in (3, 20, 50) else (6.0,)):
                r, plan, config, x, analysis = synthesize((bw,), position, boost)
                spectrum = demodulate(plan, x, plan.carriers[0])
                rbs = [LOWER[bw], UPPER[bw]][:n_carriers] if position == "both" else [LOWER[bw]]
                _, nb_mask, _, nr_gain, nb_gain = reference(nrb, 1, 10, n_id=103, rbs=rbs, boost=boost)
                n_nb, n_total = 1085 * n_carriers, nrb * 12 * 140
                assert nb_mask.sum() == n_nb
                power = np.abs(spectrum) ** 2
                p_nb, p_nr, p_all = power[nb_mask].mean(), power[~nb_mask].mean(), power.mean()
                assert 10 * math.log10(p_nb / p_all) == pytest.approx(boost, abs=1e-4)
                g = 10 ** (boost / 10)
                expected_db = boost + 10 * math.log10((n_total - n_nb) / (n_total - n_nb * g))
                assert 10 * math.log10(p_nb / p_nr) == pytest.approx(expected_db, abs=1e-4)
                assert 10 * math.log10(nr_gain ** 2) == pytest.approx(10 * math.log10((n_total - n_nb * g) / (n_total - n_nb)), abs=1e-12)
                entry = analysis.allocation[0]["nb_iot"]
                assert (entry["nb_iot_re"], entry["nr_re"], entry["frame_re"]) == (n_nb, n_total - n_nb, n_total)
                assert entry["boost_db"] == boost and entry["nb_iot_gain_db"] == pytest.approx(boost, abs=1e-9)
                assert entry["nr_gain_db"] == pytest.approx(20 * math.log10(nr_gain), abs=1e-9)
                assert [c["rb"] for c in entry["carriers"]] == rbs and [c["re_per_subframe"] for c in entry["carriers"]] == [SUBFRAME_RES] * len(rbs)
                if boost == 6.0 and position == "lower":                # documented: the NR REs of the host are less than 0.7 dB below the others
                    assert -0.7 < entry["nr_gain_db"] < 0


def test_the_frame_energy_is_that_of_a_carrier_without_nb_iot():
    for bw in (3, 10, 50):
        r, plan, config, x, analysis = synthesize((bw,), "both")
        plain = NrTestModelConfig(model="NR-FR1-TM1.1", carriers=[NrTmCarrier(scs_khz=15, bandwidth_mhz=bw)])
        plain_plan = make_plan(plain)
        with_nb, grids = baseband(plan)
        without, plain_grids = baseband(plain_plan)
        assert plan.sample_rate_hz == plain_plan.sample_rate_hz and len(with_nb) == len(without)
        # TM1.1 loads every RE of the frame with unit power, so the energy balance of the standard makes both grids equal in energy
        for grid in (grids[0][1], plain_grids[0][1]):
            assert np.sum(np.abs(grid.values) ** 2) == pytest.approx(NRB[bw] * 1680, rel=1e-12)
        # in the time domain the unequal energies of the OFDM symbols are weighted by their different cyclic-prefix lengths
        assert np.mean(np.abs(with_nb) ** 2) == pytest.approx(np.mean(np.abs(without) ** 2), rel=5e-3)


@pytest.mark.parametrize("bw", [3, 10, 20])
def test_psd_in_the_punctured_rb_rises_by_the_frame_average_of_the_boosted_re_power(bw):
    """Welch PSD of the waveform: the punctured RB against the other RBs, compared with the same ratio of the expected grid."""
    nrb = NRB[bw]
    r, plan, config, x, analysis = synthesize((bw,), "lower")
    grid, nb_mask, _, _, _ = reference(nrb, 1, 10, n_id=103, rbs=[LOWER[bw]])
    freq = np.asarray(analysis.frequency_mhz) * 1e6
    psd = 10 ** (np.asarray(analysis.psd_dbfs_hz) / 10)
    k_total = 12 * nrb

    def rb_psd(rb):                                   # inner ten subcarriers of the RB (row k sits at (k - K/2) x 15 kHz)
        low, high = (12 * rb + 0.5 - k_total / 2) * 15e3, (12 * rb + 10.5 - k_total / 2) * 15e3
        return psd[(freq >= low) & (freq <= high)].mean()

    def rb_power(rb):
        return np.mean(np.abs(grid[12 * rb:12 * rb + 12]) ** 2)

    others = [rb for rb in range(1, nrb - 1) if rb != LOWER[bw]]
    measured = 10 * math.log10(rb_psd(LOWER[bw]) / np.median([rb_psd(rb) for rb in others]))
    expected = 10 * math.log10(rb_power(LOWER[bw]) / np.median([rb_power(rb) for rb in others]))
    assert 3.5 < expected < 6 and measured == pytest.approx(expected, abs=0.2)


def test_lower_upper_both_pick_the_outermost_eligible_rbs_of_the_lowest_and_highest_carrier():
    for bw, nrb in NRB.items():
        eligible = nb_iot.eligible_rbs(nrb)
        assert eligible[0] == LOWER[bw] and eligible[-1] == UPPER[bw]
        for position, expected in (("lower", [(0, LOWER[bw])]), ("upper", [(0, UPPER[bw])]), ("both", [(0, LOWER[bw]), (0, UPPER[bw])])):
            assert make_plan(recipe(bw, position=position)).nb_iot_placements == tuple(expected)
    plan = make_plan(recipe(10, 20, 5, position="both"))
    assert plan.nb_iot_placements == ((0, 0), (2, 22))                 # lowest eligible RB of the lowest, highest of the highest carrier
    assert make_plan(recipe(10, 20, position="lower")).nb_iot_placements == ((0, 0),)
    assert make_plan(recipe(10, 20, position="upper")).nb_iot_placements == ((1, 102),)


def test_the_two_nb_iot_carriers_of_one_host_share_the_n_id_and_the_content():
    r, plan, config, x, analysis = synthesize((10,), "both")
    spectrum = demodulate(plan, x, plan.carriers[0])
    low, high = spectrum[:12], spectrum[12 * 50:12 * 51]
    _, nb_mask, _, _, nb_gain = reference(52, 1, 10, n_id=103, rbs=[0, 50])
    assert np.array_equal(nb_mask[:12], nb_mask[12 * 50:12 * 51])
    assert np.allclose(low[nb_mask[:12]], high[nb_mask[:12]], atol=1e-6 * np.abs(low).max())
    assert [c.n_id for _, overlay in plan.nb_iot_overlays for c in overlay.carriers] == [103, 103]


@pytest.mark.parametrize("bandwidths,position,ids,hosts", [
    ((10, 20), "both", None, [(0, [0], 103, 1), (1, [102], 110, 2)]),           # 97 + 6 n + max(0, m - 1): n = 1, 2 and m = 1, 2
    ((10, 20), "lower", None, [(0, [0], 103, 1)]),
    ((10, 20), "upper", None, [(1, [102], 104, 2)]),                              # the only host has rank 1, cell identity 2
    ((10, 20), "both", (7, 9), [(0, [0], 109, 7), (1, [102], 117, 9)]),
    ((5, 10, 20), "both", None, [(0, [2], 103, 1), (2, [102], 111, 3)]),         # the middle carrier hosts nothing: ranks 1, 2; 97 + 12 + 2
    ((10,), "both", (0,), [(0, [0, 50], 103, 0)]),                               # cell identity 0 counts like 1
])
def test_two_and_three_carriers_host_the_nb_iot_in_the_right_carriers_with_the_n_id_rule(bandwidths, position, ids, hosts):
    r, plan, config, x, analysis = synthesize(bandwidths, position, 6.0, ids)
    host_by_carrier = {carrier: (rbs, n_id, cell) for carrier, rbs, n_id, cell in hosts}
    assert sorted(dict(plan.nb_iot_overlays)) == sorted(host_by_carrier)
    scales = []
    for carrier in plan.carriers:
        spectrum = demodulate(plan, x, carrier)
        cell = carrier.n_cell_id
        if carrier.index in host_by_carrier:
            rbs, n_id, host_cell = host_by_carrier[carrier.index]
            assert cell == host_cell and [c.n_id for c in dict(plan.nb_iot_overlays)[carrier.index].carriers] == [n_id] * len(rbs)
            grid, _, _, _, _ = reference(carrier.nrb, cell, 10, n_id=n_id, rbs=rbs)
            assert [e["rb"] for e in analysis.allocation[carrier.index]["nb_iot"]["carriers"]] == rbs
            assert analysis.allocation[carrier.index]["nb_iot"]["n_id"] == n_id
        else:
            grid, _, _, _, _ = reference(carrier.nrb, cell, 10)
            assert "nb_iot" not in analysis.allocation[carrier.index]
        scales.append(assert_matches(spectrum, grid))
    assert max(scales) == pytest.approx(min(scales), rel=1e-6)                   # one common scale: unitary scaling, equal RE amplitude
    assert analysis.evm_percent < 1e-4 and all(e["evm_percent"] < 1e-4 for e in analysis.allocation)
    assert analysis.nr_test_model.nb_iot_rbs == [rb for _, rbs, _, _ in hosts for rb in rbs]


def test_non_host_carriers_are_plain_tm11_and_hosts_balance_their_own_frame():
    r, plan, config, x, analysis = synthesize((10, 20), "lower")
    _, _, _, nr_gain, _ = reference(52, 1, 10, n_id=103, rbs=[0])
    assert dict(plan.nb_iot_overlays)[0].nr_gain == pytest.approx(nr_gain, rel=1e-12)
    spectrum = demodulate(plan, x, plan.carriers[1])
    plain, _, _, gain, _ = reference(106, 2, 10)
    assert gain == 1.0 and assert_matches(spectrum, plain) > 0
    assert any("NR resource elements of a host are below the level of the other carriers" in w for w in plan.warnings)


def test_n_id_range_and_host_cell_id_limit():
    assert nb_iot.nb_iot_n_id(401, 1) == 503
    assert make_plan(recipe(10, ids=(401,))).nb_iot_overlays[0][1].carriers[0].n_id == 503
    with pytest.raises(ValidationError, match=r"N_ID 504.*host cell ID 402.*at most 401"):
        recipe(10, ids=(402,))
    with pytest.raises(ValidationError, match=r"Carrier 1: NB-IoT N_ID 802"):
        recipe(10, ids=(700,))
    make_plan(recipe(10, 20, ids=(1, 395), position="both"))                      # rank 2: 97 + 12 + 394 = 503
    with pytest.raises(ValidationError, match=r"Carrier 2: NB-IoT N_ID 504.*NB-IoT host #2"):
        recipe(10, 20, ids=(1, 396), position="both")
    make_plan(recipe(10, 20, ids=(1, 700), position="lower"))                     # a carrier that hosts no NB-IoT never matters
    with pytest.raises(ValidationError, match="N_ID"):
        recipe(10, 20, ids=(1, 700), position="upper")                            # 97 + 6 + 699


@pytest.mark.parametrize("host_cell,expected", [(1, 1085), (2, 1089), (3, 1085), (4, 1085), (5, 1089), (6, 1085), (7, 1085), (0, 1085)])
def test_nb_iot_re_budget_per_host_v_shift(host_cell, expected):
    r, plan, config, x, analysis = synthesize((10,), "lower", 6.0, (host_cell,))
    assert analysis.allocation[0]["nb_iot"]["nb_iot_re"] == expected
    grid, nb_mask, _, _, _ = reference(52, host_cell, 10, n_id=103 + max(0, host_cell - 1), rbs=[0])
    assert nb_mask.sum() == expected
    assert_matches(demodulate(plan, x, plan.carriers[0]), grid)


def test_nb_iot_evm_exclusion_and_the_documented_summary():
    r, plan, config, x, analysis = synthesize((10,), "both")
    _, nb_mask, evm_mask, _, _ = reference(52, 1, 10, n_id=103, rbs=[0, 50])
    carrier_grid = baseband(plan)[1][0][1]
    assert np.array_equal(carrier_grid.data_mask, evm_mask)
    base = build_carrier_grid(TM11, 52, 15, "fdd", 1, 10).data_mask
    assert 0 < (base & nb_mask).sum() < nb_mask.sum() and carrier_grid.data_mask.sum() == base.sum() - (base & nb_mask).sum()
    # had the NB-IoT REs stayed in the reference (as NR PDSCH data) the EVM would be several percent
    spectrum = demodulate(plan, x, plan.carriers[0])
    grid, _, _, _, _ = reference(52, 1, 10, n_id=103, rbs=[0, 50])
    scale = assert_matches(spectrum, grid)
    unpunctured = build_carrier_grid(TM11, 52, 15, "fdd", 1, 10).values * dict(plan.nb_iot_overlays)[0].nr_gain
    error = np.abs(spectrum - scale * unpunctured)[base]
    assert 100 * math.sqrt((error ** 2).sum() / (np.abs(scale * unpunctured[base]) ** 2).sum()) > 5 > 1e3 * analysis.evm_percent
    summary = analysis.nr_test_model
    assert summary.model == "NR-N-TM" and summary.nb_iot_rbs == [0, 50] and summary.tdd_pattern is None and not summary.truncated
    assert any("energy of the whole 10 ms frame" in w and "N_ID^Ncell 103" in w and "+6.00 dB" in w for w in summary.warnings)
    text = "\n".join(analysis.notes)
    for needle in ("NR-N-TM", "TS 36.141", "NPSS", "NSSS", "N_ID^Ncell = 97 + 6 n", "port-2001", "excluded from the reference",
                   "multiple of 100 kHz", "whole 10 ms frame", "RB 0, 50"):
        assert needle in text, needle
    entry = analysis.allocation[0]["nb_iot"]
    assert entry["carriers"][0]["subcarriers"] == analysis.allocation[0]["subcarriers"][0:12]
    assert entry["carriers"][1]["subcarriers"] == analysis.allocation[0]["subcarriers"][600:612]
    assert entry["host_cell_id"] == 1 and [c["side"] for c in entry["carriers"]] == ["lower", "upper"]


def test_the_nb_iot_frame_repeats_every_10_ms_and_short_captures_use_the_first_subframes():
    r, plan, config, x, analysis = synthesize((10,), "both", 6.0, None, 20)
    assert plan.subframes == 20
    spectrum = demodulate(plan, x, plan.carriers[0])
    assert np.allclose(spectrum[:, :140], spectrum[:, 140:], atol=1e-6 * np.abs(spectrum).max())
    grid, _, _, _, _ = reference(52, 1, 20, n_id=103, rbs=[0, 50])
    assert_matches(spectrum, grid)
    r, plan, config, x, analysis = synthesize((10,), "lower", 6.0, None, 3)
    grid, nb_mask, _, _, _ = reference(52, 1, 3, n_id=103, rbs=[0])
    assert_matches(demodulate(plan, x, plan.carriers[0]), grid)
    assert nb_mask.sum() == sum(SUBFRAME_RES[:3]) and plan.truncated


def test_determinism_and_distinct_ids():
    first, second = synthesize((20,), "both"), _synthesize.__wrapped__((20,), "both", 6.0, None, None)
    assert first[3].tobytes() == second[3].tobytes() and first[4].evm_percent == second[4].evm_percent
    ids = {nr_tm.resolve(recipe(20, position=p, boost=b)).config.preset_id for p in ("lower", "upper", "both") for b in (6.0, 3.0)}
    assert len(ids) == 6 and all(i.startswith("nr-ntm-fdd-15k-20m-c1-") and len(i) <= 64 for i in ids)
    lower, upper = synthesize((20,), "lower")[3], synthesize((20,), "upper")[3]
    assert not np.array_equal(lower, upper) and lower.shape == upper.shape
    assert nr_tm.resolve(recipe(20)).config.preset_id == nr_tm.resolve(recipe(20)).config.preset_id


# ---------------------------------------------------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------------------------------------------------

def direct(**kwargs):
    """A recipe that skipped the pydantic validator, to call validate_recipe itself."""
    carriers = kwargs.pop("carriers", [NrTmCarrier(scs_khz=15, bandwidth_mhz=10)])
    return NrTestModelConfig.model_construct(model="NR-N-TM", carriers=carriers, nb_iot=kwargs.pop("nb_iot", NrTmNbIot()), **kwargs)


def test_boost_must_leave_power_for_the_nr_resource_elements():
    limit = nb_iot.max_boost_db(15, 2 * 1085)
    assert limit == pytest.approx(10.649, abs=1e-3)
    with pytest.raises(ValidationError, match=r"boost of 11 dB is not possible with position 'both' on a 15-RB carrier.*below 10\.65 dB.*single NB-IoT carrier"):
        recipe(3, position="both", boost=11)
    with pytest.raises(ValueError, match=r"below 10\.65 dB"):
        validate_recipe(direct(carriers=[NrTmCarrier(scs_khz=15, bandwidth_mhz=3)], nb_iot=NrTmNbIot(position="both", boost_db=12)))
    plan = make_plan(recipe(3, position="both", boost=10.6))
    assert 0 < plan.nb_iot_overlays[0][1].nr_gain < 0.2
    overlay = make_plan(recipe(3, position="lower", boost=12)).nb_iot_overlays[0][1]            # one NB-IoT carrier: limit 13.66 dB
    assert overlay.nr_gain == pytest.approx(math.sqrt((25200 - 1085 * 10 ** 1.2) / (25200 - 1085)), rel=1e-12)
    for bw in (5, 10, 50):
        make_plan(recipe(bw, position="both", boost=12))
    # the limit of every bandwidth and position: the largest boost that works is just below 10 log10(N_tot / N_NB)
    for bw, nrb in NRB.items():
        for n_carriers, position in ((1, "lower"), (2, "both")):
            limit = 10 * math.log10(nrb * 1680 / (1085 * n_carriers))
            if limit < 12:
                make_plan(recipe(bw, position=position, boost=round(limit - 0.01, 2)))
                with pytest.raises(ValidationError, match="NB-IoT boost"):
                    recipe(bw, position=position, boost=min(12, round(limit + 0.01, 2)))


def test_unsupported_configurations_are_rejected_with_readable_messages():
    with pytest.raises(ValidationError, match=r"NR-N-TM needs the NB-IoT block \(nb_iot: position"):
        NrTestModelConfig(model="NR-N-TM", carriers=[NrTmCarrier(scs_khz=15, bandwidth_mhz=10)])
    with pytest.raises(ValueError, match="needs the NB-IoT block"):
        validate_recipe(direct(nb_iot=None))
    for model in ("NR-FR1-TM1.1", "NR-FR1-TM3.1", "NR-FR1-TM2"):
        with pytest.raises(ValidationError, match="NB-IoT block belongs to the NR-N-TM"):
            NrTestModelConfig(model=model, carriers=[NrTmCarrier(scs_khz=15, bandwidth_mhz=10)], nb_iot=NrTmNbIot())
    with pytest.raises(ValidationError, match=r"defined for FDD with 15 kHz.*TDD is not supported"):
        recipe(10, duplex="tdd")
    for scs, bw in ((30, 10), (60, 20)):
        with pytest.raises(ValidationError, match=rf"Carrier 1: NB-IoT operation in NR in-band is defined for 15 kHz.*{scs} kHz host"):
            NrTestModelConfig(model="NR-N-TM", carriers=[NrTmCarrier(scs_khz=scs, bandwidth_mhz=bw)], nb_iot=NrTmNbIot())
    with pytest.raises(ValidationError, match=r"Carrier 2: NB-IoT operation in NR in-band is defined for 15 kHz"):
        NrTestModelConfig(model="NR-N-TM", nb_iot=NrTmNbIot(), carriers=[NrTmCarrier(scs_khz=15, bandwidth_mhz=10),
                                                                         NrTmCarrier(scs_khz=30, bandwidth_mhz=10)])
    for bw in (60, 100):
        with pytest.raises(ValidationError, match=rf"{bw} MHz channel is not defined for 15 kHz.*3, 5, 10"):
            recipe(bw)
    with pytest.raises(ValidationError, match="Two layers"):
        recipe(10, layers=2)
    with pytest.raises(ValueError, match="FDD"):
        validate_recipe(direct(duplex="tdd"))
    with pytest.raises(ValidationError):
        NrTmNbIot(position="middle")
    with pytest.raises(ValidationError):
        NrTmNbIot(boost_db=2.9)
    with pytest.raises(ValidationError):
        NrTmNbIot(boost_db=12.1)
    with pytest.raises(ValidationError, match=r"Oversampling 2 would need 1966\.08 MS/s.*1,966,080 samples per subframe"):
        recipe(*([50] * 16), oversampling=2)                                       # 16 x 50 MHz: one subframe at 983 MS/s is the most that fits
    with pytest.raises(ValidationError, match="at most 1,000,000 fit, i.e. 1 subframe"):
        recipe(*([50] * 16), subframes=2)


def test_catalog_describes_nr_n_tm_from_the_nb_iot_module():
    catalog = nr_tm.catalog()
    info = next(m for m in catalog.models if m.model == "NR-N-TM")
    assert info.nb_iot and info.fdd_only and info.scs_khz == [15] and info.modulations == ["QPSK"]
    names = [p.name for p in info.fixed_parameters]
    assert len(set(names)) == len(names)
    for name in ("NB-IoT", "NB-IoT channels", "NB-IoT subframe map", "NB-IoT antenna port", "NB-IoT cell identity", "NB-IoT resource elements",
                 "NB-IoT power", "NR content around NB-IoT", "Duplex / numerology", "EVM"):
        assert name in names, name
    by_name = {p.name: p.value for p in info.fixed_parameters}
    assert "all data bits are zero" in by_name["NB-IoT channels"] and "port 2001" in by_name["NB-IoT antenna port"]
    assert "97 + 6 n + max(0, m - 1)" in by_name["NB-IoT cell identity"] and "1085" in by_name["NB-IoT resource elements"]
    assert "subframe 0 NPBCH" in by_name["NB-IoT subframe map"] and "9 NSSS" in by_name["NB-IoT subframe map"]
    assert catalog.nb_iot.bandwidths_mhz == sorted(NRB)
    for bw, nrb in NRB.items():
        assert catalog.nb_iot.eligible_rbs[str(bw)] == nb_iot.eligible_rbs(nrb)
        assert (catalog.nb_iot.lower_rb[str(bw)], catalog.nb_iot.upper_rb[str(bw)]) == (LOWER[bw], UPPER[bw])
    others = [m for m in catalog.models if m.model != "NR-N-TM"]
    assert all("NB-IoT subframe map" not in [p.name for p in m.fixed_parameters] for m in others)


def test_resolve_round_trip_and_editable_fields():
    r = recipe(20, position="both", boost=8.0)
    resolution = nr_tm.resolve(r, rms=0.2, snr_db=40.0)
    assert resolution.config.nr_test_model == r and resolution.summary.nb_iot_rbs == [2, 102]
    x, analysis = nr_tm.synthesize_nr_tm(resolution.config)
    assert np.isclose(analysis.rms, 0.2, rtol=1e-4) and analysis.evm_percent < 1.5
    assert resolution.config.channel_modulations == [4] and resolution.config.channel_subcarriers == [1272]


# ---------------------------------------------------------------------------------------------------------------------
# performance
# ---------------------------------------------------------------------------------------------------------------------

def test_the_widest_and_the_most_populated_captures_stay_fast():
    started = time.perf_counter()
    config = nr_tm.resolve(recipe(50, position="both")).config
    x, analysis = nr_tm.synthesize_nr_tm(config)
    single = time.perf_counter() - started
    assert len(x) == config.sample_count <= 1_000_000 and single < 20, single
    assert analysis.evm_percent < 1e-4
    started = time.perf_counter()
    config = nr_tm.resolve(recipe(*([10] * 16), position="both")).config
    x, analysis = nr_tm.synthesize_nr_tm(config)
    many = time.perf_counter() - started
    assert len(x) == config.sample_count <= 1_000_000 and many < 20, many
    assert analysis.evm_percent < 1e-4 and len(analysis.allocation) == 16
    assert [i for i, e in enumerate(analysis.allocation) if "nb_iot" in e] == [0, 15]
    assert [analysis.allocation[i]["nb_iot"]["n_id"] for i in (0, 15)] == [103, 124]            # hosts of rank 1 and 2, cell identities 1 and 16: 97 + 12 + 15
    peak = peak_rss_mb("""
from opendpd.core.waveforms import nr_tm
from opendpd.schemas.nr_test_model import NrTestModelConfig, NrTmCarrier, NrTmNbIot
for cells in ([50], [10] * 16):
    recipe = NrTestModelConfig(model="NR-N-TM", nb_iot=NrTmNbIot(position="both"), carriers=[NrTmCarrier(scs_khz=15, bandwidth_mhz=b) for b in cells])
    x, analysis = nr_tm.synthesize_nr_tm(nr_tm.resolve(recipe).config)
    assert len(x) in (614400, 983040)
""")
    assert peak < 3 * 1024, peak


def test_sixteen_wide_carriers_are_accepted_when_they_fit_the_sample_limit():
    """The legacy OFDMA description (sum of subcarriers within one FFT of at most 16,384 bins) does not apply to test-model configurations."""
    for bandwidth, subframes, rate in ((20, 2, 491.52e6), (50, 1, 983.04e6)):
        resolution = nr_tm.resolve(recipe(*([bandwidth] * 16), position="both"))
        config = resolution.config
        assert (config.sample_rate_hz, config.n_samples, resolution.summary.subframes) == (rate, 983_040, subframes)
        assert resolution.summary.truncated and len(config.channel_subcarriers) == 16 and config.fft_size == 16384
        assert config.bandwidth_hz < config.sample_rate_hz and config.channel_gap_bins <= 1024
    x, analysis = nr_tm.synthesize_nr_tm(nr_tm.resolve(recipe(*([20] * 16), position="both")).config)
    assert x.shape == (983_040,) and analysis.evm_percent < 1e-4 and analysis.nr_test_model.nb_iot_rbs == [2, 102]
    assert [e["nb_iot"]["n_id"] for e in analysis.allocation if "nb_iot" in e] == [103, 124]
