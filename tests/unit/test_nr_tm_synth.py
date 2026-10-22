"""Synthesis options: two layers, impairment tail, analysis content, the NB-IoT overlay hook, guard rails and the performance guard."""
import math
import os
import subprocess
import sys
import time

import numpy as np
import pytest

from opendpd.core.waveforms import nr_tm
from opendpd.core.waveforms.nr_tm.plan import make_plan
from opendpd.core.waveforms.nr_tm.synth import baseband
from opendpd.schemas.nr_test_model import NrTestModelConfig, NrTmCarrier, NrTmNbIot


def recipe(*cells, **kwargs):
    return NrTestModelConfig(carriers=[NrTmCarrier(scs_khz=s, bandwidth_mhz=b) for s, b in cells], **kwargs)


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


def run(r, **overrides):
    config = nr_tm.resolve(r, **overrides).config
    x, analysis = nr_tm.synthesize_nr_tm(config)
    return config, x, analysis


def test_two_layers_export_either_port_and_port_1_flips_the_dmrs_comb():
    plans = {port: make_plan(recipe((30, 20), subframes=2, layers=2, output_port=port)) for port in (0, 1)}
    grids = {port: baseband(plan)[1][0][1].values for port, plan in plans.items()}
    assert not np.array_equal(grids[0], grids[1]) and grids[0].shape == grids[1].shape
    for port in (0, 1):
        config, x, analysis = run(recipe((30, 20), subframes=2, layers=2, output_port=port))
        assert x.shape == (config.sample_count,) and analysis.evm_percent < 1e-4
        assert any("port " + str(1000 + port) in note for note in analysis.notes)
    dmrs_rows = np.arange(12 * 3, 12 * 40, 2)                                  # even subcarriers of PRB >= 3 in the first DM-RS symbol
    sign = np.where((dmrs_rows // 2) % 2 == 0, 1.0, -1.0)
    assert np.allclose(grids[1][dmrs_rows, 2], grids[0][dmrs_rows, 2] * sign, atol=1e-15)
    one_layer = baseband(make_plan(recipe((30, 20), subframes=2)))[1][0][1].values
    assert np.array_equal(grids[0][:36, :2], one_layer[:36, :2])                # the PDCCH is the same single-port PDCCH
    assert not np.array_equal(grids[0], one_layer)                             # but the PDSCH carries twice the bits per slot


def test_data_conventions_differ_only_where_documented():
    def grid(model, duplex, convention, subframes, scs=15, bw=10):
        r = recipe((scs, bw), model=model, duplex=duplex, subframes=subframes, data_convention=convention, oversampling=1)
        return baseband(make_plan(r))[1][0][1].values
    for model in ("NR-FR1-TM1.1", "NR-FR1-TM1.2", "NR-FR1-TM3.2", "NR-FR1-TM3.3"):
        assert np.array_equal(grid(model, "fdd", "standard", 10), grid(model, "fdd", "matlab", 10)), model
    assert not np.array_equal(grid("NR-FR1-TM2", "fdd", "standard", 10), grid("NR-FR1-TM2", "fdd", "matlab", 10))
    assert not np.array_equal(grid("NR-FR1-TM1.1", "tdd", "standard", 5), grid("NR-FR1-TM1.1", "tdd", "matlab", 5))
    assert not np.array_equal(grid("NR-FR1-TM1.1", "fdd", "standard", 20), grid("NR-FR1-TM1.1", "fdd", "matlab", 20))
    assert np.array_equal(grid("NR-FR1-TM1.1", "tdd", "matlab", 5), grid("NR-FR1-TM1.1", "tdd", "matlab", 5))


def test_pn_scope_changes_the_tdd_waveform_only_after_the_first_special_slot():
    def grid(scope):
        return baseband(make_plan(recipe((15, 10), duplex="tdd", subframes=10, pn_scope=scope, oversampling=1)))[1][0][1].values
    dl, everything = grid("dl_symbols"), grid("all_symbols")
    assert np.array_equal(dl[:, :14 * 3], everything[:, :14 * 3])                  # slots 0-2 are full downlink slots in both readings
    assert not np.array_equal(dl[:, 14 * 5:14 * 6], everything[:, 14 * 5:14 * 6])    # slot 5 follows the special and uplink slot
    plan = make_plan(recipe((15, 10), subframes=10, pn_scope="all_symbols"))
    assert any("pn_scope only matters for TDD" in w for w in plan.warnings)
    plan = make_plan(recipe((15, 10), duplex="tdd", subframes=10, data_convention="matlab", pn_scope="all_symbols"))
    assert any("no effect with the MATLAB data convention" in w for w in plan.warnings)


def test_seed_only_matters_with_noise_and_the_output_is_reproducible():
    r = recipe((30, 10), subframes=2)
    _, a, _ = run(r, snr_db=30, seed=1)
    _, b, _ = run(r, snr_db=30, seed=1)
    _, c, _ = run(r, snr_db=30, seed=2)
    assert a.tobytes() == b.tobytes() and a.tobytes() != c.tobytes()
    _, d, _ = run(r, seed=1)
    _, e, _ = run(r, seed=2)
    assert d.tobytes() == e.tobytes()                                               # the noiseless test model has no random part


def test_the_impairment_tail_acts_as_in_the_generic_generator():
    r = recipe((30, 20), subframes=4)
    config, clean, base = run(r)
    n = len(clean)
    occupied_fraction = 612 * 30e3 / config.sample_rate_hz
    # a phase offset of phi turns every point by phi: EVM = 100 |exp(j phi) - 1| = 200 sin(phi / 2) per cent
    _, _, rotated = run(r, phase_offset_deg=1.0)
    assert rotated.evm_percent == pytest.approx(200 * math.sin(math.radians(0.5)), rel=1e-4)
    # AWGN is added over the whole sample rate, the EVM sees the in-band share only
    _, _, noisy = run(r, snr_db=30)
    assert noisy.evm_percent == pytest.approx(100 * 10 ** (-30 / 20) * math.sqrt(occupied_fraction), rel=0.03)
    # DC offset lands in the dc_magnitude and in the EVM
    _, _, dc = run(r, dc_i=0.01)
    assert dc.dc_magnitude == pytest.approx(0.01, abs=2e-3) and dc.evm_percent > 1
    # clipping lowers the peak
    _, clipped, clip_analysis = run(r, clip_db=6)
    assert np.max(np.abs(clipped)) <= config.rms * 10 ** (6 / 20) * (1 + 1e-6) and clip_analysis.papr_db < base.papr_db
    # I/Q gain imbalance, frequency offset and phase noise change the signal and the EVM
    for override in ({"iq_gain_db": 0.5}, {"iq_phase_deg": 2.0}, {"frequency_offset_hz": 1000.0}, {"phase_noise_rms_deg": 0.5}):
        _, x, analysis = run(r, **override)
        assert x.shape == (n,) and analysis.evm_percent > 0.3, override
    # burst gating: exactly zero between the bursts (no noise, no DC) and still the exact length
    _, burst, _ = run(r, burst_on_samples=2000, burst_off_samples=500, burst_ramp_samples=0)
    positions = np.arange(n) % 2500
    assert not np.any(burst[positions >= 2000]) and np.all(np.abs(burst[positions < 2000][:100]) > 0)
    # the optional PA-input filter keeps length and RMS and removes the out-of-band part
    _, filtered, filter_analysis = run(r, filter_enabled=True)
    assert len(filtered) == n and filter_analysis.rms == pytest.approx(config.rms, rel=1e-6)
    assert np.mean(np.abs(filtered) ** 2) == pytest.approx(config.rms ** 2, rel=1e-5)


def test_analysis_content_for_a_single_carrier():
    r = recipe((30, 20), model="NR-FR1-TM3.1", subframes=4)
    config, x, analysis = run(r)
    assert analysis.sample_count == len(x) == 245_760 and analysis.sample_rate_hz == config.sample_rate_hz
    assert analysis.subcarrier_spacing_hz == pytest.approx(30e3) and analysis.useful_symbol_us == pytest.approx(1e6 / 30e3)
    assert analysis.cp_lengths_samples[:2] == [176, 144]                                           # N = 2048: 144 + 16 x 2 for the first symbol of a subframe
    assert analysis.active_carriers == 612 and analysis.pilot_carriers == 0
    assert analysis.occupied_bandwidth_99_hz == pytest.approx(612 * 30e3, rel=0.02)
    assert analysis.complete_symbols == 14 * 8 and analysis.trailing_samples == 0
    assert len(analysis.evm_per_symbol_percent) == analysis.evm_symbols == 14 * 8
    assert len(analysis.evm_per_subcarrier_percent) == len(analysis.evm_subcarrier_indices) == 612
    assert analysis.nr_test_model is not None and analysis.nr_test_model.model == "NR-FR1-TM3.1"
    entry, = analysis.allocation
    assert entry["channel"] == 1 and entry["scs_khz"] == 30 and entry["bandwidth_mhz"] == 20 and entry["n_rb"] == 51 and entry["n_cell_id"] == 1
    assert entry["modulation_order"] == 64 and entry["subcarrier_step"] == 1 and entry["evm_percent"] < 1e-4
    assert entry["center_offset_hz"] == pytest.approx(0.0) and len(entry["subcarriers"]) == 612
    assert entry["subcarriers"][0] == -306 and entry["subcarriers"][-1] == 305                    # bins relative to DC in units of 30 kHz
    text = " ".join(analysis.notes)
    assert "NR-FR1-TM3.1" in text and "4.9.2.2.5" in text and "Data convention 'standard'" in text and "not the receiver EVM" in text
    assert len(analysis.constellation_i) == len(analysis.reference_i) <= 4096
    assert max(abs(np.array(analysis.constellation_i) - np.array(analysis.reference_i))) < 1e-5
    assert len(analysis.psd_dbfs_hz) == len(analysis.frequency_mhz) and analysis.papr_db > 6


def test_analysis_content_for_several_carriers_and_mixed_numerologies():
    config, x, analysis = run(recipe((15, 10), (15, 10), (15, 10), subframes=2))
    assert [e["channel"] for e in analysis.allocation] == [1, 2, 3] and [e["n_cell_id"] for e in analysis.allocation] == [1, 2, 3]
    assert all(e["evm_percent"] < 1e-4 for e in analysis.allocation) and analysis.evm_percent < 1e-4
    offsets = [e["center_offset_hz"] for e in analysis.allocation]
    assert offsets == sorted(offsets) and offsets[1] - offsets[0] == pytest.approx(9.9e6)
    assert analysis.active_carriers == 3 * 624 and any("3 carriers" in note for note in analysis.notes)
    _, _, mixed = run(recipe((15, 10), (30, 10), subframes=2))
    assert 0.5 < mixed.evm_percent < 5                                                  # inter-numerology leakage of the unwindowed CP-OFDM symbols
    assert any("inter-numerology leakage" in w for w in mixed.nr_test_model.warnings) and any("leakage" in n for n in mixed.notes)
    _, _, wide = run(recipe((30, 100), (30, 100), subframes=1))
    assert all(e["subcarrier_step"] == 12 for e in wide.allocation) and len(wide.allocation[0]["subcarriers"]) == 273
    assert len(wide.evm_per_subcarrier_percent) == len(wide.evm_subcarrier_indices) == 2 * 273
    assert wide.evm_percent < 1e-4


def test_truncation_is_reported_in_the_analysis_and_the_summary():
    config, x, analysis = run(recipe((30, 100), model="NR-FR1-TM3.1b"))
    assert len(x) == 983_040 and analysis.nr_test_model.truncated and analysis.nr_test_model.subframes == 8
    assert any("shortened to 8 ms" in w for w in analysis.nr_test_model.warnings) and any("shortened to 8 ms" in n for n in analysis.notes)
    assert analysis.evm_percent < 1e-4


def test_synthesis_refuses_configurations_that_do_not_match_their_recipe():
    config = nr_tm.resolve(recipe((30, 10), subframes=2)).config
    with pytest.raises(ValueError, match="no longer matches"):
        nr_tm.synthesize_nr_tm(config.model_copy(update={"n_samples": config.n_samples - 256}))
    with pytest.raises(ValueError, match="no longer matches"):
        nr_tm.synthesize_nr_tm(config.model_copy(update={"sample_rate_hz": config.sample_rate_hz / 2}))
    with pytest.raises(ValueError, match="no nr_test_model"):
        nr_tm.synthesize_nr_tm(config.model_copy(update={"nr_test_model": None}))


def test_nr_n_tm_synthesises_with_the_nb_iot_carriers_punctured_into_the_host_and_excluded_from_the_evm_reference():
    r = recipe((15, 10), model="NR-N-TM", nb_iot=NrTmNbIot(position="both", boost_db=6), subframes=10)
    config, x, analysis = run(r)
    summary = analysis.nr_test_model
    assert config.preset_id.startswith("nr-ntm-fdd-15k-10m") and summary.model == "NR-N-TM"
    assert summary.nb_iot_rbs == [0, 50]                                    # 52 RB host: the lowest and the highest eligible RB
    assert len(x) == config.sample_count == 614_400 and not np.isnan(x).any() and analysis.evm_percent < 1e-4
    entry = analysis.allocation[0]["nb_iot"]
    assert [c["rb"] for c in entry["carriers"]] == [0, 50] and entry["nb_iot_re"] == 2170 and entry["boost_db"] == 6

    # the same host without NB-IoT: its EVM reference may differ from the NR-N-TM one only in the punctured RBs, and there it
    # must lose resource elements (the boosted NB-IoT signal is not a PDSCH data RE of the NR test model)
    host = baseband(make_plan(recipe((15, 10), model="NR-FR1-TM1.1", subframes=10)))[1][0][1].data_mask
    punctured = baseband(make_plan(r))[1][0][1].data_mask
    assert host.shape == punctured.shape and not (punctured & ~host).any()
    removed = host & ~punctured
    rows = np.flatnonzero(removed.any(axis=1))
    assert set(rows) == set(range(0, 12)) | set(range(12 * 50, 12 * 51))     # exactly the subcarriers of RB 0 and RB 50
    assert 0 < removed.sum() <= entry["nb_iot_re"]                           # only resource elements that carry NB-IoT signal


def test_the_largest_captures_stay_within_time_and_memory_limits():
    started = time.perf_counter()
    config, x, analysis = run(recipe((30, 100), model="NR-FR1-TM3.1b", duplex="tdd"))
    single = time.perf_counter() - started
    assert len(x) == config.sample_count <= 1_000_000 and single < 20, single
    started = time.perf_counter()
    config, x, analysis = run(recipe(*([(15, 10)] * 16)))
    many = time.perf_counter() - started
    assert len(x) == config.sample_count <= 1_000_000 and many < 20 and len(analysis.allocation) == 16, many
    assert analysis.evm_percent < 1e-4
    peak = peak_rss_mb("""
from opendpd.core.waveforms import nr_tm
from opendpd.schemas.nr_test_model import NrTestModelConfig, NrTmCarrier
for model, duplex, cells in (("NR-FR1-TM3.1b", "tdd", [(30, 100)]), ("NR-FR1-TM1.1", "fdd", [(15, 10)] * 16)):
    recipe = NrTestModelConfig(model=model, duplex=duplex, carriers=[NrTmCarrier(scs_khz=s, bandwidth_mhz=b) for s, b in cells])
    x, analysis = nr_tm.synthesize_nr_tm(nr_tm.resolve(recipe).config)
    assert len(x) == 983040
""")
    assert peak < 3 * 1024, peak                                                                # the design budget is 3 GB; about 0.3 GB are used
