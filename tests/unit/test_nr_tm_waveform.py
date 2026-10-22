"""Time domain of the test-model synthesis: cyclic prefixes, OFDM mapping, exact length, RMS, periodic structure, TDD OFF samples, oversampling."""
from fractions import Fraction

import numpy as np
import pytest

from opendpd.core.waveforms import nr_tm
from opendpd.core.waveforms.nr_tm.grid import build_carrier_grid
from opendpd.core.waveforms.nr_tm.modulate import cp_lengths, nfft_for_nrb, ofdm_demodulate, ofdm_modulate, phasor, symbol_starts
from opendpd.core.waveforms.nr_tm.plan import make_plan
from opendpd.core.waveforms.nr_tm.synth import baseband
from opendpd.core.waveforms.nr_tm.tables import MU, TDD_PATTERNS
from opendpd.schemas.nr_test_model import NrTestModelConfig, NrTmCarrier


def recipe(model="NR-FR1-TM1.1", scs=15, bw=5, duplex="fdd", **kwargs):
    return NrTestModelConfig(model=model, duplex=duplex, carriers=[NrTmCarrier(scs_khz=scs, bandwidth_mhz=bw)], **kwargs)


def literal_cp(mu, n_fft, symbol_in_subframe):
    """N_CP,l of TS 38.211 5.3.1 in units of Tc, converted to samples of a signal with n_fft samples per useful part."""
    kappa = 64
    cp_tc = Fraction(144 * kappa, 2 ** mu)
    if symbol_in_subframe in (0, 7 * 2 ** mu):
        cp_tc += 16 * kappa
    useful_tc = Fraction(2048 * kappa, 2 ** mu)
    return cp_tc * n_fft / useful_tc


@pytest.mark.parametrize("scs", [15, 30, 60])
@pytest.mark.parametrize("n_fft", [128, 512, 2048, 4096])
def test_cyclic_prefix_lengths_follow_38_211_5_3_1(scs, n_fft):
    mu = MU[scs]
    cp = cp_lengths(mu, n_fft, 4 * 2 ** mu)                                   # four subframes
    per_subframe = 14 * 2 ** mu
    for symbol, length in enumerate(cp):
        expected = literal_cp(mu, n_fft, symbol % per_subframe)
        assert expected.denominator == 1 and length == expected, (scs, n_fft, symbol)
    long_cp = np.flatnonzero(cp > cp[1])
    assert long_cp.tolist() == [s * per_subframe + p for s in range(4) for p in (0, 7 * 2 ** mu)]
    useful = cp + n_fft
    assert useful.reshape(4, per_subframe).sum(axis=1).tolist() == [30720 * 2 ** mu * n_fft // 2048] * 4        # exactly 1 ms per subframe
    ends, starts = symbol_starts(mu, n_fft, 4 * 2 ** mu)
    assert starts[0] == cp[0] and starts[1] == cp[0] + n_fft + cp[1]


def test_cyclic_prefix_is_the_tail_of_the_useful_part_and_rows_map_to_signed_bins():
    mu, k_total, n_fft = 0, 300, 512
    grid = np.zeros((k_total, 14), dtype=complex)
    grid[k_total // 2 + 3, 2] = 1.0                                             # row K/2 + 3 -> +3 subcarriers above DC
    grid[k_total // 2, 5] = 1.0                                                 # row K/2 is DC and is used
    grid[0, 9] = 1.0                                                            # lowest row -> bin -150
    samples = ofdm_modulate(grid, mu, n_fft, gain=np.sqrt(n_fft))
    cp, starts = symbol_starts(mu, n_fft, 1)
    n = np.arange(n_fft)
    for symbol, bin_index in ((2, 3), (5, 0), (9, -150)):
        useful = samples[starts[symbol]:starts[symbol] + n_fft]
        assert np.allclose(useful, np.exp(2j * np.pi * bin_index * n / n_fft) / np.sqrt(n_fft), atol=1e-15)
    for symbol in range(14):
        useful = samples[starts[symbol]:starts[symbol] + n_fft]
        assert np.array_equal(samples[starts[symbol] - cp[symbol]:starts[symbol]], useful[n_fft - cp[symbol]:])
    assert len(samples) == int(cp.sum()) + 14 * n_fft


def test_modulate_demodulate_round_trip_is_exact_for_random_grids():
    rng = np.random.default_rng(1)
    for mu, k_total, n_fft in ((0, 300, 512), (1, 3276, 4096), (2, 132, 256)):
        n_slots = 2 ** mu * 2
        grid = rng.standard_normal((k_total, 14 * n_slots)) + 1j * rng.standard_normal((k_total, 14 * n_slots))
        samples = ofdm_modulate(grid, mu, n_fft, gain=np.sqrt(n_fft))
        back = ofdm_demodulate(samples, mu, n_fft, k_total, n_slots, gain=np.sqrt(n_fft))
        assert np.max(np.abs(back - grid)) < 1e-12


def test_fft_size_rule_matches_the_matlab_ofdm_info_rule():
    # smallest power of two >= 128 with 12 N_RB / N <= 0.85
    cases = {11: 256, 25: 512, 52: 1024, 106: 2048, 133: 2048, 273: 4096, 24: 512, 51: 1024, 65: 1024, 78: 2048, 217: 4096, 245: 4096}
    for nrb, n in cases.items():
        assert nfft_for_nrb(nrb) == n, nrb
        assert 12 * nrb <= 0.85 * n and (n == 128 or 12 * nrb > 0.85 * n / 2)


def test_phasor_is_exact_for_long_records_and_large_shifts():
    n = 1_000_000
    shift = Fraction(99_900_000, 1) / 2                                         # 49.95 MHz at 245.76 MS/s: a long non-trivial phase ramp
    p = phasor(shift, 245_760_000, n)
    k = np.arange(n, dtype=np.float64)
    naive = np.exp(2j * np.pi * float(shift) * k / 245_760_000)
    assert np.max(np.abs(p - naive)) < 5e-9                                      # the float ramp itself is only accurate to ~1e-9 here
    assert np.allclose(np.abs(p), 1.0, atol=1e-15)
    # shift = fs / 4 gives the period-4 sequence 1, j, -1, -j; the integer phase reduction leaves no drift after 2e6 samples
    q = phasor(Fraction(1, 4) * 61_440_000, 61_440_000, 2_000_000)
    assert np.allclose(q[:4], [1, 1j, -1, -1j], atol=1e-15)
    assert np.allclose(q[1_999_996:], [1, 1j, -1, -1j], atol=1e-15)


@pytest.mark.parametrize("duplex,model,scs,bw", [("fdd", "NR-FR1-TM1.1", 15, 5), ("fdd", "NR-FR1-TM3.1", 30, 20), ("tdd", "NR-FR1-TM3.2", 30, 20),
                                                  ("tdd", "NR-FR1-TM2", 60, 20), ("fdd", "NR-FR1-TM1.2", 60, 10)])
def test_baseband_demodulates_back_to_the_grid_to_1e_12(duplex, model, scs, bw):
    plan = make_plan(recipe(model, scs, bw, duplex))
    x, grids = baseband(plan)
    assert len(x) == plan.n_samples and x.dtype == np.complex128
    (carrier, grid), = grids
    back = ofdm_demodulate(x, carrier.mu, carrier.n_fft, carrier.subcarriers, carrier.slots, gain=np.sqrt(carrier.n_fft))
    assert np.max(np.abs(back - grid.values)) < 1e-12


@pytest.mark.parametrize("duplex,model,scs,bw,subframes", [("fdd", "NR-FR1-TM1.1", 15, 5, None), ("fdd", "NR-FR1-TM3.1", 30, 20, 7),
                                                           ("tdd", "NR-FR1-TM3.2", 30, 10, None), ("fdd", "NR-FR1-TM2", 60, 20, 3)])
def test_exact_length_dtype_rms_and_determinism(duplex, model, scs, bw, subframes):
    resolution = nr_tm.resolve(recipe(model, scs, bw, duplex, subframes=subframes))
    config = resolution.config
    x, analysis = nr_tm.synthesize_nr_tm(config)
    assert x.dtype == np.complex64 and x.shape == (config.sample_count,) == (resolution.summary.n_samples,)
    assert float(np.sqrt(np.mean(np.abs(x.astype(np.complex128)) ** 2))) == pytest.approx(config.rms, rel=1e-6)
    assert analysis.sample_count == config.sample_count and analysis.rms == pytest.approx(config.rms, rel=1e-6)
    assert analysis.evm_percent < 1e-4                                          # complex64 rounding is the only error source
    again, _ = nr_tm.synthesize_nr_tm(config)
    assert x.tobytes() == again.tobytes()
    assert analysis.nr_test_model is not None and analysis.nr_test_model.n_samples == len(x)


def test_output_demodulates_to_the_grid_after_undoing_the_rms_scale():
    plan = make_plan(recipe("NR-FR1-TM3.1", 30, 10, subframes=4))
    config = nr_tm.resolve(recipe("NR-FR1-TM3.1", 30, 10, subframes=4)).config
    x, _ = nr_tm.synthesize_nr_tm(config)
    base, grids = baseband(plan)
    scale = config.rms / np.sqrt(np.mean(np.abs(base) ** 2))
    (carrier, grid), = grids
    back = ofdm_demodulate(x.astype(np.complex128) / scale, carrier.mu, carrier.n_fft, carrier.subcarriers, carrier.slots, gain=np.sqrt(carrier.n_fft))
    assert np.max(np.abs(back - grid.values)) < 2e-6
    assert np.allclose(x.astype(np.complex128), scale * base, atol=3e-8 * config.rms * 10)


def test_standard_convention_repeats_every_frame_matlab_convention_does_not():
    for duplex, scs in (("fdd", 15), ("tdd", 30)):
        std = nr_tm.resolve(recipe("NR-FR1-TM1.1", scs, 10, duplex, subframes=20)).config
        x, _ = nr_tm.synthesize_nr_tm(std)
        half = len(x) // 2
        assert np.array_equal(x[:half], x[half:]), duplex                                 # frame 2 == frame 1, sample for sample
        mat = nr_tm.resolve(recipe("NR-FR1-TM1.1", scs, 10, duplex, subframes=20, data_convention="matlab")).config
        y, _ = nr_tm.synthesize_nr_tm(mat)
        assert np.max(np.abs(y[:half] - y[half:])) > 0.05                                         # frame 2 carries other bits
        if duplex == "fdd":      # TM1.1 FDD: frame 1 is identical in both conventions up to the RMS normalisation of the whole record
            gain = float(np.vdot(x[:half], y[:half]).real / np.vdot(x[:half], x[:half]).real)
            assert abs(gain - 1) < 0.01 and np.max(np.abs(y[:half] - gain * x[:half])) < 1e-6
        else:                    # TDD: the MATLAB convention gives the special slot its own stream, so frame 1 differs as well
            assert np.max(np.abs(y[:half] - x[:half] * (np.linalg.norm(y[:half]) / np.linalg.norm(x[:half])))) > 0.05


def test_tdd_off_samples_are_exactly_zero():
    for scs, nrb_bw in ((15, 10), (30, 20), (60, 20)):
        r = recipe("NR-FR1-TM1.1", scs, nrb_bw, "tdd")
        config = nr_tm.resolve(r).config
        x, _ = nr_tm.synthesize_nr_tm(config)
        plan = make_plan(r)
        carrier = plan.carriers[0]
        cp, starts = symbol_starts(carrier.mu, carrier.n_fft, carrier.slots)
        pattern = TDD_PATTERNS[scs]
        n_off = 0
        for slot in range(carrier.slots):
            n_dl = pattern.dl_symbols_of_slot(slot % pattern.slots)
            for symbol in range(14):
                s = 14 * slot + symbol
                chunk = x[starts[s] - cp[s]:starts[s] + carrier.n_fft]
                if symbol >= n_dl:
                    assert not np.any(chunk), (scs, slot, symbol)
                    n_off += len(chunk)
                else:
                    assert np.any(chunk), (scs, slot, symbol)
        assert n_off / len(x) == pytest.approx(1 - 26 / 35, abs=0.02)        # about 9/35 of the capture is OFF (symbol lengths differ slightly)


def test_fdd_has_no_zero_samples_and_dc_subcarrier_is_used():
    config = nr_tm.resolve(recipe("NR-FR1-TM1.1", 15, 5, subframes=1)).config
    x, _ = nr_tm.synthesize_nr_tm(config)
    assert np.count_nonzero(x) == len(x)
    plan = make_plan(recipe("NR-FR1-TM1.1", 15, 5, subframes=1))
    grid = build_carrier_grid(plan.model, 25, 15, "fdd", 1, 1).values
    assert np.all(np.abs(grid[150, 2:14]) > 0)                              # row K/2 = 150 (the DC subcarrier) carries data or DM-RS


def test_oversampling_is_a_pure_interpolation_of_the_same_grid():
    plans = {o: make_plan(recipe("NR-FR1-TM3.1", 15, 10, subframes=3, oversampling=o)) for o in (1, 2, 4)}
    bases = {o: baseband(p)[0] for o, p in plans.items()}
    for o in (2, 4):
        assert plans[o].sample_rate_hz == o * plans[1].sample_rate_hz and plans[o].n_samples == o * plans[1].n_samples
        assert np.max(np.abs(bases[o][::o] * np.sqrt(o) - bases[1])) < 1e-12        # x_o(o m) = x_1(m) / sqrt(o) before the RMS scaling
    outputs = {}
    for o in (1, 2):
        config = nr_tm.resolve(recipe("NR-FR1-TM3.1", 15, 10, subframes=3, oversampling=o)).config
        outputs[o], analysis = nr_tm.synthesize_nr_tm(config)
        assert config.sample_rate_hz == plans[o].sample_rate_hz and analysis.evm_percent < 1e-4
    ratio = outputs[2][::2].astype(complex) / outputs[1].astype(complex)
    assert np.std(ratio) < 1e-5 and abs(np.mean(ratio) - 1) < 0.1                    # same waveform up to one real RMS-normalisation constant
