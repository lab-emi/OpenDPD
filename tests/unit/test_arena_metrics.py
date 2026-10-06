"""Arena measures added distortion, without confusing it with total RF leakage, and EVM as a receiver sees it."""

import numpy as np
import pytest

from opendpd.core.arena_metrics import adjacent_error_ratios, compute, evm_db, quality_db, symbol_windows
from opendpd.schemas import SignalSpec


def iq(z):
    return np.stack((z.real, z.imag), axis=-1)


def tones():
    n = np.arange(4096)
    main = np.exp(2j * np.pi * 8 * n / 1024)
    left = np.exp(-2j * np.pi * 128 * n / 1024)
    right = np.exp(2j * np.pi * 128 * n / 1024)
    spec = SignalSpec(sample_rate_hz=1024, bandwidth_hz=128, n_sub_ch=1, nperseg=1024)
    return main, left, right, spec


BAND = dict(useful_samples=1024, prefix_samples=0, occupied_hz=[[-64, 64]])
# 64-sample symbols behind an 8-sample cyclic prefix; subcarriers ±1…±8 of a 1,024 Hz capture are occupied.
OFDM = dict(useful_samples=64, prefix_samples=8, occupied_hz=[[-8.5 * 16, -0.5 * 16], [0.5 * 16, 8.5 * 16]])
OFDM_SPEC = SignalSpec(sample_rate_hz=1024, bandwidth_hz=256, n_sub_ch=1, nperseg=64)


def ofdm(symbols=6, seed=3, unloaded=()):
    """The waveform and the energy of its transmitted constellation points."""
    rng = np.random.default_rng(seed)
    blocks, energy = [], 0.
    for _ in range(symbols):
        grid = np.zeros(64, dtype=complex)
        bins = np.r_[1:9, -8:0]
        grid[bins] = rng.choice([-3, -1, 1, 3], 16) + 1j * rng.choice([-3, -1, 1, 3], 16)
        grid[list(unloaded)] = 0
        energy += float(np.sum(abs(grid) ** 2))
        useful = np.fft.ifft(grid)
        blocks.append(np.r_[useful[-8:], useful])
    return np.concatenate(blocks), energy


def test_perfect_linearization_is_not_penalized_for_reference_adjacent_content():
    main, left, right, spec = tones()
    reference = iq(main + 0.1 * left + 0.1 * right)
    baseline = compute(iq(main + 0.02 * left + 0.02 * right), reference, spec, BAND)
    perfect = compute(reference, reference, spec, BAND)
    # The old raw-ACLR objective would prefer the distorted baseline.
    assert baseline["aclr_l_db"] == pytest.approx(-33.9794, abs=1e-4)
    assert perfect["aclr_l_db"] == pytest.approx(-20, abs=1e-4)
    assert baseline["aer_l_db"] == pytest.approx(10 * np.log10(0.08**2), abs=1e-5)
    assert perfect["aer_l_db"] == -300
    assert perfect["aer_r_db"] == -300
    assert perfect["nmse_db"] == -300 and perfect["evm_db"] == -300
    assert baseline["evm_db"] < -250  # the adjacent tones lie outside the occupied band
    assert perfect["reference_aclr_l_db"] == perfect["aclr_l_db"]


def test_error_is_complex_waveform_difference_not_difference_of_psds():
    main, left, _, spec = tones()
    reference = iq(main + 0.1 * left)
    # Same total spectral power, but the adjacent tone has the wrong phase.
    wrong = iq(main - 0.1 * left)
    measured = adjacent_error_ratios(wrong, reference, spec)
    assert measured["aer_l_db"] == pytest.approx(10 * np.log10(0.2**2), abs=1e-5)


def test_fixed_reference_exposes_power_backoff():
    main, left, right, spec = tones()
    reference = iq(main + 0.1 * left + 0.1 * right)
    value = compute(0.5 * reference, reference, spec, BAND)
    assert value["power_error_db"] == pytest.approx(-6.020599913)
    assert value["nmse_db"] == pytest.approx(-6.020599913)
    assert value["evm_db"] < -250  # a receiver removes the common gain; the power gate sees the back-off


def test_evm_is_the_subcarrier_error_vector_of_complete_symbols_after_one_common_gain():
    reference, energy = ofdm(unloaded=[3])  # an occupied but unloaded subcarrier: the error is orthogonal to the data
    n = np.arange(len(reference))
    error = np.zeros(len(reference), dtype=complex)
    for k in range(6):  # a tone on subcarrier +3, only inside each useful window
        window = slice(k * 72 + 8, (k + 1) * 72)
        error[window] = 0.01 * np.exp(2j * np.pi * 3 * (n[window] - window.start) / 64)
    expected = 10 * np.log10(6 * (0.01 * 64) ** 2 / energy)
    assert evm_db(iq(reference + error), iq(reference), OFDM_SPEC, OFDM) == pytest.approx(expected, abs=1e-6)
    rotated = 0.9 * np.exp(0.3j) * (reference + error)
    assert evm_db(iq(rotated), iq(reference), OFDM_SPEC, OFDM) == pytest.approx(expected, abs=1e-6)
    # An error confined to cyclic prefixes or to an unoccupied subcarrier never reaches the constellation.
    prefix = np.zeros(len(reference), dtype=complex)
    prefix[[i for k in range(6) for i in range(k * 72, k * 72 + 8)]] = 0.05
    unused = 0.05 * np.exp(2j * np.pi * 20 * (n % 72 - 8) / 64)
    assert evm_db(iq(reference + prefix), iq(reference), OFDM_SPEC, OFDM) < -250
    assert evm_db(iq(reference + unused), iq(reference), OFDM_SPEC, OFDM) < -250


def test_symbol_windows_follow_the_capture_grid_not_the_split():
    assert symbol_windows(OFDM, 0, 6 * 72) == [(k * 72 + 8, 64) for k in range(6)]
    # A split that starts mid-symbol skips to the next complete useful window and drops a truncated tail.
    assert symbol_windows(OFDM, 100, 200) == [(52, 64), (124, 64)]
    assert symbol_windows(OFDM, 8, 64) == [(0, 64)]
    with pytest.raises(ValueError, match="complete OFDM"):
        symbol_windows(BAND, 100, 200)
    with pytest.raises(ValueError, match="complete OFDM symbol"):
        symbol_windows(OFDM, 9, 70)
    reference, _ = ofdm()
    shifted = evm_db(iq(reference[100:300] * 1.0 + 0.01), iq(reference[100:300]), OFDM_SPEC, OFDM, 100)
    assert shifted < -250  # DC is not an occupied subcarrier on the capture's own grid


def test_uncorrelated_or_absent_output_is_a_full_error_vector():
    reference, _ = ofdm()
    assert evm_db(iq(np.zeros_like(reference)), iq(reference), OFDM_SPEC, OFDM) == 0.0
    with pytest.raises(ValueError, match="occupied subcarrier"):
        evm_db(iq(reference), iq(np.zeros_like(reference)), OFDM_SPEC, OFDM)


def test_independently_timed_measured_carriers_keep_all_occupied_error_vectors():
    size, count, fs = 64, 512, 1024
    n = np.arange(count)
    tone = lambda f: np.exp(2j*np.pi*f*n/fs)
    reference = sum(tone(center+16)+.5*tone(center-32) for center in (-128,128))
    spec = SignalSpec(sample_rate_hz=fs, bandwidth_hz=384, n_sub_ch=2, nperseg=64)
    grid = dict(useful_samples=size, prefix_samples=8, carriers=[
        dict(frequency_shift_hz=-center, filter_half_bandwidth_hz=64,
             fft_starts=[start], occupied_bins=[1,2,62,63])
        for center,start in [(-128,80),(128,150)]])
    # A common complex gain is removable. A different, out-of-carrier signal
    # must not contaminate either independently positioned constellation.
    assert evm_db(iq((.8+.2j)*reference+tone(400)), iq(reference), spec, grid,
                  window=(50,450)) < -250
    error = .01*tone(128-16)  # occupied, zero-reference bin; orthogonal to the gain estimate
    expected = 10*np.log10(.01**2/(2*(1+.5**2)))
    assert evm_db(iq(reference+error),iq(reference),spec,grid,window=(50,450)) == pytest.approx(expected,abs=1e-8)
    # The short validation segment cannot be quietly treated as symbol EVM.
    with pytest.raises(ValueError,match='every measured carrier'):
        evm_db(iq(reference),iq(reference),spec,grid,window=(100,200))


def test_quality_weighs_in_band_and_the_worse_adjacent_side_equally():
    baseline = dict(evm_db=-20., aclr_l_db=-30., aclr_r_db=-34.)
    scores = dict(evm_db=-32., aclr_l_db=-44., aclr_r_db=-38.)
    # EVM gains 12 dB; the worse side moves from −30 (left) to −38 (right): 8 dB.
    assert quality_db(scores, baseline) == pytest.approx(10.)
    assert quality_db(baseline, baseline) == 0.


def test_empty_reference_and_nonfinite_or_mismatched_outputs_are_rejected():
    main, _, _, spec = tones()
    value = iq(main)
    for output, reference in (
        (value[:-1], value),
        (value * np.nan, value),
        (value, np.zeros_like(value)),
    ):
        with pytest.raises(ValueError):
            adjacent_error_ratios(output, reference, spec)


@pytest.mark.parametrize("metric", [lambda y, r, spec: compute(y, r, spec, BAND), adjacent_error_ratios,
                                    lambda y, r, spec: evm_db(y, r, spec, BAND)])
def test_complex_or_wrong_shaped_input_is_explicitly_rejected(metric):
    main, _, _, spec = tones()
    for invalid in (main, iq(main)[None, ...], iq(main)[:, :1]):
        with pytest.raises(ValueError, match="real IQ arrays"):
            metric(invalid, invalid, spec)
