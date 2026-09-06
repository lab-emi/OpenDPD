"""``ofdm-lte20-v1`` reference waveform and its data-aided demodulator (plan S15).

Ideal signal, known noise, distortion, timing and frequency perturbations and misconfiguration are each
checked against a closed-form expectation or an explicit refusal; nothing here compares against a standard's
own tool chain (that cross-validation is a maintainer step recorded in ``docs/protocols/waveform-profiles.md``).
"""

import math

import numpy as np
import pytest
from scipy.signal import resample_poly

from opendpd.core.waveforms import demodulate, generate, read_package, synchronize, to_baseband_rate, to_iq, write_package
from opendpd.core.waveforms.ofdm import FS, N_SC, NFFT, SAMPLES_PER_SUBFRAME, qam64_points, rate_ratio
from opendpd.schemas.waveform import WaveformSpec

SPEC = WaveformSpec(seed=11, n_subframes=2)


@pytest.fixture(scope="module")
def wf():
    return generate(SPEC)


def awgn(x, snr_db, seed=0):
    rng = np.random.default_rng(seed)
    sigma = math.sqrt(0.5 / 10 ** (snr_db / 10))
    return x + rng.normal(0, sigma, (x.size, 2)).view(np.complex128).ravel()


def test_waveform_is_deterministic_unit_power_and_has_the_fixed_layout(wf):
    assert wf.x.size == SPEC.n_subframes * SAMPLES_PER_SUBFRAME == SPEC.n_samples
    assert wf.symbols.shape == (SPEC.n_symbols, N_SC) == (28, 1200)
    assert np.mean(np.abs(wf.x) ** 2) == pytest.approx(1.0, abs=0.02)
    assert np.mean(np.abs(qam64_points()) ** 2) == pytest.approx(1.0)
    again = generate(WaveformSpec(seed=11, n_subframes=2))
    assert np.array_equal(again.x, wf.x) and again.sha256() == wf.sha256()
    other = generate(WaveformSpec(seed=12, n_subframes=2))
    assert not np.array_equal(other.symbols, wf.symbols) and other.sha256() != wf.sha256()
    # a longer waveform of the same seed starts with the shorter one: symbols are drawn in symbol order, so a
    # package can be extended without changing what was already played
    longer = generate(WaveformSpec(seed=11, n_subframes=3))
    assert np.array_equal(longer.symbols[:SPEC.n_symbols], wf.symbols) and np.allclose(longer.x[:wf.x.size], wf.x)
    # cyclic prefix: the tail of every useful part precedes it
    for l in range(SPEC.n_symbols):
        s, cp = int(wf.fft_starts[l]), int(wf.cp_lengths[l])
        assert np.allclose(wf.x[s - cp:s], wf.x[s + NFFT - cp:s + NFFT])


def test_spectrum_occupies_the_configured_band_and_leaves_the_dc_subcarrier_empty(wf):
    spectrum = np.fft.fft(wf.x[int(wf.fft_starts[3]):int(wf.fft_starts[3]) + NFFT])
    assert abs(spectrum[0]) < 1e-9                                  # DC subcarrier
    assert np.all(np.abs(spectrum[601:NFFT - 600]) < 1e-9)          # guard band empty
    assert np.all(np.abs(spectrum[1:601]) > 0) and np.all(np.abs(spectrum[NFFT - 600:]) > 0)


def test_numerology_is_fixed_and_misconfiguration_is_refused():
    with pytest.raises(ValueError, match="fixed numerology"):
        WaveformSpec(fft_size=4096)
    with pytest.raises(ValueError):
        WaveformSpec(modulation="256QAM")
    with pytest.raises(ValueError):
        WaveformSpec(n_subframes=0)
    assert rate_ratio(FS) == (1, 1) and rate_ratio(122.88e6) == (1, 4) and rate_ratio(800e6) == (24, 625)
    assert rate_ratio(15.36e6) is None                 # under-sampled
    assert rate_ratio(30.72e6 * math.pi) is None       # no small rational ratio
    assert to_baseband_rate(np.zeros(10), 15.36e6) is None


def test_ideal_signal_demodulates_to_zero_evm_with_exact_timing(wf):
    d = demodulate(wf.x, wf)
    assert d is not None and d.n_symbols == SPEC.n_symbols and d.offset == 0
    assert d.correlation == pytest.approx(1.0, abs=1e-9)
    assert d.evm_rms_pct < 1e-9 and d.cfo_hz == pytest.approx(0.0, abs=1e-6)
    assert np.allclose(d.equalizer, 1.0)


@pytest.mark.parametrize("shift", [1, 1234, SAMPLES_PER_SUBFRAME + 7])
def test_timing_offset_and_looped_playback_are_recovered(wf, shift):
    y = np.roll(wf.x, -shift)                     # y[n] = x[n + shift]: the capture started mid-waveform
    offset, corr = synchronize(y, wf)
    assert offset == shift % wf.period and corr == pytest.approx(1.0, abs=1e-9)
    d = demodulate(y, wf)
    # one period of capture holds every symbol but the one cut by the wrap-around
    assert d.evm_rms_pct < 1e-9 and d.n_symbols == SPEC.n_symbols - 1
    # a capture shorter than one period still yields every complete symbol it holds
    partial = demodulate(y[:40_000], wf)
    assert partial is not None and 0 < partial.n_symbols < SPEC.n_symbols and partial.evm_rms_pct < 1e-9


def test_white_noise_of_known_power_gives_the_predicted_evm(wf):
    """Noise power inside the occupied subcarriers is 1200/2048 of the total; the data-aided least-squares
    equaliser over L symbols removes 1/L of it. The residual frequency estimate must not add to the error."""
    snr = 10 ** (30 / 10)
    expected = 100 / math.sqrt(snr) * math.sqrt(N_SC / NFFT) * math.sqrt(1 - 1 / SPEC.n_symbols)
    values = [demodulate(awgn(wf.x, 30, seed), wf).evm_rms_pct for seed in range(3)]
    assert all(v == pytest.approx(expected, abs=0.03) for v in values), (values, expected)


def test_frequency_offset_and_phase_are_estimated_and_removed(wf):
    n = np.arange(wf.x.size)
    for cfo in (-350.0, 200.0):
        y = wf.x * np.exp(1j * (2 * math.pi * cfo * n / FS + 0.7))
        d = demodulate(y, wf)
        assert d.cfo_hz == pytest.approx(cfo, abs=0.01)
        assert d.evm_rms_pct < 1e-6
    # without correction the same offset is an error, so the estimate is not cosmetic
    y = wf.x * np.exp(2j * math.pi * 200.0 * n / FS)
    assert demodulate(y, wf, correct_cfo=False).evm_rms_pct > 1.0


def test_linear_channel_inside_the_cyclic_prefix_is_equalised_exactly(wf):
    h = np.array([0.9, 0.3j, -0.1, 0.05])
    y = np.convolve(wf.x, h)[:wf.x.size]
    d = demodulate(y, wf)
    assert d.evm_rms_pct < 1e-8
    expected = np.fft.fft(h, NFFT)
    from opendpd.core.waveforms.ofdm import subcarrier_bins
    assert np.allclose(d.equalizer, expected[subcarrier_bins()])


def test_nonlinear_distortion_raises_evm_monotonically(wf):
    mild = demodulate(wf.x - 0.05 * np.abs(wf.x) ** 2 * wf.x, wf).evm_rms_pct
    strong = demodulate(wf.x - 0.10 * np.abs(wf.x) ** 2 * wf.x, wf).evm_rms_pct
    assert 1.0 < mild < strong


def test_resampling_from_a_capture_rate_has_a_small_recorded_floor(wf):
    """Polyphase conversion from 4x and from the 800 MHz capture rate: the floor is a measurement limit of the
    profile (recorded in the protocol), far below any PA-related EVM but not zero."""
    for fs in (122.88e6, 800e6):
        up, down = rate_ratio(fs)
        captured = resample_poly(wf.x, down, up)           # what an instrument sampling at fs would record
        d = demodulate(to_baseband_rate(captured, fs), wf)
        assert d.n_symbols == SPEC.n_symbols and d.evm_rms_pct < 0.1, (fs, d.evm_rms_pct)


def test_unrelated_signals_are_refused_not_scored(wf):
    rng = np.random.default_rng(1)
    assert demodulate(rng.normal(0, 1, (60_000, 2)).view(np.complex128).ravel(), wf) is None
    assert demodulate(wf.x[:1000], wf) is None                       # shorter than one symbol
    other = generate(WaveformSpec(seed=99, n_subframes=2))
    assert demodulate(other.x, wf) is None                          # another seed of the same waveform


def test_package_round_trips_and_is_regenerated_not_trusted(tmp_path, wf):
    path = write_package(wf, tmp_path / "pkg")
    spec, digest = read_package(path)
    assert spec == SPEC and digest == wf.sha256()
    assert np.array_equal(np.load(tmp_path / "pkg" / "x.npy"), to_iq(wf.x))
    assert np.load(tmp_path / "pkg" / "symbols.npy").shape == (SPEC.n_symbols, N_SC)
    assert read_package(tmp_path / "pkg")[0] == SPEC
