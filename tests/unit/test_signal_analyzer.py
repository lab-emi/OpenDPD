"""Analytical signal fixtures, estimator units, range limits and reference semantics."""
import numpy as np
import pytest
from pydantic import ValidationError

from opendpd.core.waveforms.analyzer import band_power, inspect_signal
from opendpd.core.waveforms.generator import synthesize, rrc_taps
from opendpd.core.waveforms.modulation import Payload, qam_symbols, psk_symbols
from opendpd.schemas.signal_analyzer import AnalyzerConfig
from opendpd.schemas.signal_generator import GeneratorConfig


def values(result):
    return {m['key']: m['value'] for m in result['measurements']}


@pytest.mark.parametrize('real', [True, False])
def test_tone_power_units_and_two_sided_spectrum(real):
    t = np.arange(8192) / 8192
    x = np.cos(2*np.pi*512*t) if real else np.exp(2j*np.pi*512*t)
    cfg = AnalyzerConfig(sample_rate_hz=8192, bandwidth_hz=2048, fft_size=1024)
    result = inspect_signal(x, cfg)
    m = values(result)
    assert m['rms'] == pytest.approx(np.sqrt(.5) if real else 1)
    assert m['papr'] == pytest.approx(10*np.log10(2) if real else 0, abs=1e-10)
    assert m['rbw'] == 12 and m['bin_width'] == 8
    density = 10**(np.array(result['psd_dbfs_hz'])/10)
    assert np.sum(density)*8 == pytest.approx(.5 if real else 1)
    assert m['channel_power'] == pytest.approx(-3.01029995664 if real else 0, abs=1e-9)
    assert result['real_signal'] == real
    assert len(result['time_s']) == 2048
    assert np.shape(result['spectrogram_dbfs_hz']) == (15, 256)
    assert len(result['eye_i']) <= 64
    if not real:
        np.testing.assert_allclose(result['instantaneous_frequency_hz'][1:], 512, atol=1e-9)


def test_full_nyquist_band_preserves_every_fft_bin():
    fs = 1024
    f = np.fft.fftshift(np.fft.fftfreq(256, 1/fs))
    p = np.ones(256)
    assert band_power(f, p, fs, 0, fs) == fs
    assert band_power(f, p, fs, 100, 10) == 10
    assert band_power(f, p, fs, 500, 50) is None


def test_acpr_is_unavailable_when_complete_band_does_not_fit():
    result = inspect_signal(np.ones(4096), AnalyzerConfig(sample_rate_hz=4096, bandwidth_hz=2048))
    assert values(result)['acpr_lower'] is None and values(result)['acpr_upper'] is None


def test_zero_signal_remains_finite_and_reports_undefined_ratios():
    result = inspect_signal(np.zeros(256), AnalyzerConfig())
    m = values(result)
    assert m['rms'] == 0 and m['papr'] is None and m['obw'] is None
    assert np.isfinite(result['psd_dbfs_hz']).all()
    assert all(v is None for v in result['instantaneous_frequency_hz'])
    assert sum(result['histogram_probability']) == 1


def test_reference_fit_is_explicit_and_does_not_change_source_samples():
    rng = np.random.default_rng(7)
    ref = rng.normal(size=8192) + 1j*rng.normal(size=8192)
    x = (2+1j)*ref
    original = x.copy()
    raw = inspect_signal(x, AnalyzerConfig(), ref)
    fit = inspect_signal(x, AnalyzerConfig(reference_gain_fit=True), ref)
    assert values(raw)['reference_evm'] == pytest.approx(100*np.sqrt(2))
    assert values(fit)['reference_evm'] < 1e-10
    np.testing.assert_array_equal(x, original)
    with pytest.raises(ValueError, match='same selected'):
        inspect_signal(x, AnalyzerConfig(), ref[:-1])


def test_range_work_and_chart_sizes_are_bounded():
    x = np.exp(2j*np.pi*.1*np.arange(1_000_000))
    result = inspect_signal(x, AnalyzerConfig(fft_size=16384))
    assert result['sample_count'] == 1_000_000
    assert np.shape(result['spectrogram_dbfs_hz']) == (128, 256)
    assert len(result['scatter_i']) == 4096
    with pytest.raises(ValueError):
        inspect_signal(x[:255], AnalyzerConfig())
    with pytest.raises(ValueError):
        inspect_signal(np.full(256, np.nan), AnalyzerConfig())


@pytest.mark.parametrize('changes', [{'sample_rate_hz': float('nan')}, {'fft_size': 700},
    {'n_samples': 1_000_001}, {'symbol_offset': 8}, {'bandwidth_hz': 1e9},
    {'sample_format': 'iq', 'i_column': 0, 'q_column': 0}, {'adjacent_offset_hz': 1e6}])
def test_invalid_analysis_configs_rejected(changes):
    with pytest.raises(ValidationError):
        AnalyzerConfig(**changes)


def test_gray_bit_mapping_and_total_rrc_span():
    c = GeneratorConfig(payload_mode='bits', payload_bits='00011011')
    payload = Payload(c, np.random.default_rng(1))
    np.testing.assert_allclose(qam_symbols(payload, 4, 4), np.array([-1+1j, -1-1j, 1+1j, 1-1j])/np.sqrt(2))
    c = GeneratorConfig(payload_mode='bits', payload_bits='01')
    np.testing.assert_array_equal(qam_symbols(Payload(c, None), 2, 2), [-1, 1])
    np.testing.assert_allclose(psk_symbols(Payload(c, None), 2, 2), [1, -1], atol=1e-15)
    taps = rrc_taps(8, .25, 10)
    assert len(taps) == 81
    assert np.sum(taps**2) == pytest.approx(1)
    np.testing.assert_array_equal(taps, taps[::-1])


def test_burst_gating_constant_phase_and_dft_zero_bins_are_honest():
    c = GeneratorConfig(waveform='tone', n_samples=4096, filter_enabled=False, burst_on_samples=1024,
                        burst_off_samples=1024, burst_ramp_samples=0, phase_offset_deg=90)
    x, a = synthesize(c)
    assert x[0].imag == pytest.approx(c.rms)
    assert not np.any(x[1024:2048])
    assert a.papr_db == pytest.approx(10*np.log10(2), abs=1e-6)
    _, a = synthesize(GeneratorConfig(filter_enabled=False, dft_spreading=True, pilot_mode='none', payload_mode='bits', payload_bits='0'))
    assert np.isfinite(a.evm_per_subcarrier_percent).all()
    assert a.evm_percent < .001


@pytest.mark.parametrize('waveform', ['psk', 'fsk', 'gfsk', 'noise'])
def test_new_waveforms_are_reproducible(waveform):
    c = GeneratorConfig(waveform=waveform, samples_per_symbol=16, n_samples=8192, filter_enabled=False)
    x, a = synthesize(c)
    np.testing.assert_array_equal(x, synthesize(c)[0])
    assert len(x) == 8192 and np.isfinite(x).all()
    if waveform in ('fsk', 'gfsk'):
        assert a.papr_db < 1e-5
