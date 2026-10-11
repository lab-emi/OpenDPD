"""Independent numeric and rejection checks for the two registered LTE v2 profiles."""
import copy

import numpy as np
import pytest

from opendpd.core.metrics import ofdm_evm_v1, ofdm_evm_v2, registry
from opendpd.core.waveforms import ofdm
from opendpd.schemas.dataset import SignalSpec
from opendpd.schemas.waveform import WaveformSpec
from scripts import matlab_parity_v2 as harness


@pytest.mark.parametrize('subframes', [1, 10])
@pytest.mark.parametrize('offset_hz', [-350.0, 350.0])
@pytest.mark.parametrize('lag', [0, 937])
def test_v2_recovers_both_offset_signs_and_partial_symbol_captures(subframes, offset_hz, lag):
    wf = ofdm.generate(WaveformSpec(seed=3, n_subframes=subframes))
    captured = np.roll(wf.x, -lag) * (0.7 + 0.3j) * np.exp(2j * np.pi * offset_hz / ofdm.FS * np.arange(wf.period))
    result = ofdm.demodulate(captured, wf, frequency_tolerant=True)
    assert result is not None and result.offset == lag
    assert abs(result.cfo_hz - offset_hz) < 0.01
    assert result.evm_rms_pct < 1e-9
    if subframes == 10:
        assert ofdm.demodulate(captured, wf) is None, 'v1 must retain its recorded refusal'


def test_magnitude_candidate_does_not_authorize_a_wrong_waveform_or_noise():
    wf = ofdm.generate(WaveformSpec(seed=1, n_subframes=10))
    other = ofdm.generate(WaveformSpec(seed=2, n_subframes=10))
    noise = np.random.default_rng(173).standard_normal(wf.period) + 1j * np.random.default_rng(174).standard_normal(wf.period)
    for value in (other.x, noise, np.zeros(wf.period), wf.x[:1024]):
        assert ofdm.demodulate(value, wf, frequency_tolerant=True) is None


def test_no_offset_v2_keeps_existing_demodulation_numbers():
    wf = ofdm.generate(WaveformSpec(seed=5, n_subframes=1))
    y = wf.x * (0.7 - 0.2j)
    old, new = ofdm.demodulate(y, wf), ofdm.demodulate(y, wf, frequency_tolerant=True)
    assert (old.offset, old.cfo_hz, old.evm_rms_pct, old.correlation) == (new.offset, new.cfo_hz, new.evm_rms_pct, new.correlation)


def test_enclosing_integration_has_analytic_endpoint_and_nyquist_behavior():
    freq = np.arange(-5, 5, dtype=float)
    density = np.arange(1, 11, dtype=float)
    power, count = ofdm_evm_v2.enclosing_band_power(freq, density, -1.2, 1.2, 10)
    assert count == 5 and power == 4 + 5 + 6 + 7 + 8
    power, count = ofdm_evm_v2.enclosing_band_power(freq, density, -1, 1, 10)
    assert count == 3 and power == 5 + 6 + 7
    assert ofdm_evm_v2.enclosing_band_power(freq, density, 3, 5, 10) == (19, 2)
    assert ofdm_evm_v2.enclosing_band_power(freq, density, 3, 5.1, 10) == (0, 0)


def test_psd_tone_ratio_and_parseval_without_matlab():
    fs = 122.88e6
    n = 8192
    t = np.arange(n) / fs
    signal = np.exp(2j * np.pi * 1.92e6 * t) + 0.1 * np.exp(2j * np.pi * 19.2e6 * t)
    for auto in (False, True):
        f, p = ofdm_evm_v2._psd(signal, fs, 2048, auto)
        main, _ = ofdm_evm_v2.enclosing_band_power(f, p, -9e6, 9e6, fs)
        adjacent, _ = ofdm_evm_v2.enclosing_band_power(f, p, 11e6, 29e6, fs)
        assert 10 * np.log10(adjacent / main) == pytest.approx(-20, abs=0.002)
        assert np.sum(p) * fs / len(p) == pytest.approx(1.01, abs=1e-6)


def test_auto_psd_uses_full_record_and_ignores_dataset_segments():
    x = np.random.default_rng(6).standard_normal(1500) + 1j * np.random.default_rng(7).standard_normal(1500)
    f, p = ofdm_evm_v2._psd(x, 122.88e6, 128, True)
    f2, p2 = ofdm_evm_v2._psd(x, 122.88e6, 4096, True)
    assert len(f) == 2048
    np.testing.assert_array_equal(f, f2)
    np.testing.assert_array_equal(p, p2)


def test_registry_profiles_are_separate_and_v1_not_promoted():
    assert ofdm_evm_v1.PROFILE.version == 1
    assert ofdm_evm_v1.PROFILE.validation.value == 'pending_cross_validation'
    for name in harness.PROFILE_IDS.values():
        profile = registry.get_profile(name)
        assert profile.version == 2 and profile.profile_id == name
        assert profile.parameters['protocol'].endswith('waveform-profiles-v2.md')
    assert 'magnitude' not in ofdm_evm_v1.PROFILE.parameters['timing']
    assert registry.DEFAULT_PROFILE_ID == 'opendpd-spectral-v2'


def test_product_statuses_and_evm_are_independent_of_psd_mode():
    wf = ofdm.generate(WaveformSpec(seed=1, n_subframes=1))
    from scripts.matlab_parity import fft_interpolate
    fs = 122.88e6
    y = fft_interpolate(wf.x, 4)
    binding = ofdm.bind_input(ofdm.to_iq(y), fs, wf.spec)
    signal = SignalSpec(sample_rate_hz=fs, bandwidth_hz=18e6, n_sub_ch=1, nperseg=2048, waveform=binding)
    outputs = []
    for pid in harness.PROFILE_IDS.values():
        outputs.append({m.name: m for m in registry.evaluate(pid, ofdm.to_iq(y), None, signal)})
        bad = ofdm.to_iq(y); bad[0, 0] = np.nan
        assert all(m.status.value == 'invalid' for m in registry.evaluate(pid, bad, None, signal))
        missing = signal.model_copy(update={'waveform': None})
        assert all(m.status.value == 'missing_reference' for m in registry.evaluate(pid, ofdm.to_iq(y), None, missing))
    assert outputs[0]['EVM_RMS'].value == outputs[1]['EVM_RMS'].value
    assert all(m.status.value == 'ok' for row in outputs for m in row.values())


def passing_items():
    items = []
    for group, name in sorted(harness.expected_cases()):
        budget = 0.1 if group == 'P2' else 0.05 if 'EVM_RMS' in name else 0.01 if 'frequency offset' in name else 1e-6 if 'relative RMS' in name else 0
        mode = 'default' if name.endswith('default') else 'aligned'
        items.append(harness.score(group, name, 0.0, 0.0, budget, profile_id=harness.PROFILE_IDS[mode] if group == 'P2' else None))
    return items


def test_verdict_requires_every_case_original_budget_and_honest_values():
    good = passing_items()
    assert harness.verdict(good)['passed'] == 115
    for altered in (good[:-1], good + [good[-1]]):
        with pytest.raises(ValueError): harness.verdict(altered)
    for field, value in [('budget', 0.2), ('matlab', 1.0), ('profile_id', 'ofdm-lte20-evm-v1')]:
        altered = copy.deepcopy(good)
        item = next(i for i in altered if i['group'] == 'P2')
        item[field] = value
        with pytest.raises(ValueError): harness.verdict(altered)
    refused = copy.deepcopy(good)
    target = next(i for i in refused if i['group'] == 'P3' and 'EVM_RMS' in i['item'])
    target.update(harness.score('P3', target['item'], None, None, 0.05))
    assert not harness.verdict(refused)['all_scored_within_budget']
