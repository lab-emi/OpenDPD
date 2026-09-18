"""Independent FFT references for valid-sample carrier ACLR, including padded tails."""
import numpy as np
import pytest

from opendpd.core.metrics import evaluate
from opendpd.schemas.dataset import SignalSpec


def iq(z):
    return np.column_stack((z.real, z.imag))


def oracle(z, fs, bw, carriers, size):
    """Manual periodogram average: no production PSD/integration helpers or scipy Welch."""
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(size) / size)
    frames = np.stack([z[start:start + size] for start in range(0, len(z)-size+1, size-size//2)])
    power = np.mean(abs(np.fft.fft(frames*window, axis=-1))**2, axis=0) / (fs * sum(window**2))
    frequencies = np.fft.fftfreq(size, 1/fs)
    def integrate(lo, hi):
        return sum(power[(frequencies >= lo) & (frequencies < hi)]) * fs / size
    width = bw/carriers
    reference = max(integrate(-bw/2+i*width, -bw/2+(i+1)*width) for i in range(carriers))
    return tuple(10*np.log10(integrate(lo, hi)/reference) for lo, hi in
                 ((-bw/2-width, -bw/2), (bw/2, bw/2+width)))


@pytest.mark.parametrize('bw,carriers', [(5e6,1),(20e6,1),(80e6,4),(160e6,4),(200e6,5),(320e6,10)])
def test_known_carrier_powers_across_bandwidths(bw, carriers):
    fs, size = 4*bw, 256*carriers
    t = np.arange(3*size+size//2)/fs
    width = bw/carriers
    z = sum((1 if i==0 else .5)*np.exp(2j*np.pi*(-bw/2+(i+.5)*width)*t) for i in range(carriers))
    z += 10**(-55/20)*np.exp(2j*np.pi*(-bw/2-width/2)*t)
    z += 10**(-60/20)*np.exp(2j*np.pi*(bw/2+width/2)*t)
    spec = SignalSpec(sample_rate_hz=fs, bandwidth_hz=bw, n_sub_ch=carriers, nperseg=size)
    scores = {m.name:m for m in evaluate('opendpd-spectral-v2',iq(z),iq(z),spec)}
    assert scores['ACLR_L'].value == pytest.approx(-55, abs=1e-8)
    assert scores['ACLR_R'].value == pytest.approx(-60, abs=1e-8)
    assert scores['ACLR_AVG'].value == pytest.approx(-57.5, abs=1e-8)
    assert scores['NMSE'].value == -300


@pytest.mark.parametrize('size', [512,513,4096])
def test_partial_tail_and_nonzero_model_padding_are_excluded(size):
    rng=np.random.default_rng(12)
    z=rng.normal(size=3*size+size//2)+1j*rng.normal(size=3*size+size//2)
    valid=len(z)
    spec=SignalSpec(sample_rate_hz=80e6, bandwidth_hz=20e6,n_sub_ch=4,nperseg=size)
    padded=np.pad(iq(z),((0,4*size-valid),(0,0)),constant_values=1000).reshape(4,size,2)
    values={m.name:m.value for m in evaluate('opendpd-spectral-v2',padded,padded,spec,valid_samples=valid)}
    expected=oracle(z,80e6,20e6,4,size)
    assert (values['ACLR_L'],values['ACLR_R']) == pytest.approx(expected,abs=1e-10)
    assert values['NMSE']==-300


def test_no_fabricated_score_when_capture_cannot_measure_bands():
    z=np.ones(4096,dtype=complex)
    for spec in [SignalSpec(sample_rate_hz=40e6,bandwidth_hz=20e6,n_sub_ch=1,nperseg=512),
                 SignalSpec(sample_rate_hz=80e6,bandwidth_hz=20e6,n_sub_ch=1,nperseg=8192)]:
        scores=evaluate('opendpd-spectral-v2',iq(z),iq(z),spec)
        assert all(m.value is None and m.status.value=='not_applicable' for m in scores if m.name.startswith('ACLR'))


def test_nr20_padding_regression_does_not_invent_adjacent_leakage():
    from opendpd.core.waveforms.generator import synthesize
    from opendpd.core.waveforms.generator_presets import presets
    from modules.data_collector import IQSegmentDataset
    config = next(p.config for p in presets() if p.preset_id == 'nr-20')
    signal, _ = synthesize(config)
    real = iq(signal[int(.8*len(signal)):]).astype(np.float32)
    padded = IQSegmentDataset(real, real, nperseg=4096).features.numpy()
    spec = SignalSpec(sample_rate_hz=config.sample_rate_hz, bandwidth_hz=config.bandwidth_hz,
                      n_sub_ch=1, nperseg=4096)
    def scores(profile):
        return {m.name:m.value for m in evaluate(profile,padded,padded,spec,valid_samples=len(real))}
    old, new = scores('legacy-opendpd-v1'), scores('opendpd-spectral-v2')
    # The historic zero-padded half-window creates a false ~-36 dBc floor.
    assert -40 < old['ACLR_AVG'] < -30
    assert new['ACLR_L'] < -90 and new['ACLR_R'] < -90
    expected = oracle(real[:,0].astype(float)+1j*real[:,1], config.sample_rate_hz,
                      config.bandwidth_hz,1,4096)
    assert (new['ACLR_L'],new['ACLR_R']) == pytest.approx(expected,abs=1e-8)
