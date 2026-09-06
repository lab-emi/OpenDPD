"""``ofdm-lte20-evm-v1``: explicit statuses on every input that is not a bound capture of the reference waveform,
and numbers that agree with the demodulator on captures that are (plan S15)."""

import math

import numpy as np
import pytest
from scipy.signal import resample_poly

from opendpd.core.metrics import evaluate, get_profile, list_profiles
from opendpd.core.metrics.ofdm_evm_v1 import MIN_FS_ACLR_HZ, PROFILE_ID
from opendpd.core.waveforms import demodulate, generate, to_baseband_rate
from opendpd.core.waveforms.ofdm import rate_ratio
from opendpd.schemas import MetricStatus, SignalSpec
from opendpd.schemas.metrics import ProfileValidation
from opendpd.schemas.waveform import WaveformBinding, WaveformSpec

SPEC = WaveformSpec(seed=5, n_subframes=2)
FS_CAPTURE = 122.88e6
NPERSEG = 4096


def binding(fs=FS_CAPTURE, offset=0):
    return WaveformBinding(spec=SPEC, input_offset_samples=offset, correlation=1.0, input_sample_rate_hz=fs)


def signal(fs=FS_CAPTURE, bound=True, nperseg=NPERSEG):
    return SignalSpec(sample_rate_hz=fs, bandwidth_hz=18e6, nperseg=nperseg, amplitude_units="normalized",
                      waveform=binding(fs) if bound else None)


def iq(z):
    return np.stack([z.real, z.imag], axis=-1).astype(np.float32)


def by_name(metrics):
    return {m.name: m for m in metrics}


@pytest.fixture(scope="module")
def capture():
    """What an instrument sampling at 122.88 MS/s records while the waveform is played through a mildly
    non-linear PA: the reference of this profile is the waveform, not this capture."""
    wf = generate(SPEC)
    up, down = rate_ratio(FS_CAPTURE)
    played = resample_poly(wf.x, down, up)
    return wf, played, played - 0.04 * np.abs(played) ** 2 * played


def test_profile_is_registered_but_pending_cross_validation():
    profile = get_profile(PROFILE_ID)
    assert profile.validation == ProfileValidation.pending_cross_validation and not profile.frozen
    assert [m.name for m in profile.metrics] == ["EVM_RMS", "EVM_DB", "ACLR_L", "ACLR_R"]
    assert "not a conformance measurement" in profile.description.lower()
    assert PROFILE_ID in {p.profile_id for p in list_profiles()}
    # the two older profiles keep their validation status; nothing about them changed
    assert get_profile("legacy-opendpd-v1").validation == ProfileValidation.golden
    assert get_profile("general-spectral-v1").validation == ProfileValidation.analytic


def test_unbound_dataset_is_missing_reference_never_scored(capture):
    _, _, y = capture
    m = by_name(evaluate(PROFILE_ID, iq(y), None, signal(bound=False)))
    assert all(v.status == MetricStatus.missing_reference for v in m.values())
    assert all(v.value is None and "not bound" in v.reason for v in m.values())
    # the built-in multi-carrier capture layout (ten carriers at 800 MS/s) is exactly such a dataset
    builtin = SignalSpec(sample_rate_hz=800e6, bandwidth_hz=200e6, n_sub_ch=10, nperseg=2560)
    assert all(v.status == MetricStatus.missing_reference for v in evaluate(PROFILE_ID, iq(y), None, builtin))


def test_bound_capture_scores_like_the_demodulator_and_aclr_is_leakage(capture):
    wf, played, y = capture
    m = by_name(evaluate(PROFILE_ID, iq(y), None, signal()))
    assert all(v.status == MetricStatus.ok for v in m.values()), m
    direct = demodulate(to_baseband_rate(y, FS_CAPTURE), wf)
    assert m["EVM_RMS"].value == pytest.approx(direct.evm_rms_pct, rel=1e-9)
    assert m["EVM_DB"].value == pytest.approx(20 * math.log10(direct.evm_rms_pct / 100), abs=1e-9)
    assert 1.0 < m["EVM_RMS"].value < 20.0
    assert m["ACLR_L"].value < 0 and m["ACLR_R"].value < 0
    # the undistorted playback has a lower EVM (the rate-conversion floor) and less leakage
    clean = by_name(evaluate(PROFILE_ID, iq(played), None, signal()))
    assert clean["EVM_RMS"].value < 0.1 < m["EVM_RMS"].value
    assert clean["ACLR_L"].value < m["ACLR_L"].value and clean["ACLR_R"].value < m["ACLR_R"].value


def test_segmented_layout_and_valid_range_are_handled(capture):
    """The worker hands the profile (n_segments, nperseg, 2) arrays whose last segment is zero padded."""
    _, _, y = capture
    n = (y.size // NPERSEG) * NPERSEG
    segments = iq(y[:n]).reshape(-1, NPERSEG, 2)
    padded = np.concatenate([segments, np.zeros((1, NPERSEG, 2), dtype=np.float32)])
    m = by_name(evaluate(PROFILE_ID, padded, None, signal(), valid_samples=n))
    flat = by_name(evaluate(PROFILE_ID, iq(y[:n]), None, signal()))
    assert m["EVM_RMS"].value == pytest.approx(flat["EVM_RMS"].value, rel=1e-9)
    assert m["ACLR_R"].value == pytest.approx(flat["ACLR_R"].value, rel=1e-9)


def test_unsupported_capture_rates_and_missing_metadata_are_explicit(capture):
    wf, _, y = capture
    # at the waveform clock EVM works but the adjacent channel is not captured
    y30 = to_baseband_rate(y, FS_CAPTURE)
    m = by_name(evaluate(PROFILE_ID, iq(y30), None, signal(fs=30.72e6)))
    assert m["EVM_RMS"].status == MetricStatus.ok
    assert m["ACLR_L"].status == MetricStatus.not_applicable and f"{MIN_FS_ACLR_HZ / 1e6:.0f}" in m["ACLR_L"].reason
    # a rate without a small exact ratio to the waveform clock
    m = by_name(evaluate(PROFILE_ID, iq(y), None, signal(fs=30.72e6 * math.pi)))
    assert m["EVM_RMS"].status == MetricStatus.not_applicable and "cannot be converted" in m["EVM_RMS"].reason
    # unknown sample rate: nothing can be said
    m = by_name(evaluate(PROFILE_ID, iq(y), None, SignalSpec(waveform=binding())))
    assert all(v.status == MetricStatus.not_applicable for v in m.values())
    # nperseg missing: EVM still works, ACLR does not
    m = by_name(evaluate(PROFILE_ID, iq(y), None, signal(nperseg=None)))
    assert m["EVM_RMS"].status == MetricStatus.ok and m["ACLR_L"].status == MetricStatus.not_applicable


def test_a_bound_dataset_whose_signal_is_not_the_waveform_is_missing_reference(capture):
    rng = np.random.default_rng(3)
    noise = rng.normal(0, 1, (200_000, 2)).astype(np.float32)
    m = by_name(evaluate(PROFILE_ID, noise, None, signal()))
    assert m["EVM_RMS"].status == MetricStatus.missing_reference and "does not correlate" in m["EVM_RMS"].reason
    assert m["ACLR_L"].status == MetricStatus.ok          # the spectrum of whatever was captured is still a fact
    bad = iq(capture[2].copy())
    bad[10, 0] = np.nan
    assert all(v.status == MetricStatus.invalid for v in evaluate(PROFILE_ID, bad, None, signal()))
