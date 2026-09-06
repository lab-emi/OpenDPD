"""S16: aligning a measured PA output to the signal that was played (opendpd.core.measurement)."""

import numpy as np
import pytest

from opendpd.core.measurement import MIN_CORRELATION, align, level_db, rate_ratio, resample, rms, to_complex, to_iq

def _signals(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    x = 0.3 * (rng.normal(size=n) + 1j * rng.normal(size=n))
    u = x * (1.0 + 0.2 * np.abs(x) ** 2)                      # a pre-distorted version of x
    return x, u


def _pa(v, gain=2.0, phase=0.5, cubic=0.15):
    return gain * np.exp(1j * phase) * (v - cubic * np.abs(v) ** 2 * v)


def test_delay_and_complex_gain_are_recovered_from_a_looped_capture():
    x, u = _signals()
    y = np.roll(np.tile(_pa(u), 2), 251)                       # two periods on air, 251 samples of latency
    window, al = align(y, u, x)
    assert al.delay_samples == 251 and not al.wrapped
    assert al.correlation > 0.99
    # the gain is the least-squares fit onto x, so it absorbs the DPD's gain expansion and the PA's phase
    ref = np.vdot(x, window) / np.vdot(x, x)
    assert al.gain == pytest.approx(ref)
    assert np.degrees(np.angle(al.gain)) == pytest.approx(np.degrees(0.5), abs=3)
    np.testing.assert_allclose(window, _pa(u))                 # the window starts where the roll put sample 0


def test_a_one_period_capture_wraps_only_under_looped_playback():
    x, u = _signals()
    y = np.roll(_pa(u), 40)                                    # exactly one period, the tail came first
    window, al = align(y, u, x, loop=True)
    assert al.delay_samples == 40 and al.wrapped
    np.testing.assert_allclose(window, _pa(u))
    with pytest.raises(ValueError, match="ends 40 samples before"):
        align(y, u, x, loop=False)


def test_short_or_unrelated_captures_are_refused_with_the_reason():
    x, u = _signals()
    with pytest.raises(ValueError, match="at least one full period"):
        align(_pa(u)[:-1], u, x)
    rng = np.random.default_rng(9)
    noise = rng.normal(size=2 * u.size) + 1j * rng.normal(size=2 * u.size)
    with pytest.raises(ValueError, match=f"correlation .* < {MIN_CORRELATION}"):
        align(noise, u, x)
    with pytest.raises(ValueError, match="non-finite"):
        align(np.concatenate([_pa(u), [np.nan]]), u, x)


def test_the_capture_is_scored_as_captured_never_rescaled():
    x, u = _signals()
    window, al = align(_pa(u, gain=7.0), u, x)
    assert rms(window) == pytest.approx(rms(_pa(u, gain=7.0)))
    assert abs(al.gain) > 6.0
    assert level_db(_pa(u, gain=7.0), _pa(u, gain=3.5)) == pytest.approx(20 * np.log10(2), abs=1e-9)


def test_rate_conversion_is_exact_rational_or_refused():
    assert rate_ratio(2e6, 1e6) == (1, 2)
    assert rate_ratio(122.88e6, 30.72e6) == (1, 4)
    assert rate_ratio(1e6, 1e6) == (1, 1)
    assert rate_ratio(1e6 * np.pi, 1e6) is None
    x, u = _signals(n=4000)
    high = resample(_pa(u), 1e6, 3e6)
    assert high.size == 3 * u.size
    back = resample(high, 3e6, 1e6)
    window, al = align(back, u, x)
    assert al.delay_samples == 0 and al.correlation > 0.98    # polyphase filter transients at the edges
    with pytest.raises(ValueError, match="rational ratio"):
        resample(u, 1e6, 1e6 * np.pi)


def test_iq_round_trip():
    z = np.array([1 + 2j, -0.5 + 0.25j])
    iq = to_iq(z)
    assert iq.dtype == np.float32 and iq.shape == (2, 2)
    np.testing.assert_allclose(to_complex(iq), z)
    np.testing.assert_allclose(to_complex(np.array([[1.0, 2.0]])), [1 + 2j])
