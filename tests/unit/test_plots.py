"""``plots-v1`` derived data: what is drawn is what was measured, with fixed budgets (plan S11)."""

import numpy as np
import pytest

from opendpd.core import plots

FS, BW, NPERSEG = 800e6, 200e6, 2560
N = 8 * NPERSEG


def tone(f, amp=1.0, n=N, fs=FS):
    t = np.arange(n) / fs
    return amp * np.exp(2j * np.pi * f * t)


def iq(z):
    return np.stack([z.real, z.imag], axis=-1).astype(np.float32)


def test_spectrum_uses_the_profile_estimator_and_peaks_at_the_tone():
    data = plots.spectrum({"tone": iq(tone(50e6))}, {"tone": "primary"}, sample_rate_hz=FS, nperseg=NPERSEG, bandwidth_hz=BW)
    assert data["version"] == "plots-v1" and data["axis"] == "hz" and data["nperseg"] == NPERSEG
    freq = np.asarray(data["frequency"])
    psd = np.asarray(data["traces"][0]["psd_db"])
    assert len(freq) == len(psd) == NPERSEG
    assert abs(freq[int(np.argmax(psd))] - 50e6) <= FS / NPERSEG
    assert data["bands"] == {"main": [-BW / 2, BW / 2], "adjacent": [[-3 * BW / 2, -BW / 2], [BW / 2, 3 * BW / 2]]}
    assert data["traces"][0]["role"] == "primary"
    assert "welch" in data["estimator"]


def test_spectrum_without_sample_rate_uses_a_normalised_axis_and_no_bands():
    data = plots.spectrum({"z": iq(tone(50e6))}, {}, sample_rate_hz=None, nperseg=None, bandwidth_hz=None)
    assert data["axis"] == "normalized" and data["bands"] is None
    freq = np.asarray(data["frequency"])
    assert freq.min() >= -0.5 and freq.max() < 0.5


def test_spectrum_respects_the_valid_range_and_never_emits_minus_infinity():
    z = np.zeros(N, dtype=np.complex128)
    z[: NPERSEG * 4] = tone(10e6, n=NPERSEG * 4)
    data = plots.spectrum({"z": iq(z)}, {}, sample_rate_hz=FS, nperseg=NPERSEG, bandwidth_hz=BW, valid_samples=NPERSEG * 4)
    assert data["n_samples"] == NPERSEG * 4
    silent = plots.spectrum({"z": iq(np.zeros(N, dtype=np.complex128))}, {}, sample_rate_hz=FS, nperseg=NPERSEG, bandwidth_hz=BW)
    assert all(np.isfinite(silent["traces"][0]["psd_db"]))


def test_time_excerpt_is_a_fixed_window_with_identical_indices():
    z = tone(1e6)
    data = plots.time_excerpt({"a": iq(z), "b": iq(2 * z)}, {"a": "input"}, start=10, n=100)
    assert data["start"] == 10 and data["n"] == 100 and data["n_samples"] == N
    a, b = data["traces"]
    assert len(a["i"]) == len(b["q"]) == 100
    assert a["i"][0] == pytest.approx(z[10].real, abs=1e-6) and b["i"][0] == pytest.approx(2 * z[10].real, abs=1e-6)
    assert a["role"] == "input" and b["role"] == "primary"


def test_am_am_pm_of_a_linear_gain_and_a_phase_rotation_are_exact():
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(N) + 1j * rng.standard_normal(N)) * 0.3
    g = 2.5 * np.exp(1j * np.deg2rad(30.0))
    data = plots.am_am_pm(iq(x), {"lin": iq(g * x)}, {"lin": "primary"}, max_points=1000)
    assert data["n_points"] <= 1000 and data["stride"] == int(np.ceil(N / 1000))
    amp_in = np.asarray(data["amp_in"])
    tr = data["traces"][0]
    np.testing.assert_allclose(np.asarray(tr["amp_out"]), 2.5 * amp_in, atol=2e-5)
    np.testing.assert_allclose(np.asarray(tr["phase_deg"]), 30.0, atol=1e-2)


def test_budgets_bound_the_payload_regardless_of_capture_length():
    z = tone(1e6, n=64 * NPERSEG)
    am = plots.am_am_pm(iq(z), {"y": iq(z)}, {})
    assert am["n_points"] <= plots.AM_POINTS
    tm = plots.time_excerpt({"z": iq(z)}, {})
    assert tm["n"] == plots.TIME_EXCERPT_SAMPLES
