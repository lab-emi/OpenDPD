"""Analytic references for ``general-spectral-v1`` (plan S08).

Every number the profile can produce is checked against a closed-form
expectation on deterministic signals; statuses are checked to be explicit
(never a fabricated 0 or a silent NaN).
"""

import math

import numpy as np
import pytest

from opendpd.core.metrics import evaluate, get_profile
from opendpd.core.metrics.general_v1 import FLOOR_DB, PROFILE_ID
from opendpd.schemas import MetricStatus, SignalSpec

FS, BW, NPERSEG = 800e6, 200e6, 2560
SIGNAL = SignalSpec(sample_rate_hz=FS, bandwidth_hz=BW, n_sub_ch=10, nperseg=NPERSEG, amplitude_units="normalized")
N = 64 * NPERSEG


def tones(freqs, amps, n=N, fs=FS):
    t = np.arange(n) / fs
    z = np.zeros(n, dtype=np.complex128)
    for k, (f, a) in enumerate(zip(freqs, amps)):
        z += a * np.exp(1j * (2 * np.pi * f * t + 2 * np.pi * ((k * 0.6180339887) % 1.0)))
    return z


def iq(z):
    return np.stack([z.real, z.imag], axis=-1)


def reference():
    """32 tones well inside the main channel: total power is the sum of a_k^2."""
    freqs = np.linspace(-0.4 * BW, 0.4 * BW, 32)
    amps = np.full(32, 0.05)
    return tones(freqs, amps), float(np.sum(amps ** 2))


def by_name(metrics):
    return {m.name: m for m in metrics}


def test_zero_error_reaches_the_documented_floor_not_minus_infinity():
    r, _ = reference()
    m = by_name(evaluate(PROFILE_ID, iq(r), iq(r), SIGNAL))
    assert m["NMSE"].status == MetricStatus.ok and m["NMSE"].value == FLOOR_DB
    assert m["IBE"].status == MetricStatus.ok and m["IBE"].value == FLOOR_DB
    assert get_profile(PROFILE_ID).parameters["floor_db"] == FLOOR_DB


def test_known_complex_gain_error_is_exact():
    r, _ = reference()
    g = 1.1 * np.exp(1j * np.deg2rad(10))
    expected = 20 * math.log10(abs(g - 1))
    m = by_name(evaluate(PROFILE_ID, iq(g * r), iq(r), SIGNAL))
    assert m["NMSE"].value == pytest.approx(expected, abs=1e-9)
    assert m["IBE"].value == pytest.approx(expected, abs=1e-6)     # linear: error PSD = |g-1|^2 * reference PSD


def test_known_noise_power_gives_its_snr_pooled_and_in_band():
    r, p_ref = reference()
    rng = np.random.default_rng(0)
    snr_db = 30.0
    sigma2 = p_ref * 10 ** (-snr_db / 10)
    noise = math.sqrt(sigma2 / 2) * (rng.standard_normal(N) + 1j * rng.standard_normal(N))
    m = by_name(evaluate(PROFILE_ID, iq(r + noise), iq(r), SIGNAL))
    assert m["NMSE"].value == pytest.approx(-snr_db, abs=0.1)
    # white noise: only the BW/FS fraction of its power falls into the main channel
    assert m["IBE"].value == pytest.approx(-snr_db + 10 * math.log10(BW / FS), abs=0.15)


def test_known_adjacent_band_power_gives_the_leakage_ratio_in_dbc():
    r, p_ref = reference()
    a_right, a_left = 0.02, 0.005
    y = r + tones([BW, -BW], [a_right, a_left])      # tone centred in each adjacent channel
    m = by_name(evaluate(PROFILE_ID, iq(y), iq(r), SIGNAL))
    assert m["ACPR_R"].value == pytest.approx(10 * math.log10(a_right ** 2 / p_ref), abs=0.05)
    assert m["ACPR_L"].value == pytest.approx(10 * math.log10(a_left ** 2 / p_ref), abs=0.05)
    assert m["ACPR_R"].unit == "dBc" and m["ACPR_R"].better.value == "lower"
    assert m["ACPR_R"].value < 0, "leakage convention: adjacent / main is negative dBc"


def test_pooled_nmse_is_not_the_mean_of_per_segment_db():
    r, _ = reference()
    seg = r.reshape(-1, NPERSEG)
    err = np.empty_like(seg)
    err[: len(seg) // 2] = 0.1 * seg[: len(seg) // 2]     # power ratio 1e-2 in the first half
    err[len(seg) // 2:] = 0.01 * seg[len(seg) // 2:]      # power ratio 1e-4 in the second half
    y = (seg + err).reshape(-1)
    p1 = float(np.sum(np.abs(seg[: len(seg) // 2]) ** 2))
    p2 = float(np.sum(np.abs(seg[len(seg) // 2:]) ** 2))
    general = by_name(evaluate(PROFILE_ID, iq(y), iq(r), SIGNAL))
    legacy = by_name(evaluate("legacy-opendpd-v1", iq(y).reshape(-1, NPERSEG, 2), iq(r).reshape(-1, NPERSEG, 2), SIGNAL))
    assert general["NMSE"].value == pytest.approx(10 * math.log10((1e-2 * p1 + 1e-4 * p2) / (p1 + p2)), abs=1e-9)
    assert legacy["NMSE"].value == pytest.approx((-20 - 40) / 2, abs=1e-9)
    assert abs(general["NMSE"].value - legacy["NMSE"].value) > 5


def test_valid_range_excludes_zero_padding_but_legacy_keeps_it():
    r, _ = reference()
    y = 1.05 * r
    n_valid = N - 1000
    padded_r = np.concatenate([r[:n_valid], np.zeros(1000)])
    padded_y = np.concatenate([y[:n_valid], np.zeros(1000)])
    exact = by_name(evaluate(PROFILE_ID, iq(y[:n_valid]), iq(r[:n_valid]), SIGNAL))
    trimmed = by_name(evaluate(PROFILE_ID, iq(padded_y), iq(padded_r), SIGNAL, valid_samples=n_valid))
    assert trimmed["NMSE"].value == exact["NMSE"].value == pytest.approx(20 * math.log10(0.05), abs=1e-9)
    assert trimmed["IBE"].value == pytest.approx(exact["IBE"].value, abs=1e-9)


@pytest.mark.parametrize("case", ["no_reference", "zero_reference", "nan", "no_metadata", "band_outside", "too_short"])
def test_every_undefined_case_has_an_explicit_status(case):
    r, _ = reference()
    y, ref, signal, n_valid = 1.05 * r, r, SIGNAL, None
    if case == "no_reference":
        ref = None
    elif case == "zero_reference":
        ref = np.zeros_like(r)
    elif case == "nan":
        y = y.copy()
        y[10] = np.nan
    elif case == "no_metadata":
        signal = SignalSpec()
    elif case == "band_outside":
        signal = SignalSpec(sample_rate_hz=2.2 * BW, bandwidth_hz=BW, nperseg=NPERSEG)
    elif case == "too_short":
        n_valid = NPERSEG - 1
    m = by_name(evaluate(PROFILE_ID, iq(y), None if ref is None else iq(ref), signal, valid_samples=n_valid))
    for v in m.values():
        assert (v.status == MetricStatus.ok) == (v.value is not None), "a value exists exactly when the status is ok"
        assert v.status == MetricStatus.ok or v.reason
    expect = {
        "no_reference": {"NMSE": "missing_reference", "IBE": "missing_reference", "ACPR_L": "ok", "ACPR_R": "ok"},
        "zero_reference": {"NMSE": "invalid", "IBE": "invalid", "ACPR_L": "ok", "ACPR_R": "ok"},
        "nan": {"NMSE": "invalid", "IBE": "invalid", "ACPR_L": "invalid", "ACPR_R": "invalid"},
        "no_metadata": {"NMSE": "ok", "IBE": "not_applicable", "ACPR_L": "not_applicable", "ACPR_R": "not_applicable"},
        "band_outside": {"NMSE": "ok", "IBE": "ok", "ACPR_L": "not_applicable", "ACPR_R": "not_applicable"},
        "too_short": {"NMSE": "ok", "IBE": "not_applicable", "ACPR_L": "not_applicable", "ACPR_R": "not_applicable"},
    }[case]
    assert {k: v.status.value for k, v in m.items()} == expect
    if case == "band_outside":
        assert "exceeds the captured range" in m["ACPR_R"].reason
