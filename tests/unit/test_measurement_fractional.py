"""Versioned measurement alignment, checked against independently generated delayed tones."""

import numpy as np
import pytest

from opendpd.core.measurement import align, align_fractional, FRACTIONAL_HALF_TAPS
from opendpd.schemas.measurement import MeasurementConfig


def tones(t, n=2048):
    # Evaluate sinusoids at arbitrary times; do not use the correction implementation to generate truth.
    rng = np.random.default_rng(912)
    bins = rng.choice(np.arange(-479, 480), 64, replace=False)
    weights = np.exp(1j * rng.uniform(-np.pi, np.pi, len(bins))) / 8
    return np.exp(2j * np.pi * np.outer(t, bins / n)) @ weights


@pytest.mark.parametrize("fraction", [-.49, -.27, 0, .31, .49])
@pytest.mark.parametrize("loop", [True, False])
def test_fractional_delay_phase_and_gain_against_analytic_waveform(fraction, loop):
    n, delay = 2048, 37
    x = tones(np.arange(n))
    gain = 2.4 * np.exp(.61j)
    capture = gain * tones(np.arange(n if loop else n + 80) - delay - fraction)
    original = capture.copy()
    y, record = align_fractional(capture, x, x, loop=loop)
    assert record.integer.delay_samples == delay
    assert record.fractional_delay_samples == pytest.approx(fraction, abs=2e-5)
    edge = 0 if loop else FRACTIONAL_HALF_TAPS
    assert record.valid_sample_range == (edge, n-edge)
    assert record.gain == pytest.approx(gain, rel=2e-5)
    np.testing.assert_allclose(y, gain * x[edge:n-edge], atol=8e-5)
    np.testing.assert_array_equal(capture, original)
    assert record.correlation > .999999
    assert record.diagnostics[-1]["complex_fit_nmse_db"] < -90
    assert record.diagnostics[-1]["magnitude_fit_nmse_db"] > -10  # uncorrected phase remains visible


def test_fractional_protocol_preserves_power_and_exposes_integer_ablation():
    n = 2048
    x = tones(np.arange(n))
    capture = 7 * tones(np.arange(n) - 23.37)
    legacy, before = align(capture, x, x)
    corrected, fractional = align_fractional(capture, x, x)
    legacy_again, after = align(capture, x, x)
    np.testing.assert_array_equal(legacy, legacy_again)
    assert before == after
    assert np.linalg.norm(corrected) == pytest.approx(np.linalg.norm(capture), rel=1e-12)
    stages = fractional.diagnostics
    assert stages[2]["complex_fit_nmse_db"] < stages[1]["complex_fit_nmse_db"] - 80
    assert stages[0]["correlation"] < stages[1]["correlation"] < stages[2]["correlation"]


def test_single_shot_filter_does_not_wrap_file_edges_into_valid_interval():
    n = 2048
    x = tones(np.arange(n))
    capture = tones(np.arange(n + 80) - 37.31)
    y, record = align_fractional(capture, x, x, loop=False)
    # Explicit finite impulse response support gives a documented retained sample interval.
    assert len(y) == n - 128
    assert "no circular wrapping" in record.boundary_method
    with pytest.raises(ValueError, match="capture ends"):
        align_fractional(capture[:n], x, x, loop=False)


def test_legacy_config_defaults_to_integer_and_versions_are_closed():
    assert MeasurementConfig.model_fields["processing_version"].default == "measurement-integer-v1"
    from pydantic import TypeAdapter, ValidationError
    field = TypeAdapter(MeasurementConfig.model_fields["processing_version"].annotation)
    with pytest.raises(ValidationError):
        field.validate_python("latest")
