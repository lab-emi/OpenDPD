"""Scientific review invariants: declared power, full identities and exact band coordinates."""

import numpy as np
import pytest
from pydantic import ValidationError

from opendpd.core.metrics import incompatibilities
from opendpd.schemas import SignalSpec
from opendpd.schemas.examples import result_dpd_surrogate_mock
from opendpd.schemas.rf import RFConditions
from opendpd.schemas.review import FigurePanel, FigureSpec
from opendpd.services.review import integration_bands


def test_surrogate_identity_uses_all_64_hash_characters():
    a = result_dpd_surrogate_mock()
    b = a.model_copy(deep=True)
    pa = next(m for m in a.models if m.role == "pa")
    other = next(m for m in b.models if m.role == "pa")
    other.weights_sha256 = pa.weights_sha256[:12] + "f" * 52
    assert any(s.startswith("PA surrogate:") for s in incompatibilities(a, b))


def test_declared_conditions_do_not_silently_join_different_operating_points():
    a = result_dpd_surrogate_mock()
    b = a.model_copy(deep=True)
    a.rf_conditions = RFConditions(note="instrument reading", average_output_power_dbm=20, mode="A")
    b.rf_conditions = RFConditions(note="instrument reading", average_output_power_dbm=19, mode="B")
    reasons = incompatibilities(a, b)
    assert any("average output power dbm: 20.0 vs 19.0" in r for r in reasons)
    assert any("mode: A vs B" in r for r in reasons)


@pytest.mark.parametrize("fields", [{"vswr": 2}, {"pa_dc_power_w": -1}, {"input_power_dbm": float("inf")},
                                     {"dc_rails_w": {"PA": 2}, "included_rails": ["PA", "PA"]},
                                     {"dc_rails_w": {"PA": float("nan")}}, {"included_rails": ["unknown"]}])
def test_invalid_physical_claims_are_rejected(fields):
    with pytest.raises(ValidationError):
        RFConditions(note="test", **fields)


def test_legacy_bands_match_actual_frozen_index_slices_and_are_not_full_width_adjacent():
    sig = SignalSpec(sample_rate_hz=800e6, bandwidth_hz=200e6, nperseg=2560, n_sub_ch=10)
    bands, note = integration_bands("legacy-opendpd-v1", sig)
    freq = np.fft.fftshift(np.fft.fftfreq(sig.nperseg, 1 / sig.sample_rate_hz))
    left, right = np.where(freq >= -100e6)[0].min(), np.where(freq <= 100e6)[0].max()
    count = int((right - left) / sig.n_sub_ch)
    expected = [(freq[left - count], freq[left]), (freq[right], freq[right + count])]
    assert [b.edges_hz for b in bands if b.role == "adjacent"] == expected
    assert "strongest subchannel" in note
    assert len([b for b in bands if b.role == "subchannel"]) == 10


def test_out_of_capture_bands_are_unavailable_and_missing_metadata_is_not_guessed():
    bands, _ = integration_bands("general-spectral-v1", SignalSpec(sample_rate_hz=400e6, bandwidth_hz=200e6))
    assert [b.available for b in bands] == [True, False, False]
    assert all(b.reason for b in bands if not b.available)
    assert integration_bands("legacy-opendpd-v1", SignalSpec(sample_rate_hz=800e6, bandwidth_hz=200e6))[0] == []


def test_figures_reject_nonfinite_axes_and_foreign_references():
    trace = dict(run_id="a", trace_name="output")
    with pytest.raises(ValidationError):
        FigurePanel(traces=[trace], x_range=[0, float("nan")])
    with pytest.raises(ValidationError):
        FigurePanel(traces=[trace], y_range=[-1, -2])
    with pytest.raises(ValidationError):
        FigureSpec(title="example", reference_run_id="b", profiles={"a": "general-spectral-v1"}, panels=[dict(traces=[trace])])
