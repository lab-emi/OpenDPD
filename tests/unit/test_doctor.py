"""Dataset Doctor against the fixture protocol (tests/fixtures/manifest.json: doctor_protocol)."""

import json
from pathlib import Path

import numpy as np
import pytest

from opendpd.core.doctor import DOCTOR_VERSION, diagnose, estimates_from_report
from opendpd.schemas import SignalSpec
from tests.fixtures.synthetic import Impairments, synthesize

PROTOCOL = json.loads((Path(__file__).resolve().parents[1] / "fixtures" / "manifest.json").read_text())["doctor_protocol"]
SIGNAL = SignalSpec(sample_rate_hz=800e6, bandwidth_hz=200e6, n_sub_ch=10, nperseg=2560, amplitude_units="normalized")
N = 20_000


def report_for(imp: Impairments, signal: SignalSpec = SIGNAL, seed: int = 0):
    x, y = synthesize(N, seed, impairments=imp)
    return diagnose(x, y, signal, dataset_id="syn", raw_sha256="a" * 64)


def codes(report):
    return {i.code: i for i in report.items}


def test_clean_pa_is_not_called_broken():
    r = report_for(Impairments())
    c = codes(r)
    assert r.doctor_version == DOCTOR_VERSION and not r.evaluation_blocked
    assert not any(k.startswith("possible_") or k in ("output_outliers", "non_finite_samples", "time_misalignment") for k in c)
    assert "alignment_ok" in c and abs(c["alignment_ok"].evidence["delay_samples"]) < PROTOCOL["delay"]["tolerance_samples"]
    assert all(i.evidence for i in r.items), "every finding carries evidence"


@pytest.mark.parametrize("delay", [3, 7, 40])
def test_delay_within_protocol_tolerance(delay):
    r = report_for(Impairments(delay_samples=delay))
    est = estimates_from_report(r)
    assert abs(est.delay_samples - delay) <= PROTOCOL["delay"]["tolerance_samples"]
    assert est.delay_confidence >= PROTOCOL["delay"]["min_confidence"]
    item = codes(r)["time_misalignment"]
    assert item.severity.value == "warning" and "delay correction" in item.suggestion


def test_gain_and_phase_within_tolerance():
    g = 1.4 * np.exp(1j * np.radians(25))
    est = estimates_from_report(report_for(Impairments(gain=g)))
    assert abs(est.gain_db - 20 * np.log10(1.4)) <= PROTOCOL["gain"]["tolerance_db"]
    assert abs(est.phase_deg - 25) <= PROTOCOL["gain"]["tolerance_deg"]


def test_clipping_detected_only_when_a_plateau_exists():
    _, y_clean = synthesize(N, 0)
    peak = float(np.hypot(y_clean[:, 0], y_clean[:, 1]).max())
    c = codes(report_for(Impairments(clip_level=0.6 * peak)))
    assert "possible_output_clipping" in c
    assert c["possible_output_clipping"].severity.value == "warning"
    assert c["possible_output_clipping"].evidence["fraction_at_peak"] > 1e-3
    assert "intentional" in c["possible_output_clipping"].suggestion
    assert "possible_output_clipping" not in codes(report_for(Impairments()))


def test_outliers_found_with_indices_and_no_false_positives():
    imp = Impairments(n_outliers=5)
    c = codes(report_for(imp))
    assert c["output_outliers"].evidence["count"] == 5
    assert len(c["output_outliers"].evidence["first_indices"]) == 5
    assert "output_outliers" not in codes(report_for(Impairments()))


def test_non_finite_and_missing_metadata_block_evaluation():
    r = report_for(Impairments(nan_indices=(100, 2000)))
    c = codes(r)
    assert r.evaluation_blocked and c["non_finite_samples"].blocking
    assert c["non_finite_samples"].evidence["first_indices"] == [100, 2000]
    r2 = report_for(Impairments(), signal=SignalSpec())
    assert r2.evaluation_blocked and set(codes(r2)["metadata_missing"].evidence["missing"]) == {
        "sample_rate_hz", "bandwidth_hz", "n_sub_ch", "nperseg"}


def test_length_mismatch_and_bandwidth_checks():
    x, y = synthesize(N, 0)
    r = diagnose(x, y[:-10], SIGNAL, dataset_id="syn")
    assert codes(r)["length_mismatch"].blocking
    narrow = SignalSpec(sample_rate_hz=250e6, bandwidth_hz=200e6, n_sub_ch=10, nperseg=2560)
    x2, y2 = synthesize(N, 0, fs=250e6, bandwidth=200e6)
    assert "insufficient_oversampling" in codes(diagnose(x2, y2, narrow, dataset_id="syn"))
    wrong = SignalSpec(sample_rate_hz=800e6, bandwidth_hz=50e6, n_sub_ch=10, nperseg=2560)
    assert "bandwidth_metadata_mismatch" in codes(diagnose(x, y, wrong, dataset_id="syn"))


def test_silent_output_is_a_blocking_error_with_both_rms_values():
    x, _ = synthesize(N, 0, impairments=Impairments())
    r = diagnose(x, np.zeros_like(x), SIGNAL, dataset_id="syn", raw_sha256="a" * 64)
    item = codes(r)["silent_signal"]
    assert r.evaluation_blocked and item.blocking
    assert item.evidence["rms_output"] == 0.0 and item.evidence["rms_input"] > 0
