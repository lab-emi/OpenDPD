"""``opendpd.sdk.metrics``: the arrays-in, numbers-out entry the MATLAB toolbox uses is the run service's code."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from opendpd.core.metrics import evaluate as core_evaluate
from opendpd.core.waveforms import ofdm
from opendpd.schemas import SignalSpec
from opendpd.schemas.waveform import WaveformBinding, WaveformSpec
from opendpd.sdk import matlab, metrics
from tests.fixtures.synthetic import memory_polynomial_pa

ROOT = Path(__file__).resolve().parents[2]
FS_CAPTURE = 122.88e6


def _capture(seed=1, subframes=2, drive=0.35):
    """The played waveform (float32 as an instrument holds it), interpolated x4, through the synthetic PA."""
    wf = ofdm.generate(WaveformSpec(seed=seed, n_subframes=subframes))
    x30 = wf.x.astype(np.complex64).astype(np.complex128)
    spectrum = np.fft.fft(x30)
    n = x30.size
    padded = np.zeros(4 * n, dtype=np.complex128)
    padded[:n // 2], padded[-(n // 2):] = spectrum[:n // 2], spectrum[-(n // 2):]
    return memory_polynomial_pa(drive * np.fft.ifft(padded) * 4)


def _by_name(rows):
    return {row["name"]: row for row in rows}


def test_waveform_is_the_one_the_generator_makes():
    out = metrics.lte_waveform(seed=3, n_subframes=2)
    wf = ofdm.generate(WaveformSpec(seed=3, n_subframes=2))
    assert out["metadata"]["sha256"] == wf.sha256() and out["metadata"]["n_samples"] == wf.period == 61440
    assert out["iq"].dtype == np.float32 and out["iq"].shape == (wf.period, 2)
    np.testing.assert_array_equal(out["iq"], ofdm.to_iq(wf.x))
    layout = out["symbols_iq"].reshape(wf.symbols.shape[0], wf.symbols.shape[1], 2)
    np.testing.assert_array_equal(layout[..., 0] + 1j * layout[..., 1], wf.symbols.astype(np.complex64))
    assert out["metadata"]["occupied_subcarriers"] == 1200 and out["metadata"]["n_symbols"] == 28


def test_evaluate_is_the_core_evaluation_on_the_same_signal():
    y = _capture()
    rows = _by_name(metrics.evaluate(y, sample_rate_hz=FS_CAPTURE, nperseg=2048, waveform_seed=1, waveform_subframes=2))
    spec = WaveformSpec(seed=1, n_subframes=2)
    signal = SignalSpec(sample_rate_hz=FS_CAPTURE, nperseg=2048,
                        waveform=WaveformBinding(spec=spec, input_offset_samples=0, correlation=0.5,
                                                 input_sample_rate_hz=FS_CAPTURE))
    expected = {m.name: m for m in core_evaluate("ofdm-lte20-evm-v1", np.stack([y.real, y.imag], -1), None, signal)}
    assert set(rows) == {"EVM_RMS", "EVM_DB", "ACLR_L", "ACLR_R"} == set(expected)
    for name, value in expected.items():
        assert rows[name]["value"] == value.value and rows[name]["status"] == "ok" == value.status.value
    assert 1.0 < rows["EVM_RMS"]["value"] < 20.0 and rows["ACLR_L"]["value"] < -20.0


def test_nothing_is_guessed():
    y = _capture()
    bad_seed = _by_name(metrics.evaluate(y, sample_rate_hz=FS_CAPTURE, nperseg=2048, waveform_seed=2, waveform_subframes=2))
    assert bad_seed["EVM_RMS"]["status"] == "missing_reference" and bad_seed["EVM_RMS"]["value"] is None
    assert bad_seed["ACLR_L"]["status"] == "ok"                       # the spectrum does not need the waveform
    unbound = _by_name(metrics.evaluate(y, sample_rate_hz=FS_CAPTURE, nperseg=2048))
    # the LTE profile is defined for a capture of its waveform: without the waveform no metric of it is scored
    assert {row["status"] for row in unbound.values()} == {"missing_reference"}
    assert all(row["value"] is None and "reference waveform" in row["reason"] for row in unbound.values())
    no_segment = _by_name(metrics.evaluate(y, sample_rate_hz=FS_CAPTURE, waveform_seed=1, waveform_subframes=2))
    assert no_segment["ACLR_L"]["status"] == "not_applicable" and "nperseg" in no_segment["ACLR_L"]["reason"]
    assert no_segment["EVM_RMS"]["status"] == "ok"
    slow = _by_name(metrics.evaluate(y[::4], sample_rate_hz=FS_CAPTURE / 4, nperseg=512, waveform_seed=1,
                                     waveform_subframes=2))
    assert slow["ACLR_L"]["status"] == "not_applicable" and "58" in slow["ACLR_L"]["reason"]


@pytest.mark.parametrize("kwargs, message", [
    ({"waveform_seed": 1}, "both the seed"),
    ({"waveform_subframes": 2}, "both the seed"),
    ({"profile": "no-such-profile"}, "not registered"),
])
def test_incomplete_requests_are_refused(kwargs, message):
    with pytest.raises((ValueError, KeyError), match=message):
        metrics.evaluate(_capture(), sample_rate_hz=FS_CAPTURE, nperseg=2048, **kwargs)


def test_input_validation_and_precision():
    with pytest.raises(ValueError, match="finite"):
        metrics.evaluate(np.array([[0.0, np.nan]] * 10), sample_rate_hz=FS_CAPTURE, nperseg=4)
    with pytest.raises(ValueError, match="complex vector or an"):
        metrics.evaluate(np.zeros((10, 3)), sample_rate_hz=FS_CAPTURE, nperseg=4)
    with pytest.raises(ValueError, match="same number of samples"):
        metrics.evaluate(np.ones((10, 2)), reference=np.ones((9, 2)), sample_rate_hz=FS_CAPTURE, nperseg=4,
                         profile="general-spectral-v1")
    wide = np.random.default_rng(0).standard_normal((64, 2))
    assert metrics._iq(wide, "y").dtype == np.float64                  # no rounding to float32
    assert metrics._iq(wide[:, 0] + 1j * wide[:, 1], "y").dtype == np.float64
    assert metrics._iq(wide.astype(np.float32), "y").dtype == np.float32


def test_reference_based_metrics_use_the_reference():
    rng = np.random.default_rng(1)
    target = rng.standard_normal((4096, 2))
    noisy = target + 0.01 * rng.standard_normal((4096, 2))
    rows = _by_name(metrics.evaluate(noisy, reference=target, profile="opendpd-spectral-v2", sample_rate_hz=100e6,
                                     nperseg=256, bandwidth_hz=20e6))
    assert rows["NMSE"]["status"] == "ok" and -45 < rows["NMSE"]["value"] < -35
    no_ref = _by_name(metrics.evaluate(noisy, profile="opendpd-spectral-v2", sample_rate_hz=100e6, nperseg=256,
                                       bandwidth_hz=20e6))
    assert no_ref["NMSE"]["status"] != "ok"


def test_matlab_adapters_return_json_and_arrays():
    iq, symbols, meta = matlab.lte_waveform(1, 1)
    assert iq.shape == (30720, 2) and symbols.shape == (14 * 1200, 2) and json.loads(meta)["n_symbols"] == 14
    y = _capture(subframes=1)
    options = json.dumps({"profile": "ofdm-lte20-evm-v1", "sample_rate_hz": FS_CAPTURE, "nperseg": 1024,
                          "n_sub_ch": 1, "waveform_seed": 1, "waveform_subframes": 1})
    rows = json.loads(matlab.evaluate_metrics(np.stack([y.real, y.imag], -1), None, options))
    assert [r["name"] for r in rows] == ["EVM_RMS", "EVM_DB", "ACLR_L", "ACLR_R"]


def test_recorded_parity_values_are_reproduced_through_this_entry():
    """The product entry equals the numbers registered in docs/performance/matlab-parity.json (S4, float32 as stored)."""
    record = ROOT / "docs" / "performance" / "matlab-parity.json"
    if not record.is_file():
        pytest.skip("no parity record in this checkout")
    items = {i["item"]: i for i in json.loads(record.read_text())["items"]}
    wf = ofdm.generate(WaveformSpec(seed=1, n_subframes=10))
    x30 = ofdm.to_iq(wf.x).astype(np.float64)
    x30 = x30[:, 0] + 1j * x30[:, 1]
    spectrum = np.fft.fft(x30)
    padded = np.zeros(4 * x30.size, dtype=np.complex128)
    padded[:x30.size // 2], padded[-(x30.size // 2):] = spectrum[:x30.size // 2], spectrum[-(x30.size // 2):]
    y = memory_polynomial_pa(0.35 * np.fft.ifft(padded) * 4)
    stored = np.stack([y.real, y.imag], -1).astype(np.float32)
    rows = _by_name(metrics.evaluate(stored, sample_rate_hz=FS_CAPTURE, nperseg=2048, waveform_seed=1, waveform_subframes=10))
    assert rows["EVM_RMS"]["value"] == pytest.approx(items["S4 EVM_RMS (percent)"]["opendpd"], rel=1e-9)
    assert rows["ACLR_L"]["value"] == pytest.approx(items["S4 ACLR_L nperseg=2048 vs comm.ACPR default"]["opendpd"], rel=1e-9)
    assert rows["ACLR_R"]["value"] == pytest.approx(items["S4 ACLR_R nperseg=2048 vs comm.ACPR default"]["opendpd"], rel=1e-9)
