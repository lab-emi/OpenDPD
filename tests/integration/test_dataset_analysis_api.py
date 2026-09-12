"""Real capture inspection through Python, HTTP and CLI, with no alternate metric math."""

import contextlib
import io
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from opendpd.commands import main
from opendpd.core.doctor import ANALYSIS_SAMPLES
from opendpd.schemas import PreprocessingParams, SignalSpec
from opendpd.server.app import create_app
from opendpd.server.security import CSRF_HEADER
from opendpd.services import datasets
from opendpd.services.dataset_analysis import analyze_dataset
from opendpd.services.workspace import Workspace

pytestmark = pytest.mark.integration


def iq(z):
    return np.column_stack((z.real, z.imag))


@pytest.fixture()
def capture(tmp_path):
    ws = Workspace.open_or_create(tmp_path / "workspace")
    ws.imports_dir.mkdir(parents=True, exist_ok=True)
    t = np.arange(8192) / 8192
    x = .2 * np.exp(2j * np.pi * 512 * t)
    y = 2 * x + .002 * np.exp(2j * np.pi * 1536 * t)
    source = ws.imports_dir / "capture.npz"
    np.savez(source, input=iq(x), output=iq(y))
    datasets.import_dataset(ws, source, dataset_id="capture", signal=SignalSpec(
        sample_rate_hz=8192, bandwidth_hz=2048, n_sub_ch=1, nperseg=1024, amplitude_units="normalized"))
    return ws


def test_http_python_and_cli_inspect_the_same_capture_without_mutation(capture):
    ws = capture
    before = ws.get_dataset("capture").model_dump_json()
    app = create_app(ws.root, bootstrap_token="inspection-test", shutdown_timeout=2)
    with TestClient(app, base_url="http://127.0.0.1:8877") as client:
        assert client.get("/api/v1/datasets/capture/analysis").status_code == 401
        bootstrap = client.post("/api/v1/session/bootstrap", json={"token": "inspection-test"}).json()
        client.headers[CSRF_HEADER] = bootstrap["csrf_token"]
        response = client.get("/api/v1/datasets/capture/analysis")
        assert response.status_code == 200, response.text
        body = response.json()
        # Strict JSON: no NaN/Infinity sent to plots or rendered as a measurement.
        json.dumps(body, allow_nan=False)
        assert body["sample_range"] == [0, 8192]
        assert body["inspection_ready"] and body["metadata_complete"]
        rows = {row["name"]: row for row in body["measurements"]}
        # Known power ratio: (0.002 / 0.4)^2 in the upper adjacent band.
        assert rows["ACPR_R"]["output"]["value"] == pytest.approx(-46.020599913279625, abs=.005)
        assert rows["PAPR"]["input"]["value"] == pytest.approx(0, abs=1e-5)
        assert rows["BLA_NMSE"]["output"]["status"] == "review_required"
        assert rows["EVM_RMS"]["output"]["status"] == "missing_reference"
        assert "NMSE" not in rows
        assert len(body["time"]["traces"][0]["i"]) <= 256
        assert len(body["iq"]["traces"][0]["i"]) <= 4000
        assert body["iq"]["mode"] == "samples"
        assert body["am"]["n_points"] <= 4000
        python_result = analyze_dataset(ws, "capture")
        assert body["measurements"] == python_result.model_dump(mode="json")["measurements"]
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            assert main(["datasets", "analyze", "capture", "--workspace", str(ws.root)]) == 0
        assert json.loads(out.getvalue())["measurements"] == body["measurements"]
        assert client.get("/api/v1/datasets/capture/analysis?version=missing-v1").status_code == 409
    assert ws.get_dataset("capture").model_dump_json() == before
    assert not (ws.dataset_dir("capture") / "diagnostics").exists()


def test_selected_version_and_metadata_change_the_inspection(capture):
    ws = capture
    raw = analyze_dataset(ws, "capture")
    datasets.create_version(ws, "capture", "scaled-v1", PreprocessingParams(gain_db=6))
    scaled = analyze_dataset(ws, "capture", "scaled-v1")
    rows = lambda result: {m.name: m for m in result.measurements}
    assert scaled.data_version == "scaled-v1"
    assert rows(scaled)["RMS"].output.value == pytest.approx(rows(raw)["RMS"].output.value * 10 ** (-6 / 20), rel=1e-6)
    assert rows(scaled)["ACPR_R"].output.value == pytest.approx(rows(raw)["ACPR_R"].output.value, abs=.005)
    datasets.update_manifest(ws, "capture", signal=SignalSpec(nperseg=2560))
    missing = analyze_dataset(ws, "capture")
    assert not missing.metadata_complete and not missing.inspection_ready
    assert missing.spectrum.axis == "normalized" and missing.spectrum.bands is None
    assert np.all(np.diff(missing.spectrum.frequency) > 0), "normalised FFT bins must not collapse through display rounding"
    assert rows(missing)["ACPR_R"].output.status == "not_applicable"
    assert rows(missing)["ACPR_R"].output.value is None


def test_large_capture_uses_a_bounded_contiguous_window(capture):
    ws = capture
    n = ANALYSIS_SAMPLES + 5000
    t = np.arange(n) / 8192
    x = .2 * np.exp(2j * np.pi * 512 * t)
    source = ws.imports_dir / "long.npz"
    np.savez(source, input=iq(x), output=iq(2 * x))
    datasets.import_dataset(ws, source, dataset_id="long", signal=ws.get_dataset("capture").signal)
    result = analyze_dataset(ws, "long")
    assert result.sample_range == (2500, 2500 + ANALYSIS_SAMPLES)
    assert result.spectrum.n_samples == ANALYSIS_SAMPLES
    assert result.time.start == 2500
    assert result.total_samples == n


def test_invalid_capture_is_not_silently_cleaned_for_charts(capture):
    ws = capture
    x = np.ones((4096, 2), dtype=np.float32)
    y = x.copy()
    y[12, 0] = np.nan
    source = ws.imports_dir / "invalid.npz"
    np.savez(source, input=x, output=y)
    datasets.import_dataset(ws, source, dataset_id="invalid", signal=ws.get_dataset("capture").signal)
    result = analyze_dataset(ws, "invalid")
    assert result.spectrum is None and result.time is None and result.iq is None and result.am is None
    assert not result.inspection_ready
    assert next(m for m in result.measurements if m.name == "RMS").output.status == "invalid"
    json.dumps(result.model_dump(mode="json"), allow_nan=False)


def test_short_capture_returns_actionable_state(capture):
    ws = capture
    source = ws.imports_dir / "short.npz"
    np.savez(source, input=np.ones((20, 2)), output=np.ones((20, 2)))
    datasets.import_dataset(ws, source, dataset_id="short", signal=ws.get_dataset("capture").signal, guard_samples=0)
    result = analyze_dataset(ws, "short")
    assert not result.inspection_ready and result.spectrum is None
    assert result.diagnostics.items[0].code == "too_few_samples"
    assert all(m.output.value is None for m in result.measurements)


def test_builtin_http_returns_the_existing_dataset_demodulator_symbols(tmp_path):
    from datasets.demodulator import Demodulator

    ws = Workspace.open_or_create(tmp_path / "workspace")
    ws.register_builtin_dataset("DPA_200MHz", dataset_id="renamed-capture")
    x, y, _ = datasets.load_version_arrays(ws, "renamed-capture")
    demod = Demodulator.from_dataset("DPA_200MHz")
    xc, yc = x.astype(np.float64) @ np.array([1, 1j]), y.astype(np.float64) @ np.array([1, 1j])
    expected = [demod.demodulate(xc), demod.demodulate(yc, sync_signal=xc, equalize=True)]
    before = ws.get_dataset("renamed-capture").model_dump_json()
    with TestClient(create_app(ws.root, bootstrap_token="demod-test", shutdown_timeout=2),
                    base_url="http://127.0.0.1:8877") as client:
        client.post("/api/v1/session/bootstrap", json={"token": "demod-test"})
        response = client.get("/api/v1/datasets/renamed-capture/analysis")
        assert response.status_code == 200, response.text
        result = response.json()
    c = result["constellation"]
    assert c["status"] == "ok" and c["modulation"] == "64QAM"
    assert c["sample_range"] == c["source_sample_range"] == [0, 38400]
    assert c["fft_size"] == 2560 and c["n_carriers"] == 10 and c["active_subcarriers_per_carrier"] == 64
    assert c["traces"][0]["n_symbols"] == 9600
    assert not c["traces"][0]["equalized"] and c["traces"][1]["equalized"]
    assert "equalized" in c["traces"][1]["name"]
    for trace, (ri, rq) in zip(c["traces"], expected):
        assert len(trace["i"]) <= 4000
        np.testing.assert_array_equal(trace["i"], ri[::trace["stride"]])
        np.testing.assert_array_equal(trace["q"], rq[::trace["stride"]])
    assert result["iq"]["mode"] == "samples"
    assert next(m for m in result["measurements"] if m["name"] == "EVM_RMS")["output"]["status"] == "missing_reference"
    assert ws.get_dataset("renamed-capture").model_dump_json() == before


@pytest.mark.parametrize("delay, origin", [(-3.5, 4), (3.5, 1), (-3, 3), (3, 0)])
def test_processed_constellation_preserves_raw_frame_origin(tmp_path, delay, origin):
    from datasets.demodulator import Demodulator

    ws = Workspace.open_or_create(tmp_path / "workspace")
    ws.register_builtin_dataset("DPA_200MHz")
    raw, _, _ = datasets.load_version_arrays(ws, "dpa-200mhz")
    datasets.create_version(ws, "dpa-200mhz", "aligned-v1", PreprocessingParams(delay_samples=delay))
    c = analyze_dataset(ws, "dpa-200mhz", "aligned-v1").constellation
    assert c.status == "ok"
    assert c.source_sample_range[0] == (2560 if origin else 0)
    assert all(s % 2560 == 0 for s in c.source_sample_range)
    assert c.source_sample_range[0] - c.sample_range[0] == origin
    start, end = c.source_sample_range
    xc = raw[start:end].astype(np.float64) @ np.array([1, 1j])
    ri, rq = Demodulator.from_dataset("DPA_200MHz").demodulate(xc)
    trace = c.traces[0]
    np.testing.assert_array_equal(trace.i, ri[::trace.stride])
    np.testing.assert_array_equal(trace.q, rq[::trace.stride])


def test_constellation_does_not_guess_from_labels_or_stale_metadata(capture):
    from opendpd.services.dataset_constellation import dataset_constellation

    ws = capture
    m = ws.get_dataset("capture")
    # Even the packaged name and a 64QAM label do not bind imported code/data.
    m.source.name = "DPA_200MHz"
    m.signal.modulation = "64QAM"
    ws.save_dataset(m)
    assert analyze_dataset(ws, "capture").constellation.status == "unavailable"
    ws.register_builtin_dataset("DPA_200MHz")
    m = ws.get_dataset("dpa-200mhz")
    x, y, _ = datasets.load_version_arrays(ws, m.dataset_id)
    m.signal.nperseg = 1280
    c = dataset_constellation(m, "raw-v1", x, y)
    assert c.status == "unavailable" and "nperseg" in c.reason
    m.signal.nperseg = 2560
    y[10, 0] = np.nan
    assert dataset_constellation(m, "raw-v1", x, y).status == "invalid"


def test_long_builtin_constellation_rounds_the_analysis_window_to_frames(tmp_path):
    ws = Workspace.open_or_create(tmp_path / "workspace")
    ws.register_builtin_dataset("DPA_160MHz")
    result = analyze_dataset(ws, "dpa-160mhz")
    c = result.constellation
    assert c.status == "ok" and c.fft_size == 16384
    assert c.sample_range[0] >= result.sample_range[0]
    assert c.sample_range[1] <= result.sample_range[1]
    assert all(s % c.fft_size == 0 for s in c.source_sample_range)
    assert c.sample_range[1] - c.sample_range[0] <= ANALYSIS_SAMPLES
    assert all(len(t.i) <= 4000 for t in c.traces)


@pytest.mark.parametrize("name", ["APA_200MHz", "APA_200MHz_b"])
def test_cp_dataset_uses_its_receiver_and_rejects_an_unknown_carrier_phase_origin(tmp_path, name):
    ws = Workspace.open_or_create(tmp_path / "workspace")
    manifest = ws.register_builtin_dataset(name)
    result = analyze_dataset(ws, manifest.dataset_id)
    c = result.constellation
    assert c.status == "ok" and "OFDMCPDemodulator" in c.demodulator
    assert c.fft_size == 32768 and c.n_carriers == 5
    assert c.sample_range == (0, 98304)
    assert c.traces[0].n_symbols == 3000 and c.traces[1].equalized
    datasets.create_version(ws, manifest.dataset_id, "cropped-v1", PreprocessingParams(delay_samples=-1))
    c = analyze_dataset(ws, manifest.dataset_id, "cropped-v1").constellation
    assert c.status == "unavailable" and "carrier phase" in c.reason
    assert not c.traces
