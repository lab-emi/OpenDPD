"""MATLAB-facing numeric layouts, MAT files and SDK diagnostics."""

import json
import subprocess
import sys

import numpy as np
import pytest
from scipy.io import savemat

from opendpd.sdk import Job, Project, SDKError
from opendpd.sdk.iq import as_iq, read_mat


@pytest.mark.parametrize("shape", [(7,), (1, 7), (7, 1)])
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_complex_orientation_and_precision(shape, dtype):
    x = np.array([1 + 2j, 3 - 4j, -1j, 0, 0.25 + 0.125j, 9, -3j], dtype=dtype)
    actual = as_iq(x.reshape(shape))
    np.testing.assert_array_equal(actual[:, 0], x.real.astype(np.float32))
    np.testing.assert_array_equal(actual[:, 1], x.imag.astype(np.float32))
    assert actual.dtype == np.float32 and actual.flags.c_contiguous


@pytest.mark.parametrize("bad", [np.array([]), np.ones((2, 3)), np.ones((2, 3), complex),
    np.array([True]), np.array([1], dtype=np.int64), np.array(["1"]), np.array([1], dtype=object),
    np.array([np.nan]), np.array([np.inf]), np.array([1e40])])
def test_invalid_iq_is_rejected(bad):
    with pytest.raises(ValueError):
        as_iq(bad)


def test_single_sample_real_iq_and_real_vector_are_distinct():
    np.testing.assert_array_equal(as_iq(np.array([[1., 2.]])), [[1., 2.]])
    np.testing.assert_array_equal(as_iq(np.array([1., 2.])), [[1., 0.], [2., 0.]])


def test_mat_numeric_variables_preserve_source_dtype(tmp_path):
    path = tmp_path / "capture 中文.mat"
    x = np.arange(20, dtype=np.float64) + 0.5j
    savemat(path, {"tx": x.reshape(1, -1), "rx": (0.8 * x).reshape(-1, 1), "ignore": "unused"})
    returned, tx, rx = read_mat(path, "tx", "rx")
    assert returned == path and tx.dtype == np.complex128
    np.testing.assert_array_equal(tx, x)
    np.testing.assert_allclose(rx, 0.8 * x)


def test_mat_real_vector_is_not_misread_as_iq(tmp_path):
    path = tmp_path / "real.mat"
    savemat(path, {"x": [[1., 2.]], "y": [[3.], [4.]]})
    _, x, y = read_mat(path)
    np.testing.assert_array_equal(as_iq(x), [[1., 0.], [2., 0.]])
    np.testing.assert_array_equal(as_iq(y), [[3., 0.], [4., 0.]])


@pytest.mark.parametrize("values", [{"x": np.ones(4)}, {"x": {"v": 1}, "y": np.ones(4)},
    {"x": np.array([1], dtype=object), "y": np.ones(4)}])
def test_mat_missing_and_non_numeric_variables_are_rejected(tmp_path, values):
    path = tmp_path / "invalid.mat"
    savemat(path, values)
    with pytest.raises(ValueError, match="variable"):
        read_mat(path)


def test_mat_v73_has_actionable_error(tmp_path):
    path = tmp_path / "capture.mat"
    path.write_bytes(b"MATLAB 7.3 MAT-file" + b" " * 128)
    with pytest.raises(ValueError, match="'-v7'"):
        read_mat(path)


def test_sdk_import_does_not_load_torch_or_start_services():
    result = subprocess.run([sys.executable, "-c",
        "import sys, opendpd.sdk; assert 'torch' not in sys.modules; print(opendpd.sdk.API_VERSION)"],
        capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "1"


def test_diagnostics_can_cross_json_boundary():
    from opendpd.sdk.matlab import diagnostics

    info = json.loads(diagnostics())
    assert info["api_version"] == 1 and info["ok"], info
    # what apply() accepts is whatever has a parity test against the evaluator; streaming only where a variant is registered
    assert info["apply_models"] == ["gru", "tres_gru", "gmp", "mp_ls", "gmp_ls"]
    assert info["apply_execution"] == ["offline_segmented", "streaming_stateful"]
    assert info["apply_streaming_models"] == ["gmp", "gru"]


def test_wait_timeout_does_not_request_cancellation():
    class RunningProject:
        def _request(self, method, path):
            assert method == "GET" and path == "/runs/run-wait"
            return {"status": "running"}

    with pytest.raises(TimeoutError, match="not cancelled"):
        Job(RunningProject(), "run-wait").wait(timeout=0.01, poll_interval=0.005)


def test_wait_failure_keeps_run_id_and_cause():
    class FailedProject:
        def _request(self, method, path):
            return {"status": "failed", "error": {"message": "worker failed"}}

    with pytest.raises(SDKError, match="run-failed: worker failed"):
        Job(FailedProject(), "run-failed").wait()


def test_uncertain_import_retains_source_for_the_running_server(tmp_path):
    project = Project.__new__(Project)
    project.workspace = tmp_path
    project._closed = False

    def disconnected(*args):
        raise SDKError("connection_lost", "response timed out")

    project._request = disconnected
    with pytest.raises(SDKError, match="import_uncertain.*may still finish"):
        project.import_iq(np.zeros((1024, 2), np.float32), np.zeros((1024, 2), np.float32),
                          sample_rate_hz=80e6, bandwidth_hz=20e6, nperseg=256)
    sources = list((tmp_path / "imports").glob("sdk-*.npz"))
    assert len(sources) == 1
    with np.load(sources[0]) as archive:
        assert archive["input"].shape == (1024, 2)
