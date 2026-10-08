"""Real service, real workers, reconnection and existing evaluator parity."""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from opendpd.sdk import SDKError, open_project
from opendpd.sdk import matlab as bridge
from opendpd.services import experiments
from opendpd.services.evaluation import predict_test_split
from opendpd.services.workspace import Workspace
from opendpd.studio.launcher import Lock, LOCK_FILE
from tests.fixtures.synthetic import synthesize

pytestmark = pytest.mark.integration
TRAINING = {"epochs": 1, "frame_length": 32, "frame_stride": 32,
            "batch_size": 16, "batch_size_eval": 16}


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    p = open_project(tmp_path_factory.mktemp("matlab-sdk") / "workspace 中文")
    yield p
    p.close(stop_service=True)


@pytest.fixture(scope="module")
def trained(project):
    x, y = synthesize(4096, fs=80e6, bandwidth=20e6)
    dataset = json.loads(bridge.import_iq(project, x.astype(np.float64), y, json.dumps({
        "dataset_id": "synthetic-matlab", "sample_rate_hz": 80e6, "bandwidth_hz": 20e6,
        "nperseg": 128, "origin": "synthetic"})))
    pa = project.train_pa(dataset["dataset_id"], parameters={"hidden_size": 4}, training=TRAINING).wait(timeout=120)
    dpd = project.train_dpd(dataset["dataset_id"], pa, parameters={"hidden_size": 4}, training=TRAINING).wait(timeout=120)
    return dataset, pa, dpd


def test_import_uses_shared_workspace_and_records_scaling(project, trained):
    ds, _, _ = trained
    notes = json.loads(ds["notes"])
    assert notes["scaling"] == "none" and notes["source_arrays"]["input"]["dtype"] == "float64"
    assert ds["origin"] == "synthetic" and ds["split"]["guard_samples"] == 256
    assert ds["dataset_id"] in {d["dataset_id"] for d in project.datasets()}
    assert not list((project.workspace / "imports").glob("sdk-*.npz"))


def test_service_info_and_authenticated_links(project, trained, monkeypatch):
    from urllib.parse import parse_qs, urlparse

    info = project.studio_info()
    assert info["workspace"] == str(project.workspace) and info["studio_navigation_version"] == 1
    assert trained[0]["dataset_id"] in {d["dataset_id"] for d in project.datasets()}
    assert trained[2].run_id in {r["run_id"] for r in project.runs()}
    assert "token=" not in json.dumps(info)
    url = urlparse(project.studio_url("run", trained[2].run_id))
    query = parse_qs(url.query)
    assert url.path == "/bootstrap" and query["next"] == ["/runs/" + trained[2].run_id]
    # Every link carries its own single-use token; the lock file holds none.
    assert query["token"] != parse_qs(urlparse(project.studio_url()).query)["token"]
    assert "token=" not in (project.workspace / ".studio.lock").read_text()
    assert len(project.runs(limit=1)) == 1
    monkeypatch.setattr(project, "studio_info", lambda: {"ready": True})
    assert "next=" not in project.studio_url()  # older running services retain home support
    with pytest.raises(SDKError, match="service_update_needed"):
        project.studio_url("datasets")


def test_simultaneous_workspaces_reserve_distinct_ports(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    projects = []
    try:
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(open_project, tmp_path / f"parallel-{index}") for index in range(3)]
            for future in futures:
                projects.append(future.result())
        ports = {Lock.read(p.workspace / LOCK_FILE).port for p in projects}
        assert len(ports) == 3
        assert all(p.studio_info()["workspace"] == str(p.workspace) for p in projects)
    finally:
        for p in projects:
            p.close(stop_service=True)


def test_mat_roundtrip_into_shared_import(project, tmp_path):
    x, y = synthesize(2048, fs=80e6, bandwidth=20e6)
    x, y = x[:, 0].astype(np.float64) + 1j * x[:, 1], y[:, 0] + 1j * y[:, 1]
    path = tmp_path / "capture with spaces 中文.mat"
    savemat(path, {"tx": x.reshape(1, -1), "rx": y.reshape(-1, 1)})
    ds = json.loads(bridge.import_mat(project, str(path), json.dumps({
        "input_variable": "tx", "output_variable": "rx", "dataset_id": "mat-file",
        "sample_rate_hz": 80e6, "bandwidth_hz": 20e6, "nperseg": 128, "origin": "synthetic"})))
    notes = json.loads(ds["notes"])
    assert notes["source"]["format"] == "mat" and len(notes["source"]["sha256"]) == 64
    assert notes["source_arrays"]["input"]["dtype"] == "complex128"
    raw = next((project.workspace / "datasets" / ds["dataset_id"] / "raw").glob("*.npz"))
    with np.load(raw) as data:
        np.testing.assert_array_equal(data["input"][:, 1], x.imag.astype(np.float32))


@pytest.mark.parametrize("index", [1, 2], ids=["pa", "dpd"])
def test_apply_matches_existing_python_evaluator(project, trained, index):
    from modules.data_collector import load_dataset

    ds, _, _ = trained
    job = trained[index]
    ws = Workspace.open(project.workspace)
    config = experiments.load_resolved(ws, job.run_id)
    x = load_dataset(dataset_path=ws.dataset_version_dir(ds["dataset_id"], "raw-v1"))[4]
    cwd = Path.cwd()
    expected = predict_test_split(ws, job.run_id, config, experiments.load_artifacts(ws, job.run_id))
    expected_iq = expected.prediction if index == 1 else expected.u
    actual, meta_json = bridge.apply(job, x)
    meta = json.loads(meta_json)
    np.testing.assert_allclose(actual, expected_iq.reshape(-1, 2)[:len(x)], rtol=1e-5, atol=1e-6)
    assert actual.shape == x.shape and actual.dtype == np.float32
    assert meta["execution"] == "offline_segmented" and meta["segment_samples"] == 128
    assert meta["output_role"] == ("modeled_pa_output" if index == 1 else "predistorted_pa_input")
    assert Path.cwd() == cwd


def test_apply_resets_segments_and_trims_partial_tail(project, trained):
    job = trained[2]
    x, _ = synthesize(257, fs=80e6, bandwidth=20e6)
    full, _ = job.apply(x)
    first, _ = job.apply(x[:128])
    tail, _ = job.apply(x[128:])
    np.testing.assert_allclose(full, np.concatenate([first, tail]), rtol=1e-5, atol=1e-6)
    assert full.shape == (257, 2)
    with pytest.raises(ValueError, match="offline_segmented"):
        job.apply(x, execution="streaming_stateful")


def test_apply_uses_frozen_run_metadata_after_dataset_edit(project, trained):
    from opendpd.services.datasets import update_manifest

    ws = Workspace.open(project.workspace)
    ds, pa, _ = trained
    manifest = ws.get_dataset(ds["dataset_id"])
    x, _ = synthesize(257, fs=80e6, bandwidth=20e6)
    expected, _ = pa.apply(x)
    try:
        update_manifest(ws, manifest.dataset_id,
            signal=manifest.signal.model_copy(update={"nperseg": 64, "sample_rate_hz": 160e6}))
        actual, info = pa.apply(x)
        np.testing.assert_array_equal(actual, expected)
        assert info["segment_samples"] == 128 and info["sample_rate_hz"] == 80e6
    finally:
        update_manifest(ws, manifest.dataset_id, signal=manifest.signal)


def test_reconnect_reuses_service_and_reads_same_result(project, trained):
    lock_before = Lock.read(project.workspace / LOCK_FILE)
    attached = open_project(project.workspace, start=False)
    try:
        job = attached.job(trained[2].run_id)
        assert job.wait().result() == trained[2].result()
        assert Lock.read(project.workspace / LOCK_FILE).pid == lock_before.pid
    finally:
        attached.close()
    assert lock_before.alive() and trained[2].status()["status"] == "succeeded"


def test_standard_dpd_export_is_visible_as_run_and_artifacts(project, trained):
    exported = project.run_dpd(trained[2]).wait(timeout=120)
    result = exported.result()
    assert result["evidence_type"] == "dpd_surrogate"
    kinds = {a["kind"] for a in exported.artifacts()["artifacts"]}
    assert "dpd_output" in kinds


def test_modified_checkpoint_is_refused(project, trained):
    job = trained[1]
    ws = Workspace.open(project.workspace)
    artifact = next(a for a in job.artifacts()["artifacts"] if a["kind"] == "checkpoint")
    path = ws.run_dir(job.run_id) / artifact["file"]["path"]
    original = path.read_bytes()
    try:
        path.write_bytes(original + b"modified")
        with pytest.raises(SDKError, match="checkpoint_changed"):
            job.apply(np.zeros((3, 2), dtype=np.float32))
    finally:
        path.write_bytes(original)


def test_queue_cancel_and_reconnect(project, trained):
    ds, _, _ = trained
    first = project.train_pa(ds["dataset_id"], parameters={"hidden_size": 8},
                            training={**TRAINING, "epochs": 100, "frame_stride": 1})
    second = project.train_pa(ds["dataset_id"], parameters={"hidden_size": 8}, training=TRAINING)
    try:
        assert second.status()["status"] == "queued"
        assert second.cancel()["status"] == "cancelled"
        first.cancel()
        with pytest.raises(SDKError, match="cancelled"):
            first.wait(timeout=60)
        assert project.job(second.run_id).status()["status"] == "cancelled"
    finally:
        if first.status()["status"] not in ("succeeded", "cancelled", "failed", "interrupted"):
            first.cancel()


def test_invalid_config_preserves_service_error_details(project, trained):
    with pytest.raises(SDKError) as error:
        project.train_pa(trained[0]["dataset_id"], model="not-a-model")
    assert error.value.code == "invalid_config" and error.value.details


def test_stop_rejects_unowned_service_metadata(project):
    path = project.workspace / ".sdk-service.json"
    original = path.read_bytes()
    try:
        path.unlink()
        with pytest.raises(SDKError, match="external_service"):
            project.close(stop_service=True)
        assert project.models()
    finally:
        path.write_bytes(original)


def test_service_can_be_stopped_after_client_reconnect(tmp_path):
    first = open_project(tmp_path / "new project")
    lock = Lock.read(first.workspace / LOCK_FILE)
    try:
        first.close()
        second = open_project(first.workspace, start=False)
        second.close(stop_service=True)
        if first._process:
            first._process.wait(timeout=10)
        assert not lock.alive()
        assert not (first.workspace / LOCK_FILE).exists()
    finally:
        first.close(stop_service=True)


def test_concurrent_open_uses_one_workspace_service(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    projects = []
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(open_project, tmp_path / "concurrent workspace") for _ in range(2)]
            for future in futures:
                projects.append(future.result(timeout=40))
        assert projects[0]._base_url == projects[1]._base_url
        assert projects[0].models() == projects[1].models()
    finally:
        for project in projects:
            project.close(stop_service=True)
