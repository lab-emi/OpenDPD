"""S07 dataset routes through the real app: roots, inspect, import, doctor, preprocess, upload."""

import io
import json
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from opendpd.server.app import create_app
from opendpd.server.security import CSRF_HEADER
from opendpd.services import datasets as ds
from opendpd.services.workspace import Workspace
from tests.fixtures.synthetic import Impairments, synthesize

pytestmark = pytest.mark.integration
TOKEN = "datasets-token"
SIGNAL = {"sample_rate_hz": 800e6, "bandwidth_hz": 200e6, "n_sub_ch": 10, "nperseg": 2560, "amplitude_units": "normalized"}


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("ws")
    app = create_app(root, bootstrap_token=TOKEN, supervisor_kwargs={"poll_interval": 0.1}, shutdown_timeout=3)
    with TestClient(app, base_url="http://127.0.0.1:8765") as client:
        r = client.post("/api/v1/session/bootstrap", json={"token": TOKEN})
        client.headers[CSRF_HEADER] = r.json()["csrf_token"]
        ws = Workspace.open(root)
        x, y = synthesize(20000, 5, impairments=Impairments(delay_samples=4, n_outliers=3))
        (ws.imports_dir / "lab").mkdir(parents=True)
        pd.DataFrame({"tx_i": x[:, 0], "tx_q": x[:, 1], "rx_i": y[:, 0], "rx_q": y[:, 1]}).to_csv(ws.imports_dir / "lab" / "pa.csv", index=False)
        (ws.imports_dir / "secret.txt").write_text("not data")
        yield client, ws


def test_roots_listing_and_traversal_refused(env):
    client, ws = env
    roots = client.get("/api/v1/datasets/import-roots").json()
    assert [r["root_id"] for r in roots] == ["imports"] and roots[0]["path"] == str(ws.imports_dir)
    files = client.get("/api/v1/datasets/import-roots/imports/files").json()
    assert [f["path"] for f in files] == ["lab"]              # secret.txt is not a data file
    files = client.get("/api/v1/datasets/import-roots/imports/files", params={"path": "lab"}).json()
    assert files[0]["path"] == "lab/pa.csv" and files[0]["size_bytes"] > 0
    r = client.get("/api/v1/datasets/import-roots/imports/files", params={"path": "../../"})
    assert r.status_code == 409 and "escapes" in r.json()["error"]["message"]
    assert client.get("/api/v1/datasets/import-roots/home/files").status_code == 409
    r = client.post("/api/v1/datasets/inspect", json={"root_id": "imports", "path": "../secret.txt"})
    assert r.status_code == 409


def test_inspect_import_doctor_preprocess_flow(env):
    client, ws = env
    info = client.post("/api/v1/datasets/inspect", json={"root_id": "imports", "path": "lab/pa.csv"}).json()
    assert info["kind"] == "csv_import" and info["n_rows"] == 20000
    assert info["suggested_mapping"] == {"I_in": "tx_i", "Q_in": "tx_q", "I_out": "rx_i", "Q_out": "rx_q"}
    r = client.post("/api/v1/datasets/import", json={"source": {"root_id": "imports", "path": "lab/pa.csv"},
                                                    "dataset_id": "lab-pa", "signal": SIGNAL, "origin": "synthetic",
                                                    "display_name": "Lab PA (synthetic)"})
    assert r.status_code == 201, r.text
    manifest = r.json()
    assert manifest["n_samples"] == 20000 and manifest["versions"][0]["version"] == "raw-v1"
    assert manifest["split"]["guard_samples"] == 256 and manifest["columns"]["I_in"] == "tx_i"
    assert client.get("/api/v1/datasets/lab-pa/diagnostics").json() is None
    report = client.post("/api/v1/datasets/lab-pa/diagnostics").json()
    codes = {i["code"]: i for i in report["items"]}
    assert report["doctor_version"] == "dataset-doctor-v1" and "time_misalignment" in codes and "output_outliers" in codes
    assert abs(codes["time_misalignment"]["evidence"]["delay_samples"] - 4) < 0.1
    assert client.get("/api/v1/datasets/lab-pa/diagnostics").json()["report_id"] == report["report_id"]
    # the same report through the Python API and the CLI (one implementation)
    assert ds.latest_report(ws, "lab-pa").report_id == report["report_id"]
    from opendpd.commands import main as cli
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = cli(["datasets", "doctor", "lab-pa", "--workspace", str(ws.root), "--json"])
    cli_report = json.loads(buf.getvalue())
    assert rc == 0 and cli_report["doctor_version"] == report["doctor_version"]
    assert [i["code"] for i in cli_report["items"]] == [i["code"] for i in report["items"]]

    params = {"delay_samples": 4, "remove_outliers": True, "normalize": "peak_input"}
    preview = client.post("/api/v1/datasets/lab-pa/preprocess/preview", json={"params": params}).json()
    assert preview["n_samples_after"] == 19996
    assert not any(i["code"] in ("time_misalignment", "output_outliers") for i in preview["report_after"]["items"])
    assert ws.get_dataset("lab-pa").versions[-1].version == "raw-v1", "preview writes nothing"
    r = client.post("/api/v1/datasets/lab-pa/preprocess", json={"params": params})
    assert r.status_code == 422
    r = client.post("/api/v1/datasets/lab-pa/preprocess", json={"params": params, "version": "clean-v1"})
    assert r.status_code == 201 and r.json()["fit_range"] == [0, 11692]
    versions = client.get("/api/v1/datasets/lab-pa").json()["versions"]
    assert [v["version"] for v in versions] == ["raw-v1", "clean-v1"]
    r = client.post("/api/v1/datasets/lab-pa/manifest", json={"signal": dict(SIGNAL, bandwidth_hz=150e6), "notes": "edited"})
    assert r.status_code == 200 and r.json()["signal"]["bandwidth_hz"] == 150e6 and r.json()["notes"] == "edited"
    spec = json.loads((ws.dataset_version_dir("lab-pa", "clean-v1") / "spec.json").read_text())
    assert spec["bw_main_ch"] == 150e6, "trainer-side spec follows the manifest"


def test_upload_is_streamed_into_the_uploads_root(env):
    client, ws = env
    x, y = synthesize(3000, 6)
    buf = io.BytesIO()
    np.savez(buf, input=x, output=y)
    r = client.post("/api/v1/datasets/upload", files={"file": ("../evil.npz", buf.getvalue(), "application/octet-stream")})
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["root_id"] == "imports" and body["path"].startswith("uploads/") and body["path"].endswith("-evil.npz")
    assert (ws.imports_dir / body["path"]).exists()
    r = client.post("/api/v1/datasets/import", json={"source": body | {}, "dataset_id": "uploaded", "signal": SIGNAL})
    assert r.status_code == 201 and r.json()["n_samples"] == 3000
    r = client.post("/api/v1/datasets/upload", files={"file": ("x.py", b"print(1)", "text/plain")})
    assert r.status_code == 409
    # the general 2 MB cap still applies to JSON routes
    assert client.post("/api/v1/runs", json={"config": {}, "name": "x" * (3 * 1024 * 1024)}).status_code == 413


def test_training_reads_a_preprocessed_version(env):
    client, ws = env
    from opendpd.services.recipes import instantiate
    cfg = instantiate("pa-gru-smoke-v1", "lab-pa")
    cfg = cfg.model_copy(update={"dataset": cfg.dataset.model_copy(update={"preprocessing_version": "clean-v1"}),
                                 "training": cfg.training.model_copy(update={"epochs": 1})})
    report = client.post("/api/v1/experiments/validate", json={"config": json.loads(cfg.model_dump_json())}).json()
    assert report["ok"], report["errors"]
    from opendpd.services import experiments
    record = experiments.create_run(ws, cfg)
    resolved = experiments.load_resolved(ws, record.run_id)
    assert resolved.dataset.preprocessing_version == "clean-v1"
    command = json.loads((ws.run_dir(record.run_id) / "provenance.json").read_text())["legacy_equivalent_command"]
    assert str(ws.dataset_version_dir("lab-pa", "clean-v1")) in command


def test_validate_names_missing_versions_and_guard_shorter_than_the_frame(env):
    client, ws = env
    from opendpd.services.recipes import instantiate
    r = client.post("/api/v1/datasets/import", json={"source": {"root_id": "imports", "path": "lab/pa.csv"},
                                                    "dataset_id": "guarded", "signal": SIGNAL, "origin": "synthetic",
                                                    "guard_samples": 256})
    assert r.status_code == 201, r.text
    r = client.post("/api/v1/datasets/guarded/preprocess", json={"version": "clean-v1", "params": {"delay_samples": 4}})
    assert r.status_code == 201, r.text
    payload = json.loads(instantiate("pa-gru-smoke-v1", "guarded").model_dump_json())
    payload["dataset"]["preprocessing_version"] = "nope-v9"
    report = client.post("/api/v1/experiments/validate", json={"config": payload}).json()
    assert not report["ok"] and report["resolved"] is None
    err = next((e for e in report["errors"] if e["field"] == "dataset.preprocessing_version"), None)
    assert err is not None and "clean-v1" in err["hint"], report["errors"]
    # a frame longer than the split guard is allowed, but the context leak across the boundary is named
    payload["dataset"]["preprocessing_version"] = "clean-v1"
    payload["training"]["frame_length"] = 4096
    report = client.post("/api/v1/experiments/validate", json={"config": payload}).json()
    assert report["ok"], report["errors"]
    warn = next(w for w in report["warnings"] if w["field"] == "training.frame_length")
    assert "guard" in warn["message"] and "4096" in warn["hint"]
    assert any("frame_length 4096 exceeds" in w for w in report["resolved"]["resolution"]["warnings"])
