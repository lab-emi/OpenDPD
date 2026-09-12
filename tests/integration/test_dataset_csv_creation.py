"""Real paired CSV ingestion, split protocol and packaged-dataset coverage."""

import contextlib
import io
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from datasets.demodulator import Demodulator
from modules.data_collector import load_dataset
from opendpd.commands import main
from opendpd.core.splits import DEFAULT_GUARD_SAMPLES, DEFAULT_RATIOS, contiguous_boundaries
from opendpd.schemas import DatasetOrigin
from opendpd.schemas.importing import CsvOptions, DatasetImportDefaults
from opendpd.server.app import create_app
from opendpd.server.security import CSRF_HEADER
from opendpd.services.csv_import import inspect_csv
from opendpd.services.datasets import import_dataset, load_version_arrays
from opendpd.services.workspace import BUILTIN_DATASETS_DIR, Workspace, sha256_file

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("csv-creation")
    app = create_app(root, bootstrap_token="csv-test", allow_custom_datasets=True, shutdown_timeout=2)
    with TestClient(app, base_url="http://127.0.0.1:8877") as client:
        bootstrap = client.post("/api/v1/session/bootstrap", json={"token": "csv-test"}).json()
        client.headers[CSRF_HEADER] = bootstrap["csrf_token"]
        ws = Workspace.open(root)
        ws.imports_dir.mkdir(exist_ok=True)
        yield client, ws


def complex_capture(n=4096):
    t = np.arange(n, dtype=np.float64)
    x = .1 * np.exp(2j * np.pi * t / 32)
    y = 1.7 * x * np.exp(.2j)
    return x, y


@pytest.mark.parametrize("kind,header", [("complex_pair", True), ("complex_pair", False), ("iq_columns", True), ("iq_columns", False)])
def test_upload_validate_create_and_read_exact_samples(env, kind, header):
    client, ws = env
    x, y = complex_capture()
    # Reverse the complex columns, exercising named auto-mapping. Real columns
    # use scientific notation and the unheaded form uses positional mapping.
    if kind == "complex_pair":
        lines = ["output,input"] if header else []
        for a, b in zip(x, y):
            values = (b, a) if header else (a, b)
            lines.append(",".join(f"{v.real:.16e}{v.imag:+.16e}{'i' if i else 'j'}" for i, v in enumerate(values)))
    else:
        lines = ["I_in,Q_in,I_out,Q_out"] if header else []
        lines.extend(",".join(f"{v:.16e}" for v in row) for row in np.column_stack((x.real, x.imag, y.real, y.imag)))
    text = "\ufeff" + "\n".join(lines) + "\n"
    upload = client.post("/api/v1/datasets/upload", files={"file": (f"{kind}-{header}.csv", text.encode(), "text/csv")})
    assert upload.status_code == 201, upload.text
    source = {k: upload.json()[k] for k in ("root_id", "path")}
    defaults = client.get("/api/v1/datasets/import-defaults").json()
    assert defaults == {"ratios": DEFAULT_RATIOS, "guard_samples": DEFAULT_GUARD_SAMPLES}
    preview = client.post("/api/v1/datasets/csv/preview", json={"source": source}).json()
    assert preview["valid"] and preview["data_valid"], preview
    assert preview["n_samples"] == len(x) and preview["options"]["format"] == kind
    assert preview["options"]["header"] == ("present" if header else "absent")
    assert preview["split_counts"] == {"train": 2150, "val": 716, "test": 718}
    split = {"ratios": {"train": .7, "val": .2, "test": .1}, "guard_samples": DEFAULT_GUARD_SAMPLES}
    request = {"source": source, "options": preview["options"], "split": split}
    review = client.post("/api/v1/datasets/csv/preview", json=request).json()
    assert review["valid"] and review["split_counts"] == {"train": 2508, "val": 716, "test": 360}
    dataset_id = f"csv-{kind}-{str(header).lower()}"
    response = client.post("/api/v1/datasets/csv", json={**request, "dataset_id": dataset_id,
        "display_name": "PA CSV", "origin": "synthetic", "expected_sha256": review["sha256"]})
    assert response.status_code == 201, response.text
    manifest = ws.get_dataset(dataset_id)
    assert manifest.origin == DatasetOrigin.synthetic and "synthetic" in manifest.display_name
    assert manifest.split.boundaries == contiguous_boundaries(len(x), split["ratios"], DEFAULT_GUARD_SAMPLES)
    xi, yi, _ = load_version_arrays(ws, dataset_id)
    np.testing.assert_array_equal(xi, np.column_stack((x.real, x.imag)).astype(np.float32))
    np.testing.assert_array_equal(yi, np.column_stack((y.real, y.imag)).astype(np.float32))
    xt, yt, xv, yv, xe, ye = load_dataset(dataset_path=ws.dataset_version_dir(dataset_id))
    assert [len(xt), len(xv), len(xe)] == [2508, 716, 360]
    assert [len(yt), len(yv), len(ye)] == [2508, 716, 360]
    raw = ws.dataset_raw_dir(dataset_id) / source["path"].split("/")[-1]
    assert raw.read_bytes() == text.encode() and sha256_file(raw) == preview["sha256"]
    result = client.get(f"/api/v1/datasets/{dataset_id}/analysis").json()
    assert len(result["spectrum"]["frequency"]) > 0 and len(result["time"]["traces"]) == 2
    assert result["constellation"]["status"] == "unavailable", "a custom capture must not inherit a receiver from its filename"
    assert result["iq"]["mode"] == "samples"


@pytest.mark.parametrize("bad,code", [("NaN,0,1,0", "non_finite"), ("0,inf,1,0", "non_finite"),
    ("0,0,1e40,0", "numeric"), ("0,,1,0", "numeric"), ("0,0,1", "row_width"), ("", "row_width")])
def test_whole_file_errors_report_line_column_and_fix_and_never_create(env, bad, code):
    client, ws = env
    path = ws.imports_dir / "invalid.csv"
    lines = ["I_in,Q_in,I_out,Q_out"] + ["0.1,0.2,0.3,0.4"] * 1024
    lines[900] = bad
    path.write_text("\n".join(lines) + "\n")
    source = {"root_id": "imports", "path": path.name}
    report = client.post("/api/v1/datasets/csv/preview", json={"source": source}).json()
    assert not report["valid"] and not report["data_valid"] and report["n_samples"] == 1024
    error = report["issues"][0]
    assert error["line"] == 901 and error["code"] == code and error["message"] and error["fix"]
    response = client.post("/api/v1/datasets/csv", json={"source": source, "options": report["options"],
        "dataset_id": "invalid", "display_name": "invalid", "expected_sha256": report["sha256"]})
    assert response.status_code == 409 and not ws.dataset_dir("invalid").exists()


def test_header_mapping_encoding_split_and_changed_source_guidance(env):
    client, ws = env
    path = ws.imports_dir / "validation.csv"
    path.write_text("input,output\n" + "0.1+0.2j,0.3-0.4i\n" * 1024)
    report = inspect_csv(path, CsvOptions(mapping={"input": 0, "output": 0}))[0]
    assert report.issues[0].code == "mapping" and not report.data_valid
    for ratios in ({"train": .6, "val": .3, "test": .3}, {"train": 1., "val": 0., "test": 0.}):
        report = inspect_csv(path, split=DatasetImportDefaults(ratios=ratios))[0]
        assert report.data_valid and not report.valid and report.issues[0].code == "split"
    report = inspect_csv(path)[0]
    path.write_text(path.read_text().replace("0.3-0.4i", "0.4-0.4i", 1))
    with pytest.raises(Exception, match="changed since validation"):
        import_dataset(ws, path, dataset_id="changed", csv_options=report.options, expected_sha256=report.sha256)
    assert not ws.dataset_dir("changed").exists()
    path.write_bytes(b"input,output\n\xff,0\n")
    assert inspect_csv(path)[0].issues[0].code == "encoding"
    path.write_text("input,output\n0.1+0.2j,0.3-0.4j\n")
    short = inspect_csv(path)[0]
    assert short.data_valid and not short.valid and short.issues[0].code == "split"
    path.write_text("input;output\n0.1+0.2j;0.3-0.4j\n")
    assert inspect_csv(path)[0].issues[0].code == "columns"


def test_cli_uses_the_same_csv_validation_and_materialisation(env):
    _, ws = env
    path = ws.imports_dir / "cli.csv"
    path.write_text("input,output\n" + "0.1+0.2j,0.3-0.4i\n" * 1024)
    args = ["datasets", "import", str(path), "--workspace", str(ws.root), "--csv-format", "auto", "--ratios", ".7", ".2", ".1"]
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        assert main([*args, "--inspect-csv"]) == 0
    cli_report = json.loads(out.getvalue())
    python_report = inspect_csv(path, split=DatasetImportDefaults(ratios={"train": .7, "val": .2, "test": .1}))[0]
    assert cli_report == python_report.model_dump(mode="json")
    with contextlib.redirect_stdout(io.StringIO()):
        assert main([*args, "--id", "cli-created", "--json"]) == 0
    assert ws.get_dataset("cli-created").split.boundaries == python_report.boundaries


def test_all_packaged_datasets_read_and_demodulate_through_real_http(env):
    client, ws = env
    listed = client.get("/api/v1/datasets/builtin")
    assert listed.status_code == 200, listed.text
    catalog = listed.json()
    assert {entry["name"] for entry in catalog} == {p.parent.name for p in BUILTIN_DATASETS_DIR.glob("*/spec.json")}
    assert {entry["name"] for entry in catalog} == {"APA_200MHz", "APA_200MHz_b", "DPA_200MHz", "DPA_160MHz", "MyCustomPA"}
    for entry in catalog:
        name = entry["name"]
        registered = client.post("/api/v1/datasets/import-builtin", json={"name": name})
        assert registered.status_code in (200, 201), registered.text
        manifest = registered.json()
        assert manifest["n_samples"] == entry["n_samples"] and manifest["raw_sha256"] == entry["raw_sha256"]
        xi, yi, split = load_version_arrays(ws, manifest["dataset_id"])
        xt, yt, xv, yv, xe, ye = load_dataset(dataset_name=name)
        np.testing.assert_array_equal(xi, np.concatenate((xt, xv, xe)).astype(np.float32))
        np.testing.assert_array_equal(yi, np.concatenate((yt, yv, ye)).astype(np.float32))
        assert split.boundaries["test"][1] == len(xi)
        response = client.get(f'/api/v1/datasets/{manifest["dataset_id"]}/analysis')
        assert response.status_code == 200, response.text
        analysis = response.json()
        json.dumps(analysis, allow_nan=False)
        for key in ("time", "spectrum", "iq", "am"):
            assert analysis[key], (name, key)
        const = analysis["constellation"]
        assert const["status"] == "ok", (name, const)
        a, b = const["sample_range"]
        demod = Demodulator.from_dataset(name)
        x = xi[a:b, 0].astype(float) + 1j * xi[a:b, 1].astype(float)
        i, q = demod.demodulate(x)
        trace = const["traces"][0]
        np.testing.assert_array_equal(trace["i"], i[::trace["stride"]])
        np.testing.assert_array_equal(trace["q"], q[::trace["stride"]])
        assert const["traces"][1]["equalized"]
        if name == "MyCustomPA":
            assert manifest["origin"] == "synthetic" and "dummy dataset for tutorial purpose" in manifest["display_name"]
            # Independent reference: the tutorial uses each of the 64 grid points
            # exactly once per frame. Recover their original odd-integer levels.
            levels = np.rint(np.column_stack((i, q)) * np.sqrt(42)).astype(int)
            assert set(map(tuple, levels)) == {(re, im) for re in range(-7, 8, 2) for im in range(-7, 8, 2)}
            assert np.max(np.abs(np.column_stack((i, q)) * np.sqrt(42) - levels)) < 1e-6


def test_updated_builtin_is_added_separately_and_old_workspace_copy_survives(env):
    _, ws = env
    original = ws.register_builtin_dataset("MyCustomPA")
    old = original.model_copy(update={"raw_sha256": "0" * 64, "display_name": "Old tutorial (synthetic)"})
    ws.save_dataset(old)
    replacement = ws.register_builtin_dataset("MyCustomPA")
    assert replacement.dataset_id != original.dataset_id
    assert ws.get_dataset(original.dataset_id).display_name == "Old tutorial (synthetic)"
    assert ws.register_builtin_dataset("MyCustomPA").dataset_id == replacement.dataset_id
