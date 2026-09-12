"""Import adapters, versions and roots: GUI/CLI/API share this one implementation."""

import json

import numpy as np
import pandas as pd
import pytest

from opendpd.core.preprocess import apply
from opendpd.schemas import DatasetOrigin, PreprocessingParams, SignalSpec
from opendpd.services import datasets as ds
from opendpd.services.workspace import Workspace
from tests.fixtures.synthetic import Impairments, synthesize, write_dataset

SIGNAL = SignalSpec(sample_rate_hz=800e6, bandwidth_hz=200e6, n_sub_ch=10, nperseg=2560, amplitude_units="normalized")


@pytest.fixture
def ws(tmp_path):
    return Workspace.create(tmp_path / "ws")


def test_csv_import_with_odd_headers_and_swapped_iq(ws, tmp_path):
    x, y = synthesize(5000, 1)
    src = tmp_path / "capture.csv"
    pd.DataFrame({"time": np.arange(5000), "tx_q": x[:, 1], "tx_i": x[:, 0], "rx_i": y[:, 0], "rx_q": y[:, 1]}).to_csv(src, index=False)
    info = ds.inspect_source(src)
    assert info.suggested_mapping == {"I_in": "tx_i", "Q_in": "tx_q", "I_out": "rx_i", "Q_out": "rx_q"}
    assert info.n_rows == 5000 and len(info.preview) == 5 and not info.problems
    m = ds.import_dataset(ws, src, dataset_id="cap", signal=SIGNAL, guard_samples=64)
    assert m.n_samples == 5000 and m.columns["I_in"] == "tx_i" and m.split.guard_samples == 64
    xr, yr, split = ds.load_version_arrays(ws, "cap")
    np.testing.assert_allclose(xr, x, rtol=0, atol=1e-6)
    # a user can correct a wrong guess explicitly (here: swap I/Q of the input)
    m2 = ds.import_dataset(ws, src, dataset_id="cap-swapped", signal=SIGNAL,
                           mapping={"I_in": "tx_q", "Q_in": "tx_i"})
    xs, _, _ = ds.load_version_arrays(ws, "cap-swapped")
    np.testing.assert_allclose(xs[:, 0], x[:, 1], atol=1e-6)
    assert m2.columns["I_in"] == "tx_q"


def test_import_errors_are_explained(ws, tmp_path):
    src = tmp_path / "weird.csv"
    pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]}).to_csv(src, index=False)
    info = ds.inspect_source(src)
    assert "map them explicitly" in info.problems[0]
    with pytest.raises(ds.ImportError_, match="mapping is incomplete"):
        ds.import_dataset(ws, src, dataset_id="weird")
    with pytest.raises(ds.ImportError_, match="not in the file"):
        ds.import_dataset(ws, src, dataset_id="weird", mapping={"I_in": "a", "Q_in": "b", "I_out": "c", "Q_out": "d"})
    assert not (ws.dataset_dir("weird")).exists(), "a failed import leaves nothing behind"
    x, y = synthesize(3000, 0)
    npz = tmp_path / "pair.npz"
    np.savez(npz, input=x, output=y[:-5])
    with pytest.raises(ds.ImportError_, match="must be paired"):
        ds.import_dataset(ws, npz, dataset_id="pair")


def test_numpy_imports_and_object_arrays_refused(ws, tmp_path):
    x, y = synthesize(4000, 2)
    npz = tmp_path / "pair.npz"
    np.savez(npz, input=x[:, 0] + 1j * x[:, 1], output=y)     # complex input, (n,2) output
    m = ds.import_dataset(ws, npz, dataset_id="pair", signal=SIGNAL, origin=DatasetOrigin.synthetic)
    assert "synthetic" in m.display_name.lower() and m.n_samples == 4000
    xr, yr, _ = ds.load_version_arrays(ws, "pair")
    np.testing.assert_allclose(xr, x, atol=1e-6)
    npy = tmp_path / "wide.npy"
    np.save(npy, np.concatenate([x, y], axis=1))
    assert ds.import_dataset(ws, npy, dataset_id="wide", signal=SIGNAL).n_samples == 4000
    evil = tmp_path / "evil.npy"
    np.save(evil, np.array([{"a": 1}, None], dtype=object), allow_pickle=True)
    info = ds.inspect_source(evil)
    assert info.problems and "refused" in info.problems[0]
    with pytest.raises(ValueError):
        ds.import_dataset(ws, evil, dataset_id="evil")


def test_legacy_split_directory_import(ws):
    from opendpd.services.workspace import BUILTIN_DATASETS_DIR
    m = ds.import_dataset(ws, BUILTIN_DATASETS_DIR / "DPA_200MHz", dataset_id="legacy-copy", origin=DatasetOrigin.measured)
    assert m.source.kind.value == "legacy_dir_import" and m.signal.sample_rate_hz == 800e6
    assert m.n_samples == 38400 and m.split.boundaries["train"] == (0, 23040)
    assert ws.dataset_version_dir("legacy-copy") == ws.dataset_raw_dir("legacy-copy")


def test_versions_keep_raw_untouched_and_fit_only_on_train(ws, tmp_path):
    write_dataset(tmp_path / "s", 20000, 0, 800e6, 200e6, Impairments(delay_samples=5), fmt="csv")
    m = ds.import_dataset(ws, tmp_path / "s" / "data.csv", dataset_id="syn", signal=SIGNAL, origin=DatasetOrigin.synthetic)
    raw_files = {f.path: f.sha256 for f in m.files}
    report = ds.run_doctor(ws, "syn")
    assert ds.latest_report(ws, "syn").report_id == report.report_id
    params = PreprocessingParams(delay_samples=5, normalize="peak_input")
    preview = ds.preview_preprocess(ws, "syn", params)
    assert preview["n_samples_after"] == 20000 - 5
    assert not any(i["code"] == "time_misalignment" for i in preview["report_after"]["items"])
    v = ds.create_version(ws, "syn", "aligned-v1", params)
    m2 = ws.get_dataset("syn")
    assert {f.path: f.sha256 for f in m2.files} == raw_files and m2.raw_sha256 == m.raw_sha256
    assert [x.version for x in m2.versions] == ["raw-v1", "aligned-v1"]
    train = m.versions[0].split.boundaries["train"]
    assert v.fit_range == train and v.record["steps"][-1]["normalize"]["fit_range"] == list(train)
    assert v.code_version == "preprocess-v1" and v.params.delay_samples == 5
    x2, _, split2 = ds.load_version_arrays(ws, "syn", "aligned-v1")
    assert float(np.hypot(x2[train[0]:train[1], 0], x2[train[0]:train[1], 1]).max()) == pytest.approx(1.0, abs=1e-6)
    assert split2.boundaries["test"][1] == len(x2)
    assert (ws.dataset_version_dir("syn", "aligned-v1") / "train_input.csv").exists()
    with pytest.raises(ds.ImportError_, match="already exists"):
        ds.create_version(ws, "syn", "aligned-v1", params)
    with pytest.raises(ValueError, match="fit_range"):
        apply(x2, x2, PreprocessingParams(normalize="peak_input"), fit_range=None)


def test_preprocess_apply_corrects_delay_gain_and_spikes():
    x, y = synthesize(8000, 4, impairments=Impairments(delay_samples=3, gain=2 * np.exp(1j * np.radians(30)), n_outliers=3))
    x2, y2, record = apply(x, y, PreprocessingParams(delay_samples=3, gain_db=20 * np.log10(2), phase_deg=30, remove_outliers=True))
    _, y_ref = synthesize(8000, 4)
    assert len(x2) == 8000 - 3 and record["steps"][0]["remove_outliers"]["count"] == 3
    err = np.abs(y2[100:-100] - y_ref[100:-103]).max(axis=1)
    assert np.median(err) < 1e-4 and (err > 2e-3).sum() <= 3 * 2   # only the interpolated spike positions differ


def test_import_roots_refuse_traversal_and_unknown_roots(ws, tmp_path):
    (ws.imports_dir / "sub").mkdir(parents=True)
    (ws.imports_dir / "sub" / "a.csv").write_text("I_in,Q_in,I_out,Q_out\n0,0,0,0\n")
    (ws.imports_dir / "notes.txt").write_text("x")
    entries = ds.list_files(ws, "imports")
    assert [e.path for e in entries] == ["sub"]                     # non-data files are hidden
    assert [e.path for e in ds.list_files(ws, "imports", "sub")] == ["sub/a.csv"]
    with pytest.raises(ds.ImportError_, match="escapes"):
        ds.resolve_in_root(ws, "imports", "../../etc/passwd")
    with pytest.raises(ds.ImportError_, match="unknown import root"):
        ds.list_files(ws, "home")
    ws.add_import_root("Lab share", tmp_path / "share")
    assert "lab-share" in ws.import_roots()
    assert json.loads((ws.root / "workspace.json").read_text())["import_roots"]["lab-share"].endswith("share")


def test_upload_is_capped_and_confined(ws):
    path = ds.receive_upload(ws, "../../x.csv", [b"I_in,Q_in,I_out,Q_out\n", b"0,0,0,0\n"], max_bytes=1000)
    assert path.parent == ws.imports_dir / "uploads" and path.name.endswith("-x.csv")
    with pytest.raises(ds.ImportError_, match="exceeds"):
        ds.receive_upload(ws, "big.csv", [b"x" * 600, b"y" * 600], max_bytes=1000)
    assert not list((ws.imports_dir / "uploads").glob("*big.csv"))
    with pytest.raises(ds.ImportError_, match="only"):
        ds.receive_upload(ws, "script.py", [b"print(1)"], max_bytes=1000)


def test_numpy_imports_stream_from_the_source_without_copies(ws, tmp_path):
    """Stress-tier promise (performance protocol §8.2): parse/convert never materialises the capture."""
    n = 300_000
    x, y = synthesize(n, 3)
    src = tmp_path / "pair-stack.npy"
    stacked = np.lib.format.open_memmap(src, mode="w+", dtype=np.float32, shape=(2, n, 2))
    stacked[0], stacked[1] = x, y
    stacked.flush()
    del stacked
    xi, yi = ds._read_numpy_arrays(src, {})
    assert not xi.flags.owndata and not yi.flags.owndata           # views of the memory map, not copies
    m = ds.import_dataset(ws, src, dataset_id="stack", signal=SIGNAL, origin=DatasetOrigin.synthetic)
    assert m.n_samples == n
    xr, yr, _ = ds.load_version_arrays(ws, "stack")
    assert isinstance(xr, np.memmap) and xr.dtype == np.float32          # versions are read memory-mapped too
    np.testing.assert_allclose(np.asarray(xr[:1000]), x[:1000], atol=1e-6)
    np.testing.assert_allclose(np.asarray(yr[-1000:]), y[-1000:], atol=1e-6)


def test_doctor_analyses_a_bounded_central_window_and_says_so(ws, tmp_path, monkeypatch):
    x, y = synthesize(300_000, 3)
    npz = tmp_path / "long.npz"
    np.savez(npz, input=x, output=y)
    ds.import_dataset(ws, npz, dataset_id="stack", signal=SIGNAL, origin=DatasetOrigin.synthetic)
    monkeypatch.setattr(ds, "MAX_DOCTOR_SAMPLES", 50_000)
    assert ds.analysis_window(10_000, 50_000) == (0, 10_000)
    assert ds.analysis_window(300_000, 50_000) == (125_000, 175_000)
    report = ds.run_doctor(ws, "stack")
    note = next(i for i in report.items if i.code == "analysis_window")
    assert note.evidence == {"start": 125_000, "end": 175_000, "n_samples": 300_000}
    assert "125,000" in note.message and note.severity == "info"
    monkeypatch.setattr(ds, "MAX_DOCTOR_SAMPLES", 2_000_000)
    assert all(i.code != "analysis_window" for i in ds.run_doctor(ws, "stack").items)
