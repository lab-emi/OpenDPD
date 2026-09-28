"""Regression tests for untrusted inputs and process boundaries in 2.2.18."""
import ast
import io
import json
import os
from pathlib import Path
import zipfile

import numpy as np
import pytest

from opendpd.safe_paths import contained_path
from opendpd.schemas.common import FileRef
from opendpd.schemas.experiment import DatasetRef
from opendpd.schemas.legacy_spec import validate_spec
from opendpd.server.security import BOOTSTRAP_MAX_AGE, SessionStore
from opendpd.services.packages import PackageError, _redacted_json, _safe_member


@pytest.mark.parametrize("path", ["D:secret/x.csv", "foo/D:secret", "//host/share/x", "a///host/x", "../x", "a/../b", "C:\\data.csv", "a/./b", "a//b", "a/NUL", "a/x."])
def test_portable_metadata_paths_fail_before_filesystem_access(path, monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "resolve", lambda *_a, **_k: pytest.fail("unsafe spelling reached resolve"))
    with pytest.raises(ValueError):
        FileRef(path=path)
    with pytest.raises(ValueError):
        contained_path(tmp_path, path)
    with pytest.raises(PackageError):
        _safe_member(path)


@pytest.mark.parametrize("field,value", [("n_epochs", 999), ("quant_dir_label", "/outside"), ("dataset_path", "/private.csv"),
                                        ("init_weights", "private.pt"), ("step", "run_dpd"), ("csv_filename", "../private.csv")])
def test_dataset_metadata_cannot_supply_training_arguments(field, value):
    with pytest.raises(ValueError):
        validate_spec({field: value})


def test_dataset_versions_are_identifiers():
    with pytest.raises(ValueError):
        DatasetRef(id="data", preprocessing_version="../../outside")


def test_imported_plot_coordinates_cannot_be_html_categories():
    from opendpd.schemas.plot_input import validate_plot
    valid = {'version': 'plots-v1', 'kind': 'spectrum', 'frequency': [1, 2],
             'traces': [{'name': 'signal', 'psd_db': [-10, -20]}]}
    assert validate_plot(valid) is valid
    valid['frequency'][0] = '<a href="https://example.invalid">click</a>'
    with pytest.raises(ValueError, match='numeric arrays'):
        validate_plot(valid)


def test_project_refuses_spec_override_without_changing_args(tmp_path):
    from project import Project
    from types import SimpleNamespace
    (tmp_path / "spec.json").write_text('{"n_epochs": 999, "quant_dir_label": "/outside"}')
    project = Project.__new__(Project)
    project.dataset_path = str(tmp_path)
    project.args = SimpleNamespace(n_epochs=1)
    project.hparams = {"n_epochs": 1}
    with pytest.raises(ValueError):
        project.load_spec()
    assert project.args.n_epochs == project.hparams["n_epochs"] == 1


@pytest.mark.skipif(os.name != 'posix', reason='symlink fixture')
def test_legacy_import_rejects_linked_members(tmp_path):
    from opendpd.services.datasets import inspect_source
    source = tmp_path / 'data'
    source.mkdir()
    outside = tmp_path / 'private.csv'
    outside.write_text('I,Q\n1,2\n')
    (source / 'train_input.csv').symlink_to(outside)
    with pytest.raises(RuntimeError, match='symbolic links'):
        inspect_source(source)


def test_windows_paths_are_redacted_inside_json_values(tmp_path):
    path = tmp_path / "meta.json"
    path.write_text(json.dumps({"command": r"--workspace C:\Users\Alice\secret", "nested": ["c:/users/ALICE/secret"]}))
    result = json.loads(_redacted_json(path, [(r"C:\Users\Alice", "<home>")]))
    assert result == {"command": r"--workspace <home>\secret", "nested": ["<home>/secret"]}


def test_bootstrap_is_single_use_expires_and_can_be_minted_again(monkeypatch):
    from opendpd.server import security
    now = [1000.0]
    monkeypatch.setattr(security.time, "monotonic", lambda: now[0])
    store = SessionStore("first")
    assert store.exchange("first")
    assert store.exchange("first") is None
    assert store.mint("wrong") is None
    next_token = store.mint(store.launcher_secret)
    now[0] += BOOTSTRAP_MAX_AGE
    assert store.exchange(next_token) is None
    assert store.exchange(store.mint(store.launcher_secret))


def test_npz_headers_are_bounded_before_allocation(tmp_path, monkeypatch):
    from opendpd.services import numpy_input
    path = tmp_path / "data.npz"
    np.savez_compressed(path, input=np.zeros((1000, 2)), output=np.zeros((1000, 2)))
    monkeypatch.setattr(numpy_input, "MAX_ARRAY_BYTES", 1024)
    with pytest.raises(ValueError, match="limit"):
        numpy_input.inspect_numpy(path)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("other.txt", b"not an array")
    with pytest.raises(ValueError):
        numpy_input.inspect_numpy(path)


def test_csv_text_is_inert_but_negative_numbers_stay_numeric():
    from opendpd.services.csv_safety import safe_cell
    for value in ("=1+1", " +formula", "-formula", "@formula", "\t=1"):
        assert safe_cell(value) == "'" + value
    assert safe_cell(-10.5) == -10.5
    assert safe_cell("valid") == "valid"


def test_all_torch_loads_explicitly_disable_pickle():
    root = Path(__file__).resolve().parents[2]
    sources = [p for folder in ("opendpd", "modules", "steps", "quant", "backbones", "benchmark") for p in (root / folder).rglob("*.py")]
    sources += [root / name for name in ("main.py", "models.py", "project.py")]
    for path in sources:
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "load" and isinstance(node.func.value, ast.Name) and node.func.value.id == "torch":
                assert any(k.arg == "weights_only" and isinstance(k.value, ast.Constant) and k.value.value is True for k in node.keywords), f"{path}:{node.lineno}"


@pytest.mark.skipif(os.name != "posix", reason="GPU agent uses Linux directory descriptors")
@pytest.mark.parametrize("where", ["root", "parent", "file"])
def test_gpu_pack_never_reads_replaced_output_links(tmp_path, where):
    from opendpd.web.gpu_archive import pack
    root = tmp_path / "workspace"
    (root / "runs" / "run-a").mkdir(parents=True)
    outside = tmp_path / "private"
    outside.mkdir()
    (outside / "secret").write_text("private sentinel")
    if where == "root":
        (root / "runs" / "run-a").rmdir()
        (root / "runs" / "run-a").symlink_to(outside, target_is_directory=True)
    elif where == "parent":
        (root / "runs" / "run-a").rmdir()
        (root / "runs").rmdir()
        (root / "runs").symlink_to(outside, target_is_directory=True)
    else:
        (root / "runs" / "run-a" / "secret").symlink_to(outside / "secret")
    with pytest.raises((ValueError, OSError)):
        pack(root, subtree="runs/run-a")


@pytest.mark.parametrize("name", ["torch.py", "opendpd/web/gpu_container.py", "datasets/data/module.py", "runs/run-a/module.pth", "runs/run-a/binary.so"])
def test_gpu_input_rejects_code(tmp_path, name):
    from opendpd.web.gpu_archive import unpack
    content = io.BytesIO()
    with zipfile.ZipFile(content, "w") as archive:
        archive.writestr(name, "raise RuntimeError('untrusted module')")
    with pytest.raises(ValueError, match="layout"):
        unpack(content.getvalue(), tmp_path, input_run_id="run-a")
    assert not list(tmp_path.iterdir())


def test_bundle_extra_module_is_rejected_without_importing_it(tmp_path):
    from opendpd.services.figure_bundle import verify_directory
    (tmp_path / "manifest.json").write_text('{"files": {}}')
    (tmp_path / "textwrap.py").write_text("raise RuntimeError('untrusted module')")
    with pytest.raises(ValueError, match="unlisted"):
        verify_directory(tmp_path)
