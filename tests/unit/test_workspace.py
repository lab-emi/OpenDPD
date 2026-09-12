"""Workspace layout, atomic writes, dataset registration."""

import json
import os
import stat

import pytest

from opendpd.services.workspace import (
    WORKSPACE_VERSION,
    Workspace,
    WorkspaceError,
    slugify,
    write_json_atomic,
)


def test_create_open_and_version_check(tmp_path):
    ws = Workspace.create(tmp_path / "ws")
    assert ws.meta["workspace_version"] == WORKSPACE_VERSION
    assert ws.datasets_dir.is_dir() and ws.runs_dir.is_dir()
    with pytest.raises(WorkspaceError):
        Workspace.create(tmp_path / "ws")
    assert Workspace.open(tmp_path / "ws").root == ws.root
    with pytest.raises(WorkspaceError):
        Workspace.open(tmp_path / "nope")
    write_json_atomic(ws.meta_path, {"workspace_version": 99})
    with pytest.raises(WorkspaceError, match="version 99"):
        Workspace.open(tmp_path / "ws")


def test_atomic_write_leaves_no_temp_and_rejects_nan(tmp_path):
    target = tmp_path / "x.json"
    write_json_atomic(target, {"b": 1, "a": [1, 2]})
    assert json.loads(target.read_text()) == {"a": [1, 2], "b": 1}
    assert [p.name for p in tmp_path.iterdir()] == ["x.json"]
    with pytest.raises(ValueError):
        write_json_atomic(target, {"v": float("nan")})
    assert json.loads(target.read_text()) == {"a": [1, 2], "b": 1}


@pytest.mark.skipif(os.name == "nt" or os.geteuid() == 0, reason="permission bits are not enforced here")
def test_preflight_detects_readonly_workspace(tmp_path):
    ws = Workspace.create(tmp_path / "ws")
    assert ws.preflight() == []
    os.chmod(ws.root, stat.S_IRUSR | stat.S_IXUSR)
    try:
        problems = ws.preflight()
    finally:
        os.chmod(ws.root, stat.S_IRWXU)
    assert problems and "not writable" in problems[0]


def test_preflight_reports_low_disk(tmp_path):
    ws = Workspace.create(tmp_path / "ws")
    problems = ws.preflight(min_free_mb=10 ** 9)
    assert problems and "free" in problems[0]


def test_register_builtin_dataset_copies_hashes_and_splits(tmp_path):
    ws = Workspace.create(tmp_path / "ws")
    manifest = ws.register_builtin_dataset("DPA_200MHz")
    assert manifest.dataset_id == "dpa-200mhz"
    assert manifest.origin.value == "measured"
    assert manifest.signal.sample_rate_hz == 800e6 and manifest.signal.nperseg == 2560
    assert manifest.n_samples == 38400
    assert manifest.split.boundaries == {"train": (0, 23040), "val": (23040, 30720), "test": (30720, 38400)}
    names = {f.path for f in manifest.files}
    assert "raw/spec.json" in names and "raw/train_input.csv" in names and len(names) == 7
    assert all(f.sha256 for f in manifest.files) and manifest.raw_sha256
    assert manifest.missing_metadata() == []
    # idempotent and listed
    assert ws.register_builtin_dataset("DPA_200MHz").raw_sha256 == manifest.raw_sha256
    assert [d.dataset_id for d in ws.list_datasets()] == ["dpa-200mhz"]
    with pytest.raises(WorkspaceError, match="unknown built-in"):
        ws.register_builtin_dataset("NoSuchPA")
    with pytest.raises(WorkspaceError):
        ws.get_dataset("missing")


def test_run_ids_are_unique(tmp_path):
    ws = Workspace.create(tmp_path / "ws")
    ids = {ws.new_run_id() for _ in range(20)}
    assert len(ids) == 20 and all(i.startswith("run-") for i in ids)


def test_slugify():
    assert slugify("DPA_200MHz") == "dpa-200mhz"
    assert slugify("My PA  capture!") == "my-pa-capture"
