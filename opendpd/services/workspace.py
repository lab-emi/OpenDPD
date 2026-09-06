"""Workspace: the on-disk home of datasets, runs, caches and exports.

Layout (plan §4.3)::

    workspace/
      workspace.json
      datasets/<dataset_id>/{manifest.json, raw/, processed/<version>/, diagnostics/}
      runs/<run_id>/{config.user.json, config.resolved.json, provenance.json, run.json,
                     logs/, save/, log/, dpd_out/, artifacts.json, result.json}
      cache/  exports/

Writes use temp-file + atomic replace. Raw dataset files are never modified
after registration; their hashes live in the manifest.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import secrets
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from pydantic import BaseModel

from opendpd import __version__
from opendpd.schemas import (
    DatasetManifest,
    DatasetOrigin,
    DatasetSource,
    DatasetSourceKind,
    FileRef,
    SignalSpec,
    SoftwareProvenance,
    SplitSpec,
)

WORKSPACE_VERSION = 1
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
BUILTIN_DATASETS_DIR = PACKAGE_ROOT / "datasets"
MIN_FREE_MB = 500


class WorkspaceError(RuntimeError):
    pass


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def write_json_atomic(path: Path, data: Any) -> None:
    if isinstance(data, BaseModel):
        text = data.model_dump_json(indent=2)
    else:
        text = json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
    tmp = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
    tmp.write_text(text + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def software_provenance() -> SoftwareProvenance:
    torch_version = None
    if "torch" in sys.modules:
        torch_version = getattr(sys.modules["torch"], "__version__", None)
    commit, dirty = _git_state()
    return SoftwareProvenance(
        opendpd_version=__version__,
        python_version=platform.python_version(),
        platform=f"{platform.system()}-{platform.machine()}",
        torch_version=torch_version,
        git_commit=commit,
        git_dirty=dirty,
    )


def _git_state():
    """Best-effort git commit of the *source checkout* (None for wheels)."""
    git_dir = PACKAGE_ROOT / ".git"
    if not git_dir.is_dir():
        return None, None
    try:
        import subprocess
        commit = subprocess.run(["git", "-C", str(PACKAGE_ROOT), "rev-parse", "HEAD"],
                                capture_output=True, text=True, timeout=5)
        status = subprocess.run(["git", "-C", str(PACKAGE_ROOT), "status", "--porcelain"],
                                capture_output=True, text=True, timeout=5)
        if commit.returncode != 0:
            return None, None
        return commit.stdout.strip(), bool(status.stdout.strip())
    except Exception:  # pragma: no cover - git missing or sandboxed
        return None, None


def slugify(name: str) -> str:
    out = "".join(c.lower() if c.isalnum() else "-" for c in name).strip("-")
    while "--" in out:
        out = out.replace("--", "-")
    return out or "dataset"


class Workspace:
    def __init__(self, root: Path):
        self.root = Path(root).expanduser().resolve()
        self.datasets_dir = self.root / "datasets"
        self.runs_dir = self.root / "runs"
        self.cache_dir = self.root / "cache"
        self.exports_dir = self.root / "exports"
        self.meta_path = self.root / "workspace.json"

    # -- lifecycle ---------------------------------------------------------
    @classmethod
    def create(cls, root: Path) -> "Workspace":
        ws = cls(root)
        if ws.meta_path.exists():
            raise WorkspaceError(f"workspace already exists: {ws.root}")
        ws.root.mkdir(parents=True, exist_ok=True)
        for d in (ws.datasets_dir, ws.runs_dir, ws.cache_dir, ws.exports_dir):
            d.mkdir(exist_ok=True)
        write_json_atomic(ws.meta_path, {
            "workspace_version": WORKSPACE_VERSION,
            "workspace_id": str(uuid.uuid4()),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "created_by": f"opendpd {__version__}",
        })
        return ws

    @classmethod
    def open(cls, root: Path) -> "Workspace":
        ws = cls(root)
        if not ws.meta_path.exists():
            raise WorkspaceError(f"not a workspace (missing workspace.json): {ws.root}")
        meta = read_json(ws.meta_path)
        version = meta.get("workspace_version")
        if version != WORKSPACE_VERSION:
            raise WorkspaceError(f"workspace version {version} is not supported by this build "
                                 f"(expected {WORKSPACE_VERSION}); migration is not automatic")
        for d in (ws.datasets_dir, ws.runs_dir, ws.cache_dir, ws.exports_dir):
            d.mkdir(exist_ok=True)
        return ws

    @classmethod
    def open_or_create(cls, root: Path) -> "Workspace":
        ws = cls(root)
        return cls.open(root) if ws.meta_path.exists() else cls.create(root)

    @property
    def meta(self) -> Dict[str, Any]:
        return read_json(self.meta_path)

    def preflight(self, min_free_mb: int = MIN_FREE_MB) -> List[str]:
        """Problems that would make a run fail late: report them before starting."""
        problems: List[str] = []
        if not os.access(self.root, os.W_OK):
            problems.append(f"workspace is not writable: {self.root}")
        else:
            probe = self.root / f".write-probe-{secrets.token_hex(4)}"
            try:
                probe.write_text("ok")
                probe.unlink()
            except OSError as err:
                problems.append(f"cannot write in workspace ({err})")
        try:
            free_mb = shutil.disk_usage(self.root).free / 1e6
            if free_mb < min_free_mb:
                problems.append(f"only {free_mb:.0f} MB free in workspace; at least {min_free_mb} MB needed")
        except OSError as err:
            problems.append(f"cannot determine free disk space ({err})")
        return problems

    # -- datasets ----------------------------------------------------------
    def dataset_dir(self, dataset_id: str) -> Path:
        return self.datasets_dir / dataset_id

    def list_datasets(self) -> List[DatasetManifest]:
        out = []
        for manifest in sorted(self.datasets_dir.glob("*/manifest.json")):
            out.append(DatasetManifest.model_validate(read_json(manifest)))
        return out

    def get_dataset(self, dataset_id: str) -> DatasetManifest:
        path = self.dataset_dir(dataset_id) / "manifest.json"
        if not path.exists():
            raise WorkspaceError(f"dataset '{dataset_id}' is not registered in {self.root}")
        return DatasetManifest.model_validate(read_json(path))

    def save_dataset(self, manifest: DatasetManifest) -> None:
        write_json_atomic(self.dataset_dir(manifest.dataset_id) / "manifest.json", manifest)

    def dataset_raw_dir(self, dataset_id: str) -> Path:
        return self.dataset_dir(dataset_id) / "raw"

    def dataset_version_dir(self, dataset_id: str, version: str = "raw-v1") -> Path:
        """Directory in the trainer's split-CSV layout for one data version.
        Built-in datasets keep raw-v1 in ``raw/``; imports materialise every
        version (raw-v1 included) under ``versions/<name>/``."""
        candidate = self.dataset_dir(dataset_id) / "versions" / version
        if candidate.is_dir():
            return candidate
        if version == "raw-v1":
            return self.dataset_raw_dir(dataset_id)
        raise WorkspaceError(f"dataset '{dataset_id}' has no version '{version}'")

    # -- authorised import roots ------------------------------------------------
    @property
    def imports_dir(self) -> Path:
        return self.root / "imports"

    def import_roots(self) -> Dict[str, Path]:
        """Directories the service may read user data from: the workspace's own
        ``imports/`` plus any roots recorded in workspace.json. Nothing else."""
        roots = {"imports": self.imports_dir}
        for name, path in (self.meta.get("import_roots") or {}).items():
            roots[slugify(name)] = Path(path).expanduser()
        return roots

    def add_import_root(self, name: str, path: Path) -> None:
        meta = self.meta
        meta.setdefault("import_roots", {})[slugify(name)] = str(Path(path).expanduser().resolve())
        write_json_atomic(self.meta_path, meta)

    def register_builtin_dataset(self, name: str, dataset_id: Optional[str] = None) -> DatasetManifest:
        """Copy a packaged dataset (``datasets/<name>``) into the workspace."""
        src = BUILTIN_DATASETS_DIR / name
        spec_path = src / "spec.json"
        if not spec_path.is_file():
            known = sorted(p.parent.name for p in BUILTIN_DATASETS_DIR.glob("*/spec.json"))
            raise WorkspaceError(f"unknown built-in dataset '{name}'; available: {', '.join(known)}")
        dataset_id = dataset_id or slugify(name)
        target = self.dataset_dir(dataset_id)
        if (target / "manifest.json").exists():
            return self.get_dataset(dataset_id)
        spec = read_json(spec_path)
        raw = target / "raw"
        raw.mkdir(parents=True, exist_ok=True)
        files: List[FileRef] = []
        copied = ["spec.json"] + sorted(p.name for p in src.glob("*.csv"))
        for filename in copied:
            shutil.copy2(src / filename, raw / filename)
            files.append(FileRef(path=f"raw/{filename}", sha256=sha256_file(raw / filename),
                                 size_bytes=(raw / filename).stat().st_size))
        n_samples, boundaries = _count_samples(raw, spec)
        manifest = DatasetManifest(
            dataset_id=dataset_id,
            display_name=f"{name} (built-in, {'measured' if name.startswith(('DPA', 'APA')) else 'origin unknown'})",
            origin=DatasetOrigin.measured if name.startswith(("DPA", "APA")) else DatasetOrigin.unknown,
            source=DatasetSource(kind=DatasetSourceKind.builtin, name=name),
            signal=SignalSpec(
                sample_rate_hz=spec.get("input_signal_fs"), bandwidth_hz=spec.get("bw_main_ch"),
                sub_channel_bandwidth_hz=spec.get("bw_sub_ch"), n_sub_ch=spec.get("n_sub_ch"),
                nperseg=spec.get("nperseg"), modulation=spec.get("modulation"), standard=spec.get("standard"),
                amplitude_units="normalized",
            ),
            files=files,
            n_samples=n_samples,
            split=SplitSpec(ratios={k: float(v) for k, v in spec["split_ratios"].items()}, boundaries=boundaries),
            raw_sha256=combined_sha256(f.sha256 for f in files),
            notes=spec.get("description"),
        )
        self.save_dataset(manifest)
        return manifest

    # -- runs --------------------------------------------------------------
    def new_run_id(self) -> str:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        while True:
            run_id = f"run-{stamp}-{secrets.token_hex(3)}"
            if not (self.runs_dir / run_id).exists():
                return run_id

    def run_dir(self, run_id: str) -> Path:
        return self.runs_dir / run_id

    def list_run_ids(self) -> List[str]:
        return sorted(p.name for p in self.runs_dir.glob("run-*") if (p / "run.json").exists())


def combined_sha256(hashes: Iterable[Optional[str]]) -> str:
    h = hashlib.sha256()
    for item in sorted(x for x in hashes if x):
        h.update(item.encode("ascii"))
    return h.hexdigest()


def _count_samples(raw: Path, spec: Dict[str, Any]):
    """Row counts per split from the raw files (header excluded)."""
    def rows(path: Path) -> int:
        with open(path, "rb") as f:
            return max(sum(1 for _ in f) - 1, 0)

    if spec.get("dataset_format", "split_csv") == "split_csv":
        counts = {s: rows(raw / f"{s}_input.csv") for s in ("train", "val", "test")}
    else:
        total = rows(raw / spec.get("csv_filename", "data.csv"))
        idx = spec.get("split_indices")
        if idx:
            counts = {"train": idx["train_end"], "val": idx["val_end"] - idx["train_end"],
                      "test": total - idx["val_end"]}
        else:
            r = spec["split_ratios"]
            n_train, n_val = int(total * r["train"]), int(total * r["val"])
            counts = {"train": n_train, "val": n_val, "test": total - n_train - n_val}
    boundaries, start = {}, 0
    for split in ("train", "val", "test"):
        boundaries[split] = (start, start + counts[split])
        start += counts[split]
    return start, boundaries
