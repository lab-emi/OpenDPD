"""Dataset import, inspection, Doctor and preprocessing on top of a Workspace.

One implementation for the GUI, the CLI and the Python API, so all three
return the same manifest, the same diagnostic report and the same version
record. Sources are read incrementally (CSV in chunks, NumPy memory-mapped);
object arrays are refused (``allow_pickle=False``).
"""

from __future__ import annotations

import csv
import hashlib
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from opendpd.core import preprocess as pp
from opendpd.core.doctor import diagnose
from opendpd.core.splits import DEFAULT_GUARD_SAMPLES, DEFAULT_RATIOS, SPLIT_VERSION, contiguous_boundaries
from opendpd.schemas import (
    DatasetManifest,
    DatasetOrigin,
    DatasetSource,
    DatasetSourceKind,
    DatasetVersion,
    DiagnosticReport,
    FileRef,
    PreprocessingParams,
    SignalSpec,
    SplitSpec,
)
from opendpd.services.workspace import Workspace, WorkspaceError, combined_sha256, read_json, sha256_file, slugify, \
    write_json_atomic
from opendpd.schemas.importing import CsvOptions, DatasetImportDefaults

LOGICAL = ("I_in", "Q_in", "I_out", "Q_out")
SUPPORTED_SUFFIXES = (".csv", ".npy", ".npz")
CSV_CHUNK_ROWS = 200_000
LEGACY_SPLIT_FILES = tuple(f"{s}_{k}.csv" for s in ("train", "val", "test") for k in ("input", "output"))

# Column-name heuristics -> logical names (lower-cased, punctuation removed).
_ALIASES: Dict[str, Tuple[str, ...]] = {
    "I_in": ("i_in", "iin", "i", "in_i", "input_i", "x_i", "xi", "i_x", "re_in", "input_re", "x_re", "tx_i", "pa_in_i"),
    "Q_in": ("q_in", "qin", "q", "in_q", "input_q", "x_q", "xq", "q_x", "im_in", "input_im", "x_im", "tx_q", "pa_in_q"),
    "I_out": ("i_out", "iout", "out_i", "output_i", "y_i", "yi", "i_y", "re_out", "output_re", "y_re", "rx_i", "pa_out_i"),
    "Q_out": ("q_out", "qout", "out_q", "output_q", "y_q", "yq", "q_y", "im_out", "output_im", "y_im", "rx_q", "pa_out_q"),
}


class ImportError_(WorkspaceError):
    """Import problems are workspace errors with a field-level explanation."""


class UploadTooLarge(ImportError_):
    """The streamed upload passed the configured cap; the partial file was removed."""


# --- authorised roots ------------------------------------------------------------------

@dataclass
class FileEntry:
    path: str          # relative to the root, POSIX
    kind: str          # "file" | "dir"
    size_bytes: int = 0


def resolve_in_root(ws: Workspace, root_id: str, relative: str) -> Path:
    """Absolute path inside an authorised root; refuses traversal and symlink escapes."""
    roots = ws.import_roots()
    if root_id not in roots:
        raise ImportError_(f"unknown import root '{root_id}'; available: {', '.join(sorted(roots))}")
    base = roots[root_id].resolve()
    target = (base / relative).resolve()
    if target != base and base not in target.parents:
        raise ImportError_("path escapes the import root")
    return target


def list_files(ws: Workspace, root_id: str, relative: str = "") -> List[FileEntry]:
    base = resolve_in_root(ws, root_id, relative)
    if not base.exists():
        return []
    if base.is_file():
        return [FileEntry(path=relative, kind="file", size_bytes=base.stat().st_size)]
    entries = []
    root = ws.import_roots()[root_id].resolve()
    for p in sorted(base.iterdir(), key=lambda q: (q.is_file(), q.name.lower())):
        if p.name.startswith("."):
            continue
        resolved = p.resolve()
        if resolved != root and root not in resolved.parents:
            continue        # a symlink pointing outside the root is never listed (and never readable)
        rel = resolved.relative_to(root).as_posix()
        if p.is_dir():
            entries.append(FileEntry(path=rel, kind="dir"))
        elif p.suffix.lower() in SUPPORTED_SUFFIXES:
            entries.append(FileEntry(path=rel, kind="file", size_bytes=p.stat().st_size))
    return entries


# --- inspection ------------------------------------------------------------------------

@dataclass
class SourceInfo:
    kind: DatasetSourceKind
    path: str
    columns: List[str] = field(default_factory=list)          # CSV headers / npz keys
    suggested_mapping: Dict[str, str] = field(default_factory=dict)
    n_rows: Optional[int] = None
    preview: List[Dict[str, float]] = field(default_factory=list)
    arrays: Dict[str, Dict[str, object]] = field(default_factory=dict)   # npy/npz: name -> {dtype, shape}
    problems: List[str] = field(default_factory=list)
    legacy_files: List[str] = field(default_factory=list)


def _norm(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum() or ch == "_")


def suggest_mapping(columns: Iterable[str]) -> Dict[str, str]:
    cols = list(columns)
    normed = {_norm(c): c for c in cols}
    out: Dict[str, str] = {}
    for logical, aliases in _ALIASES.items():
        if _norm(logical) in normed:
            out[logical] = normed[_norm(logical)]
            continue
        for alias in aliases:
            if alias in normed and normed[alias] not in out.values():
                out[logical] = normed[alias]
                break
    return out


def _count_rows(path: Path) -> int:
    with open(path, "rb") as f:
        return max(sum(1 for _ in f) - 1, 0)


def inspect_source(path: Path) -> SourceInfo:
    path = Path(path)
    if path.is_dir():
        present = [f for f in LEGACY_SPLIT_FILES if (path / f).is_file()]
        info = SourceInfo(kind=DatasetSourceKind.legacy_dir_import, path=str(path), legacy_files=present)
        if len(present) != len(LEGACY_SPLIT_FILES):
            missing = sorted(set(LEGACY_SPLIT_FILES) - set(present))
            info.problems.append("not an OpenDPD split directory; missing " + ", ".join(missing))
        elif (path / "spec.json").is_file():
            info.columns = list(read_json(path / "spec.json").keys())
        return info
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with open(path, newline="") as f:
            reader = csv.reader(f)
            header = next(reader, [])
            rows = []
            for _ in range(5):
                row = next(reader, None)
                if row is None:
                    break
                rows.append(row)
        info = SourceInfo(kind=DatasetSourceKind.csv_import, path=str(path), columns=header,
                          suggested_mapping=suggest_mapping(header), n_rows=_count_rows(path))
        for row in rows:
            try:
                info.preview.append({h: float(v) for h, v in zip(header, row)})
            except ValueError:
                info.problems.append(f"non-numeric values in row: {row[:6]}")
                break
        missing = [k for k in LOGICAL if k not in info.suggested_mapping]
        if missing:
            info.problems.append("could not guess columns for " + ", ".join(missing) + "; map them explicitly")
        return info
    if suffix in (".npy", ".npz"):
        info = SourceInfo(kind=DatasetSourceKind.numpy_import, path=str(path))
        try:
            loaded = np.load(path, mmap_mode="r", allow_pickle=False)
        except ValueError as err:
            info.problems.append(f"refused: {err} (object arrays are never loaded)")
            return info
        arrays = {"array": loaded} if suffix == ".npy" else {k: loaded[k] for k in loaded.files}
        for name, arr in arrays.items():
            if arr.dtype.kind not in "fiuc":
                info.problems.append(f"{name}: dtype {arr.dtype} is not numeric")
                continue
            info.arrays[name] = {"dtype": str(arr.dtype), "shape": list(arr.shape)}
        info.columns = list(info.arrays)
        if suffix == ".npz":
            info.suggested_mapping = {k: v for k, v in suggest_mapping(info.arrays).items()}
            lowered = {k.lower(): k for k in info.arrays}
            for logical, keys in (("input", ("input", "input_iq", "x", "tx")), ("output", ("output", "output_iq", "y", "rx"))):
                for key in keys:
                    if key in lowered:
                        info.suggested_mapping[logical] = lowered[key]
                        break
        return info
    raise ImportError_(f"unsupported file type '{suffix}'; use .csv, .npy, .npz or an OpenDPD split directory")


# --- reading arrays ---------------------------------------------------------------------

def _read_csv_arrays(path: Path, mapping: Dict[str, str]) -> Tuple[np.ndarray, np.ndarray]:
    import pandas as pd

    missing = [k for k in LOGICAL if k not in mapping]
    if missing:
        raise ImportError_("mapping is incomplete for " + ", ".join(missing))
    with open(path, newline="") as f:
        header = next(csv.reader(f), [])
    unknown = [c for c in mapping.values() if c not in header]
    if unknown:
        raise ImportError_(f"columns {unknown} are not in the file; found {header}")
    parts_x, parts_y = [], []
    usecols = [mapping[k] for k in LOGICAL]
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=CSV_CHUNK_ROWS, dtype=np.float64):
        parts_x.append(chunk[[mapping["I_in"], mapping["Q_in"]]].to_numpy(dtype=np.float32))
        parts_y.append(chunk[[mapping["I_out"], mapping["Q_out"]]].to_numpy(dtype=np.float32))
    if not parts_x:
        raise ImportError_("the CSV has no data rows")
    return np.concatenate(parts_x), np.concatenate(parts_y)


def _as_iq(arr: np.ndarray, name: str) -> np.ndarray:
    """(n, 2) float view of a source array. Float/int layouts stay memory-mapped views (converted
    chunk by chunk when written); a complex array is the one layout that needs a full copy."""
    arr = np.asarray(arr)
    if arr.dtype.kind == "c":
        return np.stack([arr.real, arr.imag], -1).astype(np.float32)
    if arr.dtype.kind not in "fiu":
        raise ImportError_(f"{name}: dtype {arr.dtype} is not numeric")
    if arr.ndim == 2 and arr.shape[1] == 2:
        return arr
    if arr.ndim == 2 and arr.shape[0] == 2:
        return arr.T
    raise ImportError_(f"{name}: expected shape (n, 2) or complex (n,), got {arr.shape} {arr.dtype}")


def _read_numpy_arrays(path: Path, mapping: Dict[str, str]) -> Tuple[np.ndarray, np.ndarray]:
    loaded = np.load(path, mmap_mode="r", allow_pickle=False)
    if path.suffix.lower() == ".npy":
        arr = np.asarray(loaded)
        if arr.ndim == 2 and arr.shape[1] == 4:
            return arr[:, :2], arr[:, 2:]
        if arr.ndim == 3 and arr.shape[0] == 2:
            return _as_iq(arr[0], "input"), _as_iq(arr[1], "output")
        raise ImportError_("a single .npy needs shape (n, 4) [I_in,Q_in,I_out,Q_out] or (2, n, 2); "
                           "use .npz with 'input' and 'output' arrays otherwise")
    keys = {k.lower(): k for k in loaded.files}
    in_key = mapping.get("input") or keys.get("input") or keys.get("input_iq") or keys.get("x")
    out_key = mapping.get("output") or keys.get("output") or keys.get("output_iq") or keys.get("y")
    if not in_key or not out_key:
        raise ImportError_(f"npz needs 'input' and 'output' arrays (found {loaded.files}); map them explicitly")
    return _as_iq(loaded[in_key], in_key), _as_iq(loaded[out_key], out_key)


def load_version_arrays(ws: Workspace, dataset_id: str, version: str = "raw-v1") -> Tuple[np.ndarray, np.ndarray, SplitSpec]:
    """Continuous (x, y) of a version in time order plus its split boundaries.

    Materialised versions keep the continuous capture as ``input_iq.npy`` /
    ``output_iq.npy`` (guard samples included) next to the split CSVs the
    trainer reads; built-in datasets only have the split CSVs, which are
    contiguous with no guard, so concatenating them restores the capture.
    """
    manifest = ws.get_dataset(dataset_id)
    directory = ws.dataset_version_dir(dataset_id, version)
    v = manifest.version(version)
    if (directory / "input_iq.npy").is_file():
        x = np.load(directory / "input_iq.npy", mmap_mode="r", allow_pickle=False)
        y = np.load(directory / "output_iq.npy", mmap_mode="r", allow_pickle=False)
        return x, y, (v.split if v else manifest.split)
    # The shared legacy loader also understands packaged single_csv datasets
    # such as MyCustomPA. Reassembling its splits preserves capture time order.
    from modules.data_collector import load_dataset

    xt, yt, xv, yv, xe, ye = load_dataset(dataset_path=directory)
    return (np.concatenate((xt, xv, xe)).astype(np.float32),
            np.concatenate((yt, yv, ye)).astype(np.float32),
            v.split if v else manifest.split)


# --- writing versions ------------------------------------------------------------------

def _write_split_dir(directory: Path, x: np.ndarray, y: np.ndarray, boundaries: Dict[str, Tuple[int, int]],
                     spec: Dict[str, object]) -> List[FileRef]:
    """Legacy split-CSV layout (what modules/data_collector.load_dataset reads)."""
    directory.mkdir(parents=True, exist_ok=True)
    files: List[FileRef] = []
    for split, (start, end) in boundaries.items():
        for kind, arr, cols in (("input", x, ("I", "Q")), ("output", y, ("I", "Q"))):
            path = directory / f"{split}_{kind}.csv"
            with open(path, "w", newline="") as f:
                f.write(",".join(cols) + "\n")
                for s0 in range(start, end, CSV_CHUNK_ROWS):
                    block = arr[s0:min(s0 + CSV_CHUNK_ROWS, end)]
                    np.savetxt(f, block, delimiter=",", fmt="%.8g")
            files.append(FileRef(path=path.name, sha256=sha256_file(path), size_bytes=path.stat().st_size))
    write_json_atomic(directory / "spec.json", spec)
    files.append(FileRef(path="spec.json", sha256=sha256_file(directory / "spec.json"),
                         size_bytes=(directory / "spec.json").stat().st_size))
    return files


def _save_float32_chunked(path: Path, arr: np.ndarray) -> None:
    """Write ``arr`` as a float32 .npy without materialising it: the source may be a memory-mapped
    view of a file far larger than RAM (the Stress tier of the performance protocol)."""
    out = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=arr.shape)
    try:
        for start in range(0, len(arr), CSV_CHUNK_ROWS):
            out[start:start + CSV_CHUNK_ROWS] = arr[start:start + CSV_CHUNK_ROWS]
        out.flush()
    finally:
        del out


def _legacy_spec(signal: SignalSpec, split: SplitSpec, description: str) -> Dict[str, object]:
    return {
        "description": description, "dataset_format": "split_csv", "split_ratios": dict(split.ratios),
        "input_signal_fs": signal.sample_rate_hz, "bw_main_ch": signal.bandwidth_hz,
        "bw_sub_ch": signal.sub_channel_bandwidth_hz, "n_sub_ch": signal.n_sub_ch, "nperseg": signal.nperseg,
        "standard": signal.standard, "modulation": signal.modulation,
    }


def _materialise(ws: Workspace, manifest: DatasetManifest, version: str, x: np.ndarray, y: np.ndarray, *,
                 base_version: Optional[str], params: Optional[PreprocessingParams], record: Dict[str, object],
                 guard: int, ratios: Dict[str, float]) -> DatasetVersion:
    n = len(x)
    boundaries = contiguous_boundaries(n, ratios, guard)
    split = SplitSpec(version=SPLIT_VERSION, ratios=ratios, guard_samples=guard, boundaries=boundaries)
    directory = ws.dataset_dir(manifest.dataset_id) / "versions" / version
    if directory.exists():
        raise ImportError_(f"version '{version}' already exists for dataset '{manifest.dataset_id}'")
    files = _write_split_dir(directory, x, y, boundaries, _legacy_spec(manifest.signal, split, manifest.display_name))
    for name, arr in (("input_iq.npy", x), ("output_iq.npy", y)):
        _save_float32_chunked(directory / name, arr)
        files.append(FileRef(path=name, sha256=sha256_file(directory / name), size_bytes=(directory / name).stat().st_size))
    fit_range = record.get("fit_range")
    dv = DatasetVersion(version=version, base_version=base_version, params=params,
                        code_version=record.get("code_version"), fit_range=tuple(fit_range) if fit_range else None,
                        record=record, n_samples=n, split=split, files=files, sha256=combined_sha256(f.sha256 for f in files))
    write_json_atomic(directory / "version.json", dv)
    return dv


# --- public operations ---------------------------------------------------------------------

def import_dataset(ws: Workspace, source: Path, *, dataset_id: Optional[str] = None, display_name: Optional[str] = None,
                   mapping: Optional[Dict[str, str]] = None, signal: Optional[SignalSpec] = None,
                   origin: DatasetOrigin = DatasetOrigin.unknown, guard_samples: int = DEFAULT_GUARD_SAMPLES,
                   notes: Optional[str] = None, waveform: Optional[Path] = None,
                   ratios: Optional[Dict[str, float]] = None, csv_options: Optional[CsvOptions] = None,
                   expected_sha256: Optional[str] = None) -> DatasetManifest:
    """Copy the source into the workspace (hashed, untouched) and materialise raw-v1.

    ``waveform`` (a reference-waveform package, plan S15) binds the dataset to the waveform its input column
    was captured from: the waveform is regenerated and correlated with the input; an input that is not the
    waveform is refused, so a binding never rests on a guess."""
    source = Path(source)
    if not source.exists():
        raise ImportError_(f"{source} does not exist")
    info = (SourceInfo(kind=DatasetSourceKind.csv_import, path=str(source))
            if csv_options is not None else inspect_source(source))
    mapping = dict(info.suggested_mapping, **(mapping or {}))
    signal = signal or SignalSpec()
    ratios = dict(DEFAULT_RATIOS if ratios is None else ratios)
    dataset_id = dataset_id or slugify(source.stem if source.is_file() else source.name)
    from pydantic import TypeAdapter
    from opendpd.schemas.common import Slug
    TypeAdapter(Slug).validate_python(dataset_id)
    display_name = display_name or (source.stem if source.is_file() else source.name)
    if origin == DatasetOrigin.synthetic and "synthetic" not in display_name.lower():
        display_name = f"{display_name} (synthetic)"
    target = ws.dataset_dir(dataset_id)
    if (target / "manifest.json").exists():
        raise ImportError_(f"dataset id '{dataset_id}' already exists; choose another id")

    raw = target / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    files: List[FileRef] = []
    try:
        if info.kind == DatasetSourceKind.legacy_dir_import:
            if info.problems:
                raise ImportError_(info.problems[0])
            if waveform is not None:
                raise ImportError_("a waveform binding needs the capture file (CSV or NumPy), not a split directory")
            for name in LEGACY_SPLIT_FILES + ("spec.json",):
                if (source / name).is_file():
                    shutil.copy2(source / name, raw / name)
                    files.append(FileRef(path=f"raw/{name}", sha256=sha256_file(raw / name),
                                         size_bytes=(raw / name).stat().st_size))
            if (source / "spec.json").is_file():
                spec = read_json(source / "spec.json")
                signal = SignalSpec(**{**dict(sample_rate_hz=spec.get("input_signal_fs"), bandwidth_hz=spec.get("bw_main_ch"),
                                              sub_channel_bandwidth_hz=spec.get("bw_sub_ch"), n_sub_ch=spec.get("n_sub_ch"),
                                              nperseg=spec.get("nperseg"), modulation=spec.get("modulation"),
                                              standard=spec.get("standard")),
                                       **{k: v for k, v in signal.model_dump().items() if v not in (None, "unknown")}})
            import pandas as pd
            counts = {s: len(pd.read_csv(raw / f"{s}_input.csv", usecols=[0])) for s in ("train", "val", "test")}
            total = sum(counts.values())
            boundaries, start = {}, 0
            for s in ("train", "val", "test"):
                boundaries[s] = (start, start + counts[s])
                start += counts[s]
            manifest = DatasetManifest(
                dataset_id=dataset_id, display_name=display_name, origin=origin,
                source=DatasetSource(kind=info.kind, original_path=str(source)), signal=signal, files=files,
                n_samples=total, split=SplitSpec(ratios={s: counts[s] / total for s in counts}, guard_samples=0,
                                                 boundaries=boundaries),
                raw_sha256=combined_sha256(f.sha256 for f in files), notes=notes or "imported OpenDPD split directory; "
                "the split was made by the original author (guard 0, boundaries as found)")
            ws.save_dataset(manifest)
            return manifest

        copied = raw / source.name
        shutil.copy2(source, copied)
        files.append(FileRef(path=f"raw/{source.name}", sha256=sha256_file(copied), size_bytes=copied.stat().st_size))
        if csv_options is not None:
            from opendpd.services.csv_import import inspect_csv
            report, x, y = inspect_csv(copied, csv_options, DatasetImportDefaults(ratios=ratios, guard_samples=guard_samples), collect=True)
            if expected_sha256 is not None and report.sha256 != expected_sha256:
                raise ImportError_("The CSV changed since validation. Validate it again before creating the dataset.")
            if not report.valid:
                detail = report.issues[0]
                where = f"line {detail.line}, " if detail.line else ""
                where += f"column {detail.column}: " if detail.column else ""
                raise ImportError_(f"{where}{detail.message} {detail.fix}")
            mapping = {key: report.columns[index] for key, index in report.options.mapping.items()}
        elif info.kind == DatasetSourceKind.csv_import:
            x, y = _read_csv_arrays(copied, mapping)
        else:
            x, y = _read_numpy_arrays(copied, mapping)
        if len(x) != len(y):
            raise ImportError_(f"input has {len(x)} samples but output has {len(y)}; they must be paired")
        if waveform is not None:
            signal = signal.model_copy(update={"waveform": bind_to_waveform(x, signal, waveform)})
        manifest = DatasetManifest(
            dataset_id=dataset_id, display_name=display_name, origin=origin,
            source=DatasetSource(kind=info.kind, original_path=str(source)), signal=signal, files=files,
            n_samples=int(len(x)), columns=mapping or None,
            split=SplitSpec(version=SPLIT_VERSION, ratios=ratios, guard_samples=guard_samples),
            raw_sha256=combined_sha256(f.sha256 for f in files), notes=notes)
        record = {"code_version": None, "note": "raw samples, contiguous split with guard"}
        if csv_options is not None:
            record["csv_options"] = report.options.model_dump(mode="json")
        version = _materialise(ws, manifest, "raw-v1", x, y, base_version=None, params=None,
                               record=record,
                               guard=guard_samples, ratios=ratios)
        manifest = manifest.model_copy(update={"split": version.split, "versions": [version]})
        ws.save_dataset(manifest)
        return manifest
    except Exception:
        shutil.rmtree(target, ignore_errors=True)
        raise


def bind_to_waveform(x: np.ndarray, signal: SignalSpec, package: Path):
    """The binding of an input column to a reference-waveform package (``waveform.json`` or its directory)."""
    from opendpd.core.waveforms import bind_input, read_package

    if signal.sample_rate_hz is None:
        raise ImportError_("binding to a waveform needs the capture's sample rate (--fs)")
    try:
        spec, digest = read_package(Path(package))
    except (OSError, ValueError, KeyError) as err:
        raise ImportError_(f"{package} is not a readable waveform package: {err}") from None
    try:
        return bind_input(x, float(signal.sample_rate_hz), spec, package_sha256=digest)
    except ValueError as err:
        raise ImportError_(str(err)) from None


def update_manifest(ws: Workspace, dataset_id: str, *, signal: Optional[SignalSpec] = None,
                    display_name: Optional[str] = None, origin: Optional[DatasetOrigin] = None,
                    notes: Optional[str] = None) -> DatasetManifest:
    """Edit descriptive metadata; data files and hashes are untouched."""
    manifest = ws.get_dataset(dataset_id)
    updates = {}
    if signal is not None:
        updates["signal"] = signal
    if display_name:
        updates["display_name"] = display_name
    if origin is not None:
        updates["origin"] = origin
    if notes is not None:
        updates["notes"] = notes
    manifest = manifest.model_copy(update=updates)
    DatasetManifest.model_validate(manifest.model_dump())
    ws.save_dataset(manifest)
    for v in manifest.versions:      # keep the trainer-side spec.json in step with the manifest
        d = ws.dataset_dir(dataset_id) / "versions" / v.version
        if d.is_dir():
            write_json_atomic(d / "spec.json", _legacy_spec(manifest.signal, v.split, manifest.display_name))
    return manifest


def diagnostics_dir(ws: Workspace, dataset_id: str) -> Path:
    return ws.dataset_dir(dataset_id) / "diagnostics"


MAX_DOCTOR_SAMPLES = 2_000_000


def analysis_window(n: int, limit: int = MAX_DOCTOR_SAMPLES) -> Tuple[int, int]:
    """Central contiguous window the doctor analyses: all of a capture up to ``limit`` samples, else
    ``limit`` samples from the middle (start-up transients and trailing silence are the usual edges)."""
    if n <= limit:
        return 0, n
    start = (n - limit) // 2
    return start, start + limit


def _windowed(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Optional[Tuple[int, int]]]:
    start, end = analysis_window(len(x), MAX_DOCTOR_SAMPLES)
    if (start, end) == (0, len(x)):
        return np.asarray(x), np.asarray(y), None
    return np.array(x[start:end]), np.array(y[start:end]), (start, end)


def _with_window_note(report: DiagnosticReport, window: Optional[Tuple[int, int]], total: int) -> DiagnosticReport:
    if window is None:
        return report
    from opendpd.core.doctor import _item

    note = _item("analysis_window", "info", "Analysis window",
                 f"Diagnostics were computed on samples {window[0]:,}–{window[1]:,} of {total:,} (a central window of "
                 f"{window[1] - window[0]:,} samples); statistics outside it are not checked",
                 {"start": window[0], "end": window[1], "n_samples": total},
                 suggestion="Preprocessing versions still apply to the whole capture")
    return report.model_copy(update={"items": [*report.items, note]})


def run_doctor(ws: Workspace, dataset_id: str, version: str = "raw-v1") -> DiagnosticReport:
    manifest = ws.get_dataset(dataset_id)
    x, y, _ = load_version_arrays(ws, dataset_id, version)
    xw, yw, window = _windowed(x, y)
    report = _with_window_note(diagnose(xw, yw, manifest.signal, dataset_id=dataset_id, raw_sha256=manifest.raw_sha256),
                               window, len(x))
    d = diagnostics_dir(ws, dataset_id)
    d.mkdir(parents=True, exist_ok=True)
    payload = report.model_dump(mode="json")
    payload["data_version"] = version
    write_json_atomic(d / f"{report.report_id}.json", payload)
    write_json_atomic(d / "latest.json", payload)
    return report


def latest_report(ws: Workspace, dataset_id: str) -> Optional[DiagnosticReport]:
    path = diagnostics_dir(ws, dataset_id) / "latest.json"
    if not path.exists():
        return None
    data = read_json(path)
    data.pop("data_version", None)
    return DiagnosticReport.model_validate(data)


def preview_preprocess(ws: Workspace, dataset_id: str, params: PreprocessingParams, base_version: str = "raw-v1") -> Dict[str, object]:
    """What the version would contain and what the doctor says afterwards; nothing is written."""
    manifest = ws.get_dataset(dataset_id)
    x, y, split = load_version_arrays(ws, dataset_id, base_version)
    fit_range = split.boundaries["train"] if split.boundaries else contiguous_boundaries(len(x), split.ratios, split.guard_samples)["train"]
    x2, y2, record = pp.apply(np.array(x), np.array(y), params, fit_range=fit_range)
    xw, yw, window = _windowed(x2, y2)
    after = _with_window_note(diagnose(xw, yw, manifest.signal, dataset_id=dataset_id, raw_sha256=manifest.raw_sha256),
                              window, len(x2))
    return {"n_samples_before": int(len(x)), "n_samples_after": int(len(x2)), "record": record,
            "report_after": after.model_dump(mode="json")}


def create_version(ws: Workspace, dataset_id: str, version: str, params: PreprocessingParams,
                   base_version: str = "raw-v1") -> DatasetVersion:
    """Confirmed preprocessing: write a new data version; raw files and raw_sha256 stay unchanged."""
    manifest = ws.get_dataset(dataset_id)
    if manifest.version(version) is not None or version == "raw-v1":
        raise ImportError_(f"version '{version}' already exists")
    x, y, split = load_version_arrays(ws, dataset_id, base_version)
    fit_range = split.boundaries["train"] if split.boundaries else contiguous_boundaries(len(x), split.ratios, split.guard_samples)["train"]
    x2, y2, record = pp.apply(np.array(x), np.array(y), params, fit_range=fit_range)      # whole capture in RAM, as before
    guard = split.guard_samples if manifest.source.kind != DatasetSourceKind.builtin else DEFAULT_GUARD_SAMPLES
    dv = _materialise(ws, manifest, version, x2, y2, base_version=base_version, params=params, record=record,
                      guard=guard, ratios=dict(split.ratios))
    ws.save_dataset(manifest.model_copy(update={"versions": [*manifest.versions, dv]}))
    return dv


def receive_upload(ws: Workspace, filename: str, chunks: Iterable[bytes], max_bytes: int) -> Path:
    """Stream a browser upload into imports/uploads/ (never into a user-chosen path)."""
    safe = Path(filename).name
    if not safe or Path(safe).suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ImportError_(f"only {', '.join(SUPPORTED_SUFFIXES)} uploads are accepted")
    target_dir = ws.imports_dir / "uploads"
    target_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    target = target_dir / f"{stamp}-{safe}"
    digest, total = hashlib.sha256(), 0
    with open(target, "wb") as f:
        for chunk in chunks:
            total += len(chunk)
            if total > max_bytes:
                f.close()
                target.unlink(missing_ok=True)
                raise UploadTooLarge(f"upload exceeds {max_bytes} bytes")
            digest.update(chunk)
            f.write(chunk)
    return target
