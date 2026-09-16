"""Private signal uploads and bounded inspection of workspace-owned signals."""
from __future__ import annotations

import csv
import hashlib
import itertools
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np

from opendpd.core.waveforms.analyzer import inspect_signal
from opendpd.schemas.signal_analyzer import AnalyzerSource, AnalyzerSourceInfo, SignalAnalysis
from opendpd.services.csv_import import _is_header, _number
from opendpd.services.csv_upload import CsvUploadRejected, MAX_UPLOAD_BYTES, MAX_LINE_BYTES
from opendpd.services.workspace import WorkspaceError, read_json, sha256_file, write_json_atomic


def upload_directory(ws, identifier):
    return ws.hashed_store('signal_uploads', 'sa').directory(identifier)


def admit_signal_upload(ws, path: Path):
    """Scan every CSV field before publishing a typed array; no executable formats."""
    temporary = None
    try:
        total = 0
        with path.open("rb") as source:
            while line := source.readline(MAX_LINE_BYTES + 1):
                total += len(line)
                if total > MAX_UPLOAD_BYTES or len(line) > MAX_LINE_BYTES:
                    raise CsvUploadRejected("CSV size or line length exceeds the upload limit.")
                if any(c < 32 and c not in (9, 10, 13) for c in line) or b"\x7f" in line:
                    raise CsvUploadRejected("Binary and control characters are not allowed in CSV.")
        source_hash = sha256_file(path)
        identifier = "sa-" + source_hash
        target = upload_directory(ws, identifier)
        if (target / "manifest.json").is_file():
            return AnalyzerSourceInfo.model_validate(read_json(target / "manifest.json")["info"])
        columns, count, complex_columns = [], 0, set()
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle, strict=True)
            first = next(reader, [])
            width = len(first)
            if not 1 <= width <= 8:
                raise CsvUploadRejected("Use 1–8 numeric columns, with an optional header.")
            header = _is_header(first) or first == ["I"] or (width >= 2 and all(
                _is_header([v]) or v.strip().lower() in ('i', 'q') for v in first))
            columns = [v.strip() for v in first] if header else [f"Column {i + 1}" for i in range(width)]
            if len(set(columns)) != width or any(len(v) > 64 for v in columns):
                raise CsvUploadRejected("Use distinct column names of at most 64 characters.")
            rows = reader if header else itertools.chain([first], reader)
            for row in rows:
                if len(row) != width or any(len(v) > 256 for v in row):
                    raise CsvUploadRejected("Every row must have the same number of bounded numeric fields.")
                for i, value in enumerate(row):
                    if _number(value, True).imag:
                        complex_columns.add(i)
                count += 1
                if count > 1_000_000 or count * width > 2_000_000:
                    raise CsvUploadRejected("Use at most 1,000,000 rows and 2,000,000 numeric fields.")
        if count < 256:
            raise CsvUploadRejected("Signal analysis requires at least 256 samples.")
        target.parent.mkdir(parents=True, exist_ok=True)
        if len(list(target.parent.glob("sa-*/manifest.json"))) >= 16:
            raise CsvUploadRejected("This workspace already contains 16 uploaded signals.")
        temporary = Path(tempfile.mkdtemp(prefix=".upload-", dir=target.parent))
        values = np.lib.format.open_memmap(temporary / "samples.npy", mode="w+", dtype=np.complex128, shape=(count, width))
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle, strict=True)
            if header:
                next(reader)
            for i, row in enumerate(reader):
                values[i] = [_number(v, True) for v in row]
        values.flush()
        del values
        info = AnalyzerSourceInfo(source=AnalyzerSource(kind="upload", source_id=identifier),
            label=f"CSV signal · {source_hash[:10]}", sample_count=count, columns=columns,
            complex_columns=sorted(complex_columns), origin="uploaded")
        write_json_atomic(temporary / "manifest.json", {"info": info.model_dump(mode="json"),
            "csv_sha256": source_hash, "array_sha256": sha256_file(temporary / "samples.npy")})
        temporary.replace(target)
        return info
    except (UnicodeError, csv.Error, ValueError, ArithmeticError) as exc:
        if isinstance(exc, CsvUploadRejected):
            raise
        raise CsvUploadRejected("CSV must contain finite real numbers or complex values such as 0.1+0.2j.") from None
    finally:
        path.unlink(missing_ok=True)
        if temporary and temporary.exists():
            shutil.rmtree(temporary)


def list_sources(ws):
    from opendpd.schemas.signal_generator import GeneratedSignal
    from opendpd.schemas.virtual_pa import VirtualPASimulation
    results = []
    for path in sorted((ws.root / "signals").glob("sg-*/manifest.json"))[:100]:
        if path.is_symlink() or path.parent.is_symlink() or (path.parent / ".removed").exists():
            continue
        data = GeneratedSignal.model_validate(read_json(path))
        results.append(AnalyzerSourceInfo(source=AnalyzerSource(kind="generated", source_id=data.signal_id),
            label=data.config.preset_id + " · PA input", sample_count=data.analysis.sample_count,
            sample_rate_hz=data.config.sample_rate_hz, bandwidth_hz=data.config.bandwidth_hz, origin="synthetic"))
    for path in sorted((ws.root / "signal_uploads").glob("sa-*/manifest.json"))[:16]:
        if not path.is_symlink() and not path.parent.is_symlink():
            results.append(AnalyzerSourceInfo.model_validate(read_json(path)["info"]))
    for data in ws.list_datasets()[:100]:
        for role in ("input", "output"):
            results.append(AnalyzerSourceInfo(source=AnalyzerSource(kind="dataset", source_id=data.dataset_id, role=role),
                label=f"{data.display_name} · PA {role}", sample_count=data.n_samples or 0,
                sample_rate_hz=data.signal.sample_rate_hz, bandwidth_hz=data.signal.bandwidth_hz, origin=data.origin.value))
    for path in sorted((ws.root / "pa_simulations").glob("vpa-*/manifest.json"))[:100]:
        if path.is_symlink() or path.parent.is_symlink():
            continue
        data = VirtualPASimulation.model_validate(read_json(path))
        results.append(AnalyzerSourceInfo(source=AnalyzerSource(kind="virtual_pa", source_id=data.simulation_id, role="output"),
            label=data.model.name.en + " · PA output", sample_count=data.analysis.n_samples,
            sample_rate_hz=data.sample_rate_hz, origin="synthetic"))
    return results


def _load(ws, source):
    if source.kind == "generated":
        from opendpd.services.signal_generator import read_signal, directory
        data = read_signal(ws, source.source_id)
        info = AnalyzerSourceInfo(source=source, label=data.config.preset_id + " · PA input",
            sample_count=data.analysis.sample_count, sample_rate_hz=data.config.sample_rate_hz,
            bandwidth_hz=data.config.bandwidth_hz, origin="synthetic")
        return np.load(directory(ws, source.source_id) / "iq.npy", mmap_mode="r", allow_pickle=False), info
    if source.kind == "virtual_pa":
        from opendpd.services.virtual_pa import read_simulation, directory
        data = read_simulation(ws, source.source_id)
        info = AnalyzerSourceInfo(source=source, label=data.model.name.en + " · PA output",
            sample_count=data.analysis.n_samples, sample_rate_hz=data.sample_rate_hz, origin="synthetic")
        return np.load(directory(ws, source.source_id) / "output.npy", mmap_mode="r", allow_pickle=False), info
    if source.kind == "dataset":
        from opendpd.services.datasets import load_version_arrays
        data = ws.get_dataset(source.source_id)
        x, y, _ = load_version_arrays(ws, source.source_id, source.version)
        values = x if source.role == "input" else y
        info = AnalyzerSourceInfo(source=source, label=f"{data.display_name} · PA {source.role}",
            sample_count=len(values), sample_rate_hz=data.signal.sample_rate_hz, bandwidth_hz=data.signal.bandwidth_hz,
            origin=data.origin.value)
        return values, info
    target = upload_directory(ws, source.source_id)
    path, manifest = target / "samples.npy", target / "manifest.json"
    if path.is_symlink() or manifest.is_symlink() or not path.is_file() or not manifest.is_file():
        raise WorkspaceError("Upload this signal in the current workspace first.")
    proof = read_json(manifest)
    if sha256_file(path) != proof["array_sha256"]:
        raise WorkspaceError("Uploaded samples changed after validation.")
    return np.load(path, mmap_mode="r", allow_pickle=False), AnalyzerSourceInfo.model_validate(proof["info"])


def _select(values, info, config):
    start, end = config.start_sample, min(len(values), config.start_sample + config.n_samples)
    if end - start < 256:
        raise WorkspaceError("The selected range must contain at least 256 samples.")
    window = np.asarray(values[start:end])
    if info.source.kind != "upload":
        if window.ndim == 1:
            return window.astype(complex)
        return window[:, 0].astype(complex) + 1j * window[:, 1]
    fmt = config.sample_format
    if fmt == "auto":
        fmt = "complex" if info.complex_columns or window.shape[1] == 1 else "iq"
    cols = [config.i_column, config.q_column] if fmt == "iq" else [config.i_column]
    if max(cols) >= window.shape[1]:
        raise WorkspaceError("Select columns that exist in this CSV.")
    if fmt in ("real", "iq") and any(col in info.complex_columns for col in cols):
        raise WorkspaceError("Selected columns contain complex samples; choose Complex format to preserve Q.")
    return (window[:, config.i_column].astype(complex) + 1j * window[:, config.q_column].real
            if fmt == "iq" else window[:, config.i_column].astype(complex))


def analyze(ws, request):
    try:
        values, info = _load(ws, request.source)
        raw, ref_info = _load(ws, request.reference) if request.reference else (None, None)
    except (WorkspaceError, OSError, ValueError) as exc:
        raise WorkspaceError("The selected signal or reference is unavailable in this workspace. Select a current source and data version.") from exc
    x = _select(values, info, request.config)
    reference, reference_hash = None, None
    if request.reference:
        if info.sample_rate_hz and ref_info.sample_rate_hz and info.sample_rate_hz != ref_info.sample_rate_hz:
            raise WorkspaceError("Reference and signal sample rates differ; no implicit resampling is performed.")
        reference = _select(raw, ref_info, request.config)
        reference_hash = hashlib.sha256(reference.tobytes()).hexdigest()
    try:
        result = inspect_signal(x, request.config, reference)
    except ValueError as exc:
        raise WorkspaceError(str(exc)) from exc
    if info.sample_rate_hz and info.sample_rate_hz != request.config.sample_rate_hz:
        result["notes"].append(f"The analysis rate overrides source metadata ({info.sample_rate_hz:g} Hz); samples are reinterpreted, not resampled.")
    return SignalAnalysis(source=info, config=request.config, source_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
        reference_sha256=reference_hash, sample_range=(request.config.start_sample, request.config.start_sample + len(x)), **result)
