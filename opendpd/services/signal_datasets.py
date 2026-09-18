"""Persist input/output dataset membership independently of immutable signal bytes."""
import hashlib
import json
import re
import shutil
import tempfile
import threading
import zipfile
from pathlib import Path

from opendpd.schemas.signal_dataset import SignalDataset
from opendpd.services.workspace import read_json, write_json_atomic

LOCK = threading.RLock()


def list_datasets(ws):
    store = ws.hashed_store("signal_datasets", "sds")
    if store.root.is_symlink():
        return []
    return [store.manifest(p.parent.name, SignalDataset)
            for p in sorted(store.root.glob("sds-*/manifest.json"))
            if not p.is_symlink() and not p.parent.is_symlink()]


def read_dataset(ws, identifier):
    return ws.hashed_store("signal_datasets", "sds").manifest(identifier, SignalDataset)


def unique_name(name, existing):
    candidate, index = name, 2
    while candidate in existing:
        suffix = f"_{index}"
        candidate = name[:96-len(suffix)] + suffix
        index += 1
    return candidate


def save_dataset(ws, name, kind, signals):
    with LOCK:
        identity = {"name": name, "kind": kind, "signals": [s.source.model_dump(mode="json") for s in signals]}
        identifier = "sds-" + hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        target = ws.hashed_store("signal_datasets", "sds").directory(identifier)
        if (target / "manifest.json").exists():
            return read_dataset(ws, identifier)
        chosen = unique_name(name, {d.name for d in list_datasets(ws)})
        result = SignalDataset(dataset_id=identifier, name=chosen, requested_name=name, kind=kind,
            signals=signals, download_url=f"/api/v1/signal-generator/datasets/{identifier}/download")
        target.mkdir(parents=True, exist_ok=True)
        write_json_atomic(target / "manifest.json", result)
        return result


def export_dataset(ws, identifier):
    """Separate CSVs and full metadata per signal; never join different rates."""
    from opendpd.services import signal_generator, virtual_pa
    dataset = read_dataset(ws, identifier)
    temporary = Path(tempfile.mkdtemp(prefix="signal-dataset-", dir=ws.exports_dir))
    try:
        filename = re.sub(r"[^A-Za-z0-9_.-]", "-", dataset.name)
        path = temporary / f"{filename}.zip"
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
            for index, member in enumerate(dataset.signals, 1):
                exporter = signal_generator.export_input if member.source.kind == "generated" else virtual_pa.export
                kind = "csv" if member.source.kind == "generated" else "output"
                base = f"{index:02d}"
                archive.write(exporter(ws, member.source.source_id, kind), base + ".csv")
                metadata = read_json(exporter(ws, member.source.source_id, "metadata"))
                archive.writestr(base + ".json", json.dumps(metadata, indent=2))
            archive.writestr("manifest.json", dataset.model_dump_json(indent=2))
        return path, temporary
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
