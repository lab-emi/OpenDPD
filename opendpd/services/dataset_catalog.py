"""One data-only format for bundled synthetic and community-reviewed datasets.

No arbitrary uploaded source files, paths, notes, Python modules or credentials
are copied to the catalog. CSV output contains finite, paired float32 samples.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from opendpd.schemas.benchmark import canonical_sha256
from opendpd.schemas.common import FileRef
from opendpd.schemas.dataset_catalog import CatalogDataset, DatasetPublicationDraft
from opendpd.schemas.importing import BuiltinDatasetInfo, CsvOptions
from opendpd.services.datasets import import_dataset, load_version_arrays
from opendpd.services.workspace import PACKAGE_ROOT, Workspace, WorkspaceError, read_json, sha256_file, write_json_atomic

CATALOG_ROOT = PACKAGE_ROOT / "dataset"
MAX_PACKAGE_BYTES = 64 * 1024 * 1024


def checked_file(root: Path, ref: FileRef):
    path = root / ref.path
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise WorkspaceError("A dataset source file is missing or is not a regular file inside its dataset.")
    if path.stat().st_size != ref.size_bytes or sha256_file(path) != ref.sha256:
        raise WorkspaceError("Dataset source bytes changed; validate the source again before publishing.")
    return path


def write_package(ws: Workspace, draft: DatasetPublicationDraft, directory: Path):
    manifest = ws.get_dataset(draft.dataset_id)
    if not manifest.raw_sha256 or not manifest.files:
        raise WorkspaceError("The dataset needs recorded raw-source hashes before publication.")
    if not manifest.n_samples or manifest.n_samples > 1_000_000:
        raise WorkspaceError("Public dataset contributions support 1 to 1,000,000 paired samples.")
    for ref in manifest.files:
        checked_file(ws.dataset_dir(draft.dataset_id), ref)
    version = manifest.version("raw-v1")
    if version:
        # These are the arrays used below. spec.json is descriptive metadata and
        # may be updated by the existing manifest editor independently.
        for ref in version.files:
            if ref.path in ("input_iq.npy", "output_iq.npy"):
                checked_file(ws.dataset_version_dir(draft.dataset_id), ref)
    x, y, split = load_version_arrays(ws, draft.dataset_id, "raw-v1")
    if (x.shape != y.shape or x.ndim != 2 or x.shape[1] != 2 or len(x) != manifest.n_samples
            or not np.isfinite(x).all() or not np.isfinite(y).all()):
        raise WorkspaceError("Publication requires finite, paired raw IQ arrays with the recorded sample count.")
    directory.mkdir(parents=True, exist_ok=False)
    try:
        data = directory / "data.csv"
        with data.open("w", encoding="utf-8", newline="") as stream:
            stream.write("I_in,Q_in,I_out,Q_out\n")
            for start in range(0, len(x), 8192):
                np.savetxt(stream, np.column_stack((x[start:start+8192], y[start:start+8192])), fmt="%.9g", delimiter=",")
                if stream.tell() > MAX_PACKAGE_BYTES:
                    raise WorkspaceError("The public dataset package exceeds 64 MiB. Create a smaller declared dataset.")
        catalog = CatalogDataset(dataset_id=manifest.dataset_id, display_name=manifest.display_name,
            origin=manifest.origin, description=draft.description, license=draft.license, attribution=draft.attribution,
            signal=manifest.signal, split=split, n_samples=len(x),
            data=FileRef(path="data.csv", sha256=sha256_file(data), size_bytes=data.stat().st_size),
            source_raw_sha256=manifest.raw_sha256, simulation=manifest.simulation)
        write_json_atomic(directory / "dataset.json", catalog)
        # No interpreted user Markdown: JSON holds the chosen attribution and
        # description. This fixed README cannot ping people or close issues.
        (directory / "README.md").write_text(
            "# OpenDPD dataset contribution\n\n"
            f"Origin: **{catalog.origin.value}**. Samples: **{catalog.n_samples}**. License: **{catalog.license}**.\n\n"
            "See `dataset.json` for the contributor's description, attribution, signal metadata, split and hashes. "
            "`data.csv` has finite I_in,Q_in,I_out,Q_out values; preprocessing is raw-v1.\n\n"
            "Synthetic data demonstrate software behavior and are not physical RF measurements. "
            "A contributor's measured declaration is not independent certification.\n\n"
            "Inclusion requires human review of data, rights and provenance. No automatic merge is requested. "
            "Questions: emi.lab@outlook.com.\n", encoding="utf-8")
        files = [FileRef(path=p.name, sha256=sha256_file(p), size_bytes=p.stat().st_size) for p in sorted(directory.iterdir())]
        digest = canonical_sha256([f.model_dump(mode="json") for f in files])
        return catalog, files, digest
    except BaseException:
        shutil.rmtree(directory, ignore_errors=True)
        raise


def entries():
    """Only committed data manifests in this installed checkout are discoverable."""
    result = {}
    for path in sorted(CATALOG_ROOT.glob("**/dataset.json")):
        if path.is_symlink() or not path.resolve().is_relative_to(CATALOG_ROOT.resolve()):
            continue
        try:
            catalog = CatalogDataset.model_validate(read_json(path))
            # IDs remain short enough for the existing run schema; the full
            # package/data hashes, not this display key, establish identity.
            name = f"catalog-{catalog.dataset_id[:36]}-{sha256_file(path)[:12]}"
            if name in result:
                raise WorkspaceError("Duplicate catalog identifier; review dataset manifests.")
            result[name] = (path.parent, catalog)
        except (ValueError, OSError):
            continue
    return result


def list_catalog():
    result = []
    for name, (directory, catalog) in entries().items():
        problem = None
        try:
            checked_file(directory, catalog.data)
        except WorkspaceError as err:
            problem = str(err)
        result.append(BuiltinDatasetInfo(name=name, description=catalog.description, dataset_format="single_csv",
            n_samples=catalog.n_samples, signal=catalog.signal, problem=problem, has_demodulator=False,
            origin=catalog.origin, raw_sha256=catalog.data.sha256))
    return result


def register_catalog(ws: Workspace, name: str, dataset_id: str | None = None):
    entry = entries().get(name)
    if entry is None:
        raise WorkspaceError("Unknown catalog dataset. Refresh the catalog and select an entry.")
    directory, catalog = entry
    source = checked_file(directory, catalog.data)
    identifier = dataset_id or name
    if (ws.dataset_dir(identifier) / "manifest.json").exists():
        old = ws.get_dataset(identifier)
        if old.source.name == name and any(f.sha256 == catalog.data.sha256 for f in old.files):
            return old
        raise WorkspaceError("This dataset identifier already contains another dataset.")
    manifest = import_dataset(ws, source, dataset_id=identifier, display_name=catalog.display_name,
        signal=catalog.signal, origin=catalog.origin, ratios=catalog.split.ratios, guard_samples=catalog.split.guard_samples,
        notes=catalog.description, csv_options=CsvOptions(), expected_sha256=catalog.data.sha256)
    # The raw-source digest in normal CSV manifests is a combined hash; keep
    # that convention. Catalog identity is independently bound by the name.
    manifest.source = manifest.source.model_copy(update={"name": name, "original_path": None})
    manifest.simulation = catalog.simulation
    ws.save_dataset(manifest)
    return manifest
