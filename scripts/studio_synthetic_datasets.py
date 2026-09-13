"""Generate the explicitly synthetic Studio research catalog; no network or instruments."""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from opendpd.schemas.dataset_catalog import DatasetPublicationDraft, SyntheticSuiteRequest
from opendpd.services.dataset_catalog import write_package
from opendpd.services.synthetic_datasets import generate_suite
from opendpd.services.workspace import Workspace, sha256_file, write_json_atomic


def generate(workspace, catalog):
    ws = Workspace.open_or_create(workspace)
    suite = generate_suite(ws, SyntheticSuiteRequest())
    if catalog:
        catalog.mkdir(parents=True, exist_ok=True)
        for dataset in suite.datasets:
            draft = DatasetPublicationDraft(dataset_id=dataset.dataset_id,
                description="SYNTHETIC Studio research fixture: three normalized drive settings and independent random realizations. No RF measurement, dBm, power efficiency, hardware cost or standard EVM evidence.",
                license="CC0-1.0", attribution="OpenDPD synthetic-memory-pa-v1 generator")
            destination = catalog / dataset.dataset_id
            with tempfile.TemporaryDirectory() as temporary:
                staging = Path(temporary) / "package"
                _, files, _ = write_package(ws, draft, staging)
                if destination.exists():
                    if any(not (destination / f.path).is_file() or sha256_file(destination / f.path) != f.sha256 for f in files):
                        raise RuntimeError(f"Refusing to replace a changed fixture: {destination}")
                else:
                    import shutil
                    shutil.copytree(staging, destination)
        card_path = catalog / "condition-set.json"
        # created_at is not part of condition identity; preserve the first copy.
        if not card_path.exists():
            write_json_atomic(card_path, suite.condition_set)
    print(f"Created/verified {len(suite.datasets)} synthetic datasets; no data were published.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--catalog", type=Path)
    args = parser.parse_args()
    if args.workspace:
        generate(args.workspace, args.catalog)
    else:
        with tempfile.TemporaryDirectory(prefix="opendpd-synthetic-workspace-") as root:
            generate(Path(root), args.catalog)
