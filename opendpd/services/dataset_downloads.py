"""Bounded dataset downloads with a frozen, standalone Virtual PA replay script."""
from __future__ import annotations

import json
import re
from pathlib import Path
import shutil
import tempfile
import zipfile

import numpy as np

from opendpd.services.datasets import load_version_arrays
from opendpd.services.workspace import WorkspaceError, sha256_file


def _safe_file(root, path):
    if root.is_symlink() or path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise WorkspaceError("Dataset export source is unavailable.")
    for parent in path.parents:
        if parent == root:
            break
        if parent.is_symlink():
            raise WorkspaceError("Dataset export source must not use symbolic links.")
    return path


def _csv(ws, manifest, version, target):
    root = ws.dataset_dir(manifest.dataset_id)
    directory = ws.dataset_version_dir(manifest.dataset_id, version)
    selected = manifest.version(version)
    if not selected and version != "raw-v1":
        raise WorkspaceError("Dataset version not found.")
    refs = selected.files if selected else manifest.files
    for ref in refs:
        path = _safe_file(root, directory / ref.path if selected else root / ref.path)
        if ref.sha256 and sha256_file(path) != ref.sha256:
            raise WorkspaceError("Dataset bytes changed. Restore the source before downloading.")
    x, y, _ = load_version_arrays(ws, manifest.dataset_id, version)
    if len(x) != len(y) or len(x) > 4_000_000:
        raise WorkspaceError("Dataset download supports at most 4,000,000 aligned samples.")
    with target.open("w") as stream:
        stream.write("I_in,Q_in,I_out,Q_out\n")
        for start in range(0, len(x), 65536):
            np.savetxt(stream, np.column_stack((x[start:start+65536], y[start:start+65536])),
                       delimiter=",", fmt="%.9g")


def _replay(ws, manifests, filenames):
    from opendpd.services.virtual_pa import directory, read_simulation
    configs, kernel = {}, None
    for manifest, filename in zip(manifests, filenames):
        provenance = manifest.simulation or {}
        result = read_simulation(ws, provenance.get("simulation_id", ""))
        root = directory(ws, result.simulation_id)
        path = _safe_file(root, root / "kernel.py")
        if not result.kernel_sha256 or sha256_file(path) != result.kernel_sha256:
            raise WorkspaceError("The frozen PA formula is unavailable for this capture.")
        source = path.read_text()
        if kernel is not None and source != kernel:
            raise WorkspaceError("Captures use different formula versions. Download captures separately.")
        kernel = source
        configs[filename] = {"model_id": result.config.model_id, "parameters": result.config.parameters,
            "sample_rate_hz": result.sample_rate_hz,
            "parameter_bounds": {p.key: [p.minimum, p.maximum, p.integer] for p in result.model.parameters}}
    header = '''# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=1.24", "scipy>=1.10"]
# ///
"""Reproduce synthetic PA outputs: uv run simulate_pa.py INPUT.csv --output OUTPUT.csv.
Use --preset NAME.csv when the input filename differs from the exported capture.
Every capture starts from zero memory state. This is an illustrative PA model.
"""
import argparse
import json
from pathlib import Path

'''
    # JSON is loaded as data; user-facing labels never become Python expressions.
    data = "\nCAPTURES = json.loads(" + repr(json.dumps(configs, sort_keys=True)) + ")\n"
    cli = '''

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="CSV with I,Q or I_in,Q_in columns")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preset", choices=sorted(CAPTURES))
    parser.add_argument("--sample-rate", type=float, help="Override this capture's sample rate in Hz")
    parser.add_argument("--parameter", action="append", default=[], metavar="NAME=VALUE")
    args = parser.parse_args()
    name = args.preset or (args.input.name if args.input.name in CAPTURES else None)
    if name is None and len(CAPTURES) == 1:
        name = next(iter(CAPTURES))
    if name is None:
        parser.error("Select --preset for this input filename.")
    if args.input.resolve() == args.output.resolve():
        parser.error("Choose an output path different from the input.")
    config = CAPTURES[name]
    parameters = dict(config["parameters"])
    for override in args.parameter:
        key, separator, raw = override.partition("=")
        try:
            value = float(raw)
            low, high, integer = config["parameter_bounds"][key]
            if not separator or not math.isfinite(value) or not low <= value <= high or (integer and value != int(value)):
                raise ValueError()
        except (KeyError, ValueError):
            parser.error("Invalid or out-of-range PA parameter: " + override)
        parameters[key] = value
    raw = np.genfromtxt(args.input, delimiter=",", names=True, dtype=np.float64, encoding="utf-8-sig", max_rows=1_000_001)
    raw = np.atleast_1d(raw)
    names = raw.dtype.names or ()
    pair = ("I_in", "Q_in") if {"I_in", "Q_in"}.issubset(names) else ("I", "Q")
    if not set(pair).issubset(names):
        parser.error("CSV requires I,Q or I_in,Q_in columns.")
    x = raw[pair[0]].astype(np.float32).astype(np.float64) + 1j*raw[pair[1]].astype(np.float32).astype(np.float64)
    fs = config["sample_rate_hz"] if args.sample_rate is None else args.sample_rate
    try:
        y, _ = simulate_resolved(x, fs, config["model_id"], parameters)
    except ValueError as exc:
        parser.error(str(exc))
    output = np.column_stack((y.real, y.imag)).astype(np.float32)
    np.savetxt(args.output, output, delimiter=",", header="I,Q", comments="", fmt="%.9g")


if __name__ == "__main__":
    main()
'''
    return header + (kernel or "") + data + cli


def export_dataset(ws, dataset_id, version="raw-v1", collection=True):
    """Return a temporary file and its directory; the HTTP response removes both."""
    parent = ws.get_dataset(dataset_id)
    manifests = [parent]
    if collection and len(parent.captures) > 1:
        manifests = [ws.get_dataset(c.dataset_id) for c in parent.captures]
        if (manifests[0].dataset_id != parent.dataset_id
                or any(m.parent_dataset_id != parent.dataset_id for m in manifests[1:])
                or len({m.dataset_id for m in manifests}) != len(manifests)):
            raise WorkspaceError("Invalid dataset collection membership.")
        if version != "raw-v1":
            raise WorkspaceError("A collection ZIP contains original captures. Download a selected processed capture as CSV.")
    if sum(m.n_samples or 0 for m in manifests) > 4_000_000:
        raise WorkspaceError("Dataset download supports at most 4,000,000 total samples.")
    target = Path(tempfile.mkdtemp(prefix="dataset-download-", dir=ws.exports_dir))
    try:
        basename = parent.display_name if re.fullmatch(r"syn_pa_inout_[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", parent.display_name) else dataset_id
        if len(manifests) == 1:
            path = target / f"{basename}.csv"
            _csv(ws, parent, version, path)
            return path, target
        # Preset labels are never used as paths. IDs have already passed the schema.
        filenames = [f"{i+1:02d}-" + re.sub(r"[^A-Za-z0-9_-]", "-", c.preset_id)[:64] + ".csv"
                     for i, c in enumerate(parent.captures)]
        script = _replay(ws, manifests, filenames)
        path = target / f"{basename}.zip"
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
            for manifest, filename in zip(manifests, filenames):
                csv = target / filename
                _csv(ws, manifest, "raw-v1", csv)
                archive.write(csv, filename)
                archive.writestr(filename[:-4] + ".json", json.dumps({
                    "dataset": manifest.model_dump(mode="json"), "csv_sha256": sha256_file(csv),
                    "columns": ["I_in", "Q_in", "I_out", "Q_out"],
                    "pa_script": "simulate_pa.py", "preset_argument": filename}, indent=2))
                csv.unlink()
            archive.writestr("simulate_pa.py", script)
            archive.writestr("manifest.json", parent.model_dump_json(indent=2))
            archive.writestr("README.txt", "SYNTHETIC PA input/output captures. Each CSV has its own length and sample rate.\n"
                "Select a capture in Studio to analyze or train at its own sample rate; do not concatenate different numerologies.\n"
                "Run: uv run simulate_pa.py <capture.csv> --output output.csv\n"
                "Or install numpy and scipy, then run with python. Use --help for parameter overrides.\n"
                "Input accepts I,Q or I_in,Q_in. The output is float32 I,Q, cold-start state per capture.\n"
                "Formulas and parameters are frozen in the script. Numeric libraries/platforms may differ at floating-point roundoff.\n")
        return path, target
    except Exception:
        shutil.rmtree(target, ignore_errors=True)
        raise
