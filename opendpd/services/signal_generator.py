"""Private generated waveforms, exports, and explicitly simulated PA datasets."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import re
import tempfile
import threading
import zipfile

import numpy as np
import scipy

from opendpd.core.splits import contiguous_boundaries
from opendpd.core.waveforms.generator import synthesize
from opendpd.core.waveforms.generator_presets import coverage
from opendpd.schemas.dataset import DatasetOrigin, SignalSpec
from opendpd.schemas.importing import CsvOptions
from opendpd.schemas.signal_generator import (DatasetSampleCounts, GeneratedSignal, GeneratorConfig,
    GeneratorDatasetRequest, GeneratorDatasetResponse)
from opendpd.services.datasets import import_dataset, load_version_arrays
from opendpd.services.workspace import Workspace, WorkspaceError, read_json, sha256_file, write_json_atomic

_LOCK = threading.RLock()


def input_summary(result):
    from opendpd.schemas.virtual_pa import PAInputDataset
    base = f"/api/v1/signal-generator/signals/{result.signal_id}"
    return PAInputDataset(signal_id=result.signal_id, name=result.config.preset_id,
        n_samples=result.analysis.sample_count, sample_rate_hz=result.config.sample_rate_hz,
        bandwidth_hz=result.config.bandwidth_hz, iq_sha256=result.iq_sha256,
        csv_url=base + "/input.csv", metadata_url=base + "/metadata.json")


def list_inputs(ws):
    root = ws.root / "signals"
    paths = sorted(root.glob("sg-*/manifest.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [input_summary(read_signal(ws, p.parent.name)) for p in paths[:100]
            if not p.is_symlink() and not p.parent.is_symlink() and not (p.parent / ".removed").exists()]


def export_input(ws, identifier, kind):
    """Two independent downloads, explicitly identifying an input-only dataset."""
    if kind not in ("csv", "metadata"):
        raise WorkspaceError("Unknown PA input export.")
    with _LOCK:
        result = read_signal(ws, identifier)
        target = directory(ws, identifier)
        csv_path = target / "pa-input.csv"
        raw = np.load(target / "iq.npy", allow_pickle=False)
        temporary = target / "pa-input.csv.tmp"
        np.savetxt(temporary, raw, delimiter=",", fmt="%.9g", header="I,Q", comments="")
        temporary.replace(csv_path)
        metadata = {
            "schema": "pa-input-dataset-v1", "dataset_kind": "pa_input", "signal_role": "pa_input",
            "signal_id": identifier, "has_pa_output": False, "origin": "synthetic",
            "n_samples": result.analysis.sample_count, "sample_rate_hz": result.config.sample_rate_hz,
            "bandwidth_hz": result.config.bandwidth_hz, "carrier_frequency_hz": result.config.carrier_frequency_hz,
            "amplitude_units": "normalized", "columns": ["I", "Q"], "sample_format": "float32 complex I/Q pairs",
            "iq_npy_sha256": result.iq_sha256, "csv_sha256": sha256_file(csv_path),
            "generator_config": result.config.model_dump(mode="json"),
            "provenance": read_json(target / "provenance.json"),
            "pairing": "A training dataset requires both PA input x and PA output y with matching sample rate, count and alignment. Obtain y from Virtual PA simulation or a real PA capture.",
        }
        metadata_path = target / "pa-input-metadata.json"
        write_json_atomic(metadata_path, metadata)
        return csv_path if kind == "csv" else metadata_path


def directory(ws, identifier):
    if not re.fullmatch(r"sg-[a-f0-9]{64}", identifier):
        raise WorkspaceError("Unknown generated signal.")
    target = ws.root / "signals" / identifier
    if target.is_symlink():
        raise WorkspaceError("Generated signal directory cannot be a symbolic link.")
    return target


def read_signal(ws, identifier) -> GeneratedSignal:
    target = directory(ws, identifier)
    if not (target / "manifest.json").is_file() or (target / "manifest.json").is_symlink():
        raise WorkspaceError("Generated signal not found. Generate the waveform first.")
    result = GeneratedSignal.model_validate(read_json(target / "manifest.json"))
    source = target / "iq.npy"
    if source.is_symlink() or not source.is_file() or sha256_file(source) != result.iq_sha256:
        raise WorkspaceError("Generated IQ bytes have changed. Regenerate and review the waveform.")
    return result


def generate(ws: Workspace, config: GeneratorConfig) -> GeneratedSignal:
    from opendpd.core.waveforms import generator
    source_hash = hashlib.sha256(b"".join((Path(generator.__file__).parent / name).read_bytes()
        for name in ("generator.py", "modulation.py", "generator_presets.py"))).hexdigest()
    serialized = json.dumps(config.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    identifier = "sg-" + hashlib.sha256((source_hash + np.__version__ + scipy.__version__ + serialized).encode()).hexdigest()
    with _LOCK:
        target = directory(ws, identifier)
        if (target / "manifest.json").is_file():
            (target / ".removed").unlink(missing_ok=True)
            return read_signal(ws, identifier)
        try:
            iq, analysis = synthesize(config)
        except ValueError as exc:
            raise WorkspaceError(str(exc)) from exc
        target.mkdir(parents=True, exist_ok=True)
        np.save(target / "iq.npy", np.column_stack((iq.real, iq.imag)), allow_pickle=False)
        result = GeneratedSignal(signal_id=identifier, config=config, iq_sha256=sha256_file(target / "iq.npy"),
            coverage=coverage(config), analysis=analysis,
            download_url=f"/api/v1/signal-generator/signals/{identifier}/download")
        write_json_atomic(target / "manifest.json", result)
        write_json_atomic(target / "provenance.json", {"generator_source_sha256": source_hash,
            "numpy_version": np.__version__, "scipy_version": scipy.__version__, "physical_measurement": False, "sample_format": "float32 [I,Q]",
            "coverage": result.coverage, "full_protocol_frames": False, "config": config.model_dump(mode="json")})
        return result


def export_signal(ws, identifier):
    with _LOCK:
        result = read_signal(ws, identifier)
        target = directory(ws, identifier)
        output = target / "waveform.zip"
        if output.exists():
            return output
        iq = np.load(target / "iq.npy", allow_pickle=False)
        csv = io.StringIO()
        np.savetxt(csv, iq, delimiter=",", fmt="%.9g", header="I,Q", comments="")
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(target / "iq.npy", "iq.npy")
            archive.write(target / "manifest.json", "manifest.json")
            archive.write(target / "provenance.json", "provenance.json")
            archive.writestr("iq.csv", csv.getvalue())
            archive.writestr("config.json", result.config.model_dump_json(indent=2))
            archive.writestr("README.txt", "SYNTHETIC complex-baseband stimulus\n\n" + "\n".join(result.analysis.notes)
                + f"\n\nSamples: {result.analysis.sample_count}\nSample rate: {result.config.sample_rate_hz} Hz\n"
                + f"RF carrier metadata: {result.config.carrier_frequency_hz} Hz\nIQ NPY SHA256: {result.iq_sha256}\n"
                + "\nLoad config.json in Studio Signal Generator to regenerate. The same generator implementation and NumPy version are required for byte reproduction.\n"
                + "The CSV contains a stimulus only; it has no measured PA response. Create a synthetic PA dataset in Studio or capture a real response for training.\n")
        return output


def create_dataset(ws, identifier, request: GeneratorDatasetRequest) -> GeneratorDatasetResponse:
    with _LOCK:
        result = read_signal(ws, identifier)
        if result.analysis.sample_count < 8192:
            raise WorkspaceError("Generate at least 8,192 samples before creating a PA training dataset.")
        try:
            ratios = {"train": request.train_ratio, "val": request.val_ratio, "test": 1-request.train_ratio-request.val_ratio}
            boundaries = contiguous_boundaries(result.analysis.sample_count, ratios, request.guard_samples)
            if min(b-a for a, b in boundaries.values()) < 256:
                raise ValueError("Each split needs at least 256 samples for the default training frame sizes.")
        except ValueError as exc:
            raise WorkspaceError(str(exc)) from exc
        simulation = {"generator": "signal-generator-pa-v1", "signal_id": identifier,
            "waveform_iq_sha256": result.iq_sha256, "waveform": result.config.model_dump(mode="json"),
            "pa": request.model_dump(mode="json", exclude={"dataset_id", "display_name"}),
            "physical_measurement": False, "full_protocol_frames": False,
            "equation": "z=gain*x/(1+compression*abs(x)^2)*exp(j*am_pm*abs(x)^2); y=z+memory*(1-0.5j)*z[n-1]+noise",
            "noise_reference": "independent PCG64 / SeedSequence([seed, 7391]); dB relative to unnoised output RMS"}
        if (ws.dataset_dir(request.dataset_id) / "manifest.json").exists():
            existing = ws.get_dataset(request.dataset_id)
            if existing.simulation != simulation:
                raise WorkspaceError("This dataset name already exists with different settings. Choose another name.")
            for ref in existing.files:
                file = ws.dataset_dir(existing.dataset_id) / ref.path
                if file.is_symlink() or not file.is_file() or sha256_file(file) != ref.sha256:
                    raise WorkspaceError("Existing generated dataset bytes changed. Choose a new dataset name.")
            return GeneratorDatasetResponse(dataset=existing, test_samples=boundaries["test"][1]-boundaries["test"][0])
        raw = np.load(directory(ws, identifier) / "iq.npy", allow_pickle=False)
        x = raw[:, 0].astype(complex) + 1j*raw[:, 1]
        z = request.pa_gain*x / (1+request.compression*np.abs(x)**2) * np.exp(1j*request.am_pm*np.abs(x)**2)
        y = z.copy()
        y[1:] += request.memory*(1-.5j)*z[:-1]
        rng = np.random.default_rng(np.random.SeedSequence([result.config.seed, 7391]))
        noise_rms = np.sqrt(np.mean(np.abs(y)**2))*10**(request.noise_db/20)
        y += noise_rms/np.sqrt(2)*(rng.standard_normal(len(y))+1j*rng.standard_normal(len(y)))
        config = result.config
        with tempfile.TemporaryDirectory(prefix="opendpd-generator-") as tmp:
            source = Path(tmp) / "data.csv"
            np.savetxt(source, np.column_stack((x.real, x.imag, y.real, y.imag)).astype(np.float32), delimiter=",",
                       fmt="%.9g", header="I_in,Q_in,I_out,Q_out", comments="")
            manifest = import_dataset(ws, source, dataset_id=request.dataset_id, display_name=request.display_name,
                origin=DatasetOrigin.synthetic, guard_samples=request.guard_samples, ratios=ratios, csv_options=CsvOptions(),
                signal=SignalSpec(sample_rate_hz=config.sample_rate_hz, bandwidth_hz=config.bandwidth_hz,
                    # OFDMA users are not separate adjacent RF carriers in the metric profile.
                    sub_channel_bandwidth_hz=config.bandwidth_hz, n_sub_ch=1,
                    nperseg=min(4096, max(512, config.fft_size*config.oversampling)),
                    modulation=f"{config.waveform.upper()} synthetic stimulus", amplitude_units="normalized"),
                notes="SYNTHETIC stimulus and illustrative nonlinear memory PA. No physical capture, calibrated RF power or standard conformance claim. " + " ".join(result.analysis.notes))
        manifest = manifest.model_copy(update={"simulation": simulation,
            "source": manifest.source.model_copy(update={"original_path": None})})
        ws.save_dataset(manifest)
        return GeneratorDatasetResponse(dataset=manifest, test_samples=boundaries["test"][1]-boundaries["test"][0])


def sample_counts(ws, dataset_id, version):
    manifest = ws.get_dataset(dataset_id)
    item = manifest.version(version)
    if version != "raw-v1" and item is None:
        raise WorkspaceError("This preprocessing version does not exist.")
    split, total = (item.split, item.n_samples) if item else (manifest.split, manifest.n_samples)
    bounds = split.boundaries
    if bounds is None or total is None:
        x, _, actual = load_version_arrays(ws, dataset_id, version)
        total = len(x)
        bounds = actual.boundaries or contiguous_boundaries(total, actual.ratios, actual.guard_samples)
    return DatasetSampleCounts(dataset_id=dataset_id, version=version, total_samples=total,
        counts={name: end-start for name, (start, end) in bounds.items()},
        sample_rate_hz=manifest.signal.sample_rate_hz, guard_samples=split.guard_samples)


def archive_input(ws, identifier, *, restore=False):
    """Hide an input from the picker; retain bytes so simulations and Undo remain valid."""
    with _LOCK:
        result = read_signal(ws, identifier)
        marker = directory(ws, identifier) / '.removed'
        if marker.is_symlink():
            raise WorkspaceError('Invalid PA input removal marker.')
        if restore:
            marker.unlink(missing_ok=True)
            return input_summary(result)
        marker.touch()
        return {'signal_id': identifier, 'removed': True}
