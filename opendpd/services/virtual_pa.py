"""Explicit PA-input -> virtual PA -> PA-output -> paired dataset pipeline."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import threading

import numpy as np

from opendpd.core import virtual_pa as engine, virtual_pa_kernel
from opendpd.core.splits import contiguous_boundaries
from opendpd.schemas.dataset import DatasetCapture, DatasetOrigin, SignalSpec
from opendpd.schemas.signal_generator import GeneratorDatasetResponse
from opendpd.schemas.virtual_pa import VirtualPARequest, VirtualPASimulation, PairedDatasetRequest, VirtualPADatasetRequest
from opendpd.services import signal_generator as inputs
from opendpd.services.datasets import import_arrays
from opendpd.services.workspace import WorkspaceError, read_json, sha256_file, write_json_atomic

_LOCK = threading.RLock()


def directory(ws, identifier):
    return ws.hashed_store('pa_simulations', 'vpa').directory(identifier)


def read_simulation(ws, identifier):
    path = directory(ws, identifier)
    if not (path / "manifest.json").is_file() or (path / "manifest.json").is_symlink():
        raise WorkspaceError("PA output not found. Simulate the virtual PA first.")
    result = VirtualPASimulation.model_validate(read_json(path / "manifest.json"))
    signal = inputs.read_signal(ws, result.config.input_signal_id)
    if signal.iq_sha256 != result.input_iq_sha256:
        raise WorkspaceError("The PA input has changed; simulate the output again.")
    output = path / "output.npy"
    if output.is_symlink() or not output.is_file() or sha256_file(output) != result.output_iq_sha256:
        raise WorkspaceError("PA output bytes changed; simulate the output again.")
    return result


def preview(ws, request: VirtualPARequest):
    with _LOCK:
        signal = inputs.read_signal(ws, request.input_signal_id)
        try:
            model, parameters = engine.resolve(request.model_id, request.parameters)
        except ValueError as exc:
            raise WorkspaceError(str(exc)) from exc
        config = request.model_copy(update={"parameters": parameters})
        kernel = Path(virtual_pa_kernel.__file__).read_bytes()
        kernel_hash = hashlib.sha256(kernel).hexdigest()
        source_hash = hashlib.sha256(Path(engine.__file__).read_bytes() + kernel).hexdigest()
        identity = {"version": "virtual-pa-v1", "config": config.model_dump(mode="json"),
            "input_iq_sha256": signal.iq_sha256, "sample_rate_hz": signal.config.sample_rate_hz,
            "simulator_source_sha256": source_hash}
        identifier = "vpa-" + hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        target = directory(ws, identifier)
        if (target / "manifest.json").exists():
            return read_simulation(ws, identifier)
        raw = np.load(inputs.directory(ws, request.input_signal_id) / "iq.npy", allow_pickle=False)
        x = raw[:, 0].astype(complex) + 1j*raw[:, 1]
        try:
            y, states = engine.simulate(x, signal.config.sample_rate_hz, config.model_id, parameters)
        except ValueError as exc:
            raise WorkspaceError(str(exc)) from exc
        output = np.column_stack((y.real, y.imag)).astype(np.float32)
        # Analysis refers to the actual exported float32 output.
        y = output[:, 0].astype(complex) + 1j*output[:, 1]
        analysis = engine.analyze(x, y, signal.config.sample_rate_hz, states, parameters)
        target.mkdir(parents=True, exist_ok=True)
        (target / "kernel.py").write_bytes(kernel)
        np.save(target / "output.npy", output, allow_pickle=False)
        base = f"/api/v1/pa-library/simulations/{identifier}"
        result = VirtualPASimulation(simulation_id=identifier, config=config, model=model,
            input_iq_sha256=signal.iq_sha256, output_iq_sha256=sha256_file(target / "output.npy"),
            kernel_sha256=kernel_hash, simulator_source_sha256=source_hash, sample_rate_hz=signal.config.sample_rate_hz,
            analysis=analysis, output_csv_url=base + "/output.csv", paired_csv_url=base + "/paired.csv",
            metadata_url=base + "/metadata.json")
        write_json_atomic(target / "manifest.json", result)
        return result


def export(ws, identifier, kind):
    if kind not in ("output", "paired", "metadata"):
        raise WorkspaceError("Unknown PA simulation export.")
    with _LOCK:
        result = read_simulation(ws, identifier)
        target = directory(ws, identifier)
        if kind == "metadata":
            path = target / "metadata.json"
            write_json_atomic(path, {"schema": "virtual-pa-output-v1", "origin": "synthetic",
                "signal_role": "pa_output", "has_measured_output": False,
                "amplitude_units": "normalized", "n_samples": result.analysis.n_samples,
                "initial_state": "cold start: all envelope states zero; effective temperature at ambient",
                "simulation": result.model_dump(mode="json")})
            return path
        output = np.load(target / "output.npy", allow_pickle=False)
        if kind == "paired":
            raw = np.load(inputs.directory(ws, result.config.input_signal_id) / "iq.npy", allow_pickle=False)
            data, header = np.column_stack((raw, output)), "I_in,Q_in,I_out,Q_out"
        else:
            data, header = output, "I,Q"
        path = target / f"{kind}.csv"
        temporary = target / f"{kind}.csv.tmp"
        np.savetxt(temporary, data, delimiter=",", fmt="%.9g", header=header, comments="")
        temporary.replace(path)
        return path


def create_dataset(ws, identifier, request: PairedDatasetRequest, *, _created=None):
    with _LOCK:
        result = read_simulation(ws, identifier)
        signal = inputs.read_signal(ws, result.config.input_signal_id)
        if result.analysis.n_samples < 8192:
            raise WorkspaceError("Use at least 8,192 PA input samples to create a training dataset.")
        ratios = {"train": request.train_ratio, "val": request.val_ratio, "test": 1-request.train_ratio-request.val_ratio}
        try:
            bounds = contiguous_boundaries(result.analysis.n_samples, ratios, request.guard_samples)
            if min(end-start for start, end in bounds.values()) < 256:
                raise ValueError("Each split needs at least 256 samples after guard intervals.")
        except ValueError as exc:
            raise WorkspaceError(str(exc)) from exc
        provenance = {"generator": "virtual-pa-v1", "simulation_id": result.simulation_id,
            "input_signal_id": result.config.input_signal_id, "input_iq_sha256": result.input_iq_sha256,
            "output_iq_sha256": result.output_iq_sha256, "simulator_source_sha256": result.simulator_source_sha256,
            "pa_model": result.model.model_dump(mode="json"), "parameters": result.config.parameters,
            "waveform": signal.config.model_dump(mode="json"), "physical_measurement": False,
            "initial_state": "cold start", "split": {"ratios": ratios, "guard_samples": request.guard_samples}}
        if (ws.dataset_dir(request.dataset_id) / "manifest.json").exists():
            existing = ws.get_dataset(request.dataset_id)
            if existing.simulation != provenance:
                raise WorkspaceError("This dataset ID already has different data or parameters. Choose a new ID.")
            for ref in existing.files:
                path = ws.dataset_dir(existing.dataset_id) / ref.path
                if path.is_symlink() or not path.is_file() or sha256_file(path) != ref.sha256:
                    raise WorkspaceError("Existing paired dataset bytes have changed.")
            return GeneratorDatasetResponse(dataset=existing, test_samples=bounds["test"][1]-bounds["test"][0])
        x = np.load(inputs.directory(ws, result.config.input_signal_id) / "iq.npy", allow_pickle=False)
        y = np.load(directory(ws, identifier) / "output.npy", allow_pickle=False)
        config = signal.config
        manifest = import_arrays(ws, x, y, dataset_id=request.dataset_id, display_name=request.display_name,
            origin=DatasetOrigin.synthetic, guard_samples=request.guard_samples, ratios=ratios,
            signal=SignalSpec(sample_rate_hz=config.sample_rate_hz, bandwidth_hz=config.bandwidth_hz,
                sub_channel_bandwidth_hz=config.bandwidth_hz, n_sub_ch=1,
                nperseg=min(4096, max(512, config.fft_size*config.oversampling)),
                modulation=f"{config.waveform.upper()} synthetic PA input", amplitude_units="normalized"),
            notes="SYNTHETIC paired PA input x and virtual PA output y. " + result.model.limitations.en + " " + " ".join(result.analysis.notes))
        if _created is not None:
            _created.append(request.dataset_id)
        manifest = manifest.model_copy(update={"simulation": provenance,
            "source": manifest.source.model_copy(update={"original_path": None})})
        ws.save_dataset(manifest)
        return GeneratorDatasetResponse(dataset=manifest, test_samples=bounds["test"][1]-bounds["test"][0])


def simulate_dataset(ws, request: VirtualPADatasetRequest):
    """Validate captures before registration and roll back newly created data on failure."""
    import shutil
    from opendpd.core.waveforms.generator_presets import presets
    from opendpd.core.waveforms.dataset_names import dataset_name
    from opendpd.schemas.signal_analyzer import AnalyzerSource, AnalyzerSourceInfo
    from opendpd.services import signal_datasets

    with _LOCK, signal_datasets.LOCK:
        signals = [inputs.read_signal(ws, identifier) for identifier in request.input_signal_ids]
        if any(s.analysis.sample_count < 8192 for s in signals):
            raise WorkspaceError("Each preset needs at least 8,192 samples to create a training dataset.")
        if sum(s.analysis.sample_count for s in signals) > 4_000_000:
            raise WorkspaceError("Use at most 4,000,000 samples across all presets.")
        if len({s.config.preset_id for s in signals}) != len(signals):
            raise WorkspaceError("Choose distinct presets for a multi-preset dataset.")
        simulations = [preview(ws, VirtualPARequest(input_signal_id=s.signal_id,
            model_id=request.model_id, parameters=request.parameters)) for s in signals]
        requested_name = request.dataset_name or dataset_name([s.config for s in signals], "inout", request.model_id)
        identity = json.dumps([requested_name, [r.simulation_id for r in simulations]], separators=(",", ":"))
        parent_id = "vpa-set-" + hashlib.sha256(identity.encode()).hexdigest()[:32]
        name = (ws.get_dataset(parent_id).display_name if (ws.dataset_dir(parent_id) / "manifest.json").exists()
                else signal_datasets.unique_name(requested_name, {d.display_name for d in ws.list_datasets()}))
        identifiers = [parent_id] + [f"{parent_id}-{i+1}" for i in range(1, len(signals))]
        labels = {p.preset_id: p.label for p in presets()}
        created, results = [], []
        try:
            for identifier, signal, result in zip(identifiers, signals, simulations):
                response = create_dataset(ws, result.simulation_id, PairedDatasetRequest(
                    dataset_id=identifier, display_name=name if identifier == parent_id else
                    dataset_name([signal.config], "inout", request.model_id)), _created=created)
                results.append(response)
            captures = [DatasetCapture(dataset_id=identifier, preset_id=s.config.preset_id,
                label=labels.get(s.config.preset_id, s.config.preset_id), n_samples=s.analysis.sample_count,
                sample_rate_hz=s.config.sample_rate_hz, bandwidth_hz=s.config.bandwidth_hz)
                for identifier, s in zip(identifiers, signals)]
            for response in results[1:]:
                ws.save_dataset(response.dataset.model_copy(update={"parent_dataset_id": parent_id}))
            parent = results[0].dataset.model_copy(update={"captures": captures, "display_name": name})
            ws.save_dataset(parent)
            output_members = [AnalyzerSourceInfo(source=AnalyzerSource(kind="virtual_pa", source_id=r.simulation_id, role="output"),
                label=s.config.preset_id, sample_count=r.analysis.n_samples, sample_rate_hz=s.config.sample_rate_hz,
                bandwidth_hz=s.config.bandwidth_hz, origin="synthetic") for s, r in zip(signals, simulations)]
            signal_datasets.save_dataset(ws, name.replace("syn_pa_inout_", "syn_pa_out_", 1), "pa_output", output_members)
            return results[0].model_copy(update={"dataset": parent})
        except Exception:
            for identifier in created:
                shutil.rmtree(ws.dataset_dir(identifier), ignore_errors=True)
            raise
