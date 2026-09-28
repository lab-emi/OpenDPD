"""Admit an entire validated CSV capture as a reusable, immutable PA input."""
from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

from opendpd.core.waveforms.analyzer import inspect_signal
from opendpd.schemas.signal_analyzer import AnalyzerConfig, AnalyzerSource, AnalyzerSourceInfo
from opendpd.schemas.signal_generator import GeneratedSignal, GeneratorAnalysis, ImportedSignalConfig
from opendpd.services import signal_analyzer, signal_datasets, signal_generator
from opendpd.services.workspace import WorkspaceError, sha256_file, write_json_atomic, workspace_lock


def _analysis(x, config):
    """Reuse signal inspection without inventing modulation or reference symbols."""
    data = inspect_signal(x, AnalyzerConfig(sample_rate_hz=config.sample_rate_hz,
        bandwidth_hz=config.bandwidth_hz, n_samples=len(x)))
    metrics = {m["key"]: m["value"] for m in data["measurements"]}
    if metrics["power"] is None:
        raise WorkspaceError("The selected signal has zero power. Upload a nonzero PA input.")
    return GeneratorAnalysis(sample_count=len(x), duration_ms=len(x) / config.sample_rate_hz * 1000,
        sample_rate_hz=config.sample_rate_hz, subcarrier_spacing_hz=None, useful_symbol_us=None,
        cp_lengths_samples=[], complete_symbols=0, trailing_samples=0, active_carriers=0,
        data_carriers=0, pilot_carriers=0, rms=metrics["rms"], peak=metrics["peak"],
        papr_db=metrics["papr"], mean_power_dbfs=metrics["power"], occupied_bandwidth_99_hz=metrics["obw"],
        dc_magnitude=metrics["dc"], evm_percent=None, evm_symbols=0,
        time_us=[t * 1e6 for t in data["time_s"]], time_i=data["time_i"], time_q=data["time_q"],
        time_envelope=data["envelope"], frequency_mhz=[f / 1e6 for f in data["frequency_hz"]],
        psd_dbfs_hz=data["psd_dbfs_hz"], constellation_i=data["scatter_i"], constellation_q=data["scatter_q"],
        reference_i=[], reference_q=[], ccdf_db=data["ccdf_db"], ccdf_probability=data["ccdf_probability"],
        allocation=[], notes=["UPLOADED PA input: every selected-column sample is retained in order as float32 I/Q, without normalization, filtering or resampling.",
            "Sampling rate and bandwidth are supplied by the user. No PA output or physical measurement is inferred.", *data["notes"]])


def import_signal(ws, request):
    # Share the generator lock so a waveform is never read before its manifest.
    with workspace_lock(ws):
        source = AnalyzerSource(kind="upload", source_id=request.upload_id)
        values, info = signal_analyzer._load(ws, source)
        selection = AnalyzerConfig(sample_rate_hz=request.sample_rate_hz, bandwidth_hz=request.bandwidth_hz,
            n_samples=info.sample_count, sample_format=request.sample_format,
            i_column=request.i_column, q_column=request.q_column)
        # Use the same float32 input as training exports and the standalone PA replay.
        selected = signal_analyzer._select(values, info, selection)
        raw = np.column_stack((selected.real, selected.imag)).astype(np.float32)
        x = raw[:, 0].astype(complex) + 1j * raw[:, 1]
        config = ImportedSignalConfig(preset_id=info.label[:64], n_samples=len(x),
            sample_rate_hz=request.sample_rate_hz, bandwidth_hz=request.bandwidth_hz,
            carrier_frequency_hz=request.carrier_frequency_hz)
        provenance = {"importer": "csv-pa-input-v1", "csv_sha256": request.upload_id[3:],
            "source_name": info.label, "columns": info.columns,
            "selection": {"sample_format": request.sample_format, "i_column": request.i_column,
                "q_column": request.q_column if request.sample_format == "iq" else None},
            "config": config.model_dump(mode="json"), "origin": "uploaded",
            "sample_format": "float32 [I,Q]", "normalization": "none", "resampling": "none"}
        identity = json.dumps(provenance, sort_keys=True, separators=(",", ":"))
        identifier = "sg-" + hashlib.sha256(identity.encode()).hexdigest()
        target = signal_generator.directory(ws, identifier)
        if (target / "manifest.json").exists():
            result = signal_generator.read_signal(ws, identifier)
            (target / ".removed").unlink(missing_ok=True)
        else:
            analysis = _analysis(x, config)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = Path(tempfile.mkdtemp(prefix=".import-", dir=target.parent))
            try:
                np.save(temporary / "iq.npy", raw, allow_pickle=False)
                result = GeneratedSignal(signal_id=identifier, config=config, origin="uploaded", coverage="custom",
                    iq_sha256=sha256_file(temporary / "iq.npy"), analysis=analysis,
                    download_url=f"/api/v1/signal-generator/signals/{identifier}/download")
                write_json_atomic(temporary / "provenance.json", provenance)
                write_json_atomic(temporary / "manifest.json", result)
                temporary.replace(target)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
        member = AnalyzerSourceInfo(source=AnalyzerSource(kind="generated", source_id=identifier),
            label=info.label, sample_count=len(x), sample_rate_hz=config.sample_rate_hz,
            bandwidth_hz=config.bandwidth_hz, origin="uploaded")
        dataset = signal_datasets.save_dataset(ws, request.dataset_name, "pa_input", [member])
        return signal_generator.input_summary(result).model_copy(update={
            "dataset_id": dataset.dataset_id, "dataset_name": dataset.name})
