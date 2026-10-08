"""Isolated CPU GRU inference using recorded weights and segment semantics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .client import SDKError
from .iq import as_iq


def infer(workspace, run_id, x):
    import torch
    from models import CoreModel
    from opendpd.schemas import ArtifactKind, RunStatus, TaskType
    from opendpd.services.experiments import load_artifacts, load_resolved, load_result, load_run
    from opendpd.services.legacy_adapter import load_checkpoint
    from opendpd.services.workspace import Workspace, sha256_file

    ws = Workspace.open(Path(workspace))
    run = load_run(ws, run_id)
    config = load_resolved(ws, run_id)
    if run.status != RunStatus.succeeded or run.task not in (TaskType.train_pa, TaskType.train_dpd):
        raise SDKError("model_unavailable", "Apply needs a succeeded train_pa or train_dpd run")
    if config.model.key != "gru" or (config.quantization and config.quantization.enabled):
        raise SDKError("unsupported_model", "This preview applies unquantized GRU models only")
    manifest = load_artifacts(ws, run_id)
    checkpoints = manifest.by_kind(ArtifactKind.checkpoint) if manifest else []
    if len(checkpoints) != 1:
        raise SDKError("checkpoint_missing", "Run needs one registered model checkpoint")
    artifact = checkpoints[0]
    checkpoint = (ws.run_dir(run_id) / artifact.file.path).resolve()
    if (not checkpoint.is_relative_to(ws.run_dir(run_id).resolve()) or not checkpoint.is_file()
            or sha256_file(checkpoint) != artifact.file.sha256):
        raise SDKError("checkpoint_changed", "Checkpoint is missing or its recorded hash does not match")
    result = load_result(ws, run_id)
    if result is None or result.nperseg is None or result.evaluated_signal is None:
        raise SDKError("metadata_missing", "Run needs a stored result with frozen signal and segment metadata")
    nperseg = result.nperseg
    x = as_iq(x)
    parameters = config.model.parameters
    model = CoreModel(input_size=2, hidden_size=int(parameters["hidden_size"]),
                      num_layers=int(parameters["num_layers"]), backbone_type="gru")
    model.load_state_dict(load_checkpoint(checkpoint))
    model.eval()
    y = np.empty_like(x)
    # Bound temporary padding/batching independently of waveform length.
    batch_samples = nperseg * config.training.batch_size_eval
    with torch.inference_mode():
        for start in range(0, len(x), batch_samples):
            take = min(batch_samples, len(x) - start)
            padded = np.zeros((-(-take // nperseg) * nperseg, 2), dtype=np.float32)
            padded[:take] = x[start:start + take]
            output = model(torch.from_numpy(padded.reshape(-1, nperseg, 2))).numpy()
            y[start:start + take] = output.reshape(-1, 2)[:take]
    if not np.isfinite(y).all():
        raise SDKError("nonfinite_output", "Model produced nonfinite samples")
    metadata = {
        "sdk_iq_version": 1, "run_id": run_id, "model": "gru", "task": run.task.value,
        "checkpoint_sha256": artifact.file.sha256,
        "input_sha256": hashlib.sha256(x.astype("<f4").tobytes()).hexdigest(),
        "output_sha256": hashlib.sha256(y.astype("<f4").tobytes()).hexdigest(),
        "hash_layout": "row-major Nx2 little-endian float32 I,Q",
        "n_samples": len(x), "dtype": "float32", "device": "cpu",
        "sample_rate_hz": result.evaluated_signal.sample_rate_hz,
        "execution": "offline_segmented", "segment_samples": nperseg,
        "state_reset": "zero at each segment and at the start of each apply call",
        "tail": "zero padded; output trimmed to input length",
        "preprocessing_version": config.dataset.preprocessing_version,
        "training_scaling": result.scaling.model_dump(mode="json") if result.scaling else None,
        "input_scaling": "as supplied; use the training dataset's preprocessing and sample rate",
        "output_role": "predistorted_pa_input" if run.task == TaskType.train_dpd else "modeled_pa_output",
        "evidence": "model inference; no hardware measurement or new metric evaluation",
    }
    return y, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    try:
        import torch

        torch.set_num_threads(1)
        x = np.load(args.directory / "input.npy", allow_pickle=False)
        y, metadata = infer(args.workspace, args.run_id, x)
        np.save(args.directory / "output.npy", y, allow_pickle=False)
        (args.directory / "metadata.json").write_text(json.dumps(metadata, allow_nan=False), encoding="utf-8")
        return 0
    except Exception as err:
        (args.directory / "error.json").write_text(json.dumps({
            "code": getattr(err, "code", "inference_failed"), "message": str(err)}), encoding="utf-8")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
