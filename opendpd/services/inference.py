"""Apply a finished run's model to any waveform, using the evaluator's own model construction.

``apply_waveform`` rebuilds the network through ``trained_model`` (the code path that scored the run and that the
deployment export and ILC already use), so what is applied is what was evaluated. The result states which of two
execution semantics produced it:

* ``offline_segmented`` -- how every run is scored. The waveform is cut into ``nperseg`` segments (the run's frozen
  segment length), each starts from a zero state, the last is zero padded and the padding is trimmed. A model that
  reads future samples sees zeros beyond a segment edge, exactly as in the scored run.
* ``streaming_stateful`` -- one continuous state carried across chunks, for models with a registered streaming
  variant (``gru`` as ``gru_stream``, ``gmp`` as ``gmp_stream``). It is a different signal from the one the stored
  report scored; the streaming evidence (warm-up, look-ahead, chunk consistency) is returned with it.

Nothing is normalised, aligned or rescaled: supply the waveform in the training dataset's units and sample rate.
This runs in a worker or the SDK's isolated process, never inside the HTTP server process, because rebuilding a
network changes the working directory.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from opendpd.core.registry import STREAMING, get_model, streaming_variant_of
from opendpd.safe_paths import contained_path
from opendpd.schemas import ArtifactKind, RunStatus, TaskType
from opendpd.services.workspace import Workspace, WorkspaceError, sha256_file

OFFLINE = "offline_segmented"
EXECUTIONS = (OFFLINE, STREAMING)
ALIASES = {"streaming": STREAMING}
# Models whose output is checked against the evaluator in tests/integration/test_apply_parity.py. A model joins this
# list together with its parity test, not before.
APPLY_MODELS = ("gru", "tres_gru", "gmp", "mp_ls", "gmp_ls")
MAX_SAMPLES = 1 << 25            # 33.5 M complex samples, about 270 MB as float32 I/Q
HASH_LAYOUT = "row-major Nx2 little-endian float32 I,Q"
EVIDENCE = "model inference; no hardware measurement or new metric evaluation"


class InferenceError(WorkspaceError):
    """A refusal with a stable machine-readable ``code``."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def normalise_execution(execution: str) -> str:
    execution = ALIASES.get(execution, execution)
    if execution not in EXECUTIONS:
        raise ValueError(f"execution must be one of {', '.join(EXECUTIONS)} (or 'streaming'), got {execution!r}")
    return execution


def _iq(x: Any, max_samples: int) -> np.ndarray:
    """Finite float32 ``(N, 2)`` I/Q; floating-point input of another precision is converted, never rescaled."""
    arr = np.asarray(x)
    if arr.dtype.kind != "f" or arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] == 0:
        raise ValueError(f"expected a nonempty floating-point I/Q array of shape (N, 2), got {arr.dtype} {arr.shape}")
    if arr.shape[0] > max_samples:
        raise ValueError(f"{arr.shape[0]} samples exceeds the limit of {max_samples}; apply the waveform in parts")
    if not np.isfinite(arr).all() or np.max(np.abs(arr)) > np.finfo(np.float32).max:
        raise ValueError("samples must be finite and within the float32 range")
    return np.ascontiguousarray(arr, dtype=np.float32)


def _checkpoint(ws: Workspace, run_id: str):
    from opendpd.services.experiments import load_artifacts

    manifest = load_artifacts(ws, run_id)
    checkpoints = manifest.by_kind(ArtifactKind.checkpoint) if manifest else []
    if len(checkpoints) != 1:
        raise InferenceError("checkpoint_missing", "Run needs exactly one registered model checkpoint")
    artifact = checkpoints[0]
    try:
        path = contained_path(ws.run_dir(run_id), artifact.file.path)
    except ValueError:
        raise InferenceError("checkpoint_changed", "Checkpoint path is not inside its run") from None
    if not path.is_file() or sha256_file(path) != artifact.file.sha256:
        raise InferenceError("checkpoint_changed", "Checkpoint is missing or its recorded hash does not match")
    return artifact


def _offline(core, x: np.ndarray, nperseg: int, batch_segments: int) -> np.ndarray:
    """Segment, forward and trim like ``IQSegmentDataset`` evaluation; the temporary copy is bounded by the batch."""
    import torch

    from opendpd.services.streaming import segments

    padded = segments(x, nperseg)
    out = np.empty_like(padded)
    step = max(1, int(batch_segments))
    with torch.inference_mode():
        for start in range(0, len(padded), step):
            out[start:start + step] = core(torch.from_numpy(padded[start:start + step])).numpy()
    return out.reshape(-1, 2)[:len(x)]


def _limitations(key: str, execution: str, nperseg: int, streaming: Optional[List[str]]) -> List[str]:
    model = get_model(key)
    notes = [EVIDENCE]
    if execution == OFFLINE:
        if model.lookahead_samples:
            notes.append(f"{key} reads {model.lookahead_samples} future samples: within that distance of a segment end "
                         f"(every {nperseg} samples) it sees zero padding instead of the waveform, as in the scored run")
        elif model.lookahead_samples is None:
            notes.append(f"temporal reach of {key} is not characterised as a constant: {model.lookahead_note}")
    else:
        notes.extend(streaming or [])
    return notes


def apply_waveform(ws: Workspace, run_id: str, x: Any, *, execution: str = OFFLINE,
                   chunk_samples: Optional[int] = None, max_samples: int = MAX_SAMPLES) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Apply a succeeded ``train_pa`` / ``train_dpd`` run to ``x``; returns ``(float32 (N, 2), metadata)``."""
    from opendpd.services.evaluation import trained_model
    from opendpd.services.experiments import load_resolved, load_result, load_run

    execution = normalise_execution(execution)
    if chunk_samples is not None and (isinstance(chunk_samples, bool) or int(chunk_samples) != chunk_samples
                                      or chunk_samples < 1):
        raise ValueError("chunk_samples must be a positive integer")
    x = _iq(x, max_samples)
    run = load_run(ws, run_id)
    resolved = load_resolved(ws, run_id)
    if run.status != RunStatus.succeeded or resolved.task not in (TaskType.train_pa, TaskType.train_dpd):
        raise InferenceError("model_unavailable", "Apply needs a succeeded train_pa or train_dpd run")
    key = resolved.model.key
    if key not in APPLY_MODELS:
        raise InferenceError("unsupported_model", f"apply supports {', '.join(APPLY_MODELS)}; '{key}' has no parity "
                             "test against the evaluator yet")
    if resolved.quantization and resolved.quantization.enabled:
        raise InferenceError("unsupported_model", "Quantisation-aware runs are not supported by apply")
    variant = None
    if execution == STREAMING:
        variant = streaming_variant_of(key)
        if variant is None:
            streams = ", ".join(sorted(m.weights_from for m in streaming_variants()))
            raise InferenceError("no_streaming_variant", f"'{key}' has no registered streaming variant (models with one: "
                                 f"{streams}); use execution='offline_segmented', which is how the run was scored")
    artifact = _checkpoint(ws, run_id)
    result = load_result(ws, run_id)
    if result is None or result.nperseg is None or result.evaluated_signal is None:
        raise InferenceError("metadata_missing", "Run needs a stored result with frozen signal and segment metadata")
    # Segment length and sample rate are the run's frozen values, not the dataset manifest's current ones: a later
    # metadata edit must not change what a finished run produces. The network itself does not depend on either.
    nperseg = int(result.nperseg)
    sample_rate = result.evaluated_signal.sample_rate_hz
    core = trained_model(ws, run_id).evaluated
    meta: Dict[str, Any] = {}
    streaming_notes: Optional[List[str]] = None
    if execution == OFFLINE:
        y = _offline(core, x, nperseg, resolved.training.batch_size_eval)
        meta.update(segment_samples=nperseg,
                    state_reset="zero at the start of every segment and of every apply call",
                    tail="the last segment is zero padded; the padding is trimmed from the output")
    else:
        from opendpd.services.streaming import stream_outputs, streaming_limitations

        y, evidence = stream_outputs(core.cpu(), variant.key, x, chunk_samples=chunk_samples, sample_rate_hz=sample_rate)
        streaming_notes = streaming_limitations(variant.key, evidence)
        meta.update(streaming_variant=variant.key, streaming=evidence.model_dump(mode="json"),
                    state_reset="only at the start of this apply call; chunk boundaries never reset it",
                    tail="a causal variant has no tail; look-ahead outputs are flushed with zeros")
    y = np.ascontiguousarray(y, dtype=np.float32)
    if y.shape != x.shape or not np.isfinite(y).all():
        raise InferenceError("nonfinite_output", "Model produced nonfinite or misshapen samples")
    meta.update(
        apply_version=2, sdk_iq_version=1, run_id=run_id, model=key, task=run.task.value, execution=execution,
        checkpoint_sha256=artifact.file.sha256, input_sha256=hashlib.sha256(x.astype("<f4").tobytes()).hexdigest(),
        output_sha256=hashlib.sha256(y.astype("<f4").tobytes()).hexdigest(), hash_layout=HASH_LAYOUT,
        n_samples=len(x), dtype="float32", device="cpu", sample_rate_hz=sample_rate,
        lookahead_samples=get_model(key).lookahead_samples,
        preprocessing_version=resolved.dataset.preprocessing_version,
        training_scaling=result.scaling.model_dump(mode="json") if result.scaling else None,
        input_scaling="as supplied; use the training dataset's preprocessing and sample rate",
        output_role="predistorted_pa_input" if run.task == TaskType.train_dpd else "modeled_pa_output",
        limitations=_limitations(key, execution, nperseg, streaming_notes), evidence=EVIDENCE)
    return y, meta


def streaming_variants():
    from opendpd.core.registry import list_models

    return [m for m in list_models() if m.execution_semantics == STREAMING]
