"""Streaming variants of trained weights (plan S18): adapters over the legacy modules, the run that scores a
finished PA or DPD run under streaming semantics, and the evidence such a result carries."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from opendpd.core.registry import STREAMING, get_model, streaming_variant_of
from opendpd.core.streaming import (
    CONSISTENCY_TOLERANCE,
    DEFAULT_CHUNK_SAMPLES,
    LOOKAHEAD_NOTE,
    RecurrentStream,
    StreamingModel,
    WindowedStream,
    chunk_consistency,
    measure_warmup,
    run_stream,
)
from opendpd.schemas import (
    DatasetRef,
    DPDReference,
    EvaluationConfig,
    EvidenceType,
    ExecutionConfig,
    ExecutionEvidence,
    ExperimentConfig,
    ModelSpec,
    PAReference,
    RunStatus,
    StreamConsistency,
    TaskType,
)
from opendpd.services.workspace import Workspace, WorkspaceError


class GRUStream(RecurrentStream):
    """backbones/gru.py (nn.GRU + linear head) with the hidden state carried across chunks."""

    def __init__(self, core):
        super().__init__()
        self._rnn, self._fc = core.backbone.rnn, core.backbone.fc_out

    def _run(self, x: np.ndarray, h: Any) -> Tuple[np.ndarray, Any]:
        import torch

        with torch.inference_mode():
            out, h_next = self._rnn(torch.from_numpy(x).unsqueeze(0), h)
            y = self._fc(out)
        return y.squeeze(0).numpy(), h_next


class GMPStream(WindowedStream):
    """backbones/gmp.py (memory length 11): its envelope windows lag the signal windows by up to another memory
    length, so an output reads 2 * (11 - 1) = 20 past samples; that history is carried instead of zero-filled per
    segment. The reach was measured with ``measure_warmup`` (20), not read off the parameter."""

    def __init__(self, core):
        super().__init__(history_samples=2 * (int(core.backbone.memory_length) - 1))
        self._backbone = core.backbone

    def _run(self, block: np.ndarray) -> np.ndarray:
        import torch

        with torch.inference_mode():
            return self._backbone(torch.from_numpy(block).unsqueeze(0), None).squeeze(0).numpy()


ADAPTERS = {"gru_stream": GRUStream, "gmp_stream": GMPStream}


def is_streaming(key: str) -> bool:
    return get_model(key).execution_semantics == STREAMING


def streaming_model(key: str, core) -> StreamingModel:
    """The streaming adapter for a registered variant, over the legacy CoreModel that holds the weights."""
    try:
        return ADAPTERS[key](core)
    except KeyError:
        raise WorkspaceError(f"model '{key}' has no streaming adapter") from None


def stream_config(ws: Workspace, run_id: str, *, chunk_samples: Optional[int] = None, device: str = "cpu",
                  profile_id: Optional[str] = None) -> ExperimentConfig:
    """The run that scores a finished train_pa / train_dpd run's weights under its streaming variant: evaluate_pa
    for a PA, run_dpd (through the training surrogate) for a DPD. A new, separately stored result."""
    from opendpd.services.experiments import load_resolved, load_run

    record = load_run(ws, run_id)
    resolved = load_resolved(ws, run_id)
    if record.status != RunStatus.succeeded or resolved.task not in (TaskType.train_pa, TaskType.train_dpd):
        raise WorkspaceError(f"run '{run_id}' is a {record.status.value} {resolved.task.value} run; streaming evaluation "
                             "starts from a succeeded train_pa or train_dpd run")
    variant = streaming_variant_of(resolved.model.key)
    if variant is None:
        raise WorkspaceError(f"model '{resolved.model.key}' has no registered streaming variant "
                             "(see `opendpd models` for keys with execution_semantics streaming_stateful)")
    model = ModelSpec(key=variant.key, parameters=dict(resolved.model.parameters))
    evaluation = EvaluationConfig(profile_id=profile_id or resolved.evaluation.profile_id, chunk_samples=chunk_samples,
                                  evidence_type=EvidenceType.pa_modeling if resolved.task == TaskType.train_pa
                                  else EvidenceType.dpd_surrogate)
    common = dict(dataset=DatasetRef(id=resolved.dataset.id, preprocessing_version=resolved.dataset.preprocessing_version),
                  model=model, evaluation=evaluation, execution=ExecutionConfig(device=device),
                  name=f"stream {run_id} as {variant.key}",
                  notes=f"streaming-v1: weights of {run_id} ({resolved.model.key}) executed as {variant.key}")
    if resolved.task == TaskType.train_pa:
        return ExperimentConfig(task=TaskType.evaluate_pa, pa_reference=PAReference(run_id=run_id), **common)
    return ExperimentConfig(task=TaskType.run_dpd, dpd_reference=DPDReference(run_id=run_id), **common)


def stream_outputs(core, key: str, x: np.ndarray, *, chunk_samples: Optional[int], sample_rate_hz: float,
                   tolerance: float = CONSISTENCY_TOLERANCE) -> Tuple[np.ndarray, ExecutionEvidence]:
    """Run ``x`` through the variant in chunks and record the evidence: chunk consistency against the full-sequence
    run of the same variant, the measured warm-up, the look-ahead as samples and seconds."""
    model = streaming_model(key, core)
    chunk = int(chunk_samples or DEFAULT_CHUNK_SAMPLES)
    y = run_stream(model, x, chunk)
    check = chunk_consistency(model, x, [chunk], tolerance=tolerance)
    spec = model.spec
    evidence = ExecutionEvidence(
        semantics=STREAMING, state=spec.state, chunk_samples=chunk, lookahead_samples=spec.lookahead_samples,
        lookahead_s=spec.latency_s(sample_rate_hz), history_samples=spec.history_samples,
        warmup_samples=measure_warmup(model, x, tolerance=tolerance), tail_policy=spec.tail_policy,
        consistency=StreamConsistency(chunk_samples=chunk, max_abs_error=check["max_abs_error"][str(chunk)],
                                      tolerance=tolerance, within_tolerance=check["within_tolerance"]),
        note=LOOKAHEAD_NOTE)
    return y, evidence


def segments(x: np.ndarray, nperseg: int) -> np.ndarray:
    """(n_segments, nperseg, 2) with the last segment zero-padded, as modules.data_collector.IQSegmentDataset."""
    n = x.shape[0]
    n_seg = -(-n // nperseg)
    out = np.zeros((n_seg * nperseg, 2), dtype=np.float32)
    out[:n] = x
    return out.reshape(n_seg, nperseg, 2)


def streaming_limitations(key: str, evidence: ExecutionEvidence) -> list:
    base = get_model(key).weights_from
    lims = [f"streaming variant {key}: {evidence.state} state carried across chunks of {evidence.chunk_samples} samples; "
            f"scored under streaming semantics, not comparable with offline_segmented results of {base} and not inherited "
            "from them"]
    if evidence.lookahead_samples:
        lims.append(f"look-ahead of {evidence.lookahead_samples} samples ({evidence.lookahead_s * 1e6:.3f} µs at the dataset "
                    "rate): a real-time implementation buffers at least this much; an algorithmic bound, not a measured latency")
    if not evidence.consistency.within_tolerance:
        lims.append(f"chunk consistency failed: max |streamed - full sequence| = {evidence.consistency.max_abs_error:.3g} "
                    f"exceeds {evidence.consistency.tolerance:g}; the streamed outputs depend on the chunking")
    return lims


def describe(evidence: Dict[str, Any]) -> str:
    """One line for the CLI."""
    c = evidence["consistency"]
    return (f"{evidence['semantics']} ({evidence['state']} state), chunk {evidence['chunk_samples']} samples, look-ahead "
            f"{evidence['lookahead_samples']} samples, warm-up {evidence['warmup_samples']} samples; chunk consistency "
            f"max error {c['max_abs_error']:.3g} ({'within' if c['within_tolerance'] else 'BEYOND'} {c['tolerance']:g})")
