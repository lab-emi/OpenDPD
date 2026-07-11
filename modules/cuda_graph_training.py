"""Opt-in CUDA-graph training for the supported DPD cascade.

Only the existing clean-eager forward, MSE loss, gradient zeroing, and backward
are captured.  Batch copies, clipping, and optimizer updates remain ordinary
PyTorch operations so parameters and optimizer state can change in place
between replays.  Every unsupported condition returns ``None`` for a strict
clean-eager fallback.
"""

from __future__ import annotations

import contextlib
import os
import threading
import weakref
from dataclasses import dataclass

import torch
from torch import nn

from backbones.cuda_graph_frozen_dgru import disable_cuda_graph_frozen_dgru


_DISABLE_ENV = "OPENDPD_DISABLE_CUDA_GRAPH_TRAINING"
_TRUE_VALUES = {"1", "true", "yes", "on"}
_CACHES = weakref.WeakKeyDictionary()
_CACHES_LOCK = threading.Lock()
_CAPTURE_FAILED = object()


@dataclass(frozen=True)
class CudaGraphTrainingResult:
    """Result of one captured forward/loss/backward replay."""

    loss: torch.Tensor
    output: torch.Tensor


def _disabled() -> bool:
    return os.getenv(_DISABLE_ENV, "").strip().lower() in _TRUE_VALUES


def _cuda_autocast_enabled() -> bool:
    try:
        return torch.is_autocast_enabled("cuda")
    except TypeError:  # pragma: no cover - compatibility with older PyTorch.
        return torch.is_autocast_enabled()


def _tres_layers(module: nn.Module) -> tuple[nn.Module, ...]:
    """Return only actual TRes recurrence layers, not lookalike attributes."""

    from backbones.tres_deltagru import DeltaGRULayer

    return tuple(
        child for child in module.modules()
        if isinstance(child, DeltaGRULayer)
    )


@contextlib.contextmanager
def force_clean_eager_tres(module: nn.Module):
    """Temporarily disable fused TRes arithmetic, restoring it afterward."""

    layers = _tres_layers(module)
    previous = tuple(layer.use_triton for layer in layers)
    for layer in layers:
        layer.use_triton = False
    try:
        yield
    finally:
        for layer, enabled in zip(layers, previous, strict=True):
            layer.use_triton = enabled


def _supported_cascade(module: nn.Module) -> bool:
    """Limit capture to the measured, state-free DPD/PA architecture."""

    from backbones.dgru import DGRU
    from backbones.tres_deltagru import TResDeltaGRU
    from models import CascadedModel, CoreModel

    if not isinstance(module, CascadedModel):
        return False
    dpd = module.dpd_model
    pa = module.pa_model
    if not isinstance(dpd, CoreModel) or not isinstance(pa, CoreModel):
        return False
    if (
        dpd.backbone_type != "tres_deltagru"
        or pa.backbone_type != "dgru"
        or not isinstance(dpd.backbone, TResDeltaGRU)
        or not isinstance(pa.backbone, DGRU)
        or dpd.backbone.rnn.debug
        or any(parameter.requires_grad for parameter in pa.parameters())
    ):
        return False
    return tuple(_tres_layers(module)) == (dpd.backbone.rnn,)


def _can_capture(
    module: nn.Module,
    features: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
    parameters: tuple[torch.Tensor, ...],
) -> bool:
    if (
        _disabled()
        or not module.training
        or not torch.is_grad_enabled()
        or not isinstance(criterion, nn.MSELoss)
        or criterion.reduction != "mean"
        or not _supported_cascade(module)
        or not parameters
        or features.requires_grad
        or targets.requires_grad
        or not features.is_cuda
        or not targets.is_cuda
        or features.device != targets.device
        or features.dtype != torch.float32
        or targets.dtype != torch.float32
        or not features.is_contiguous()
        or not targets.is_contiguous()
        or torch.version.cuda is None
        or getattr(torch.version, "hip", None) is not None
        or _cuda_autocast_enabled()
        or not hasattr(torch.cuda, "CUDAGraph")
    ):
        return False
    try:
        if torch.cuda.is_current_stream_capturing():
            return False
    except RuntimeError:
        return False

    model_parameters = tuple(module.parameters())
    if any(
        child._forward_hooks
        or child._forward_pre_hooks
        or child._backward_hooks
        or child._backward_pre_hooks
        for child in module.modules()
    ):
        return False
    trainable = tuple(
        parameter for parameter in model_parameters
        if parameter.requires_grad
    )
    if parameters != trainable:
        return False
    return all(
        parameter.device == features.device
        and parameter.dtype == torch.float32
        for parameter in model_parameters
    )


def _cache_key(
    module: nn.Module,
    features: torch.Tensor,
    targets: torch.Tensor,
    parameters: tuple[torch.Tensor, ...],
):
    return (
        features.device.index,
        features.dtype,
        tuple(features.shape),
        tuple(features.stride()),
        tuple(targets.shape),
        tuple(targets.stride()),
        tuple(parameter.data_ptr() for parameter in module.parameters()),
        tuple(parameter.data_ptr() for parameter in parameters),
    )


class _GraphCache:
    def __init__(self):
        self.entries = {}
        self.capture_streams = {}
        self.lock = threading.Lock()

    def get_or_create(
        self, module, features, targets, criterion, parameters
    ):
        key = _cache_key(module, features, targets, parameters)
        entry = self.entries.get(key)
        if entry is _CAPTURE_FAILED:
            return None
        if entry is not None and entry.failed:
            return None
        if entry is not None and entry.is_valid():
            return entry

        # A simultaneous first call falls back rather than attempting a second
        # capture of the same module and parameter storage.
        if not self.lock.acquire(blocking=False):
            return None
        try:
            entry = self.entries.get(key)
            if entry is _CAPTURE_FAILED:
                return None
            if entry is not None and entry.failed:
                return None
            if entry is None or not entry.is_valid():
                try:
                    device_index = features.device.index
                    capture_stream = self.capture_streams.get(device_index)
                    if capture_stream is None:
                        capture_stream = torch.cuda.Stream(
                            device=features.device
                        )
                        self.capture_streams[device_index] = capture_stream
                    entry = _CapturedTrainingStep(
                        module, features, targets, criterion, parameters,
                        capture_stream,
                    )
                except (RuntimeError, torch.cuda.OutOfMemoryError):
                    self.entries[key] = _CAPTURE_FAILED
                    return None
                self.entries[key] = entry
            return entry
        finally:
            self.lock.release()


class _CapturedTrainingStep:
    def __init__(
        self, module, example_features, example_targets,
        criterion, parameters, capture_stream
    ):
        self.parameters = parameters
        self.parameter_pointers = tuple(
            parameter.data_ptr() for parameter in parameters
        )
        self.lock = threading.Lock()
        self.busy = False
        self.failed = False
        self.pending_event = None
        self.pending_stream = None

        with torch.cuda.device(example_features.device), torch.enable_grad():
            self.capture_stream = capture_stream
            self.capture_stream.wait_stream(torch.cuda.current_stream())
            with (
                force_clean_eager_tres(module),
                disable_cuda_graph_frozen_dgru(),
                torch.cuda.stream(self.capture_stream),
            ):
                self.static_features = example_features.detach().clone()
                self.static_targets = example_targets.detach().clone()
                # Initialize cuDNN/cuBLAS and persistent parameter-grad buffers
                # on the exact stream that will own the capture.
                for _ in range(2):
                    for parameter in parameters:
                        if parameter.grad is not None:
                            parameter.grad.zero_()
                    warm_output = module(self.static_features)
                    warm_loss = criterion(warm_output, self.static_targets)
                    warm_loss.backward()
                del warm_output, warm_loss

            torch.cuda.current_stream().wait_stream(self.capture_stream)
            torch.cuda.synchronize(example_features.device)
            if any(parameter.grad is None for parameter in parameters):
                raise RuntimeError(
                    "captured parameters must all receive gradients"
                )

            self.graph = torch.cuda.CUDAGraph()
            with (
                force_clean_eager_tres(module),
                disable_cuda_graph_frozen_dgru(),
                torch.cuda.graph(self.graph, stream=self.capture_stream),
            ):
                for parameter in parameters:
                    parameter.grad.zero_()
                self.static_output = module(self.static_features)
                self.static_loss = criterion(
                    self.static_output, self.static_targets
                )
                self.static_loss.backward()

        self.gradient_pointers = tuple(
            parameter.grad.data_ptr() for parameter in parameters
        )

    def is_valid(self) -> bool:
        return (
            not self.failed
            and self.parameter_pointers == tuple(
                parameter.data_ptr() for parameter in self.parameters
            )
            and all(
                parameter.grad is not None for parameter in self.parameters
            )
            and self.gradient_pointers == tuple(
                parameter.grad.data_ptr() for parameter in self.parameters
            )
        )

    def reserve(self) -> bool:
        with self.lock:
            if not self.is_valid() or self.busy:
                return False
            if self.pending_event is not None:
                stream = torch.cuda.current_stream(
                    self.static_features.device
                )
                same_stream = stream.cuda_stream == self.pending_stream
                if not same_stream and not self.pending_event.query():
                    return False
                self.pending_event = None
                self.pending_stream = None
            self.busy = True
            return True

    def release(self, *, failed: bool = False) -> None:
        with self.lock:
            self.failed = self.failed or failed
            stream = torch.cuda.current_stream(self.static_features.device)
            event = torch.cuda.Event()
            event.record(stream)
            self.pending_event = event
            self.pending_stream = stream.cuda_stream
            self.busy = False

    def run(self, module, features, targets):
        if not self.reserve():
            return None
        try:
            self.static_features.copy_(features)
            self.static_targets.copy_(targets)
            with force_clean_eager_tres(module):
                self.graph.replay()
            # Each logged loss needs independent storage because replay updates
            # the static scalar on the next batch.
            result = CudaGraphTrainingResult(
                loss=self.static_loss.detach().clone(),
                output=self.static_output.detach(),
            )
        except (RuntimeError, torch.cuda.OutOfMemoryError):
            self.release(failed=True)
            return None
        self.release()
        return result


def try_cuda_graph_training_step(
    module: nn.Module,
    features: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
    parameters: tuple[torch.Tensor, ...],
) -> CudaGraphTrainingResult | None:
    """Replay one captured training body, or return ``None`` for fallback."""

    if not _can_capture(module, features, targets, criterion, parameters):
        return None
    with _CACHES_LOCK:
        cache = _CACHES.get(module)
        if cache is None:
            cache = _GraphCache()
            _CACHES[module] = cache
    step = cache.get_or_create(
        module, features, targets, criterion, parameters
    )
    if step is None:
        return None
    return step.run(module, features, targets)


def clear_cuda_graph_training_cache() -> None:
    """Drop all captured training steps; intended primarily for tests."""

    with _CACHES_LOCK:
        _CACHES.clear()
