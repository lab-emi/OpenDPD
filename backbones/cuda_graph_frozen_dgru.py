"""Exact CUDA-graph replay for a frozen DGRU used during DPD training.

The optimization records the existing cuDNN-backed DGRU forward and its input
backward as two CUDA graphs.  It neither replaces the recurrence nor changes
the module tree.  Unsupported or unsafe calls return ``None`` so the caller can
execute its ordinary eager implementation.
"""

from __future__ import annotations

import contextlib
import contextvars
import os
import threading
import weakref

import torch


_DISABLE_ENV = "OPENDPD_DISABLE_CUDA_GRAPH_FROZEN_DGRU"
_TRUE_VALUES = {"1", "true", "yes", "on"}
_MANAGERS = weakref.WeakKeyDictionary()
_MANAGERS_LOCK = threading.Lock()
_CAPTURE_FAILED = object()
_LOCAL_DISABLE_DEPTH = contextvars.ContextVar(
    "opendpd_frozen_dgru_graph_disable_depth", default=0
)


def _disabled() -> bool:
    return (
        _LOCAL_DISABLE_DEPTH.get() > 0
        or os.getenv(_DISABLE_ENV, "").strip().lower() in _TRUE_VALUES
    )


@contextlib.contextmanager
def disable_cuda_graph_frozen_dgru():
    """Temporarily bypass the inner graph without changing process state."""

    token = _LOCAL_DISABLE_DEPTH.set(_LOCAL_DISABLE_DEPTH.get() + 1)
    try:
        yield
    finally:
        _LOCAL_DISABLE_DEPTH.reset(token)


def _cuda_autocast_enabled() -> bool:
    try:
        return torch.is_autocast_enabled("cuda")
    except TypeError:  # pragma: no cover - compatibility with older PyTorch.
        return torch.is_autocast_enabled()


def _can_replay(module, input: torch.Tensor, h_0: torch.Tensor | None) -> bool:
    if (
        _disabled()
        or not module.training
        or not torch.is_grad_enabled()
        or not input.requires_grad
        or input.ndim != 3
        or not input.is_cuda
        or input.dtype != torch.float32
        or not input.is_contiguous()
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

    parameters = tuple(module.parameters())
    if not parameters or any(
        parameter.requires_grad
        or parameter.device != input.device
        or parameter.dtype != input.dtype
        for parameter in parameters
    ):
        return False
    if h_0 is not None and (
        h_0.requires_grad
        or h_0.device != input.device
        or h_0.dtype != input.dtype
        or not h_0.is_contiguous()
    ):
        return False
    return True


def _cache_key(module, input: torch.Tensor, h_0: torch.Tensor | None):
    parameter_storage = tuple(
        parameter.data_ptr() for parameter in module.parameters()
    )
    h0_signature = (
        None if h_0 is None else (tuple(h_0.shape), tuple(h_0.stride()))
    )
    return (
        input.device.index,
        input.dtype,
        tuple(input.shape),
        tuple(input.stride()),
        h0_signature,
        parameter_storage,
    )


class _ReplayCache:
    def __init__(self):
        self.entries = {}
        self.lock = threading.Lock()

    def get_or_create(self, module, input, h_0):
        key = _cache_key(module, input, h_0)
        entry = self.entries.get(key)
        if entry is _CAPTURE_FAILED:
            return None
        if entry is not None:
            return entry

        # Capturing is expensive and unsafe to duplicate.  A concurrent first
        # call simply uses eager execution and can use the cache next time.
        if not self.lock.acquire(blocking=False):
            return None
        try:
            entry = self.entries.get(key)
            if entry is _CAPTURE_FAILED:
                return None
            if entry is None:
                try:
                    entry = _FrozenDGRUReplay(module, input, h_0)
                except (RuntimeError, torch.cuda.OutOfMemoryError):
                    self.entries[key] = _CAPTURE_FAILED
                    return None
                self.entries[key] = entry
            return entry
        finally:
            self.lock.release()


class _FrozenDGRUReplay:
    def __init__(self, module, example_input, example_h0):
        self.lock = threading.Lock()
        self.busy = False
        self.failed = False
        self.generation = 0
        self.pending_event = None
        self.pending_stream = None

        with torch.cuda.device(example_input.device), torch.enable_grad():
            # Warm cuDNN and autograd on a side stream before capture.  Using
            # that same stream for both captures keeps retained autograd nodes
            # from acquiring stale cross-stream dependencies.
            self.capture_stream = torch.cuda.Stream(
                device=example_input.device
            )
            self.capture_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.capture_stream):
                self.static_input = (
                    example_input.detach().clone().requires_grad_()
                )
                self.static_h0 = (
                    None if example_h0 is None else example_h0.detach().clone()
                )
                for _ in range(3):
                    warm_output = module._forward_eager(
                        self.static_input, self.static_h0
                    )
                    warm_gradient = torch.ones_like(warm_output)
                    torch.autograd.grad(
                        warm_output,
                        self.static_input,
                        grad_outputs=warm_gradient,
                    )
                del warm_output, warm_gradient
            torch.cuda.current_stream().wait_stream(self.capture_stream)
            torch.cuda.synchronize(example_input.device)

            self.forward_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(
                self.forward_graph, stream=self.capture_stream
            ):
                self.static_output = module._forward_eager(
                    self.static_input, self.static_h0
                )

            self.static_grad_output = torch.empty_like(self.static_output)
            self.backward_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(
                self.backward_graph, stream=self.capture_stream
            ):
                (self.static_grad_input,) = torch.autograd.grad(
                    self.static_output,
                    self.static_input,
                    grad_outputs=self.static_grad_output,
                )

    def reserve(self) -> int | None:
        with self.lock:
            if self.failed or self.busy:
                return None
            if self.pending_event is not None:
                current_stream = torch.cuda.current_stream(
                    self.static_input.device
                )
                same_stream = current_stream.cuda_stream == self.pending_stream
                if not same_stream and not self.pending_event.query():
                    return None
                self.pending_event = None
                self.pending_stream = None
            self.busy = True
            self.generation += 1
            return self.generation

    def release(self, generation: int, *, failed: bool = False) -> None:
        with self.lock:
            if generation != self.generation:
                return
            self.failed = self.failed or failed
            if torch.cuda.is_available():
                event = torch.cuda.Event()
                stream = torch.cuda.current_stream(self.static_input.device)
                event.record(stream)
                self.pending_event = event
                self.pending_stream = stream.cuda_stream
            self.busy = False

    def forward(self, input, h_0, generation):
        if generation != self.generation or not self.busy:
            raise RuntimeError("stale frozen-DGRU CUDA-graph reservation")
        self.static_input.copy_(input)
        if self.static_h0 is not None:
            self.static_h0.copy_(h_0)
        self.forward_graph.replay()
        return self.static_output.clone()

    def backward(self, grad_output, generation):
        if generation != self.generation or not self.busy:
            raise RuntimeError(
                "frozen-DGRU CUDA graph cannot be replayed twice"
            )
        self.static_grad_output.copy_(grad_output)
        self.backward_graph.replay()
        return self.static_grad_input.clone()


class _FrozenDGRUReplayFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, h_0, replay, generation):
        ctx.replay = replay
        ctx.generation = generation
        ctx.used = False
        try:
            return replay.forward(input, h_0, generation)
        except Exception:
            replay.release(generation, failed=True)
            raise

    @staticmethod
    def backward(ctx, grad_output):
        replay = ctx.replay
        generation = ctx.generation
        if ctx.used:
            raise RuntimeError(
                "frozen-DGRU CUDA graph supports one backward pass"
            )
        ctx.used = True
        if torch.is_grad_enabled():
            replay.release(generation)
            raise RuntimeError(
                "CUDA-graph frozen DGRU supports first-order training only; "
                f"set {_DISABLE_ENV}=1 for higher-order autograd"
            )
        try:
            grad_input = replay.backward(grad_output, generation)
        except Exception:
            replay.release(generation, failed=True)
            raise
        replay.release(generation)
        return grad_input, None, None, None


def try_cuda_graph_frozen_dgru(
    module, input: torch.Tensor, h_0: torch.Tensor | None
) -> torch.Tensor | None:
    """Return a graph-replayed output, or ``None`` for eager fallback."""

    if not _can_replay(module, input, h_0):
        return None
    with _MANAGERS_LOCK:
        manager = _MANAGERS.get(module)
        if manager is None:
            manager = _ReplayCache()
            _MANAGERS[module] = manager
    replay = manager.get_or_create(module, input, h_0)
    if replay is None:
        return None
    generation = replay.reserve()
    if generation is None:
        return None
    try:
        return _FrozenDGRUReplayFunction.apply(input, h_0, replay, generation)
    except (RuntimeError, torch.cuda.OutOfMemoryError):
        # A replay error disables this cache entry.  Eager execution still has
        # the original input and is safe unless CUDA itself is unrecoverable.
        return None


def clear_cuda_graph_frozen_dgru_cache() -> None:
    """Drop replay caches; primarily useful for focused tests."""

    with _MANAGERS_LOCK:
        _MANAGERS.clear()
