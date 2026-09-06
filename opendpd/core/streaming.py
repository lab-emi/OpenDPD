"""Streaming execution contract (plan S18): ``reset / state / chunk / flush`` over a stream of I/Q samples.

An offline model scores fixed segments that each start from a zero state (``offline_segmented``). A streaming
variant of the same weights consumes the signal chunk by chunk and carries what it needs between chunks: a
recurrent state, or a window of past samples. Its outputs therefore differ from the offline segment scores at
every segment boundary, which is why a streaming variant is a separate registry entry whose results are never
compared with, or inherited from, the offline ones.

Semantics every variant declares in its ``StreamSpec``:

* ``lookahead_samples``: future samples an output needs. The runner holds back that many inputs, so the output for
  sample ``t`` is finalised once ``x[t + lookahead]`` has arrived; ``flush`` finalises the tail. This is the
  algorithmic look-ahead, an information bound; it is not the latency of any implementation, which is a measurement.
* ``history_samples``: past samples a window model keeps between chunks (zero-filled after ``reset``).
* ``warmup_samples``: leading outputs after ``reset`` that depend on the initial state; measured with
  :func:`measure_warmup`, never assumed.
* valid range: every output after warm-up. The chunk-consistency check compares the streamed outputs with the
  full-sequence run of the same variant from the same reset; the two must agree within ``tolerance`` everywhere.
* tail: ``flush`` feeds zeros for the missing future samples (the one tail policy of streaming-v1).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np

OFFLINE_SEMANTICS = "offline_segmented"
STREAMING_SEMANTICS = "streaming_stateful"
DEFAULT_CHUNK_SAMPLES = 1024
CONSISTENCY_TOLERANCE = 1e-4          # max |streamed - full sequence| in normalised units (float32 arithmetic)


@dataclass(frozen=True)
class StreamSpec:
    state: str                                  # "recurrent" | "window" | "none"
    lookahead_samples: int = 0
    history_samples: Optional[int] = None       # window models only
    warmup_samples: Optional[int] = None        # measured (measure_warmup); None = not measured

    def latency_s(self, sample_rate_hz: float) -> float:
        return self.lookahead_samples / float(sample_rate_hz)

    def valid_start(self) -> int:
        return int(self.warmup_samples or 0)


def _iq(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 1 and np.iscomplexobj(x):
        x = np.stack([x.real, x.imag], axis=-1)
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError(f"a stream is an (n, 2) I/Q array, got shape {x.shape}")
    return np.ascontiguousarray(x, dtype=np.float32)


class StreamingModel(ABC):
    """One stream at a time: ``reset`` before the first chunk, ``chunk`` per block of samples, ``flush`` at the end."""

    spec: StreamSpec

    @abstractmethod
    def reset(self) -> None:
        """Forget everything: zero recurrent state, zero history, no pending input."""

    @abstractmethod
    def get_state(self) -> Any:
        """A copy of everything ``chunk`` carries between calls (state, history, pending input)."""

    @abstractmethod
    def set_state(self, state: Any) -> None:
        """Restore a copy taken with ``get_state``."""

    @abstractmethod
    def chunk(self, x: np.ndarray) -> np.ndarray:
        """Consume ``x`` (n, 2) and return every output that is now finalised: ``n`` samples for a causal model,
        fewer while the look-ahead buffer fills. Outputs are (m, 2) float32 in stream order."""

    @abstractmethod
    def flush(self) -> np.ndarray:
        """Finalise the outputs still held back by the look-ahead buffer with zeros for the missing future
        (empty for a causal model)."""

    def step(self, sample: Sequence[float]) -> np.ndarray:
        """One sample in; (0 or 1, 2) out."""
        return self.chunk(np.asarray(sample, dtype=np.float32).reshape(1, 2))


class RecurrentStream(StreamingModel):
    """A causal recurrent model: the hidden state is the only thing carried across chunks."""

    def __init__(self, warmup_samples: Optional[int] = None):
        self.spec = StreamSpec(state="recurrent", lookahead_samples=0, warmup_samples=warmup_samples)
        self._h: Any = None

    @abstractmethod
    def _run(self, x: np.ndarray, h: Any) -> "tuple[np.ndarray, Any]":
        """Outputs for ``x`` from state ``h`` (None = zero state) and the state after the last sample."""

    def reset(self) -> None:
        self._h = None

    def get_state(self) -> Any:
        return None if self._h is None else _copy_state(self._h)

    def set_state(self, state: Any) -> None:
        self._h = None if state is None else _copy_state(state)

    def chunk(self, x: np.ndarray) -> np.ndarray:
        x = _iq(x)
        if x.shape[0] == 0:
            return np.zeros((0, 2), dtype=np.float32)
        out, h = self._run(x, self._h)
        out = np.asarray(out, dtype=np.float32)
        if out.shape != x.shape:
            raise RuntimeError(f"the model returned outputs of shape {out.shape} for {x.shape[0]} inputs")
        self._h = h
        return out

    def flush(self) -> np.ndarray:
        return np.zeros((0, 2), dtype=np.float32)


class WindowedStream(StreamingModel):
    """A model whose output at ``t`` reads ``x[t - history .. t + lookahead]``: the runner keeps the history and
    holds back the look-ahead; the model itself only maps a contiguous block to the outputs of its samples."""

    def __init__(self, history_samples: int, lookahead_samples: int = 0, warmup_samples: Optional[int] = 0):
        self.spec = StreamSpec(state="window", lookahead_samples=int(lookahead_samples), history_samples=int(history_samples),
                               warmup_samples=warmup_samples)
        self.reset()

    @abstractmethod
    def _run(self, block: np.ndarray) -> np.ndarray:
        """Outputs (len(block), 2) for a contiguous block, with the model's own zero padding outside the block."""

    def reset(self) -> None:
        self._history = np.zeros((self.spec.history_samples, 2), dtype=np.float32)
        self._pending = np.zeros((0, 2), dtype=np.float32)

    def get_state(self) -> Any:
        return {"history": self._history.copy(), "pending": self._pending.copy()}

    def set_state(self, state: Any) -> None:
        self._history, self._pending = state["history"].copy(), state["pending"].copy()

    def chunk(self, x: np.ndarray) -> np.ndarray:
        x = _iq(x)
        h, la = self.spec.history_samples, self.spec.lookahead_samples
        fresh = np.concatenate([self._pending, x])                     # inputs without a finalised output yet
        m = fresh.shape[0] - la                                        # how many of them can be finalised now
        if m <= 0:
            self._pending = fresh
            return np.zeros((0, 2), dtype=np.float32)
        block = np.concatenate([self._history, fresh])
        out = np.asarray(self._run(block), dtype=np.float32)[h:h + m]
        consumed = np.concatenate([self._history, fresh[:m]])          # everything before the first pending sample
        self._history = consumed[-h:] if h else consumed[:0]
        self._pending = fresh[m:]
        return out

    def flush(self) -> np.ndarray:
        n = self._pending.shape[0]
        if n == 0:
            return np.zeros((0, 2), dtype=np.float32)
        h, la = self.spec.history_samples, self.spec.lookahead_samples
        block = np.concatenate([self._history, self._pending, np.zeros((la, 2), dtype=np.float32)])
        out = np.asarray(self._run(block), dtype=np.float32)[h:h + n]
        self._history = np.concatenate([self._history, self._pending])[-h:] if h else self._history
        self._pending = self._pending[:0]
        return out


def _copy_state(state: Any) -> Any:
    if hasattr(state, "clone"):
        return state.clone()
    if isinstance(state, (list, tuple)):
        return type(state)(_copy_state(s) for s in state)
    if isinstance(state, dict):
        return {k: _copy_state(v) for k, v in state.items()}
    return np.array(state, copy=True) if isinstance(state, np.ndarray) else state


def run_stream(model: StreamingModel, x: np.ndarray, chunk_samples: Union[int, Iterable[int]] = DEFAULT_CHUNK_SAMPLES) -> np.ndarray:
    """Feed ``x`` in chunks (one size, or a sequence of sizes cycled), flush, and return one output per input."""
    x = _iq(x)
    sizes = [int(chunk_samples)] if isinstance(chunk_samples, (int, np.integer)) else [int(c) for c in chunk_samples]
    if not sizes or any(c < 1 for c in sizes):
        raise ValueError("chunk sizes must be positive")
    model.reset()
    outputs: List[np.ndarray] = []
    pos, i = 0, 0
    while pos < x.shape[0]:
        size = sizes[i % len(sizes)]
        outputs.append(model.chunk(x[pos:pos + size]))
        pos += size
        i += 1
    outputs.append(model.flush())
    y = np.concatenate(outputs) if outputs else np.zeros((0, 2), dtype=np.float32)
    if y.shape[0] != x.shape[0]:
        raise RuntimeError(f"the stream returned {y.shape[0]} outputs for {x.shape[0]} inputs")
    return y


def chunk_consistency(model: StreamingModel, x: np.ndarray, chunk_sizes: Sequence[Union[int, Sequence[int]]],
                      tolerance: float = CONSISTENCY_TOLERANCE) -> Dict[str, Any]:
    """Max |streamed - full sequence| per chunking, over the valid range (after warm-up), against the same variant run
    in one chunk from the same reset. The reference is the streaming model itself: an offline segment score is not
    a reference for a stream."""
    x = _iq(x)
    reference = run_stream(model, x, x.shape[0])
    start = model.spec.valid_start()
    errors = {}
    for sizes in chunk_sizes:
        y = run_stream(model, x, sizes)
        label = str(sizes) if isinstance(sizes, (int, np.integer)) else "+".join(str(s) for s in sizes)
        errors[label] = float(np.max(np.abs(y[start:] - reference[start:]))) if x.shape[0] > start else 0.0
    return {"reference": "full_sequence_same_reset", "valid_from": start, "tolerance": tolerance,
            "max_abs_error": errors, "within_tolerance": all(e <= tolerance for e in errors.values())}


def measure_warmup(model: StreamingModel, x: np.ndarray, tolerance: float = CONSISTENCY_TOLERANCE) -> int:
    """How many outputs after a reset in the middle of ``x`` differ from the uninterrupted stream by more than
    ``tolerance``: the number of leading samples whose output depends on the initial state."""
    x = _iq(x)
    n = x.shape[0]
    if n < 2:
        return 0
    split = n // 2
    continuous = run_stream(model, x, n)
    model.reset()
    model.chunk(x[:split])
    model.reset()                                     # the second half starts from the initial state
    restarted = np.concatenate([model.chunk(x[split:]), model.flush()])
    diff = np.max(np.abs(restarted - continuous[split:]), axis=1) > tolerance
    return int(np.max(np.nonzero(diff)[0]) + 1) if diff.any() else 0
