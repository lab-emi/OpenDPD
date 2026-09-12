"""Bit-exact fixed-point reference of the GRU streaming variant under ``fixed-point-v1`` (plan S19).

Integer arithmetic only (numpy int64), every operator defined by the specification: input / state / output
formats, per-tensor weight fractions, biases at the pre-activation fraction, exact dot products in a bounded
accumulator, one rounding rule for every rescale, saturation of every stored quantity, and the two
non-linearities as tables. The C99 backend (``opendpd.export.c_backend``) implements the same steps in the same
order; ``golden`` vectors and a per-step trace let a mismatch be located at a sample and a signal.

GRU (PyTorch gate order r, z, n)::

    r = sigmoid(W_ir x + b_ir + W_hr h + b_hr)
    z = sigmoid(W_iz x + b_iz + W_hz h + b_hz)
    n = tanh(W_in x + b_in + r * (W_hn h + b_hn))
    h' = (1 - z) * n + z * h
    y = W_out h' + b_out
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from opendpd.schemas.fixed_point import FixedPointSpec, TableSpec, TensorFormat, WordFormat

GATES = ("r", "z", "n")


# --- primitive operators (the C backend mirrors each one) ------------------------------------------

def rescale(v: np.ndarray, from_frac: int, to_frac: int) -> np.ndarray:
    """Change the fraction: exact left shift, or round-half-up right shift (floor of v + half)."""
    v = np.asarray(v, dtype=np.int64)
    if to_frac >= from_frac:
        return v << np.int64(to_frac - from_frac)
    s = from_frac - to_frac
    return np.floor_divide(v + np.int64(1 << (s - 1)), np.int64(1 << s))


def saturate(v: np.ndarray, fmt: WordFormat) -> np.ndarray:
    return np.clip(np.asarray(v, dtype=np.int64), fmt.min_int, fmt.max_int)


def to_fixed(x: np.ndarray, fmt: WordFormat) -> np.ndarray:
    """Float to the word format: round half away from zero, then saturate (numpy int64)."""
    scaled = np.asarray(x, dtype=np.float64) * fmt.scale
    return saturate(np.where(scaled >= 0, np.floor(scaled + 0.5), np.ceil(scaled - 0.5)).astype(np.int64), fmt)


def to_float(v: np.ndarray, fmt: WordFormat) -> np.ndarray:
    return np.asarray(v, dtype=np.float64) / fmt.scale


def table(spec: TableSpec) -> np.ndarray:
    """The lookup table: entry i holds f((i - offset) / 2^index_frac) in the value format."""
    offset = int(spec.range * (1 << spec.index_frac))
    xs = (np.arange(spec.entries, dtype=np.float64) - offset) / (1 << spec.index_frac)
    f = (lambda t: 1.0 / (1.0 + np.exp(-t))) if spec.function == "sigmoid" else np.tanh
    return to_fixed(f(xs), spec.value).astype(np.int64)


def lookup(pre: np.ndarray, spec: TableSpec, tab: np.ndarray, pre_frac: int) -> np.ndarray:
    """Pre-activation (pre_frac) -> table index (index_frac, rounded, saturated to the range) -> value."""
    offset = int(spec.range * (1 << spec.index_frac))
    idx = np.clip(rescale(pre, pre_frac, spec.index_frac), -offset, offset - 1) + offset
    return tab[idx]


def weight_fraction(max_abs: float, bits: int) -> int:
    """The largest fraction under which max |w| still fits the word; tiny tensors are capped at bits + 8."""
    fmt_max = (1 << (bits - 1)) - 1
    frac = bits - 1 - int(math.ceil(math.log2(max_abs))) if max_abs > 0 else bits + 8
    frac = min(frac, bits + 8)
    while frac > 0 and round(max_abs * (1 << frac)) > fmt_max:
        frac -= 1
    return frac


# --- quantised weights --------------------------------------------------------------------------------

@dataclass
class QuantisedGRU:
    """Integer weights of one GRU layer plus the output projection, with their formats."""

    spec: FixedPointSpec
    hidden: int
    inputs: int
    outputs: int
    w_ih: np.ndarray            # (3H, IN) int64 holding weight_bits values, fraction f_ih
    w_hh: np.ndarray            # (3H, H)
    w_out: np.ndarray           # (OUT, H)
    f_ih: int
    f_hh: int
    f_out: int
    b_ih: np.ndarray            # (3H,) int64 at spec.pre.frac
    b_hh: np.ndarray            # (3H,)
    b_out: np.ndarray           # (OUT,)
    tensors: List[TensorFormat] = field(default_factory=list)
    sigmoid_table: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    tanh_table: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))

    @property
    def mac_per_sample(self) -> int:
        return 3 * self.hidden * (self.inputs + self.hidden) + self.outputs * self.hidden

    @property
    def table_lookups_per_sample(self) -> int:
        return 3 * self.hidden

    def storage_bytes(self) -> Dict[str, int]:
        wb = (self.spec.weight_bits + 7) // 8
        return {"weight_bytes": (self.w_ih.size + self.w_hh.size + self.w_out.size) * wb,
                "bias_bytes": (self.b_ih.size + self.b_hh.size + self.b_out.size) * ((self.spec.pre.bits + 7) // 8),
                "state_bytes": self.hidden * ((self.spec.h.bits + 7) // 8),
                "table_bytes": (self.sigmoid_table.size + self.tanh_table.size) * ((self.spec.sigmoid.value.bits + 7) // 8)}


def quantise(state: Dict[str, np.ndarray], spec: Optional[FixedPointSpec] = None) -> QuantisedGRU:
    """Quantise a one-layer GRU state dict (``backbone.rnn.*``, ``backbone.fc_out.*``) under the spec."""
    spec = spec or FixedPointSpec()
    arrays = {k: np.asarray(v, dtype=np.float64) for k, v in state.items()}
    try:
        w_ih, w_hh = arrays["backbone.rnn.weight_ih_l0"], arrays["backbone.rnn.weight_hh_l0"]
        b_ih, b_hh = arrays["backbone.rnn.bias_ih_l0"], arrays["backbone.rnn.bias_hh_l0"]
        w_out, b_out = arrays["backbone.fc_out.weight"], arrays["backbone.fc_out.bias"]
    except KeyError as err:
        raise ValueError(f"not a one-layer GRU state dict: missing {err}") from None
    if "backbone.rnn.weight_ih_l1" in arrays:
        raise ValueError("fixed-point-v1 covers one GRU layer; this model has more")
    hidden = w_hh.shape[1]
    tensors: List[TensorFormat] = []
    quantised: Dict[str, np.ndarray] = {}
    fracs: Dict[str, int] = {}
    for name, w in (("w_ih", w_ih), ("w_hh", w_hh), ("w_out", w_out)):
        max_abs = float(np.max(np.abs(w))) if w.size else 0.0
        frac = weight_fraction(max_abs, spec.weight_bits)
        fmt = WordFormat(bits=spec.weight_bits, frac=frac)
        exact = np.asarray(w, dtype=np.float64) * fmt.scale
        rounded = np.where(exact >= 0, np.floor(exact + 0.5), np.ceil(exact - 0.5)).astype(np.int64)
        q = saturate(rounded, fmt)
        quantised[name], fracs[name] = q, frac
        tensors.append(TensorFormat(name=name, shape=list(w.shape), bits=spec.weight_bits, frac=frac, max_abs_float=max_abs,
                                    saturated=int(np.sum(q != rounded))))
    biases = {}
    for name, b in (("b_ih", b_ih), ("b_hh", b_hh), ("b_out", b_out)):
        q = to_fixed(b, spec.pre)
        biases[name] = q
        tensors.append(TensorFormat(name=name, shape=list(b.shape), bits=spec.pre.bits, frac=spec.pre.frac,
                                    max_abs_float=float(np.max(np.abs(b))) if b.size else 0.0,
                                    saturated=int(np.sum(np.abs(np.asarray(b) * spec.pre.scale) > spec.pre.max_int))))
    return QuantisedGRU(spec=spec, hidden=hidden, inputs=w_ih.shape[1], outputs=w_out.shape[0],
                        w_ih=quantised["w_ih"], w_hh=quantised["w_hh"], w_out=quantised["w_out"],
                        f_ih=fracs["w_ih"], f_hh=fracs["w_hh"], f_out=fracs["w_out"],
                        b_ih=biases["b_ih"], b_hh=biases["b_hh"], b_out=biases["b_out"], tensors=tensors,
                        sigmoid_table=table(spec.sigmoid), tanh_table=table(spec.tanh))


# --- the reference ----------------------------------------------------------------------------------

class FixedGRU:
    """Bit-exact stepper. ``h`` is the carried state (spec.h); ``reset`` zeroes it."""

    def __init__(self, q: QuantisedGRU):
        self.q = q
        self.spec = q.spec
        self._acc_limit = 1 << (self.spec.accumulator_bits - 1)
        self.h = np.zeros(q.hidden, dtype=np.int64)
        H = q.hidden
        self._slices = {g: slice(i * H, (i + 1) * H) for i, g in enumerate(GATES)}

    def reset(self) -> None:
        self.h[:] = 0

    def _dot(self, w: np.ndarray, v: np.ndarray) -> np.ndarray:
        acc = w @ v
        if np.any(np.abs(acc) >= self._acc_limit):
            raise OverflowError(f"accumulator exceeds {self.spec.accumulator_bits} bits")
        return acc

    def step(self, x_q: np.ndarray, trace: Optional[Dict[str, List[np.ndarray]]] = None) -> np.ndarray:
        """One sample in (spec.x integers), one sample out (spec.y integers); the state advances."""
        s, q = self.spec, self.q
        x_q = saturate(x_q, s.x)
        pre = s.pre.frac
        acc_i = rescale(self._dot(q.w_ih, x_q), q.f_ih + s.x.frac, pre)
        acc_h = rescale(self._dot(q.w_hh, self.h), q.f_hh + s.h.frac, pre)
        ai = saturate(acc_i + q.b_ih, s.pre)
        ah = saturate(acc_h + q.b_hh, s.pre)
        r_ = self._slices["r"]
        z_ = self._slices["z"]
        n_ = self._slices["n"]
        r = lookup(saturate(ai[r_] + ah[r_], s.pre), s.sigmoid, q.sigmoid_table, pre)
        z = lookup(saturate(ai[z_] + ah[z_], s.pre), s.sigmoid, q.sigmoid_table, pre)
        t = rescale(r * ah[n_], s.h.frac + pre, pre)
        n = lookup(saturate(ai[n_] + t, s.pre), s.tanh, q.tanh_table, pre)
        one = np.int64(1 << s.h.frac)
        h_new = saturate(rescale((one - z) * n + z * self.h, 2 * s.h.frac, s.h.frac), s.h)
        acc_o = rescale(self._dot(q.w_out, h_new), q.f_out + s.h.frac, pre)
        y = saturate(rescale(saturate(acc_o + q.b_out, s.pre), pre, s.y.frac), s.y)
        self.h = h_new
        if trace is not None:
            for name, value in (("r", r), ("z", z), ("n", n), ("h", h_new)):
                trace.setdefault(name, []).append(value.copy())
        return y

    def run(self, x_q: np.ndarray, resets_at: Tuple[int, ...] = (), *, trace: bool = False) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Every sample of ``x_q`` (n, IN) in order; the state is reset before each index in ``resets_at``."""
        x_q = np.asarray(x_q, dtype=np.int64)
        resets = set(int(i) for i in resets_at)
        out = np.zeros((x_q.shape[0], self.q.outputs), dtype=np.int64)
        tr: Optional[Dict[str, List[np.ndarray]]] = {} if trace else None
        for i in range(x_q.shape[0]):
            if i in resets:
                self.reset()
            out[i] = self.step(x_q[i], tr)
        traces = {k: np.stack(v) for k, v in (tr or {}).items()}
        return out, traces
