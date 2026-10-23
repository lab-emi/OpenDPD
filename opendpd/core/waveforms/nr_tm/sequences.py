"""Bit sequences and symbol mappers of the NR test models (all vectorised, numpy only).

* PN23 (ITU-T O.150 PRBS23, TS 38.141-1 4.9.2.3): s(i) = s(i-23) xor s(i-18), s(0..22) = 1, output s(0), s(1), ...
* Gold sequence of TS 38.211 5.2.1 (scrambling, DM-RS)
* QAM mapper of TS 38.211 5.1 (QPSK ... 1024QAM)
* DM-RS sequences of TS 38.211 7.4.1.1.1 (PDSCH) and 7.4.1.3.1 (PDCCH)

Linear recurrences over GF(2) are extended in doubling blocks: if p(x) annihilates a sequence then so does
p(x)^(2^j) = p(x^(2^j)), which gives x(m + P 2^j) = XOR_t x(m + t 2^j) and therefore blocks of (P - max tap) 2^j bits with
one vectorised XOR per tap. This makes a 10-million-bit sequence a matter of milliseconds.
"""
from __future__ import annotations

import math

import numpy as np

NC = 1600   # TS 38.211 5.2.1


def _extend(state: np.ndarray, taps: tuple[int, ...], degree: int, n: int) -> np.ndarray:
    """First n terms of x(m + degree) = XOR_{t in taps} x(m + t) started from `state` (len(state) == degree)."""
    out = np.zeros(max(n, degree), dtype=np.uint8)
    out[:degree] = state
    have = degree
    top = max(taps)
    while have < n:
        j = max(0, int(math.floor(math.log2(have / degree))))
        step = 1 << j
        chunk = min((degree - top) * step, n - have)
        base = have - degree * step
        block = out[base + taps[0] * step: base + taps[0] * step + chunk].copy()
        for t in taps[1:]:
            block ^= out[base + t * step: base + t * step + chunk]
        out[have:have + chunk] = block
        have += chunk
    return out[:n]


def pn23(n: int) -> np.ndarray:
    """First n bits of the PN23 sequence started from the all-ones state (the first 23 bits are ones)."""
    return _extend(np.ones(23, dtype=np.uint8), (0, 5), 23, n)


def gold(c_init: int, n: int) -> np.ndarray:
    """c(0..n-1) of TS 38.211 5.2.1: x1(0) = 1, x2 = binary expansion of c_init, N_C = 1600."""
    total = n + NC
    x1 = _extend(np.array([1] + [0] * 30, dtype=np.uint8), (0, 3), 31, total)
    x2 = _extend(np.array([(c_init >> i) & 1 for i in range(31)], dtype=np.uint8), (0, 1, 2, 3), 31, total)
    return x1[NC:] ^ x2[NC:]


def qam_map(bits: np.ndarray, qm: int) -> np.ndarray:
    """Bits -> unit-average-power constellation points, TS 38.211 5.1.3 ... 5.1.7.

    Bits b0 b2 b4 ... drive I and b1 b3 b5 ... drive Q. With s = 1 - 2b the per-axis amplitude is the nested form
    s0 (2^(h-1) - s2 (2^(h-2) - s4 (...))), h = qm / 2, normalised by sqrt(2 (M - 1) / 3), M = 2^qm.
    The normalisation is applied as a multiplication, which gives the same last bit as the reference implementation
    (QPSK points are +-sqrt(0.5)).
    """
    b = np.asarray(bits, dtype=np.int64).reshape(-1, qm)
    h = qm // 2

    def axis(first: int) -> np.ndarray:
        x = (1 - 2 * b[:, first + 2 * (h - 1)]).astype(np.float64)
        for j in range(h - 2, -1, -1):
            x = (1 - 2 * b[:, first + 2 * j]) * (2.0 ** (h - 1 - j) - x)
        return x

    inv_norm = math.sqrt(3.0 / (2.0 * ((1 << qm) - 1)))
    return axis(0) * inv_norm + 1j * (axis(1) * inv_norm)


# Two spellings of 1/sqrt(2) differ in the last bit; the reference implementation uses the first for the PDSCH DM-RS and
# the second for QPSK data and for the PDCCH DM-RS.
_INV_SQRT2_PDSCH_DMRS = 1.0 / math.sqrt(2.0)
_INV_SQRT2 = math.sqrt(0.5)


def _r(c: np.ndarray, count: int, scale: float) -> np.ndarray:
    return (1 - 2 * c[0:2 * count:2].astype(np.int64)) * scale + 1j * ((1 - 2 * c[1:2 * count:2].astype(np.int64)) * scale)


def dmrs_cinit_pdsch(slot_in_frame: int, symbol: int, n_id: int, n_scid: int = 0) -> int:
    """TS 38.211 7.4.1.1.1 with N_symb^slot = 14 (lambda-bar = 0, no dmrs-Downlink: nSCID-bar = n_SCID)."""
    return ((1 << 17) * (14 * slot_in_frame + symbol + 1) * (2 * n_id + 1) + 2 * n_id + n_scid) % (1 << 31)


def dmrs_cinit_pdcch(slot_in_frame: int, symbol: int, n_id: int) -> int:
    """TS 38.211 7.4.1.3.1."""
    return ((1 << 17) * (14 * slot_in_frame + symbol + 1) * (2 * n_id + 1) + 2 * n_id) % (1 << 31)


def dmrs_pdsch(slot_in_frame: int, symbol: int, n_id: int, count: int) -> np.ndarray:
    """r(0..count-1) of the PDSCH DM-RS: r(m) = ((1 - 2c(2m)) + j(1 - 2c(2m+1))) / sqrt(2)."""
    return _r(gold(dmrs_cinit_pdsch(slot_in_frame, symbol, n_id), 2 * count), count, _INV_SQRT2_PDSCH_DMRS)


def dmrs_pdcch(slot_in_frame: int, symbol: int, n_id: int, count: int) -> np.ndarray:
    return _r(gold(dmrs_cinit_pdcch(slot_in_frame, symbol, n_id), 2 * count), count, _INV_SQRT2)


def scrambling_cinit_pdsch(rnti: int, n_id: int, codeword: int = 0) -> int:
    """TS 38.211 7.3.1.1: c_init = n_RNTI 2^15 + q 2^14 + n_ID."""
    return (rnti * (1 << 15) + codeword * (1 << 14) + n_id) % (1 << 31)


def scrambling_cinit_pdcch(rnti: int, n_id: int) -> int:
    """TS 38.211 7.3.2.3: c_init = (n_RNTI 2^16 + n_ID) mod 2^31."""
    return (rnti * (1 << 16) + n_id) % (1 << 31)


class PnStreams:
    """Independent PN23 read pointers over the one maximal-length sequence (every stream starts at the all-ones state)."""

    def __init__(self) -> None:
        self._seq = pn23(1 << 16)
        self._pos: dict[object, int] = {}

    def take(self, key: object, n: int) -> np.ndarray:
        start = self._pos.get(key, 0)
        end = start + n
        if end > len(self._seq):
            self._seq = pn23(max(end, 2 * len(self._seq)))
        self._pos[key] = end
        return self._seq[start:end]

    def restart(self) -> None:
        self._pos.clear()
