"""Explicit bit sources and normalized digital modulation alphabets."""
import math
import numpy as np
from scipy import signal


class Payload:
    def __init__(self, config, rng):
        self.mode, self.rng, self.cursor = config.payload_mode, rng, 0
        self.sequence = np.array([int(v) for v in config.payload_bits], dtype=np.uint8)
        if self.mode in ("prbs9", "prbs15"):
            degree = 9 if self.mode == "prbs9" else 15
            taps = [5] if degree == 9 else [14]
            self.sequence = signal.max_len_seq(degree, taps=taps)[0]

    def integers(self, order, count):
        if self.mode == "random":
            return self.rng.integers(0, order, count)
        width = int(math.log2(order))
        indices = (self.cursor + np.arange(count * width)) % len(self.sequence)
        bits = self.sequence[indices].reshape(count, width)
        self.cursor += count * width
        return bits @ (1 << np.arange(width - 1, -1, -1))


def _gray_decode(values):
    result = values.copy()
    shifted = values >> 1
    while np.any(shifted):
        result ^= shifted
        shifted >>= 1
    return result


def qam_symbols(payload, order, count):
    labels = payload.integers(order, count)
    if order == 2:
        return (2 * labels - 1).astype(complex)
    side = math.isqrt(order)
    real = 2 * _gray_decode(labels // side) - side + 1
    imag = side - 1 - 2 * _gray_decode(labels % side)
    return (real + 1j * imag) / math.sqrt(2 * (order - 1) / 3)


def psk_symbols(payload, order, count):
    return np.exp(2j * np.pi * _gray_decode(payload.integers(order, count)) / order)
