"""Split protocol ``contiguous-v1`` (protected path: changing it changes results).

The capture is cut into three *contiguous* ranges, in time order, with a
guard band of ``guard_samples`` dropped between them. Splitting happens on
raw samples, **before** overlapping training frames are generated, so no
frame (or its memory context) can contain samples from two splits.
"""

from __future__ import annotations

import math
from typing import Dict, Mapping, Tuple

SPLIT_VERSION = "contiguous-v1"
DEFAULT_RATIOS: Dict[str, float] = {"train": 0.6, "val": 0.2, "test": 0.2}
DEFAULT_GUARD_SAMPLES = 256   # >= the longest frame length used by the recipes (200)

Boundaries = Dict[str, Tuple[int, int]]


def contiguous_boundaries(n_samples: int, ratios: Mapping[str, float] = DEFAULT_RATIOS,
                          guard_samples: int = DEFAULT_GUARD_SAMPLES) -> Boundaries:
    """Half-open ``[start, end)`` ranges for train, val and test.

    ``ratios`` apply to the usable length (total minus two guard bands); the
    remainder after integer truncation goes to the test split so no sample is
    lost by rounding.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if guard_samples < 0:
        raise ValueError("guard_samples must be non-negative")
    if set(ratios) != {"train", "val", "test"} or abs(sum(ratios.values()) - 1.0) > 1e-6:
        raise ValueError("ratios must cover train, val and test and sum to 1")
    usable = n_samples - 2 * guard_samples
    if usable < 3:
        raise ValueError(f"{n_samples} samples cannot hold three splits with guard {guard_samples}")
    n_train = math.floor(usable * ratios["train"])
    n_val = math.floor(usable * ratios["val"])
    n_test = usable - n_train - n_val
    if min(n_train, n_val, n_test) < 1:
        raise ValueError("every split needs at least one sample")
    train = (0, n_train)
    val = (n_train + guard_samples, n_train + guard_samples + n_val)
    test = (val[1] + guard_samples, n_samples)
    assert test[1] - test[0] == n_test
    return {"train": train, "val": val, "test": test}


def min_gap(boundaries: Boundaries) -> int:
    """Smallest number of dropped samples between two consecutive splits."""
    ordered = sorted(boundaries.values())
    return min(b[0] - a[1] for a, b in zip(ordered, ordered[1:]))


def context_is_isolated(boundaries: Boundaries, frame_length: int) -> bool:
    """True when no frame of ``frame_length`` in one split can overlap in time with
    the memory context of a frame in another split."""
    return min_gap(boundaries) >= frame_length


def guard_for(frame_length: int) -> int:
    """Guard that isolates frames of ``frame_length`` (rounded up to the default)."""
    return max(DEFAULT_GUARD_SAMPLES, int(frame_length))
