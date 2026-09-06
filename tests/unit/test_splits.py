"""contiguous-v1: splits are contiguous, time ordered and guard-isolated."""

import pytest

from opendpd.core.splits import DEFAULT_GUARD_SAMPLES, context_is_isolated, contiguous_boundaries, guard_for, min_gap


def test_boundaries_are_contiguous_in_time_and_guarded():
    b = contiguous_boundaries(20_000, {"train": 0.6, "val": 0.2, "test": 0.2}, guard_samples=256)
    assert b["train"][0] == 0 and b["test"][1] == 20_000
    assert b["train"][1] < b["val"][0] < b["val"][1] < b["test"][0]
    assert min_gap(b) == 256
    usable = 20_000 - 2 * 256
    assert (b["train"][1] - b["train"][0]) == int(usable * 0.6)
    assert sum(e - s for s, e in b.values()) == usable


def test_no_frame_context_crosses_a_boundary():
    b = contiguous_boundaries(20_000, guard_samples=DEFAULT_GUARD_SAMPLES)
    for frame_length in (50, 200, 256):
        assert context_is_isolated(b, frame_length)
    assert not context_is_isolated(b, 257)
    assert guard_for(200) == DEFAULT_GUARD_SAMPLES and guard_for(1000) == 1000


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError):
        contiguous_boundaries(100, guard_samples=60)
    with pytest.raises(ValueError):
        contiguous_boundaries(1000, {"train": 0.5, "val": 0.5, "test": 0.5})
    with pytest.raises(ValueError):
        contiguous_boundaries(0)
