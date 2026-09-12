"""fixed-point-v1: the primitive operators, the table approximation, the reference and the C99 backend on random
weights (bit-exact; a deliberately altered backend is caught at a sample and a signal)."""

import shutil
from pathlib import Path

import numpy as np
import pytest

from opendpd.core.fixed_point import FixedGRU, lookup, quantise, rescale, saturate, table, to_fixed, to_float, weight_fraction
from opendpd.export import c_backend
from opendpd.schemas.fixed_point import FixedPointSpec, WordFormat


def _state(hidden=6, seed=0, scale=0.7):
    rng = np.random.default_rng(seed)
    return {"backbone.rnn.weight_ih_l0": rng.standard_normal((3 * hidden, 2)) * scale,
            "backbone.rnn.weight_hh_l0": rng.standard_normal((3 * hidden, hidden)) * scale,
            "backbone.rnn.bias_ih_l0": rng.standard_normal(3 * hidden) * 0.2,
            "backbone.rnn.bias_hh_l0": rng.standard_normal(3 * hidden) * 0.2,
            "backbone.fc_out.weight": rng.standard_normal((2, hidden)) * scale,
            "backbone.fc_out.bias": rng.standard_normal(2) * 0.1}


def _inputs(n=500, seed=1, amplitude=9000):
    rng = np.random.default_rng(seed)
    return np.clip((rng.standard_normal((n, 2)) * amplitude).astype(np.int64), -32768, 32767)


def test_rounding_is_half_up_and_saturation_is_per_format():
    assert rescale(np.array([5, -5, 6, -6, 7, -7]), 2, 0).tolist() == [1, -1, 2, -1, 2, -2]   # /4: 1.25, -1.25, 1.5, -1.5, 1.75, -1.75
    assert rescale(np.array([3]), 0, 4).tolist() == [48]
    fmt = WordFormat(bits=16, frac=14)
    assert saturate(np.array([40000, -40000, 12]), fmt).tolist() == [32767, -32768, 12]
    assert to_fixed(np.array([1.0, -1.0, 2.5, 0.00003]), fmt).tolist() == [16384, -16384, 32767, 0]
    assert to_float(np.array([16384]), fmt).tolist() == [1.0]
    assert weight_fraction(0.7, 16) == 15 and weight_fraction(1.0, 16) == 14 and weight_fraction(3.9, 16) == 13
    assert weight_fraction(0.0, 16) == 24 and weight_fraction(1e-9, 16) == 24


def test_the_tables_are_the_functions_at_every_index_and_saturate_at_the_range():
    spec = FixedPointSpec()
    sig, tanh = table(spec.sigmoid), table(spec.tanh)
    assert sig.shape == (4096,) and tanh.shape == (2048,)
    assert sig[2048] == 16384 and tanh[1024] == 0                               # f(0)
    # the ends are f(-range) and f(range - 1/256): the last entry is not a saturated 1.0 but the function there
    assert sig[0] == 11 and sig[-1] == 32757 and tanh[0] == -32746 and tanh[-1] == 32746
    xs = (np.arange(4096) - 2048) / 256
    assert np.max(np.abs(sig / 32768 - 1 / (1 + np.exp(-xs)))) < 1.6e-5
    pre = np.array([0, 1 << 20, -(1 << 20), 100 << 20, -100 << 20])              # 0, +1, -1, far beyond the range
    assert lookup(pre, spec.sigmoid, sig, 20).tolist() == [16384, int(sig[2048 + 256]), int(sig[2048 - 256]), 32757, 11]


def test_quantisation_records_every_tensor_and_refuses_what_the_spec_does_not_cover():
    q = quantise(_state())
    assert {t.name for t in q.tensors} == {"w_ih", "w_hh", "w_out", "b_ih", "b_hh", "b_out"}
    assert all(t.saturated == 0 for t in q.tensors)
    assert q.mac_per_sample == 3 * 6 * (2 + 6) + 2 * 6 and q.table_lookups_per_sample == 18
    assert q.storage_bytes() == {"weight_bytes": (36 + 108 + 12) * 2, "bias_bytes": (36 + 2) * 4, "state_bytes": 12, "table_bytes": 6144 * 2}
    two_layers = dict(_state(), **{"backbone.rnn.weight_ih_l1": np.zeros((18, 6))})
    with pytest.raises(ValueError, match="one GRU layer"):
        quantise(two_layers)
    with pytest.raises(ValueError, match="not a one-layer GRU"):
        quantise({"backbone.rnn.weight_ih_l0": np.zeros((18, 2))})


def test_the_reference_is_deterministic_reset_exact_and_bounded():
    q = quantise(_state())
    x = _inputs()
    a, tr = FixedGRU(q).run(x, trace=True)
    b, _ = FixedGRU(q).run(x)
    assert np.array_equal(a, b) and set(tr) == {"r", "z", "n", "h"} and tr["h"].shape == (500, 6)
    assert np.all(np.abs(a) <= 32767) and np.all(np.abs(tr["h"]) <= 32767)
    two, _ = FixedGRU(q).run(np.concatenate([x[:100], x[:100]]), resets_at=(0, 100))
    assert np.array_equal(two[:100], two[100:])                                     # reset is exact
    carried, _ = FixedGRU(q).run(np.concatenate([x[:100], x[:100]]))
    assert not np.array_equal(carried[:100], carried[100:])                          # and only a reset resets
    narrow = FixedGRU(q)
    narrow._acc_limit = 1 << 20                                                      # a 21-bit accumulator cannot hold 16 x 16-bit products
    with pytest.raises(OverflowError, match="accumulator exceeds"):
        narrow.run(_inputs(amplitude=32000))


@pytest.mark.skipif(c_backend.compiler() is None, reason="no C compiler on PATH")
def test_the_c_backend_is_bit_exact_and_a_broken_one_is_located(tmp_path):
    q = quantise(_state(hidden=9, seed=3))
    cases = [("normal", _inputs(400, 4), ()), ("reset", _inputs(300, 5), (0, 100, 200)), ("extreme", np.full((64, 2), 32767), ())]
    verification, seconds = c_backend.verify(q, cases, tmp_path / "ok")
    assert verification.status == "bit_exact" and verification.cases_checked == 3 and set(seconds) == {"normal", "reset", "extreme"}
    # the same sources with one rounding turned into truncation: caught at the first sample where it matters, in the state
    work = tmp_path / "broken"
    work.mkdir()
    for name, text in c_backend.generate(q).items():
        (work / name).write_text(text.replace("return floor_div_pow2(v + ((int64_t)1 << (s - 1)), s);", "return floor_div_pow2(v, s);"),
                                 encoding="utf-8")
    exe = c_backend.build(work)
    x = cases[0][1]
    ref_y, traces = FixedGRU(q).run(x, trace=True)
    got_y, got_h, _ = c_backend.run_case(exe, x, (), work / "case", q.hidden)
    where = c_backend.first_mismatch(ref_y, traces["h"], got_y, got_h)
    assert where is not None and where[0] == 0 and where[1] == "h"
    assert c_backend.first_mismatch(ref_y, traces["h"], ref_y, traces["h"]) is None
    assert c_backend.first_mismatch(ref_y, traces["h"], ref_y[:-1], traces["h"][:-1]) == (399, "length")


def test_without_a_compiler_the_sources_are_generated_and_the_verification_says_not_run(tmp_path, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    verification, seconds = c_backend.verify(quantise(_state()), [("normal", _inputs(50), ())], tmp_path)
    assert verification.status == "not_run" and "no C compiler" in verification.detail and seconds == {}
    assert (tmp_path / "gru_fixed.c").exists() and (tmp_path / "harness.c").exists()
    assert Path(tmp_path / "gru_fixed.h").read_text().count("#define GRU_HIDDEN 6") == 1
