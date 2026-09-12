"""streaming-v1 contract: chunk consistency, state carried (never a circular shift), look-ahead buffering, flush,
warm-up measurement. Synthetic models with known arithmetic; the trained variants are covered in
tests/integration/test_streaming_eval.py."""

import numpy as np
import pytest

from opendpd.core.streaming import (
    CONSISTENCY_TOLERANCE,
    RecurrentStream,
    StreamSpec,
    WindowedStream,
    chunk_consistency,
    measure_warmup,
    run_stream,
)


class LeakyIntegrator(RecurrentStream):
    """y[t] = a * y[t-1] + x[t]: a causal recurrence whose state is one output sample."""

    def __init__(self, a=0.9, warmup_samples=None):
        super().__init__(warmup_samples=warmup_samples)
        self.a = a

    def _run(self, x, h):
        y = np.zeros_like(x)
        prev = np.zeros(2, dtype=np.float32) if h is None else h
        for t in range(x.shape[0]):
            prev = self.a * prev + x[t]
            y[t] = prev
        return y, prev.copy()


class Fir(WindowedStream):
    """y[t] = sum_k c[k] * x[t - history + k]: taps over history past samples, the current one and lookahead future ones."""

    def __init__(self, taps, history, lookahead):
        super().__init__(history_samples=history, lookahead_samples=lookahead)
        self.taps = np.asarray(taps, dtype=np.float32)
        assert len(self.taps) == history + 1 + lookahead

    def _run(self, block):
        h, la = self.spec.history_samples, self.spec.lookahead_samples
        padded = np.concatenate([np.zeros((h, 2), np.float32), block, np.zeros((la, 2), np.float32)])
        return np.stack([sum(self.taps[k] * padded[t + k] for k in range(len(self.taps))) for t in range(block.shape[0])])


def _signal(n=500, seed=0):
    return np.random.default_rng(seed).standard_normal((n, 2)).astype(np.float32)


def test_every_chunking_matches_the_full_sequence_run():
    x = _signal()
    for model in (LeakyIntegrator(), Fir([0.2, -0.5, 1.0, 0.3, 0.1], history=2, lookahead=2)):
        check = chunk_consistency(model, x, [1, 7, 64, 500, 1000, [3, 50, 1]])
        assert check["within_tolerance"], check
        assert set(check["max_abs_error"]) == {"1", "7", "64", "500", "1000", "3+50+1"}
        assert max(check["max_abs_error"].values()) <= CONSISTENCY_TOLERANCE


def test_the_state_carried_is_the_previous_chunks_tail_never_this_chunks():
    x = _signal(200)
    model = LeakyIntegrator()
    reference = run_stream(model, x, 200)
    # a per-chunk reset (the offline segment semantics) is a different signal at every boundary after the first
    model.reset()
    reset_each = np.concatenate([(model.reset(), model.chunk(x[i:i + 50]))[1] for i in range(0, 200, 50)])
    assert np.max(np.abs(reset_each[50:] - reference[50:])) > 0.1
    # changing this chunk's tail leaves its head untouched; changing the previous chunk's tail changes this head
    altered = x.copy()
    altered[99] += 5.0
    y_tail = run_stream(model, altered, 50)
    assert np.array_equal(y_tail[50:99], reference[50:99]) and abs(y_tail[100, 0] - reference[100, 0]) > 1.0
    for model_ in (LeakyIntegrator(), Fir([0.5, 1.0, 0.25], history=1, lookahead=1)):
        model_.reset()
        first = model_.chunk(x[:50])
        state = model_.get_state()
        second = np.concatenate([model_.chunk(x[50:]), model_.flush()])
        model_.set_state(state)
        again = np.concatenate([model_.chunk(x[50:]), model_.flush()])
        assert np.array_equal(second, again) and first.shape[0] + second.shape[0] == 200


def test_lookahead_holds_back_outputs_until_the_future_arrives_and_flush_finalises_the_tail():
    x = _signal(20)
    fir = Fir([1.0, 2.0, 4.0, 8.0], history=1, lookahead=2)
    fir.reset()
    assert fir.chunk(x[:1]).shape[0] == 0 and fir.chunk(x[1:2]).shape[0] == 0     # 2 inputs pending, none finalised
    out = fir.chunk(x[2:5])                                                        # 5 in, 3 out: y[0..2]
    assert out.shape[0] == 3
    expected0 = 1.0 * 0 + 2.0 * x[0] + 4.0 * x[1] + 8.0 * x[2]                     # zero history before the stream
    assert np.allclose(out[0], expected0, atol=1e-6)
    tail = fir.flush()                                                             # y[3], y[4] with zero future
    assert tail.shape[0] == 2 and np.allclose(tail[1], 1.0 * x[3] + 2.0 * x[4], atol=1e-6)
    assert fir.flush().shape[0] == 0
    assert fir.spec.latency_s(800e6) == pytest.approx(2 / 800e6)
    full = run_stream(fir, x, [1, 2, 5])
    offline = fir._run(x)                                                          # the same zero padding at both ends
    assert np.allclose(full, offline, atol=1e-6)


def test_step_is_a_chunk_of_one():
    model = LeakyIntegrator()
    model.reset()
    outs = np.concatenate([model.step(s) for s in _signal(30)])
    assert np.allclose(outs, run_stream(model, _signal(30), 30), atol=1e-6)


def test_warmup_is_measured_not_assumed():
    x = _signal(400)
    assert measure_warmup(Fir([0.5, 1.0], history=1, lookahead=0), x) == 1        # one sample of zero history
    assert measure_warmup(Fir([1.0, 0.5], history=0, lookahead=1), x) == 0        # no past: nothing to warm up
    warm = measure_warmup(LeakyIntegrator(a=0.5), x)
    assert 8 <= warm <= 40                                                         # 0.5^k decays below 1e-4 near k = 14..17
    assert StreamSpec(state="recurrent", warmup_samples=warm).valid_start() == warm


def test_a_stream_that_loses_or_invents_samples_is_refused():
    class Broken(LeakyIntegrator):
        def _run(self, x, h):
            y, h = super()._run(x, h)
            return y[:-1] if x.shape[0] > 1 else y, h

    with pytest.raises(RuntimeError):
        run_stream(Broken(), _signal(10), 5)
    with pytest.raises(ValueError):
        run_stream(LeakyIntegrator(), _signal(10), 0)
