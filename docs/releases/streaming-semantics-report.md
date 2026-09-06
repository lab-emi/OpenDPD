# Execution semantics report (`streaming-v1`, S18)

What the streaming variants do, measured; and what every other model
declares about its temporal context. Numbers come from
`tests/unit/test_streaming.py`, `tests/integration/test_streaming_eval.py`
and one CPU run of the smoke recipes over a synthetic capture (40 000
samples, test split 7 975 samples, 800 MS/s); they are properties of the
contract and of the variants, not benchmark scores.

## 1. Verified variants

| Variant | Executes | State carried | Look-ahead | Warm-up (measured, tolerance 1e-4) | Status |
|---|---|---|---|---|---|
| `gru_stream` | `gru` weights (`nn.GRU` hidden 23, 1 layer + linear head) | hidden state | 0 samples | 32 samples | experimental |
| `gmp_stream` | `gmp` weights (memory 11, degree 5) | last 20 inputs | 0 samples | 19–20 samples (signal dependent: the farthest lags are the weakest terms) | experimental |

The GMP's reach was first declared as 10 samples (memory length minus one)
and the measured warm-up of 20 showed the envelope windows lag the signal
windows by another memory length; the adapter now carries 20 samples. This
is the reason the contract measures warm-up instead of reading it off a
parameter.

## 2. Chunk consistency

Maximum |streamed − full sequence| over the valid range, the same variant
from the same reset, float32 arithmetic (tolerance 1e-4):

| Chunk size | `gru_stream` | `gmp_stream` |
|---|---|---|
| 1 | 1.2e-7 | 3.0e-8 |
| 16 | 8.9e-8 | 7.5e-9 |
| 128 | 1.2e-7 | 1.9e-9 |
| 1024 | 8.9e-8 | 0 |
| 8192 (> signal) | 0 | 0 |
| 7975 (= signal) | 0 | 0 |
| cycled 3, 50, 1000 | 8.9e-8 | 0 |

Chunk size 1 within tolerance means no output used a sample that had not
arrived: both variants are causal in execution, not only by declaration.
Every streaming result stores this check for its own chunk size
(`execution.consistency`); the unit tests additionally show that a per-chunk
reset (the offline segment semantics) is a different signal, that changing
the current chunk's tail leaves its head untouched while changing the previous
chunk's tail changes it, that `get_state` / `set_state` reproduce a chunk
exactly, and that the look-ahead buffer and `flush` produce the same outputs
as one offline pass with the same zero padding.

## 3. Cost of the reference implementation

The Python reference runs `gru_stream` at about 27 µs per sample and
`gmp_stream` at about 44 µs per sample on one CPU core (chunk 1024,
`torch` inference mode, no batching). This is the cost of the reference used
to define the semantics; it says nothing about the latency or throughput of
a deployed implementation, which are measured on that implementation.

## 4. Look-ahead of every registered model

`lookahead_samples` is the number of future samples an output needs (the
buffering a real-time implementation cannot avoid); the time is that number
at the dataset's rate. It is not a latency.

| Model | Look-ahead (samples) | at 800 MS/s | at 983.04 MS/s | Streaming variant |
|---|---|---|---|---|
| `gru`, `lstm`, `vdlstm`, `dgru`, `pgjanet`, `dvrjanet`, `deltagru`, `deltajanet`, `qgru`, `qgru_amp1`, `bojanet`, `apnrru` | 0 | 0 | 0 | `gru_stream` for `gru`; none for the others |
| `gmp` | 0 (20-sample past window) | 0 | 0 | `gmp_stream` |
| `mp_ls` | 0 (Q−1 past samples, zero-filled per segment) | 0 | 0 | none |
| `gmp_ls` | Mc (leading envelope terms; 0 when Kc = 0), recorded per result | Mc / 800 MS/s | Mc / 983.04 MS/s | none |
| `tres_gru`, `tres_deltagru` | 16 | 20 ns | 16.3 ns | none |
| `tcn` | 30 | 37.5 ns | 30.5 ns | none |
| `rvtdcnn`, `mcldnn` | not characterised | — | — | none |

A streaming variant of a look-ahead model would hold back that many samples
(`WindowedStream` with `lookahead_samples` declared) and end the stream with
zeros, which is not the offline symmetric padding at segment ends: one
more reason it is a new entry with its own evidence.

## 5. What this report does not claim

- No variant is `supported`: promotion, and any change to a model's
  causality, is a maintainer decision (plan §10).
- The smoke weights behind the numbers are not trained models; the
  properties measured (consistency, warm-up, reach) do not depend on training
  quality, the scores do and are not reported here.
- No implementation latency was measured.
