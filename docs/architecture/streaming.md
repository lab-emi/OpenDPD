# Streaming execution contract (`streaming-v1`, S18)

Status: contract implemented (`opendpd/core/streaming.py`) with two verified
variants, `gru_stream` and `gmp_stream`, both **experimental**. Promoting a
variant to `supported`, and any change to a model's causality, is a
maintainer decision (plan §10: model causality needs human approval); the
report in `docs/releases/streaming-semantics-report.md` records the evidence
that decision would rest on.

## 1. Offline segments versus a stream

Every trained model in OpenDPD is scored offline on fixed segments of
`nperseg` samples that each start from a zero state (`execution_semantics =
offline_segmented` in the registry). A deployed DPD does not see segments: it
sees one stream, and whatever it carries between blocks of samples decides
what it computes at every block boundary. A streaming variant of the same
weights therefore produces a different signal, and:

- it is a **separate registry entry** (`weights_from` names the offline key)
  with `execution_semantics = streaming_stateful`;
- it is never trained (`train_pa` / `train_dpd` with a variant key is
  refused); its weights come from a finished run of the offline key through
  `evaluate_pa` (PA) or `run_dpd` (DPD, through the training surrogate);
- its results carry an `execution` block and are never ranked against the
  offline key's results: the comparison key includes the execution semantics
  (`opendpd/core/metrics/compare.py`), so nothing is inherited from the
  offline benchmark.

## 2. The interface

```
reset()                 forget everything: zero state, zero history, nothing pending
get_state() / set_state(s)   a copy of what chunk() carries between calls
chunk(x) -> y           consume (n, 2) I/Q samples; return every output that is now final
flush() -> y            finalise what the look-ahead buffer still holds (empty for a causal model)
step(sample)            chunk() of one sample
```

`run_stream(model, x, chunk_samples)` feeds a signal in chunks of one size (or
a cycled list of sizes), flushes, and checks that exactly one output came back
per input. Two base classes implement the buffering so a variant only maps a
block of samples to outputs:

| Base | Carries | Variant declares |
|---|---|---|
| `RecurrentStream` | the recurrent state after the last sample | `_run(x, h) -> (y, h_next)` |
| `WindowedStream` | the last `history_samples` inputs and up to `lookahead_samples` pending inputs | `_run(block) -> y` for a contiguous block, zero-padded outside it |

## 3. Semantics every variant declares (`StreamSpec`)

| Term | Definition | Where it shows |
|---|---|---|
| `lookahead_samples` | future samples an output needs. The runner holds back that many inputs: `y[t]` is final once `x[t + lookahead]` arrived. | registry `lookahead_samples`; result `execution.lookahead_samples` and `lookahead_s = lookahead / sample rate` |
| latency | **not** defined by this contract. The look-ahead is an information bound; the latency of an implementation is a measurement of that implementation. Every result says so (the result page and the tutorial). | |
| `history_samples` | past inputs a window model keeps between chunks (zero after `reset`) | `execution.history_samples` |
| `warmup_samples` | leading outputs after `reset` that differ from an uninterrupted stream by more than the tolerance; **measured** on the evaluated signal by `measure_warmup`, never assumed | `execution.warmup_samples` |
| valid range | every output from `warmup_samples` on; the chunk-consistency check uses it | `chunk_consistency(...)["valid_from"]` |
| tail | `flush` feeds zeros for the missing future samples (the one tail policy of v1; a variant that needs another is a v2 of the contract) | `flush()` |
| padding | a window model's own zero padding applies only outside the stream; inside it the history buffer replaces the padding | `WindowedStream` |
| state reset | only `reset()` resets; a chunk boundary never does (the offline segment reset is exactly what a stream does not do) | `test_the_state_carried_is_the_previous_chunks_tail_never_this_chunks` |

## 4. Chunk consistency

`chunk_consistency(model, x, chunk_sizes)` runs the same variant from the
same reset once over the whole signal and once per chunking and reports the
maximum absolute difference over the valid range against `tolerance =
1e-4` (normalised units, float32 arithmetic). The reference is the streaming
variant itself: an offline segment score is not a reference for a stream.
Every streaming result records the check for its own chunk size
(`execution.consistency`); a failure is a limitation on the result, not a
silent number. The unit tests add sizes 1, 7, 64, the whole signal, more than
the signal, and a cycled mix, and a check that altering the tail of the
current chunk leaves its head untouched while altering the previous chunk's
tail changes it (no circular shift).

## 5. Variants

| Key | Executes | State carried | Look-ahead | Verified by |
|---|---|---|---|---|
| `gru_stream` | `gru` (`backbones/gru.py`: `nn.GRU` + linear head) | hidden state | 0 | `tests/unit/test_streaming.py`, `tests/integration/test_streaming_eval.py` |
| `gmp_stream` | `gmp` (`backbones/gmp.py`, memory 11) | the last 10 inputs (zero-filled per segment offline) | 0 | same |

Models with a look-ahead (`tres_gru`, `tres_deltagru`: 16 samples; `tcn`: 30)
have no streaming variant yet. Their buffering cost is stated in the registry
and in the semantics report; a variant would use `WindowedStream` with the
look-ahead declared, and `flush` would end the stream with `zero_pad`, which
differs from the offline symmetric padding at segment ends and is one more
reason such a variant is a new entry.

## 6. Adding a variant

1. Write the adapter (`opendpd/services/streaming.py`, `ADAPTERS`) over the
   legacy module that holds the weights, on one of the two base classes.
2. Register the key with `weights_from`, `execution_semantics =
   STREAMING`, the look-ahead and its note, `status = "experimental"` and the
   tests as evidence.
3. Add it to the unit and integration tests (chunk consistency, state carry,
   warm-up) and to the semantics report with its numbers.
4. Do not touch the offline key: its behaviour and scores stay as they are.
