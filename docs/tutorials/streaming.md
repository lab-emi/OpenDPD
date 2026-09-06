# Streaming semantics: score trained weights as a stream

An offline result scores fixed segments that each start from a zero state. A
deployed model sees one stream and carries its state across blocks. This
tutorial re-scores a finished run under its registered streaming variant and
reads the evidence that comes with the number.

Every command below is executed by `tests/integration/test_docs_commands.py`.

## 1. Which models have a streaming variant

```bash
opendpd models --json
```

Entries with `execution_semantics: "streaming_stateful"` name in
`weights_from` the offline key whose weights they execute: `gru_stream` for
`gru`, `gmp_stream` for `gmp`. Models with a look-ahead (`tres_gru`,
`tres_deltagru`, `tcn`) have none yet; the registry states their look-ahead in
samples, and every result of theirs shows it.

## 2. Score a PA run as a stream

Train a PA as in the headless tutorial (`pa-gru-smoke-v1`), then:

```bash
opendpd stream <pa-run-id> --workspace ws --chunk 512
```

This submits an `evaluate_pa` run with model `gru_stream` bound to the PA
run's weights, feeds the test split in chunks of 512 samples with the hidden
state carried across chunks, and prints:

- the chunk-consistency check: the maximum difference between the streamed
  outputs and the same variant run over the whole split in one chunk, against
  the contract's tolerance (`1e-4`);
- the look-ahead in samples and the warm-up measured on this signal;
- the metrics, and the limitations, the first of which says that the result is
  not comparable with the offline `gru` result and was not inherited from it.

A DPD run streams the same way (`run_dpd` with `gru_stream` through the
training surrogate; the surrogate itself stays the offline module because it
is not the model under evaluation). In the Studio the same action is the run
page's **Score under streaming semantics** button, offered only when the
registry names a variant; the result page shows the **Execution** panel with
the chunk, look-ahead (samples and µs), history, warm-up and the consistency
verdict.

## 3. What the look-ahead is, and is not

`lookahead_samples` is the number of future samples an output needs, an
information bound of the model; `lookahead_s` converts it at the dataset's
sample rate. Neither is the latency of an implementation, which only a
measurement of that implementation gives. The contract, the definitions and
the way to add a variant are in `docs/architecture/streaming.md`; the
evidence per variant in `docs/releases/streaming-semantics-report.md`.
