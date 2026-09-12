# Deployment export: a bit-exact fixed-point package

A trained GRU is a float model. A deployment runs integers. This tutorial
exports a finished run under `fixed-point-v1`: quantised weights, the six
golden vectors, a C99 reference verified bit for bit against the software
reference, and a report whose numbers say how they were obtained.

Every command below is executed by `tests/integration/test_docs_commands.py`.

## 1. Which runs can be exported

`opendpd models --json` lists `export_formats` per model: `fixed-point-v1`
for `gru` (one layer, executed as `gru_stream`). A run of another model is
refused with the reason, in the CLI, the API and the Studio.

## 2. Export

Train a PA as in the headless tutorial (`pa-gru-smoke-v1`), then:

```bash
opendpd deploy <pa-run-id> --workspace ws --out deploy.zip
```

The command quantises the weights (per-tensor fractions, saturation counts
recorded), writes the golden vectors with the state after every sample,
generates and compiles the C99 reference, replays every vector through it and
compares every output sample and every state step with the software
reference. It prints the verdict (`bit_exact`, or the case, sample and signal
of the first mismatch, or `not_run` without a compiler), the float-to-fixed
loss per metric of the run's profile, and the resources with their labels.

In the Studio the same export is the result page's **Deployment** panel: the
button is offered when the model has the format; otherwise the panel says
why not.

## 3. Read the package

```
manifest.json    spec, weight formats, golden index, verification, report, hash per file
spec.json        every format and rule of fixed-point-v1
weights.json     integer weights, biases and tables, with their fractions
README.md        the report in Markdown
c/               gru_fixed.h, gru_fixed.c, harness.c
golden/<case>/   x.i16, y.i16, h_final.i16, h_trace.i16, meta.json
```

To check an implementation of your own: replay `golden/<case>/x.i16` and
compare your outputs and states with `y.i16` and `h_trace.i16`; the first
differing sample and signal locate the fault. The specification, the
verification rule and the approval record are in
`docs/protocols/fixed-point-v1.md`.
