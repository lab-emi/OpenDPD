# Fixed-point specification `fixed-point-v1`: the bit-exact bridge, S19

Status: specification implemented as a software reference
(`opendpd/core/fixed_point.py`) and a C99 backend
(`opendpd/export/c_backend.py`) that is verified bit for bit on every golden
vector of every package. **The fixed-point rules below are pending a
maintainer's approval** (plan §10 lists fixed-point rules among the items
that need human approval); until then packages are research exports, not a
release format. Changing any rule is a new specification id.

## 1. Scope

One model, one precision scheme: the one-layer GRU (`gru`) executed under the
streaming semantics of `gru_stream` (state carried across samples; see
`docs/architecture/streaming.md`). Nothing else is covered: a run of another
model, or a GRU with more than one layer, is refused with the reason
(`opendpd/services/deploy.py::support`, `quantise`).

## 2. Formats

| Quantity | Word | Fraction | Range | Where |
|---|---|---|---|---|
| input `x` (I, Q) | 16-bit | 14 | ±2 | `spec.x` |
| state `h` and gate values `r`, `z`, `n` | 16-bit | 15 | (−1, 1) | `spec.h`; tables return this format |
| output `y` (I, Q) | 16-bit | 14 | ±2 | `spec.y` |
| weights `W_ih`, `W_hh`, `W_out` | 16-bit | per tensor: the largest fraction under which max |w| fits the word (`weight_fraction`), recorded in the package with the count of saturated values (0 for a sound export) | | `TensorFormat` |
| biases `b_ih`, `b_hh`, `b_out` | 32-bit | 20 (the pre-activation fraction) | ±2048 | `spec.pre` |
| pre-activation | 32-bit | 20 | ±2048 | `spec.pre` |
| accumulator | 48-bit declared; exact integer dot products | | bound checked: N·2^15·2^15 < 2^47 for N < 2^17 terms; the reference raises, never wraps | `spec.accumulator_bits` |
| table index | `index_frac` = 8 (one entry per 1/256) | | sigmoid over [−8, 8): 4096 entries; tanh over [−4, 4): 2048 entries | `spec.sigmoid`, `spec.tanh` |

## 3. Operators, in order, for one sample

With `rescale(v, f_from, f_to)` = exact left shift when `f_to ≥ f_from`,
otherwise `floor((v + 2^(s−1)) / 2^s)` with `s = f_from − f_to` (round half
up), `sat_b` = saturation to `b` bits, `LUT_f(pre)` = table entry at
`clamp(rescale(pre, 20, 8), −range·256, range·256 − 1) + range·256`:

1. `acc_i[g] = Σ_j W_ih[g, j] · x[j]` and `acc_h[g] = Σ_j W_hh[g, j] · h[j]` for every gate row `g` (exact);
2. `a_i[g] = sat_32(rescale(acc_i[g], f_ih + 14, 20) + b_ih[g])`, `a_h[g] = sat_32(rescale(acc_h[g], f_hh + 15, 20) + b_hh[g])`;
3. `r = LUT_sigmoid(sat_32(a_i[r] + a_h[r]))`, `z = LUT_sigmoid(sat_32(a_i[z] + a_h[z]))`;
4. `t = rescale(r · a_h[n], 15 + 20, 20)`, `n = LUT_tanh(sat_32(a_i[n] + t))`;
5. `h' = sat_16(rescale((2^15 − z) · n + z · h, 30, 15))`;
6. `y[k] = sat_16(rescale(sat_32(rescale(Σ_j W_out[k, j] · h'[j], f_out + 15, 20) + b_out[k]), 20, 14))`;
7. `h ← h'`.

Gate order is PyTorch's (`r`, `z`, `n`); `reset` sets `h = 0`. Every
stored quantity saturates; the accumulator never does (its bound is a
theorem of the widths, checked at run time). There is no interpolation in
the tables and no other approximation. Float-to-integer conversion of the
weights, biases and table values rounds half away from zero; every rescale
inside the step rounds half up (floor of `v + half`), which the C backend
implements with a division so that no arithmetic-shift behaviour is assumed.

## 4. Golden vectors

Every package carries six vectors, each with inputs, expected outputs, the
final state and the state after every sample (`h_trace.i16`), all hashed:

| Case | Content | What it checks |
|---|---|---|
| `normal` | the first 4096 samples of the test split | the arithmetic on real data |
| `extreme` | constant +max, constant −max, alternating full scale | input saturation and the largest pre-activations |
| `saturation` | random full-scale signs | table-range clamping and the state at its bounds |
| `all_zero` | zeros | the bias path; the state stays consistent |
| `state_reset` | one block four times with a reset before each | reset exactness: the four outputs are identical (the reference asserts it) |
| `long_sequence` | 65 536 samples | no drift, no overflow over a long stream |

## 5. Verification

`c_backend.verify` generates the sources with the package's weights and
tables, compiles them with the system compiler, replays every vector and
compares **every output sample and every state step** with the software
reference recomputed at that moment. The result is `bit_exact`, `mismatch`
(with the case, the first differing sample and the signal, `h` before `y`
because `y` is computed from it) or `not_run` (no compiler: the sources are
still in the package, unverified). A metric difference is never the
evidence: `tests/unit/test_fixed_point.py` shows a backend with one rounding
turned into truncation being located at sample 0 in `h`.

## 6. The report and its labels

| Label | Content | Never |
|---|---|---|
| quality loss | the run's metric profile on the float streaming model and on the fixed-point reference over the test split (the fixed model sees quantised inputs), metric by metric with the difference | mixed with the offline float result |
| `theoretical` | MAC per sample, table lookups per sample, weight / bias / state / table bytes from the shapes and the spec; sparsity: none exploited | presented as measured |
| `measured_execution_time` | the compiled C reference's samples per second on the machine that built the package | presented as a deployment number |
| `synthesis_estimate` | not available: nothing was synthesised | inferred |
| `measured_power` | not available: nothing was measured; energy is never inferred from MAC or parameter counts | inferred |
| execution assumptions | sequential one sample per step, state carried, floor shifts by division, exact accumulation, no parallelism or sparsity, input already in `spec.x` | omitted |

## 7. Backends

One backend is verified: portable C99 (`c/gru_fixed.h`, `c/gru_fixed.c`,
`c/harness.c`). ONNX, HLS and RTL are not provided and not promised; an
implementer of any of them verifies against the same golden vectors and
trace with the same rule (first differing sample and signal).

## 8. Approval record

| Item | Status |
|---|---|
| fixed-point rules of §2–§3 approved by a maintainer | **pending human** |
| a target other than the C99 reference verified on the golden vectors | none |
