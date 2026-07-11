# TRes-DeltaGRU training-speed benchmark

## Result

In the documented production-shape run, the TRes-DeltaGRU backbone (`B=64`,
`T=200`, `H=15`, `thx=.01`, `thh=.05`, FP32) was **97.1x faster** with the fused implementation:
**164.30 ms -> 1.692 ms per training step**.  The timed step includes forward,
MSE backward, gradient clipping, and AdamW.  The recurrent layer by itself
improved from **176.69 ms -> 0.721 ms** in a separate forward/backward
microbenchmark (245x).

The implementation has the same 999 parameters and unchanged state-dict keys.
It preserves the feature extractor, thresholds, delta state updates, gate
equations, TCN residual, and output layer.  Floating-point reduction order,
FMA contraction, and sigmoid/tanh implementations differ.

This headline applies to dense FP32, first-order training on a supported NVIDIA
CUDA/Triton system. CPU, AMP, QAT, higher-order autograd, and unsupported shapes
take or should take the eager path; see the limitations below.

## Test system and method

| Field | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4090 Laptop GPU, compute capability 8.9 |
| Software | Python 3.13.14, PyTorch 2.13.0+cu132, CUDA 13.2, cuDNN 9.20, Triton 3.7.1 |
| Host | WSL2, Linux 6.6.114.1 |
| Precision | FP32; cuBLAS and cuDNN TF32 disabled |
| Step | `zero_grad(set_to_none=True)`, forward, MSE, backward, clip norm 200, AdamW |
| Data | Seeded synthetic IQ and targets, resident on GPU |
| Timing | 5 warmups; 7 synchronized CUDA-event blocks; median and IQR reported |
| Eager block | 5 steps per timing block |
| Fused block | 100 steps per timing block |

The table reports the IQR of block-average step times. Because eager blocks
average 5 steps and fused blocks average 100, those IQRs describe stability
within each implementation and should not be compared as equal-size samples.
The script runs eager before fused; repeated runs on this power-managed laptop
produced roughly 81x-99x, so the defensible conclusion is "about two orders of
magnitude," not a universal exact 98.5x.

This isolates backbone compute and optimizer cost.  It intentionally excludes
CSV loading, frame construction, DataLoader work, and host-to-device transfer.
The real DPD pipeline also backpropagates through its frozen PA model; that is a
separate cost and can cap end-to-end gains even after this backbone is fast.

## Production-shape result

| Implementation | Median ms/step | IQR of block means | Sequence positions/s | Peak allocated | Speedup |
|---|---:|---:|---:|---:|---:|
| Clean eager reference | 164.302 | 3.184 | 77,905 | 25.3 MiB | 1.00x |
| Persistent Triton | **1.692** | 0.043 | **7,565,624** | 27.5 MiB | **97.11x** |

Raw block means, environment details, and the equivalence summary are archived
in [`results/tres_deltagru_rtx4090_laptop.json`](results/tres_deltagru_rtx4090_laptop.json).

The original implementation was slower than the clean eager reference because
it collected delta statistics and performed two unused hidden-delta reductions
at every timestep.  An exploratory pre-change measurement was about 275 ms for
forward plus MSE backward, but it is not included in the comparable table
because it used a different step boundary.

## Scaling experiments

These shorter runs used five timing blocks (three for `T=1000`). B=16 used
3/100 eager/fused steps per block, B=256 used 3/50, T=50 used 5/100, and T=1000
used 1/20 with two warmups. Their IQRs are omitted because the block settings
differ.

### Batch scaling at T=200

| Batch | Clean eager ms | Fused ms | Speedup |
|---:|---:|---:|---:|
| 16 | 165.060 | 1.694 | 97.45x |
| 64 | 164.302 | 1.692 | 97.11x |
| 256 | 168.910 | 1.939 | 87.12x |

Eager latency is nearly flat across these batch sizes because fixed Python and
kernel-launch overhead dominates, so its throughput still rises with batch.
The persistent-kernel latency is also nearly flat across this measured range.

### Sequence scaling at B=64

| Sequence | Clean eager ms | Fused ms | Speedup |
|---:|---:|---:|---:|
| 50 | 42.754 | 1.727 | 24.75x |
| 200 | 164.302 | 1.692 | 97.11x |
| 1000 | 1,041.712 | 3.657 | 284.86x |

The fused runtime grows with arithmetic rather than with Python and CUDA-launch
overhead, so its advantage increases for longer sequences.

## Numerical-equivalence result

The production-length check used identical weights, `B=8`, the production
thresholds, MSE, and TF32 disabled.

| Quantity | Maximum absolute error | Relative L2 error |
|---|---:|---:|
| Output | 6.08e-6 | 1.34e-6 |
| Input gradient | 1.46e-7 | 1.50e-5 |
| `x2h.weight` gradient | 7.15e-7 | 8.60e-7 |
| `h2h.weight` gradient | 1.07e-7 | 6.16e-7 |
| `fc_out.weight` gradient | 4.77e-7 | 4.48e-7 |
| TCN parameter gradients | <=1.49e-8 | <=1.28e-7 |

The absolute loss difference was `1.19e-7`.  Additional checks covered sequence
lengths 1, 17, 50, 200, and 1000; thresholds `(0,0)` and `(.01,.05)`; custom
initial recurrent states; recurrent-state gradients; delta-statistics equality;
and a 19,662-sample inference sequence.  At `T=1000`, output relative L2 error
was `2.44e-6` and all parameter-gradient relative errors remained below
`2.9e-6`.  These differences come from reduction/activation ordering in FP32.

## What was tried

Rows other than the selected kernel are exploratory scratch experiments; the
committed benchmark directly reproduces the eager-versus-fused full-backbone
rows, while the candidate observations explain why those routes were not kept.

| Candidate | Experiment | Decision |
|---|---|---|
| Remove diagnostic/dead work | Fixed `set_debug(0)`, removed two unused reductions, Python scalar thresholds, and the discarded outer hidden-state allocation. | Kept as output/gradient-equivalent portable cleanup. Diagnostic collection is now explicit. |
| Algebraic held-state rewrite | Mathematically telescoped delta accumulators into held-state projections. About 1.66x faster in the recurrent experiment; FP32 max output error about 5.2e-6. | Useful fallback idea, but far below the fused kernel's gain and not valid across QAT fake-quant boundaries. |
| `torch.compile(..., fullgraph=True, mode="reduce-overhead")` | The 200-step Python loop produced an enormous unrolled graph; compilation did not finish in a practical window on this host. PyTorch's prototype `scan` has no Inductor lowering in this version. | Rejected for this implementation. |
| cuDNN `nn.GRU` fast path | With both thresholds exactly zero, the delta accumulators telescope to a bias-free GRU. RNN-only: about 178.2 -> 1.12 ms (159x), max FP32 output error 5.84e-6 with TF32 off. | Excellent special case, but it cannot represent the production nonzero thresholds. The persistent kernel is faster here and supports both cases. |
| BF16/TF32 | Lower precision can accelerate dense GEMMs, but this tiny cell is launch-bound and threshold decisions can change near the boundary. | Not needed; fused FP32 already gives the large gain while minimizing trajectory changes. |
| Persistent custom recurrence | One Triton program owns each batch sequence; a reverse program implements BPTT. | **Selected.** Highest speed and supports nonzero thresholds. |

## Implementation notes

The forward kernel keeps `x_p`, `h`, `h_p`, `dm`, and `dm_nh` local while it
scans time.  It fuses thresholding, both small projections, accumulator updates,
sigmoid/tanh gates, and the hidden update.  It saves only the intermediates
needed by BPTT.

The backward kernel scans time in reverse and emits input gradients plus the two
linear-output adjoints.  Two final matrix multiplications reduce weight
gradients over batch and time.  Inference under `no_grad` or `inference_mode`
uses a forward-only path and does not allocate the training workspace. Initial
Triton compilation is cache-dependent and took roughly 0.6 seconds on this
machine; it is a one-time cost per kernel specialization and is excluded from
steady-state timings.

The fused path currently dispatches only for:

- NVIDIA CUDA with Triton available;
- FP32 dense-float modules;
- one recurrent layer;
- input width up to 16 and hidden width up to 32.

CPU, unsupported shapes/dtypes, multi-layer cells, and quantization-aware models
use the cleaned eager recurrence.  QAT deliberately stays eager because its
fake-quantized Linear/Sigmoid/Tanh/Add/Mul boundaries are part of that model's
actual computation; bypassing or reassociating them would not be equivalent.
Temporal sparsity counters are disabled by default; pass
`--collect_delta_stats` to restore `SP_T_DX`, `SP_T_DH`, `SP_T_DV`, and
`HW_PARAM` training-log fields when that diagnostic overhead is wanted.
CUDA autocast deliberately falls back to eager so AMP keeps its original
fake/mixed-precision boundaries. The custom backward is explicitly
first-order-only; use `OPENDPD_DISABLE_TRITON_DELTAGRU=1` for higher-order
autograd, JVP/vmap-style transforms, or gradient-penalty/meta-learning code.

## Reproduction

```bash
.venv/bin/python benchmark/benchmark_tres_deltagru.py
```

Useful variants:

```bash
.venv/bin/python benchmark/benchmark_tres_deltagru.py --batch-size 256
.venv/bin/python benchmark/benchmark_tres_deltagru.py --sequence-length 1000 \
  --blocks 3 --reference-steps-per-block 1 --fused-steps-per-block 20
```

Set `OPENDPD_DISABLE_TRITON_DELTAGRU=1` for normal application runs that should
force the portable eager path.  The benchmark itself explicitly chooses each
implementation so it can compare them in one process.
