# Studio 2.2.1 CUDA performance

Measured on 2026-09-13 with the same RTX 4090 Laptop GPU, PyTorch 2.8.0 / CUDA 12.8 production container, 4 CPU quota, 6 GiB RAM limit and 50% CUDA allocator memory ceiling. The memory ceiling does not cap GPU compute to 50%.

The baseline is the released 2.2.0 backend (image `sha256:a90b492d2da3c3de5b16c8c7edee2d3c121ab9f757f339958b549ffe878ff206`). Both versions run the real Studio experiment service, including live observations, validation, test metrics, checkpoint selection and final artifacts.

| Workload | 2.2.0 | 2.2.1 | End-to-end speedup |
| --- | ---: | ---: | ---: |
| PA GRU, 4 epochs | 39.54 s | 10.73 s | 3.69× |
| DPD GRU through frozen GRU PA, 4 epochs | 35.44 s | 15.52 s | 2.28× |

Both pairs use DPA_160MHz, hidden size 23, one recurrent layer, batch size 64, frame length 200, stride 1, seed 0, AdamW and the same learning-rate schedule. Each epoch has 4,605 batches. The DPD pair starts with bitwise-identical PA weights. All final metric values and every tensor in the selected checkpoints were **bitwise identical** between versions. Quick/full preset reductions are not included in these speedup figures: both sides train for four epochs.

These are single paired measurements on a laptop that also drives the desktop, not a guarantee for every model, device, dataset or workload. Initial graph capture and file/model setup are included. After startup, the PA epoch interval decreased from about 9 s to about 2 s. Raw timings and equivalence checks are in [the measurement record](studio-2.2.1-measurements.json). Progress events are deduplicated by epoch when deriving the epoch intervals.

## What changed

- Native GRU, DGRU, LSTM, TRes-GRU and RVTDCNN, including compatible frozen-PA cascades, can replay forward/loss/backward/clipping as a CUDA graph. The existing optimizer and scheduler still run unchanged. No mixed precision, learning-rate changes, frame subsampling or altered batch order are used for acceleration.
- In-memory IQ batches use vectorized row indexing instead of a Python call for every individual sequence. The original DataLoader sampler, random-number consumption and partial final batch are preserved.
- Default plots reuse the epoch's validation predictions. They no longer copy a shadow network or launch an extra 16,384-sample inference every two seconds. Explicit batch previews remain available with a red performance warning.
- Batch previews repack copied cuDNN RNN weights once, avoiding repeated weight-compaction warnings. A web preview event refreshes the latest plot snapshot.

Unsupported models, CPU/MPS, deterministic mode and unsupported capture conditions retain eager training. A short final batch also stays eager. Explicit `cuda_graph_training` continues to use the existing separate opt-in implementation. The internal environment overrides `OPENDPD_DISABLE_CUDA_FAST_PATH=1` and `OPENDPD_DISABLE_BATCHED_LOADER=1` allow diagnostic comparisons. They are operator settings and are not exposed as arbitrary environment input in the public API.

## Validation

CUDA update/gradient/optimizer-state equivalence covers all five supported native model families, frozen-PA cascades, clipping, changing learning rates and incomplete batches. It ran on both production PyTorch 2.8 / CUDA 12.8 and host PyTorch 2.13 / CUDA 13.2. Additional tests cover exact DataLoader values/order/RNG, preview counts, no extra inference in default epoch mode, input validation and the UI warning.

The public deployment keeps its queue, container isolation, resource limits, session authorization and 24-hour cleanup. Performance work does not loosen those boundaries.
