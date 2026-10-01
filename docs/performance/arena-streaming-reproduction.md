# Arena v6 measured-input streaming reproduction

All 15 stateful cases were replayed from their frozen v6 checkpoints. The auditor compares 200- and 137-sample DPD chunks, retains the recorded PA device, checks IQ continuity and hashes, and independently recomputes complete-symbol EVM.

| DPD | Dataset | Cases | Stored metric max difference (dB) | Chunk max difference (dB) | Chunk max IQ difference |
|---|---|---:|---:|---:|---:|
| gmp_stream | apa-200mhz-b | 3 | 0 | 0 | 0 |
| gru_stream | apa-200mhz-b | 12 | 0 | 9.7792701e-06 | 2.3841858e-07 |

Limits are unchanged: 0.001 dB for metric replay and 1e-4 for chunk IQ. All inputs are original measured test samples. CPU and CUDA kernels are not assumed bit-equivalent.

[Per-case evidence](arena-streaming-reproduction.json) · [Complete audit](arena-reference-audit.json)

Protocol SHA-256: `646c31e0546b6a6754e0ad06c381aed21989dcf4ffa2c52bbb6321db67065b51`.

Training SHA-256: `71294fba81c968d2762f5349178cba8f23a38a8aad942b530269c783b9d9bf94`.
