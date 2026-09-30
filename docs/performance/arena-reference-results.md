# OpenDPD 2.3 Arena reference results

Arena uses only APA_200MHz_b. One frozen TRes-GRU PA is shared by all DPD architectures. PA and DPD train, validation and test inputs are unchanged slices of original measured captures. ILC-based entries are excluded. MP/GMP least-squares models use training-only PA feedback, not ILC data. EVM is measured on complete held-out symbols and ACLR on the emitted PA output. These are fixed-PA simulations, not hardware measurements.

Every neural configuration is retrained for 240 complete passes over all training windows, batch 64 and stride 1. These v6 results use the same full-window recipe throughout; a fixed budget still does not establish each architecture's optimum.

23 backbone/board entries; 233 completed condition/seed/configuration cases.

```text
q = ½ · (ΔEVM + ΔACLR)
FoM = mean(q) − 5 log10(P/1000) − 5 log10(OPs/2000)
```

Every size uses the same cost reference. Three seeds report mean and standard deviation; the main score does not subtract the standard deviation. Require output power within ±0.5 dB for every case and positive mean quality to rank. Negative FoM is preserved; unavailable configurations are not zero scores. Costs include DPD only, with OPs = MUL + ADD at the published nonlinear reference price.

## Frozen PA models

| Condition | Parameters | Validation NMSE (dB) | Test NMSE (dB) |
|---|---:|---:|---:|
| apa-200mhz-b | 4871 | -40.0727 | -39.7183 |

The PA is selected by validation NMSE only. Its held-out test NMSE is reported after selection and is not a hard EVM or ACLR ceiling.

## Training budget

All valid 200-sample windows are shuffled without replacement each epoch; the last partial batch is included. Validation-only spectral quality selects checkpoints, with feasible output power first. All final metrics use test data.

| Condition | Full epochs | Windows/epoch | Updates/seed |
|---|---:|---:|---:|
| apa-200mhz-b | 240 | 25,108 | 94,320 |

## Leaders

Best observed configurations in the offline cohort; backbone summaries do not replace the full sweep below.

| Ranking | APA_200MHz_b |
|---|---|
| **Overall FoM** | TRes-GRU · 28.02 |
| **Best linearization** | QGRU amp1 (FP32) · 29.45 |
| **Parameter efficiency** | TRes-GRU · 28.23 |
| **Arithmetic efficiency** | TRes-GRU · 27.80 |
| **Best EVM improvement** | QGRU amp1 (FP32) · 33.30 |
| **Best ACLR improvement** | DeltaGRU (thresholds 0) · 25.79 |
| **≤ 250 parameters** | TCN · 24.54 |
| **≤ 500 parameters** | TRes-GRU · 28.02 |
| **≤ 1,000 parameters** | TRes-GRU · 28.02 |
| **≤ 2,000 parameters** | TRes-GRU · 28.02 |

## Overall configuration results

Every row below is a trained configuration. EVM dB and ACLR average seeds within one measured-source dataset; percent is converted from mean EVM dB. Studio's four Pareto plots compare EVM and ACLR against parameters and operations and execution cohorts. Diamonds mark each two-dimensional front, with seed error bars. Budget subrankings include every configuration whose actual parameter count fits the cap.

## APA_200MHz_b

### offline_overlap_200_100

| Backbone | P | OPs/sample | Quality ± seed SD (dB) | EVM (%) | EVM (dB) | ACLR (dBc) | FoM (dB) | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| TRes-GRU | 447 | 987 | 24.74 ± 0.34 | 0.37 | -48.61 | -49.49 | 28.02 | Ranked |
| DeltaGRU (thresholds 0) | 954 | 1954 | 27.47 ± 0.40 | 0.28 | -51.02 | -52.54 | 27.63 | Ranked |
| QGRU amp1 (FP32) | 977 | 1986 | 27.49 ± 0.10 | 0.27 | -51.36 | -52.24 | 27.56 | Ranked |
| TRes-DeltaGRU (thresholds 0) | 447 | 1053 | 24.04 ± 0.46 | 0.40 | -47.93 | -48.77 | 27.18 | Ranked |
| QGRU (FP32) | 977 | 1984 | 27.04 ± 0.15 | 0.29 | -50.89 | -51.80 | 27.10 | Ranked |
| TRes-GRU | 999 | 2139 | 27.23 ± 0.24 | 0.28 | -51.11 | -51.98 | 27.09 | Ranked |
| TRes-DeltaGRU (thresholds 0) | 999 | 2241 | 27.33 ± 0.13 | 0.27 | -51.21 | -52.07 | 27.09 | Ranked |
| PGJANET | 959 | 2055 | 26.66 ± 0.20 | 0.30 | -50.50 | -51.43 | 26.69 | Ranked |
| QGRU amp1 (FP32) | 1894 | 3834 | 29.45 ± 0.07 | 0.22 | -53.18 | -54.33 | 26.65 | Ranked |
| DeltaGRU (thresholds 0) | 1871 | 3802 | 29.37 ± 0.11 | 0.23 | -52.81 | -54.54 | 26.61 | Ranked |
| BOJANET | 488 | 1442 | 24.09 ± 0.12 | 0.37 | -48.63 | -48.16 | 26.35 | Ranked |
| PGJANET | 1954 | 4105 | 29.18 ± 0.07 | 0.23 | -52.73 | -54.25 | 26.17 | Ranked |
| TRes-DeltaGRU (thresholds 0) | 1916 | 4173 | 29.16 ± 0.03 | 0.22 | -53.01 | -53.92 | 26.15 | Ranked |
| QGRU (FP32) | 1894 | 3832 | 28.65 ± 0.54 | 0.24 | -52.41 | -53.51 | 25.85 | Ranked |
| DVRJANET | 929 | 2080 | 25.72 ± 1.77 | 0.33 | -49.63 | -50.42 | 25.79 | Ranked |
| DGRU | 1971 | 3948 | 28.73 ± 0.25 | 0.24 | -52.44 | -53.65 | 25.78 | Ranked |
| APNRRU | 483 | 1228 | 23.13 ± 1.18 | 0.39 | -48.08 | -46.79 | 25.77 | Ranked |
| TRes-GRU | 1916 | 4029 | 28.56 ± 0.17 | 0.23 | -52.64 | -53.10 | 25.63 | Ranked |
| DeltaGRU (thresholds 0) | 479 | 994 | 22.44 ± 0.27 | 0.57 | -44.90 | -48.60 | 25.56 | Ranked |
| GRU | 994 | 2016 | 25.56 ± 0.39 | 0.33 | -49.60 | -50.14 | 25.56 | Ranked |
| Template GRU (bundled example) | 994 | 2018 | 25.54 ± 0.56 | 0.34 | -49.31 | -50.38 | 25.53 | Ranked |
| QGRU amp1 (FP32) | 425 | 870 | 21.77 ± 0.10 | 0.59 | -44.61 | -47.55 | 25.44 | Ranked |
| DVRJANET | 397 | 936 | 21.75 ± 0.39 | 0.54 | -45.43 | -46.70 | 25.41 | Ranked |
| QGRU (FP32) | 425 | 868 | 21.67 ± 0.52 | 0.60 | -44.47 | -47.48 | 25.34 | Ranked |
| DGRU | 914 | 1834 | 24.69 ± 0.38 | 0.41 | -47.65 | -50.34 | 25.07 | Ranked |
| BOJANET | 978 | 2464 | 25.46 ± 0.04 | 0.33 | -49.70 | -49.85 | 25.06 | Ranked |
| Template GRU (bundled example) | 1911 | 3866 | 27.74 ± 0.28 | 0.26 | -51.66 | -52.43 | 24.90 | Ranked |
| DVRJANET | 1909 | 4140 | 27.88 ± 0.60 | 0.25 | -51.87 | -52.50 | 24.90 | Ranked |
| LSTM | 1962 | 3960 | 27.84 ± 0.16 | 0.26 | -51.83 | -52.47 | 24.90 | Ranked |
| BOJANET | 1346 | 3224 | 26.47 ± 0.42 | 0.30 | -50.37 | -51.19 | 24.79 | Ranked |
| TCN | 232 | 586 | 18.70 ± 0.54 | 0.80 | -41.99 | -44.03 | 24.54 | Ranked |
| DGRU | 486 | 978 | 21.33 ± 0.60 | 0.68 | -43.37 | -47.90 | 24.45 | Ranked |
| LSTM | 912 | 1846 | 23.96 ± 0.86 | 0.45 | -46.99 | -49.55 | 24.34 | Ranked |
| GRU | 1911 | 3864 | 27.14 ± 0.22 | 0.27 | -51.26 | -51.65 | 24.31 | Ranked |
| MCLDNN | 969 | 3598 | 25.50 ± 0.30 | 0.34 | -49.28 | -50.34 | 24.30 | Ranked |
| GMP (least squares) | 500 | 2043 | 22.59 ± 0.00 | 0.42 | -47.55 | -46.25 | 24.05 | Ranked |
| TCN | 493 | 1234 | 21.18 ± 0.22 | 0.65 | -43.68 | -47.30 | 23.76 | Ranked |
| Template GRU (bundled example) | 442 | 902 | 20.19 ± 0.08 | 0.69 | -43.17 | -45.84 | 23.69 | Ranked |
| GRU | 442 | 900 | 20.08 ± 0.86 | 0.69 | -43.17 | -45.62 | 23.59 | Ranked |
| PGJANET | 415 | 919 | 19.90 ± 0.48 | 0.81 | -41.81 | -46.60 | 23.50 | Ranked |
| LSTM | 488 | 990 | 20.04 ± 1.10 | 0.71 | -42.96 | -45.74 | 23.12 | Ranked |
| TRes-GRU | 199 | 459 | 16.18 ± 1.03 | 1.27 | -37.94 | -43.04 | 22.88 | Ranked |
| MP (least squares) | 250 | 1013 | 18.39 ± 0.00 | 0.75 | -42.53 | -42.87 | 22.88 | Ranked |
| TCN | 986 | 2458 | 22.88 ± 0.04 | 0.54 | -45.31 | -49.06 | 22.46 | Ranked |
| GMP (least squares) | 1000 | 4053 | 23.87 ± 0.00 | 0.36 | -48.95 | -47.41 | 22.34 | Ranked |
| APNRRU | 973 | 2334 | 22.60 ± 0.29 | 0.41 | -47.74 | -46.07 | 22.32 | Ranked |
| TRes-DeltaGRU (thresholds 0) | 199 | 501 | 15.48 ± 0.51 | 1.39 | -37.15 | -42.42 | 21.99 | Ranked |
| DVRJANET | 215 | 532 | 15.10 ± 8.34 | 1.21 | -38.31 | -40.51 | 21.32 | Ranked |
| TCN | 1972 | 4906 | 24.66 ± 0.19 | 0.42 | -47.51 | -50.43 | 21.24 | Ranked |
| GRU | 247 | 504 | 14.96 ± 0.28 | 1.27 | -37.92 | -40.62 | 20.99 | Ranked |
| MCLDNN | 1919 | 11438 | 25.90 ± 0.19 | 0.32 | -49.80 | -50.62 | 20.70 | Ranked |
| QGRU (FP32) | 230 | 472 | 14.21 ± 0.95 | 1.89 | -34.47 | -42.57 | 20.54 | Ranked |
| QGRU amp1 (FP32) | 230 | 474 | 14.08 ± 0.21 | 2.33 | -32.66 | -44.11 | 20.39 | Ranked |
| GMP (least squares) | 2000 | 8072 | 24.70 ± 0.00 | 0.35 | -49.02 | -49.00 | 20.17 | Ranked |
| Template GRU (bundled example) | 247 | 506 | 13.77 ± 2.03 | 1.93 | -34.30 | -41.85 | 19.79 | Ranked |
| VDLSTM | 218 | 441 | 12.88 ± 0.31 | 3.22 | -29.86 | -44.51 | 19.47 | Ranked |
| DeltaJANET (thresholds 0) | 442 | 924 | 16.00 ± 0.10 | 1.26 | -38.01 | -42.61 | 19.45 | Ranked |
| MP (least squares) | 490 | 1979 | 17.88 ± 0.00 | 0.75 | -42.51 | -41.86 | 19.45 | Ranked |
| DGRU | 249 | 504 | 13.40 ± 0.99 | 2.28 | -32.83 | -42.58 | 19.41 | Ranked |
| DeltaGRU (thresholds 0) | 207 | 442 | 12.60 ± 0.36 | 2.51 | -32.00 | -41.82 | 19.30 | Ranked |
| APNRRU | 1953 | 4546 | 22.50 ± 0.06 | 0.42 | -47.63 | -46.00 | 19.27 | Ranked |
| LSTM | 192 | 390 | 11.74 ± 0.39 | 2.72 | -31.32 | -40.78 | 18.88 | Ranked |
| DeltaJANET (thresholds 0) | 974 | 2002 | 18.79 ± 0.62 | 0.85 | -41.41 | -44.80 | 18.85 | Ranked |
| PGJANET | 227 | 519 | 12.57 ± 0.60 | 2.43 | -32.30 | -41.46 | 18.72 | Ranked |
| DeltaJANET (thresholds 0) | 1946 | 3964 | 20.67 ± 0.36 | 0.72 | -42.87 | -47.10 | 17.74 | Ranked |
| GMP (least squares) | 250 | 1033 | 12.96 ± 0.00 | 1.40 | -37.06 | -37.48 | 17.41 | Ranked |
| DeltaJANET (thresholds 0) | 226 | 484 | 10.94 ± 0.28 | 2.88 | -30.81 | -39.70 | 17.25 | Ranked |
| RVTDCNN | 227 | 1008 | 12.09 ± 0.27 | 3.17 | -29.97 | -42.83 | 16.80 | Ranked |
| VDLSTM | 446 | 903 | 13.01 ± 0.13 | 3.19 | -29.93 | -44.72 | 16.49 | Ranked |
| MP (least squares) | 1000 | 4028 | 17.77 ± 0.00 | 0.75 | -42.48 | -41.68 | 16.25 | Ranked |
| RVTDCNN | 500 | 1554 | 12.84 ± 0.12 | 3.15 | -30.04 | -44.26 | 14.89 | Ranked |
| MP (least squares) | 1500 | 6043 | 17.80 ± 0.00 | 0.75 | -42.49 | -41.72 | 14.52 | Ranked |
| VDLSTM | 986 | 1993 | 12.66 ± 0.10 | 3.15 | -30.04 | -43.89 | 12.70 | Ranked |
| RVTDCNN | 968 | 2490 | 12.89 ± 0.06 | 3.14 | -30.06 | -44.34 | 12.49 | Ranked |
| RVTDCNN | 1982 | 4518 | 12.81 ± 0.11 | 3.15 | -30.02 | -44.21 | 9.56 | Ranked |
| VDLSTM | 1898 | 3829 | 12.31 ± 0.12 | 3.06 | -30.28 | -42.97 | 9.51 | Ranked |
| BOJANET | 224 | 878 | 3.54 ± 0.01 | 4.54 | -26.85 | -28.84 | 8.58 | Ranked |
| GMP (gradient-trained) | 495 | 2073 | 0.14 ± 0.09 | 9.67 | -20.29 | -28.60 | 1.59 | Ranked |

### streaming_stateful

| Backbone | P | OPs/sample | Quality ± seed SD (dB) | EVM (%) | EVM (dB) | ACLR (dBc) | FoM (dB) | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| GRU (streaming, stateful) | 994 | 2016 | 25.56 ± 0.39 | 0.33 | -49.60 | -50.14 | 25.56 | Ranked |
| GRU (streaming, stateful) | 1911 | 3864 | 27.14 ± 0.22 | 0.27 | -51.26 | -51.65 | 24.31 | Ranked |
| GRU (streaming, stateful) | 442 | 900 | 20.08 ± 0.86 | 0.69 | -43.17 | -45.62 | 23.59 | Ranked |
| GRU (streaming, stateful) | 247 | 504 | 14.96 ± 0.28 | 1.27 | -37.92 | -40.62 | 20.99 | Ranked |
| GMP (streaming, windowed) | 495 | 2073 | 0.14 ± 0.09 | 9.67 | -20.29 | -28.60 | 1.59 | Ranked |

## Cost sensitivity

Best observed FoM when MUL is priced as 1, 4 or 16 ADDs, using fixed references of 1000 MUL and 1000 ADD. PA cost is always excluded.

| Board | Backbone | MUL:ADD 1:1 | 4:1 | 16:1 |
|---|---|---:|---:|---:|
| APA_200MHz_b | TRes-GRU | 28.02 | 27.96 | 27.94 |
| APA_200MHz_b | DeltaGRU (thresholds 0) | 27.63 | 27.65 | 27.66 |
| APA_200MHz_b | QGRU amp1 (FP32) | 27.56 | 27.58 | 27.58 |
| APA_200MHz_b | TRes-DeltaGRU (thresholds 0) | 27.18 | 27.21 | 27.23 |
| APA_200MHz_b | QGRU (FP32) | 27.10 | 27.12 | 27.13 |
| APA_200MHz_b | PGJANET | 26.69 | 26.67 | 26.67 |
| APA_200MHz_b | BOJANET | 26.35 | 26.32 | 26.30 |
| APA_200MHz_b | DVRJANET | 25.79 | 25.78 | 25.77 |
| APA_200MHz_b | DGRU | 25.78 | 25.80 | 25.80 |
| APA_200MHz_b | APNRRU | 25.77 | 25.73 | 25.72 |
| APA_200MHz_b | GRU | 25.56 | 25.58 | 25.59 |
| APA_200MHz_b | Template GRU (bundled example) | 25.53 | 25.55 | 25.56 |
| APA_200MHz_b | LSTM | 24.90 | 24.91 | 24.91 |
| APA_200MHz_b | TCN | 24.54 | 24.46 | 24.42 |
| APA_200MHz_b | MCLDNN | 24.30 | 24.30 | 24.30 |
| APA_200MHz_b | GMP (least squares) | 24.05 | 24.02 | 24.01 |
| APA_200MHz_b | MP (least squares) | 22.88 | 22.86 | 22.86 |
| APA_200MHz_b | VDLSTM | 19.47 | 19.46 | 19.45 |
| APA_200MHz_b | DeltaJANET (thresholds 0) | 19.45 | 19.49 | 19.51 |
| APA_200MHz_b | RVTDCNN | 16.80 | 16.80 | 16.80 |
| APA_200MHz_b | GMP (gradient-trained) | 1.59 | 1.53 | 1.50 |

## Reproducibility

[Protocol](../protocols/dpd-arena-v6.md) · [PA qualification](arena-pa-qualification.md) · [Independent audit](arena-reference-audit.json)

Protocol SHA-256: `646c31e0546b6a6754e0ad06c381aed21989dcf4ffa2c52bbb6321db67065b51`.

Training SHA-256: `71294fba81c968d2762f5349178cba8f23a38a8aad942b530269c783b9d9bf94`.

Reference bundle content seal: `1a85036a51aa66e2a14b1e6a899bb7183039f133f8b0a03abf1d3aaf541b4408`.

No production deployment or hardware RF measurement was performed.
