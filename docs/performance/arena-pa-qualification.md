# Arena frozen PA qualification

APA_200MHz_b uses one frozen H37 TRes-GRU (4,871 real parameters). All three fresh candidates complete 240 full training epochs before PA selection is frozen. Validation NMSE alone selects the checkpoint and seed; held-out test NMSE is observed afterwards. No gain, phase or delay alignment is fitted.

| Dataset | Selected source | Seed | Validation NMSE (dB) | Test NMSE (dB) |
|---|---|---:|---:|---:|
| APA_200MHz_b | Fresh 240-epoch fit | 1 | -40.0727 | -39.7183 |

Each fresh candidate uses seeds 0, 1 or 2, all stride-1 200-sample windows, batch 64 including its partial tail, and AdamW at initial learning rate 0.005. Whole-validation-record NMSE drives a plateau scheduler and checkpoint selection. Older PAs fitted to incompatible partitions remain excluded.

| Dataset | Fresh seed | Selected epoch | Validation NMSE (dB) | Best-NMSE improvement in final 30 epochs (dB) |
|---|---:|---:|---:|---:|
| APA_200MHz_b | 0 | 239 | -39.9710 | 0.0052 |
| APA_200MHz_b | 1 | 220 | -40.0727 | 0.0070 |
| APA_200MHz_b | 2 | 207 | -39.9643 | 0.0000 |

The fixed budget does not establish convergence for every seed. A low short-window training loss also does not guarantee good continuous-record PA validation; selection uses the latter. DPD results never choose the PA.

| Dataset | Validation residual AER L (dB) | Validation residual AER R (dB) |
|---|---:|---:|
| APA_200MHz_b | -48.3167 | -46.9271 |

Residual AER is a measured-output model-error diagnostic using the fixed validation context and 4,096-sample Welch segments. It is distinct from output ACLR. PA NMSE and residual AER are not hard bounds on DPD cascade EVM or ACLR, and PA complexity is excluded from DPD FoM.

PA validation/test inference uses each full original measured split from zero recurrent state. APA partitions retain complete independently timed test symbols: train/validation/test lengths are 25,307 / 6,327 / 66,270, with 200-sample gaps. DPD inputs remain those original capture slices.

[Protocol](../protocols/dpd-arena-v6.md) · [Numerical qualification](arena-pa-qualification.json)
