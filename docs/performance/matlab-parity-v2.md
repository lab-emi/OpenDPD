# Version 2 OpenDPD–MATLAB parity

Status: **115/115 scored items passed** on the registered digital signals
(2026-10-11, MATLAB R2026a). Scientific review is required before merging the
proposed validated profiles. This result belongs to the separately registered
[v2 procedure](../protocols/waveform-profiles-v2.md); the original v1 comparison
remains a recorded failure and its profile remains pending cross-validation.

## What was validated

Two explicit production profiles pair like-for-like estimator configurations:
`ofdm-lte20-evm-v2` (dataset-length symmetric Hann) and
`ofdm-lte20-evm-auto-v2` (full-record symmetric Hamming). Both use enclosing
FFT-bin band integration and the independently specified frequency-tolerant
acquisition. All original signal hashes, 115 scored cases and budgets were
retained, including the +350 Hz cases that v1 refused.

The Python implementation was written by Codex. Claude Code implemented the
MATLAB comparator from the committed written protocol without reading the new
Python implementation or its outputs (session
`claude:515477ff-8a91-42b2-ab0f-a65431301132`). Neither side calls the other.
The MATLAB reference `comm.ACPR` configurations were unchanged. The existing
MathWorks OFDM demodulation and EVM functions remain the independent backend.

## Run log

| Run | Date | Source | Outcome |
| --- | --- | --- | --- |
| v1 baseline rerun | 2026-10-11 | original implementation plus review-policy merge | 102/115 within budget, seven outside, six unevaluable; same result as the archived v1 record |
| v2 run 1 | 2026-10-11 | `ba5819b` (clean numerical source) | 115/115 passed; [machine-readable record](matlab-parity-v2-run1.json) |

Run 1 maximum absolute ACLR difference: **7.105427357601002e-15 dB**
(budget 0.1 dB). Maximum EVM difference: **4.036622438989357e-7 percentage
points** (budget 0.05). All timing/count checks were exact; the clean frequency
offsets met 0.01 Hz. Every scored item was evaluable.

This is not retrospective success of the old v1 criteria. One old common-value
comparison was mathematically impossible: the two MATLAB reference values
were 0.280827902 dB apart against two 0.1 dB budgets. V2 makes those two
measurement configurations explicit and independently validates each.

## Reproduce

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
python scripts/matlab_parity_v2.py --work /tmp/opendpd-parity-v2 \
  --matlab /path/to/MATLAB/bin/matlab --matlab-prefdir /tmp/opendpd-parity-v2-prefs \
  --date YYYY-MM-DD
```

The harness writes `report.json`, `report.md`, all independent MATLAB results
and the MATLAB log beneath the scratch directory. It rejects a changed signal
hash, missing/duplicate case, wrong product configuration, changed tolerance,
non-finite/refused score, or a verdict inconsistent with its actual numbers.
Additional diagnostic values do not decide the scored verdict.

## Limits

This validates the registered digital procedure and configurations, not a 3GPP
conformance EVM procedure, LTE physical channels, 5G NR, real instruments or
arbitrary unbound/multi-carrier captures. Both profiles require the known waveform
binding; incompatible data produces explicit unavailable statuses. The old
legacy/general/carrier profiles and default training/checkpoint metric are
unchanged. The Auto profile uses the full valid record, rather than dataset
nperseg, so its spectral estimate may differ from a segmented display plot.

Python regression coverage additionally checks both CFO signs, 1/10 ms frames,
partial-symbol capture delays, wrong references/noise, analytic tone power and
Nyquist boundaries, mode-independent EVM, and exact-case/tolerance integrity.
