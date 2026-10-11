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
| v2 run 2 | 2026-10-11 | `cd9ccb3` (proposed validated profile metadata; clean numerical source) | 115/115 passed; [run2 record and diagnostics](matlab-parity-v2-run2.json) |
| v2 run 3 | 2026-10-12 | `e6012df` (application catalogue isolated from frozen Arena; clean numerical source) | 115/115 passed; [complete record and diagnostics](matlab-parity-v2.json) |

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

## Complete scored results

<!-- parity-v2-results:begin -->
Run 3 — 2026-10-12: **115/115 passed**; 0 failed.

This is the separately registered v2 result; it does not change the old v1 verdict.
For frequency-offset rows the first value is the injected offset; the second is the estimate by the tool named in the item.

| Group | Scored item | OpenDPD | MATLAB | Difference | Budget | Pass |
| --- | --- | --- | --- | --- | --- | --- |
| P1 | sample count | 307200 | 307200 | 0 | 0 | yes |
| P1 | first-sample alignment (peak lag) | 0 | 0 | 0 | 0 | yes |
| P1 | relative RMS difference after one real scale | 0 | 2.85482688436e-08 | 2.85482688436e-08 | 1e-06 | yes |
| P3 | S0 EVM_RMS (percent) | 1.93005662701e-06 | 2.33371887091e-06 | 4.03662243899e-07 | 0.05 | yes |
| P3 | S0 integer timing | 0 | 0 | 0 | 0 | yes |
| P3 | S1 EVM_RMS (percent) | 0.0298635358911 | 0.0298635365801 | 6.89032195883e-10 | 0.05 | yes |
| P3 | S1 integer timing | 0 | 0 | 0 | 0 | yes |
| P2 | S2 ACLR_L nperseg=1024 vs comm.ACPR aligned | -41.1966539645 | -41.1966539645 | 0 | 0.1 | yes |
| P2 | S2 ACLR_R nperseg=1024 vs comm.ACPR aligned | -41.3320607114 | -41.3320607114 | 0 | 0.1 | yes |
| P2 | S2 ACLR_L nperseg=1024 vs comm.ACPR default | -41.3348054335 | -41.3348054335 | 0 | 0.1 | yes |
| P2 | S2 ACLR_R nperseg=1024 vs comm.ACPR default | -41.3280846744 | -41.3280846744 | 0 | 0.1 | yes |
| P2 | S2 ACLR_L nperseg=2048 vs comm.ACPR aligned | -41.0539775315 | -41.0539775315 | 0 | 0.1 | yes |
| P2 | S2 ACLR_R nperseg=2048 vs comm.ACPR aligned | -41.2147242922 | -41.2147242922 | 0 | 0.1 | yes |
| P2 | S2 ACLR_L nperseg=2048 vs comm.ACPR default | -41.3348054335 | -41.3348054335 | 0 | 0.1 | yes |
| P2 | S2 ACLR_R nperseg=2048 vs comm.ACPR default | -41.3280846744 | -41.3280846744 | 0 | 0.1 | yes |
| P2 | S2 ACLR_L nperseg=4096 vs comm.ACPR aligned | -41.1694572583 | -41.1694572583 | 0 | 0.1 | yes |
| P2 | S2 ACLR_R nperseg=4096 vs comm.ACPR aligned | -41.3170629337 | -41.3170629337 | 0 | 0.1 | yes |
| P2 | S2 ACLR_L nperseg=4096 vs comm.ACPR default | -41.3348054335 | -41.3348054335 | 0 | 0.1 | yes |
| P2 | S2 ACLR_R nperseg=4096 vs comm.ACPR default | -41.3280846744 | -41.3280846744 | 0 | 0.1 | yes |
| P3 | S2 EVM_RMS (percent) | 0.983656768452 | 0.983656767379 | -1.07288300466e-09 | 0.05 | yes |
| P3 | S2 integer timing | 0 | 0 | 0 | 0 | yes |
| P2 | S3 ACLR_L nperseg=1024 vs comm.ACPR aligned | -37.3448269858 | -37.3448269858 | 0 | 0.1 | yes |
| P2 | S3 ACLR_R nperseg=1024 vs comm.ACPR aligned | -37.242812401 | -37.242812401 | 0 | 0.1 | yes |
| P2 | S3 ACLR_L nperseg=1024 vs comm.ACPR default | -37.4707838609 | -37.4707838609 | 0 | 0.1 | yes |
| P2 | S3 ACLR_R nperseg=1024 vs comm.ACPR default | -37.2891823665 | -37.2891823665 | 0 | 0.1 | yes |
| P2 | S3 ACLR_L nperseg=2048 vs comm.ACPR aligned | -37.3256934491 | -37.3256934491 | 0 | 0.1 | yes |
| P2 | S3 ACLR_R nperseg=2048 vs comm.ACPR aligned | -37.2310527509 | -37.2310527509 | 0 | 0.1 | yes |
| P2 | S3 ACLR_L nperseg=2048 vs comm.ACPR default | -37.4707838609 | -37.4707838609 | 0 | 0.1 | yes |
| P2 | S3 ACLR_R nperseg=2048 vs comm.ACPR default | -37.2891823665 | -37.2891823665 | 0 | 0.1 | yes |
| P2 | S3 ACLR_L nperseg=4096 vs comm.ACPR aligned | -37.3708942349 | -37.3708942349 | 0 | 0.1 | yes |
| P2 | S3 ACLR_R nperseg=4096 vs comm.ACPR aligned | -37.275051419 | -37.275051419 | 0 | 0.1 | yes |
| P2 | S3 ACLR_L nperseg=4096 vs comm.ACPR default | -37.4707838609 | -37.4707838609 | 0 | 0.1 | yes |
| P2 | S3 ACLR_R nperseg=4096 vs comm.ACPR default | -37.2891823665 | -37.2891823665 | 0 | 0.1 | yes |
| P3 | S3 EVM_RMS (percent) | 2.72583424914 | 2.72583424809 | -1.04984065885e-09 | 0.05 | yes |
| P3 | S3 integer timing | 0 | 0 | 0 | 0 | yes |
| P2 | S4 ACLR_L nperseg=1024 vs comm.ACPR aligned | -32.7500456134 | -32.7500456134 | 0 | 0.1 | yes |
| P2 | S4 ACLR_R nperseg=1024 vs comm.ACPR aligned | -32.5281587006 | -32.5281587006 | 7.1054273576e-15 | 0.1 | yes |
| P2 | S4 ACLR_L nperseg=1024 vs comm.ACPR default | -32.8674335507 | -32.8674335507 | 0 | 0.1 | yes |
| P2 | S4 ACLR_R nperseg=1024 vs comm.ACPR default | -32.5988686269 | -32.5988686269 | 0 | 0.1 | yes |
| P2 | S4 ACLR_L nperseg=2048 vs comm.ACPR aligned | -32.7893392641 | -32.7893392641 | 0 | 0.1 | yes |
| P2 | S4 ACLR_R nperseg=2048 vs comm.ACPR aligned | -32.5639905285 | -32.5639905285 | 0 | 0.1 | yes |
| P2 | S4 ACLR_L nperseg=2048 vs comm.ACPR default | -32.8674335507 | -32.8674335507 | 0 | 0.1 | yes |
| P2 | S4 ACLR_R nperseg=2048 vs comm.ACPR default | -32.5988686269 | -32.5988686269 | 0 | 0.1 | yes |
| P2 | S4 ACLR_L nperseg=4096 vs comm.ACPR aligned | -32.8040338408 | -32.8040338408 | 0 | 0.1 | yes |
| P2 | S4 ACLR_R nperseg=4096 vs comm.ACPR aligned | -32.5841163398 | -32.5841163398 | 0 | 0.1 | yes |
| P2 | S4 ACLR_L nperseg=4096 vs comm.ACPR default | -32.8674335507 | -32.8674335507 | 0 | 0.1 | yes |
| P2 | S4 ACLR_R nperseg=4096 vs comm.ACPR default | -32.5988686269 | -32.5988686269 | 0 | 0.1 | yes |
| P3 | S4 EVM_RMS (percent) | 5.31524716274 | 5.31524716174 | -9.9977004453e-10 | 0.05 | yes |
| P3 | S4 integer timing | 0 | 0 | 0 | 0 | yes |
| P2 | S5 ACLR_L nperseg=1024 vs comm.ACPR aligned | -27.1547328151 | -27.1547328151 | 0 | 0.1 | yes |
| P2 | S5 ACLR_R nperseg=1024 vs comm.ACPR aligned | -26.8605519596 | -26.8605519596 | 0 | 0.1 | yes |
| P2 | S5 ACLR_L nperseg=1024 vs comm.ACPR default | -27.268878264 | -27.268878264 | 0 | 0.1 | yes |
| P2 | S5 ACLR_R nperseg=1024 vs comm.ACPR default | -26.9447069485 | -26.9447069485 | 0 | 0.1 | yes |
| P2 | S5 ACLR_L nperseg=2048 vs comm.ACPR aligned | -27.2172902399 | -27.2172902399 | 0 | 0.1 | yes |
| P2 | S5 ACLR_R nperseg=2048 vs comm.ACPR aligned | -26.9144876756 | -26.9144876756 | 0 | 0.1 | yes |
| P2 | S5 ACLR_L nperseg=2048 vs comm.ACPR default | -27.268878264 | -27.268878264 | 0 | 0.1 | yes |
| P2 | S5 ACLR_R nperseg=2048 vs comm.ACPR default | -26.9447069485 | -26.9447069485 | 0 | 0.1 | yes |
| P2 | S5 ACLR_L nperseg=4096 vs comm.ACPR aligned | -27.2215408218 | -27.2215408218 | 0 | 0.1 | yes |
| P2 | S5 ACLR_R nperseg=4096 vs comm.ACPR aligned | -26.9278039661 | -26.9278039661 | 0 | 0.1 | yes |
| P2 | S5 ACLR_L nperseg=4096 vs comm.ACPR default | -27.268878264 | -27.268878264 | 0 | 0.1 | yes |
| P2 | S5 ACLR_R nperseg=4096 vs comm.ACPR default | -26.9447069485 | -26.9447069485 | 0 | 0.1 | yes |
| P3 | S5 EVM_RMS (percent) | 10.6382449213 | 10.6382449204 | -9.03558117216e-10 | 0.05 | yes |
| P3 | S5 integer timing | 0 | 0 | 0 | 0 | yes |
| P2 | S6 ACLR_L nperseg=1024 vs comm.ACPR aligned | -32.7431926628 | -32.7431926628 | 0 | 0.1 | yes |
| P2 | S6 ACLR_R nperseg=1024 vs comm.ACPR aligned | -32.5185633667 | -32.5185633667 | 0 | 0.1 | yes |
| P2 | S6 ACLR_L nperseg=1024 vs comm.ACPR default | -32.8470666555 | -32.8470666555 | 0 | 0.1 | yes |
| P2 | S6 ACLR_R nperseg=1024 vs comm.ACPR default | -32.5798863517 | -32.5798863517 | 0 | 0.1 | yes |
| P2 | S6 ACLR_L nperseg=2048 vs comm.ACPR aligned | -32.7826206671 | -32.7826206671 | 0 | 0.1 | yes |
| P2 | S6 ACLR_R nperseg=2048 vs comm.ACPR aligned | -32.5543121548 | -32.5543121548 | 0 | 0.1 | yes |
| P2 | S6 ACLR_L nperseg=2048 vs comm.ACPR default | -32.8470666555 | -32.8470666555 | 0 | 0.1 | yes |
| P2 | S6 ACLR_R nperseg=2048 vs comm.ACPR default | -32.5798863517 | -32.5798863517 | 0 | 0.1 | yes |
| P2 | S6 ACLR_L nperseg=4096 vs comm.ACPR aligned | -32.7965080082 | -32.7965080082 | 0 | 0.1 | yes |
| P2 | S6 ACLR_R nperseg=4096 vs comm.ACPR aligned | -32.5739466347 | -32.5739466347 | 0 | 0.1 | yes |
| P2 | S6 ACLR_L nperseg=4096 vs comm.ACPR default | -32.8470666555 | -32.8470666555 | 0 | 0.1 | yes |
| P2 | S6 ACLR_R nperseg=4096 vs comm.ACPR default | -32.5798863517 | -32.5798863517 | 0 | 0.1 | yes |
| P3 | S6 EVM_RMS (percent) | 5.31521474069 | 5.3152147397 | -9.93350290912e-10 | 0.05 | yes |
| P3 | S6 integer timing | 92160 | 92160 | 0 | 0 | yes |
| P2 | S7 ACLR_L nperseg=1024 vs comm.ACPR aligned | -31.67301359 | -31.67301359 | 0 | 0.1 | yes |
| P2 | S7 ACLR_R nperseg=1024 vs comm.ACPR aligned | -31.5068562752 | -31.5068562752 | 0 | 0.1 | yes |
| P2 | S7 ACLR_L nperseg=1024 vs comm.ACPR default | -31.7846587227 | -31.7846587227 | 0 | 0.1 | yes |
| P2 | S7 ACLR_R nperseg=1024 vs comm.ACPR default | -31.5759143039 | -31.5759143039 | 0 | 0.1 | yes |
| P2 | S7 ACLR_L nperseg=2048 vs comm.ACPR aligned | -31.7095439799 | -31.7095439799 | 0 | 0.1 | yes |
| P2 | S7 ACLR_R nperseg=2048 vs comm.ACPR aligned | -31.5427536127 | -31.5427536127 | 0 | 0.1 | yes |
| P2 | S7 ACLR_L nperseg=2048 vs comm.ACPR default | -31.7846587227 | -31.7846587227 | 0 | 0.1 | yes |
| P2 | S7 ACLR_R nperseg=2048 vs comm.ACPR default | -31.5759143039 | -31.5759143039 | 0 | 0.1 | yes |
| P2 | S7 ACLR_L nperseg=4096 vs comm.ACPR aligned | -31.7245924946 | -31.7245924946 | 0 | 0.1 | yes |
| P2 | S7 ACLR_R nperseg=4096 vs comm.ACPR aligned | -31.5616574697 | -31.5616574697 | 3.5527136788e-15 | 0.1 | yes |
| P2 | S7 ACLR_L nperseg=4096 vs comm.ACPR default | -31.7846587227 | -31.7846587227 | 0 | 0.1 | yes |
| P2 | S7 ACLR_R nperseg=4096 vs comm.ACPR default | -31.5759143039 | -31.5759143039 | 0 | 0.1 | yes |
| P3 | S7 EVM_RMS (percent) | 5.4497361357 | 5.44973613442 | -1.28037846991e-09 | 0.05 | yes |
| P3 | S7 integer timing | 0 | 0 | 0 | 0 | yes |
| P2 | S8 ACLR_L nperseg=2048 vs comm.ACPR aligned | -32.5392568545 | -32.5392568545 | 0 | 0.1 | yes |
| P2 | S8 ACLR_R nperseg=2048 vs comm.ACPR aligned | -32.4170402013 | -32.4170402013 | 0 | 0.1 | yes |
| P2 | S8 ACLR_L nperseg=2048 vs comm.ACPR default | -32.653257327 | -32.653257327 | 0 | 0.1 | yes |
| P2 | S8 ACLR_R nperseg=2048 vs comm.ACPR default | -32.4852801226 | -32.4852801226 | 0 | 0.1 | yes |
| P2 | S8 ACLR_L nperseg=4096 vs comm.ACPR aligned | -32.5785344073 | -32.5785344073 | 0 | 0.1 | yes |
| P2 | S8 ACLR_R nperseg=4096 vs comm.ACPR aligned | -32.4514532128 | -32.4514532128 | 0 | 0.1 | yes |
| P2 | S8 ACLR_L nperseg=4096 vs comm.ACPR default | -32.653257327 | -32.653257327 | 0 | 0.1 | yes |
| P2 | S8 ACLR_R nperseg=4096 vs comm.ACPR default | -32.4852801226 | -32.4852801226 | 0 | 0.1 | yes |
| P2 | S8 ACLR_L nperseg=8192 vs comm.ACPR aligned | -32.5920534999 | -32.5920534999 | 0 | 0.1 | yes |
| P2 | S8 ACLR_R nperseg=8192 vs comm.ACPR aligned | -32.4714540464 | -32.4714540464 | 0 | 0.1 | yes |
| P2 | S8 ACLR_L nperseg=8192 vs comm.ACPR default | -32.653257327 | -32.653257327 | 0 | 0.1 | yes |
| P2 | S8 ACLR_R nperseg=8192 vs comm.ACPR default | -32.4852801226 | -32.4852801226 | 0 | 0.1 | yes |
| P3 | S8 EVM_RMS (percent) | 5.32129909371 | 5.32129909274 | -9.73950697869e-10 | 0.05 | yes |
| P3 | S8 integer timing | 0 | 0 | 0 | 0 | yes |
| P3 | S9 EVM_RMS (percent) | 0.0264295413763 | 0.0264295433964 | 2.02012303066e-09 | 0.05 | yes |
| P3 | S9 integer timing | 92160 | 92160 | 0 | 0 | yes |
| P3 | S9 frequency offset, OpenDPD | 350 | 350.000039384 | 3.93841402229e-05 | 0.01 | yes |
| P3 | S9 frequency offset, MATLAB | 350 | 350.000039384 | 3.93841609707e-05 | 0.01 | yes |
| P3 | S6b EVM_RMS (percent) | 5.31521475814 | 5.31521475714 | -9.93349402734e-10 | 0.05 | yes |
| P3 | S6b integer timing | 92160 | 92160 | 0 | 0 | yes |
| P3 | S9b EVM_RMS (percent) | 0.0264295260791 | 0.0264295280991 | 2.01999395336e-09 | 0.05 | yes |
| P3 | S9b integer timing | 92160 | 92160 | 0 | 0 | yes |
| P3 | S9b frequency offset, OpenDPD | 30 | 30.0000393898 | 3.93898376423e-05 | 0.01 | yes |
| P3 | S9b frequency offset, MATLAB | 30 | 30.0000393899 | 3.93898583724e-05 | 0.01 | yes |
<!-- parity-v2-results:end -->

## Evidence interpretation and Arena isolation

The 84 ACLR scored entries comprise 42 segmented comparisons and 42 Auto entries.
Auto ignores nperseg, so those 42 entries are 14 distinct values repeated across
the three segment-size rows for correspondence with the original matrix. They
are not 84 statistically independent observations. Both implementations share
the written procedure, so agreement does not independently validate every
assumption of that procedure. The waveform is assumed to repeat: captures with
silence-prefixed, non-circular framing are outside this registration.

Independent review caught that directly extending the old registry altered an
Arena-hashed file. The application catalogue is therefore extended separately in
`opendpd.core.metric_profiles`; Arena keeps the original registry, evaluator and
source bytes. No bundled hash or benchmark result is rewritten. Regression tests
verify the original evaluator binding, profile catalogue and sealed Arena rows.
