# 2.2.14 measurement validation

The numerical record is [stored in the repository](https://github.com/lab-emi/OpenDPD/blob/main/benchmark/results/studio-2.2.14-metric-validation.json),
including input hashes, generated waveform settings, checkpoint hashes,
sample counts and independent-check errors.

All experiments use TRes-GRU PA H27 and DPD H15, training/evaluation batch
16, seed 0, AdamW, initial learning rate 0.005, ReduceLROnPlateau, frame length
200. The eight-condition matrix uses 20 PA / 30 DPD epochs; synthetic stride
4, measured stride 16. These short runs check the full pipeline and scoring;
they are not optimized model comparisons. The synthetic stimuli use 65,536
samples each and the generator's disclosed engineering waveform conventions,
not complete standard-conformant RF packets.

| Input | QAM | PA data/model | DPD test carrier ACLR average (dBc) |
|---|---:|---|---:|
| NR 5 MHz | 16 | Rapp soft limiter | −63.44 |
| NR 20 MHz | 64 | Memory polynomial | −63.03 |
| NR 100 MHz | 256 | Saleh TWTA | −54.46 |
| Wi-Fi 6 20 MHz | 64 | Rapp AM/AM + AM/PM | −59.96 |
| Wi-Fi 6 80 MHz | 1024 | Generalized memory | −64.35 |
| Wi-Fi 7 320 MHz | 4096 | Doherty two-path | −46.78 |
| DPA 160 MHz | 1024 | Measured capture | −41.45 |
| DPA 200 MHz | 64 | Measured capture | −32.00 |

Each stored result and every raw train/validation/test PA-output split was
checked against a separately written NumPy FFT periodogram average. The
oracle builds its Hann window, FFTs and band masks directly, without SciPy
Welch or production PSD/integration helpers. Every difference was below
1e-8 dB. The test suite also checks known −55/−60 dBc adjacent tones,
carrier counts 1/4/5/10, odd FFT lengths, unavailable Nyquist bands and padding
containing arbitrary nonzero model output.

## APA reproduction

APA 200 MHz uses the bundled measured 256-QAM capture at 983.04 MSa/s,
5 × 40 MHz carrier bands and `nperseg=19662`. Both models trained for
300 epochs, frame stride 1. The validation-selected DPD checkpoint is epoch
222 (zero-based 221), scored on all 19,662 held-out test samples.

| Metric | Test value |
|---|---:|
| Carrier ACLR left | −55.537388 dBc |
| Carrier ACLR right | −55.689390 dBc |
| Carrier ACLR average | −55.613389 dBc |
| Legacy carrier ACLR average, same checkpoint | −55.612043 dBc |
| Full-band ACPR left (`general-spectral-v1`) | −57.431678 dBc |
| Full-band ACPR right (`general-spectral-v1`) | −58.822323 dBc |
| NMSE | −45.380323 dB |

The oracle difference is below 2e-14 dB. Legacy/new agreement on APA shows
that reaching the requested ~−55 dBc level does not come from changing
measurement bands to improve a number. Full-band ACPR is reported separately
because its integration bands and reference differ from carrier ACLR.

All DPD rows above are **PA-model predictions**, including models trained
on measured captures. They do not establish hardware DPD performance. One
seed is a reproduction check, not a statistical performance guarantee.
The compute used an immutable candidate source snapshot, whose Python-file
manifest hash is included with the numerical record.
