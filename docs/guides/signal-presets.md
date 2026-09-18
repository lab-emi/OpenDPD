# Signal preset reference

Studio 2.2.11 offers **1,186 engineering presets**: 940 NR, 96 Wi-Fi 6, 140 Wi-Fi 7 and 10 custom waveforms. Matrix columns are **nominal baseband bandwidth**, not RF carrier frequency; rows are modulation. Separate matrices select 1, 2, 4 or 8 OFDMA channels. Numerology buttons select the NR frequency range and subcarrier spacing.

These are **continuous, uncoded complex-baseband PA stimuli**. The catalog adopts published bandwidths, resource sizes, modulation orders and OFDM timing. It does not implement transport coding, NR synchronization/control channels, WLAN preambles, standard pilot sequences or the full RU-placement bitmap. It is not a collection of certified standard test models. Generic per-channel pilots and allocation placement are disclosed in every export. An OFDMA channel here is an independent allocation inside one RF band; it is not a separate adjacent RF carrier or a MIMO spatial stream.

## NR bandwidths

The RB counts below follow **3GPP TS 38.104 V18.9.0, Tables 5.3.2-1 and 5.3.2-2**. Normal cyclic-prefix timing and QAM mappings follow **TS 38.211 V18.5.0, §§5.1 and 5.3.1**. All available rows offer QPSK, 16/64/256/1024-QAM. 1024-QAM is a Release 18 modulation option; the catalog does not imply that every RF band, device, direction or coded MCS supports every displayed combination.

| FR1 bandwidth (MHz) | 15 kHz RBs | 30 kHz RBs | 60 kHz RBs |
| --- | ---: | ---: | ---: |
| 3 | 15 | — | — |
| 5 | 25 | 11 | — |
| 10 | 52 | 24 | 11 |
| 15 | 79 | 38 | 18 |
| 20 | 106 | 51 | 24 |
| 25 | 133 | 65 | 31 |
| 30 | 160 | 78 | 38 |
| 35 | 188 | 92 | 44 |
| 40 | 216 | 106 | 51 |
| 45 | 242 | 119 | 58 |
| 50 | 270 | 133 | 65 |
| 60 | — | 162 | 79 |
| 70 | — | 189 | 93 |
| 80 | — | 217 | 107 |
| 90 | — | 245 | 121 |
| 100 | — | 273 | 135 |

| FR2-1 bandwidth (MHz) | 60 kHz RBs | 120 kHz RBs |
| --- | ---: | ---: |
| 50 | 66 | 32 |
| 100 | 132 | 66 |
| 200 | 264 | 132 |
| 400 | — | 264 |

With N allocations, each receives `floor(RBs / N) × 12` active tones. Remainder RBs are unused. The FFT is the next power of two that spans the nominal bandwidth, with 4× output oversampling. Default RF metadata is 3.5 GHz for FR1 and 28 GHz for FR2-1. RF carrier metadata does not digitally upconvert the signal. FR2-2 and band-specific scheduling rules are outside this catalog.

## WLAN bandwidths and allocations

Wi-Fi 6 offers 20/40/80/160 MHz and BPSK, QPSK, 16/64/256/1024-QAM. Wi-Fi 7 adds 320 MHz and 4096-QAM. Both use 78.125 kHz spacing, a 12.8 µs useful symbol and a 0.8 µs guard interval by default. Advanced controls can change the guard interval; the fixed CP field is in samples **before** oversampling.

| Bandwidth (MHz) | Tones / allocation: 1 channel | 2 channels | 4 channels | 8 channels |
| --- | ---: | ---: | ---: | ---: |
| 20 | 242 | 106 | 52 | 26 |
| 40 | 484 | 242 | 106 | 52 |
| 80 | 996 | 484 | 242 | 106 |
| 160 | 1992 | 996 | 484 | 242 |
| 320 (Wi-Fi 7) | 3984 | 1992 | 996 | 484 |

These are RU-sized engineering allocations. Aggregate 1992/3984-tone payloads represent 2×996/4×996 active tones; their contiguous placement, four-bin inter-allocation gaps and generic comb pilots **do not reproduce a standard packet tone map**. The selected QAM is the constellation order, not an MCS index or a coding rate. Default sample rate is four times nominal bandwidth.

The public IEEE task-group records and specification-framework links identify the HE/EHT development basis. The final IEEE standards and framework downloads were not accessible to the release validation environment; full clause-by-clause PHY validation is therefore not claimed. Public task-group allocation records and the published HE timing description support the limited engineering scope above.

## Length and filtering

Defaults approximate 0.25 ms, bounded to 16,384–196,608 samples. Each selected preset can have its own exact length or duration. The batch limit is 16 presets and 4,000,000 total samples; one preset allows 256–1,000,000 samples, with at least 8,192 needed for a training dataset. Hosted storage admission may require a smaller batch.

The default ideal PA-input filter is a periodic-record, zero-phase FFT low-pass, centered on the configured frequency offset. Its gain is one to 96% of the nominal half-bandwidth, a cosine transition to 100%, then zero. It preserves the exact sample count and pre-filter RMS. It runs after impairments, so it also band-limits noise and can change EVM, peak amplitude and burst edges. It is not a causal hardware filter. Disable it per preset for unfiltered numerical experiments. **Virtual PA output is not filtered:** nonlinear spectral regrowth remains visible.

Stop-band energy is tested on the full-record DFT; the displayed Welch PSD includes finite-window leakage and should not be interpreted as the exact filter response. Float32 export introduces a small numerical floor.

## Dataset names and analysis

The editable name beside **Generate & preview** is saved with the entire selected batch. A single-spec example is `syn_pa_in_nr_bw20M_q64_c1_n1`. A mixed example is `syn_pa_in_nr-w7_bw20-80M_q64-256_c1-4_s30-78p125k_n2`: NR and Wi-Fi 7, 20–80 MHz bandwidth, modulation orders 64–256, 1–4 OFDMA channels, 30–78.125 kHz subcarrier spacing, and two signals. Ranges describe the selected members, not every possible combination. `p` denotes PSK order when applicable; in numbers it replaces the decimal point. The exact configurations remain in the metadata.

Automatic names follow setup changes until edited. **Use automatic name** resumes this behavior. The PA Library suggests an editable `syn_pa_inout_` name, including the PA model; the corresponding output dataset uses `syn_pa_out_`. Different data with an existing name receives a numbered suffix. Existing waveforms and datasets are preserved.

Signal Analyzer selects a **Source dataset**, then a **Signal in dataset**. Paired datasets offer input and output for every capture. Switching signals updates the sampling rate, bandwidth, and sample range; click **Analyze signal** to update the results. Named input/output datasets persist in the workspace and download as ZIPs with separate I/Q CSVs and metadata for each signal. Signals with different sampling rates are never concatenated.

## Primary references

- [ETSI / 3GPP TS 38.104 V18.9.0](https://www.etsi.org/deliver/etsi_ts/138100_138199/138104/18.09.00_60/ts_138104v180900p.pdf), Tables 5.3.2-1 and 5.3.2-2.
- [ETSI / 3GPP TS 38.211 V18.5.0](https://www.etsi.org/deliver/etsi_ts/138200_138299/138211/18.05.00_60/ts_138211v180500p.pdf), modulation and OFDM signal generation.
- [IEEE TGax official records and specification framework](https://grouper.ieee.org/groups/802/11/Reports/tgax_update.htm).
- [IEEE TGbe official records and specification framework](https://grouper.ieee.org/groups/802/11/Reports/tgbe_update.htm).
- [IEEE TGbe resource-allocation discussion](https://www.ieee802.org/11/email/stds-802-11-tgbe/msg02302.html), aggregate RU sizes and zero-user signaling context.
- [NI: HE waveform timing and modulation](https://www.ni.com/en/solutions/semiconductor/wireless-connectivity-test/introduction-to-802-11ax-high-efficiency-wireless.html), supplemental vendor technical documentation.
