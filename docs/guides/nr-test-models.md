# NR standard test models

Studio 2.4.1 adds the NR FR1 base-station test models of **3GPP TS 38.141-1 V18.14.0, clause 4.9.2** to the [Signal Generator](signal-generator.md). The **5G NR** tab opens on **Standard 5G NR test model**: choose NR-FR1-TM1.1, 1.2, 2, 2a, 2b, 3.1, 3.1a, 3.1b, 3.2, 3.3 or NR-N-TM, then set the parameters of the model that the standard leaves open and that a baseband stimulus can express: duplex mode, number of carriers, subcarrier spacing and channel bandwidth of each carrier, gaps between carriers, rank for the time-alignment model, the PN-sequence reading, NB-IoT placement and power, oversampling and capture length. Everything the standard fixes is shown read-only under **Defined by the standard**. Some open parameters cannot be expressed, and Studio adds an extension and limits of its own (a free cell identity, up to 16 carriers, the NB-IoT boost range, the sample limit); [What you cannot set](#what-you-cannot-set) lists both.

!!! warning "What this is, and is not"
    The output is a **complex-baseband stimulus** that follows the standard's definition of each test model. It is meant for PA and DPD experiments: a signal with the resource-block structure, boost levels, DM-RS and PN23 data that the standard prescribes for its tests. It is **not** a certified conformance waveform, it was not compared with an instrument or a base station, and the **EVM** Studio shows is an ideal-demodulation diagnostic on a known grid, not the receiver EVM of TS 38.141-1 clause 6.5.3. The level is RMS-normalised, not a calibrated power. At default settings 35 of the 80 combinations of duplex mode, subcarrier spacing and bandwidth (10 of 40 FDD, 25 of 40 TDD) are also **shorter than the standard's 10 or 20 ms**, because a capture holds at most 1,000,000 samples ([how to read this](#capture-length-oversampling-and-the-sample-limit)). See [Limitations](#limitations).

## Generate a test model

1. Open **Signal Generator → Generate** and choose **5G NR**. The first visit selects NR-FR1-TM1.1, FDD, one 30 kHz carrier of 20 MHz.
2. Pick a model card. Settings the new model does not allow are repaired when you switch: NR-N-TM forces FDD and 15 kHz, and layers return to 1 for every model but TM1.1.
3. Set duplex, the carriers and the options below. Studio sends the recipe to the server about 0.4 s after your last edit and shows the derived sample rate, sample count, duration and a **Shortened** chip when the standard duration does not fit the sample limit. **Generate & preview** stays disabled while any selected test model has an edit that is not part of the signal yet (the server is still checking it, has refused it, or it makes the entry identical to another selected one), and **Save configuration** is disabled meanwhile for the highlighted entry, because it would write the previous settings. An unacceptable setup shows the server's reason, prefixed with the field it concerns (for example "Carrier 1 · Cell ID: ..."). An entry in one of these states carries a marker on its chip, and while it is not the one on screen a line under the button names it, gives the reason and offers **Show it**, which returns to that entry.
4. **Defined by the standard** lists the fixed parameters of the model and, for your settings, the resolved numbers per carrier: N_RB, subcarriers, FFT size, cell ID, centre offset, RBG size P, number and indices of boosted or de-boosted RBGs, every user with modulation, PRB ranges, start symbol and level, and for TDD the slot pattern.
5. **Add another test model** appends a second entry (a different model by default). A batch holds up to 16 signals; they keep their own sample rates and lengths and are never concatenated. The selected chips read like "TM3.1a · 100 MHz · 30 kHz · FDD"; when two entries would read alike, the chip adds the settings in which they differ. Two entries with identical settings are refused, because a dataset holds each signal once.
6. **Generate & preview**. The automatic dataset name carries the model: 100 MHz TM3.1a at 30 kHz gives `syn_pa_in_nrtm3p1a_bw98p31M_q256_c1_n1` (`nrntm` for NR-N-TM; the number after `c` counts carriers); see [dataset names](signal-presets.md#dataset-names-and-analysis).

The earlier preset **matrix** (940 NR presets by bandwidth, QAM and OFDMA channel count) is still available behind **Generic numerology matrix…** below the panel. It is labelled as generic engineering stimuli, selecting its cells works as before, and they join the same selection. See [Signal preset reference](signal-presets.md). Wi-Fi 6, Wi-Fi 7 and Custom are unchanged. If the server does not provide the test-model catalogue (a hosted Studio that predates the routes refuses the request with 403, an older server with 404), the NR tab silently falls back to the matrix. Any other failure of that request (a server error, no connection, a reply that is not a catalogue) also leaves the matrix in place but says so in a one-line notice with a **Retry** button. The Signal Generator page waits for the catalogue request to finish before it shows any family, Wi-Fi 6, Wi-Fi 7 and Custom included.

**Language.** The labels of the panel are translated into the nine interface languages. Text that the server writes is **English only** and appears unchanged in every language: model titles, descriptions and "used for" lists, the table of fixed parameters, validation messages, summary warnings and analysis notes.

## The eleven models

Common to all models (TS 38.141-1 tables 4.9.2.2-2, 4.9.2.2-3, 4.9.2.2-4): normal cyclic prefix; N_RB = the maximum transmission bandwidth configuration of TS 38.104 table 5.3.2-1; a **PDCCH** of one CCE (aggregation level 1) on PRB 0–2 in symbols 0 and 1 of every downlink slot, QPSK, PDCCH EPRE equal to its DM-RS EPRE; **PDSCH** mapping type A with DM-RS type 1, `dmrs-TypeA-Position` pos2 and one additional position, PDSCH EPRE equal to DM-RS EPRE; PDSCH users distinguished by n_RNTI, where n_RNTI 0 and 1 start at symbol 0 and n_RNTI 2 occupies PRB 0–2 from symbol 2; all levels relative to the PDCCH at 0 dB.

| Model | Clause | PDSCH content | Defined for tests on |
| --- | --- | --- | --- |
| NR-FR1-TM1.1 | 4.9.2.2.1 | QPSK. n_RNTI 0 on PRB 3…N_RB−1 and n_RNTI 2 on PRB 0–2, both 0 dB. | BS output power; transmit ON/OFF power; time alignment error (TAE); unwanted emissions (occupied bandwidth, ACLR, operating band unwanted emissions, spurious emissions); transmitter intermodulation; receiver spurious emissions |
| NR-FR1-TM1.2 | 4.9.2.2.2 | QPSK. x = 40 % of the PRBs 3…N_RB−1, in resource-block groups, boosted by +3 dB (n_RNTI 0); the remaining PRBs de-boosted (n_RNTI 1) so that the total power equals N_RB−3 PRBs at 0 dB; n_RNTI 2 on PRB 0–2. | ACLR; operating band unwanted emissions |
| NR-FR1-TM2 | 4.9.2.2.3 | One 64QAM PRB per slot (n_RNTI 2), 0 dB: PRB 0 in slots 3n, PRB ⌊N_RB/2⌋ in slots 3n+1, PRB N_RB−1 in slots 3n+2; all other PRBs unallocated. | Total power dynamic range (lower OFDM symbol TX power limit, at minimum power); EVM of a single 64QAM PRB; frequency error |
| NR-FR1-TM2a | 4.9.2.2.4 | As TM2 with a 256QAM PRB. | As TM2, 256QAM |
| NR-FR1-TM2b | 4.9.2.2.4A | As TM2 with a 1024QAM PRB. | As TM2, 1024QAM |
| NR-FR1-TM3.1 | 4.9.2.2.5 | As TM1.1 with both PDSCH users in 64QAM (the PDCCH stays QPSK). | Total power dynamic range (upper OFDM symbol TX power limit, at maximum power, all 64QAM PRBs); frequency error; EVM for 64QAM |
| NR-FR1-TM3.1a | 4.9.2.2.6 | As TM3.1 in 256QAM. | As TM3.1, 256QAM |
| NR-FR1-TM3.1b | 4.9.2.2.6A | As TM3.1 in 1024QAM. | As TM3.1, 1024QAM |
| NR-FR1-TM3.2 | 4.9.2.2.7 | x = 60 % of the PRBs as 16QAM resource-block groups de-boosted by 3 dB (n_RNTI 1, the EVM-measured user); the other PRBs QPSK, boosted to keep the total power at N_RB−3 PRBs (n_RNTI 0); n_RNTI 2 QPSK on PRB 0–2. | Frequency error; EVM for 16QAM |
| NR-FR1-TM3.3 | 4.9.2.2.8 | x = 50 % of the PRBs as QPSK resource-block groups de-boosted by 6 dB (n_RNTI 1, EVM-measured); the other PRBs QPSK boosted to keep the power (n_RNTI 0); n_RNTI 2 on PRB 0–2. | Frequency error; EVM for QPSK |
| NR-N-TM | 4.9.2.2.9 | NR-FR1-TM1.1 on the NR carrier with one NR resource block punctured by an NB-IoT carrier (the N-TM of TS 36.141), boosted by X dB. Generated for FDD at 15 kHz only; the standard also defines a TDD form (table 4.9.2.2-1a) that Studio does not implement. | NB-IoT in-band tests: occupied bandwidth, ACLR, operating band unwanted emissions, spurious emissions, transmitter intermodulation, receiver spurious emissions, NB-IoT RB power dynamic range |

The rest of each model is derived from the clause; the **Defined by the standard** table of the panel prints the exact numbers for your bandwidth.

### Boosted and de-boosted groups (TM1.2, TM3.2, TM3.3)

With N = N_RB−3 PRBs from PRB 3, the nominal RBG size P comes from TS 38.214 table 5.1.2.2.1-1, configuration 1: 2, 4, 8 or 16 PRBs for N up to 36, 72, 144 or 275. The first RBG is the partial group that starts at PRB 3 and has P − (3 mod P) PRBs. The number of target groups is

`N_RBG = min( ⌊x·N / P⌋ , ½(⌊(N + (3 mod P) − P) / P⌋ − (⌊(N + (3 mod P) − P) / P⌋ mod 2)) + 1 )`

and the target groups are (0-based) `⌊(N + (3 mod P)) / P⌋ − 1` and, if N_RBG > 1, groups 1, 3, …, 2(N_RBG−2)+1. The complementary PRBs get the level `10·log10((N − 10^(L/10)·P·N_RBG) / (N − P·N_RBG))` dB, which keeps the PRB powers summing to N. The generator evaluates this with integer arithmetic. For TM1.2 the target groups are boosted by L = +3 dB (n_RNTI 0); for TM3.2 (L = −3 dB) and TM3.3 (L = −6 dB) the target groups are n_RNTI 1 and the complement is n_RNTI 0. DM-RS follow the PDSCH level of their allocation.

## What you can set

Entries marked *Studio extension* or *Studio limit* are not options of the standard.

| Parameter | Allowed values | Notes |
| --- | --- | --- |
| Test model | the eleven models above | |
| Duplex | FDD or TDD | The standard duration, which is the default capture length, is one radio frame (10 ms) for FDD and two (20 ms) for TDD. NR-N-TM is generated for FDD only: the standard also defines a TDD form of it (Table 4.9.2.2-1a) that Studio does not implement. |
| Number of carriers | 1–16 | Each carrier is an independent copy of the chosen model with its own settings. The standard asks for as many carriers as the base station supports and sets no upper bound: 16 is a *Studio limit*. |
| Subcarrier spacing (per carrier) | 15, 30 or 60 kHz | FR1 numerologies. NR-N-TM: 15 kHz. |
| Channel bandwidth (per carrier) | 15 kHz: 3, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50 MHz · 30 kHz: 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100 MHz · 60 kHz: 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100 MHz | The 40 cells of TS 38.104 table 5.3.2-1; the bandwidth sets N_RB (and with it the number of subcarriers, 12·N_RB). NR-N-TM: 3–50 MHz. |
| Cell identity (per carrier) | automatic, or 0–1007 | Automatic = N_ID^cell 1 for the lowest carrier, 2 for the next, … The standard fixes this rule (TS 38.141-1 4.9.2.3), so any other value is a *Studio extension*: every deviating carrier adds a warning to the summary, and the signal keeps its *3GPP test model* label. The identity drives scrambling and DM-RS. |
| Gap before carrier (carriers 2 and up) | 0, or a multiple of 0.3 MHz up to 999.9 MHz | Added to the nominal channel spacing; > 0 starts a new sub-block (non-contiguous operation). A value within 10⁻⁹ MHz of a multiple is stored as that exact multiple, so equal gaps give equal signals and identifiers. Carriers and gaps together must still fit the sample limit, which allows about 835 MHz in all (see [Several carriers](#several-carriers)). |
| Layers | 1 or 2 | Two layers (rank 2, ports 1000 and 1001, no precoding) exist for NR-FR1-TM1.1 only, as the standard defines it for the TAE test. |
| Exported port | 1000 or 1001 | With two layers the generator exports one of the two streams. |
| PN23 data convention | Standard text or MATLAB 5G Toolbox | [Below](#data-content-and-the-two-pn-conventions). |
| PN23 scope (TDD) | downlink symbols only, or all symbols | The standard allows both. Offered for TDD with the Standard text convention only; the MATLAB convention ignores it. See [the special slot](#data-content-and-the-two-pn-conventions). |
| NB-IoT position (NR-N-TM) | lower, upper or both | The eligible in-band RB nearest the lower or upper minimum guard band, or one at each edge. |
| NB-IoT power boost X (NR-N-TM) | 6–12 dB for hosts of 3 to 20 MHz, 3–12 dB for hosts of 25 MHz and wider; default 6 | X is the declared NB-IoT RB power dynamic range, at least +6 dB (+3 dB at other positions; TS 38.104 Table 6.3.4.2-1, TS 38.141-1 D.43). Studio refuses a boost below the minimum that applies to the lowest and highest eligible RB of the host bandwidth and says why. The 12 dB maximum is a *Studio limit*; the standard sets none. |
| Oversampling | automatic, 1, 2, 4 or 8 | Not part of the standard; see [capture length](#capture-length-oversampling-and-the-sample-limit). |
| Capture length | standard duration (default), or 1–20 subframes | Standard: 10 ms FDD, 20 ms TDD. Fewer subframes give a shorter capture; more than the standard repeat the pattern frame by frame. |

Through **Advanced** you can still set the generic impairment and level controls of any Signal Generator configuration: RMS (default 0.2), seed, RF carrier metadata, burst envelope, constant phase and phase jitter, frequency offset, I gain and Q phase mismatch, I/Q DC offsets, SNR (white Gaussian noise), clipping and the optional ideal band-edge filter (**off** for test models: the standard specifies no filtering or windowing for them, so they are generated unfiltered, and filtering changes ACLR and EVM). The FFT size, sample rate, bandwidth, channel list, pilots and length controls disappear for a test model because they are derived from the recipe. The seed only changes the added noise and phase jitter; the test-model data themselves are deterministic.

**Fixed by the standard** (shown, not editable): the modulation of every user, the boost target x (40 / 60 / 50 %), the boost and de-boost levels, the RBG size P, the number of users and the n_RNTI mapping of table 4.9.2.3.2-1, the PDCCH and PDSCH configuration of the common tables, DM-RS type and positions, PDSCH/PDCCH power ratios, normal CP and the standard duration (10 ms FDD, 20 ms TDD), which is the default capture length.

## What you cannot set

The panel does not offer the following. The first group are parameters that the standard leaves open and that Studio does not express; the second group are choices Studio makes or adds.

**Open in the standard, not offered by Studio**

* **Channel-spacing rules.** TS 38.141-1 only says that the nominal channel spacing of TS 38.104 clause 5.4.1 applies. Studio always uses clause 5.4.1.2 for operating bands with a 100 kHz channel raster. Not offered: the rule of clause 5.4.1.1 for adjacent carriers, (BW₁ + BW₂)/2, which gives 20.0 MHz for two 20 MHz carriers; the 15 kHz- and 60 kHz-raster variants of clause 5.4.1.2 (19.98 or 19.995 MHz for two 20 MHz carriers on a 15 kHz raster, where Studio gives 19.8 MHz); and an operating band. A gap cannot reach these spacings: it adds multiples of 0.3 MHz to the nominal spacing, and they lie 0.2, 0.18 and 0.195 MHz above Studio's 19.8 MHz. The guard bands are a separate matter. Clause 5.4.1.2 takes them at μ₀, the largest subcarrier spacing that the operating band supports for both bandwidths. Studio has no operating-band field and reads table 5.3.3-1 by a fixed rule with three branches: at the larger of the two carriers' subcarrier spacings if both bandwidths exist for it; otherwise at the smaller one if both exist for it; otherwise each carrier takes the guard band of its own subcarrier spacing (30 of the 1,600 ordered pairs of table cells, for example 3 MHz at 15 kHz next to 60–100 MHz at 30 kHz). Carriers of equal bandwidth are not affected. The guard bands of another subcarrier spacing at which both bandwidths exist would give a different nominal spacing for 526 of the 1,600 ordered pairs of table cells (34 of the 115 pairs of bandwidths that exist at two or more spacings), by 0.3 or 0.6 MHz, and by 0.9 MHz for 5 MHz at 15 kHz next to 60 to 90 MHz at 60 kHz (5 + 30 MHz: 17.1 MHz with the guard bands of 15 kHz, 16.8 MHz with those of 30 kHz). A gap reaches such a variant when it is larger than the spacing Studio gives, 328 of the 526 pairs (5 and 30 MHz carriers at 30 kHz: Studio gives 16.8 MHz, and a gap of 0.3 MHz gives 17.1 MHz). It does not reach a smaller one, 192 pairs (5 and 30 MHz carriers at 15 kHz: Studio gives 17.1 MHz, not the 16.8 MHz of the guard bands of 30 kHz), because the spacing cannot be reduced below the nominal one, although clause 5.4.1.2 allows smaller multiples of the least common multiple of raster and subcarrier spacing for contiguous aggregation. In the remaining 6 pairs one variant is larger and one smaller.
* **Test configurations NRTC1 to NRTC6.** You choose the carriers; Studio does not pick the number of carriers that fit a declared Base Station RF Bandwidth, does not place them from the upper edge, and does not generate the ON-OFF patterns of NRTC6 (n46, n96, n102).
* **NR-N-TM with TDD.** The standard defines a TDD form of the NB-IoT in-band test model (TS 38.141-1 Table 4.9.2.2-1a: 15 kHz, 5 ms, D S U U D). Studio does not implement it in this release, so NR-N-TM is FDD only.
* **PN23 polarity and stream sharing.** The sequence is the non-inverted shift-register output, like MATLAB's, and there is no inversion option ([below](#data-content-and-the-two-pn-conventions)). The standard says the PN sequence "can be generated per user"; the readings offered are one stream per user and one stream per PDSCH allocation, not a single stream shared by all users.
* **One exported antenna port.** Rank 2 of TM1.1 needs two streams (ports 1000 and 1001); Studio exports one per signal, so the time-alignment stimulus is two separately generated signals. The NB-IoT reference signals are exported for one port (NRS port 2000).
* **Equal power for NRTC3.** TS 38.141-1 4.7.3.2 and 4.7.4.2 (NRTC1, NRTC2) ask for the same power spectral density on every carrier; 4.7.5.2 (NRTC3) asks for the same *power* on every carrier. Studio always uses equal power spectral density, which is the same thing for carriers of equal bandwidth.

**Added by Studio, not options of the standard**

* **A cell identity** other than N_ID^cell = n for the n-th lowest carrier (TS 38.141-1 4.9.2.3). Each deviating carrier adds a warning; the signal keeps its *3GPP test model* label.
* **Up to 16 carriers.** The standard sets no maximum.
* **An NB-IoT boost of at most 12 dB.** The standard declares a minimum, not a maximum; Studio enforces the minimum that applies to the host bandwidth.
* **The sample limit.** A capture holds at most 1,000,000 samples and the sample rate is at most 2 GS/s, which shortens or refuses wide captures (see [Capture length](#capture-length-oversampling-and-the-sample-limit)).

## Resource blocks, FFT size and sample rate

N_RB per bandwidth and subcarrier spacing is the table 5.3.2-1 reproduced in the [preset reference](signal-presets.md#nr-bandwidths): 11 bandwidths at 15 kHz (3–50 MHz, N_RB 15–270), 15 at 30 kHz (5–100 MHz, 11–273) and 14 at 60 kHz (10–100 MHz, 11–135). 3 MHz exists only at 15 kHz, 5 MHz only at 15 and 30 kHz, and bandwidths above 50 MHz only at 30 and 60 kHz. Any pair outside the table is refused with the list of valid bandwidths. The table is offered in full; TS 38.104 table 5.3.5-1 additionally restricts which cells each operating band uses (for instance the 30 kHz / 5 MHz cell, N_RB 11, is in table 5.3.2-1 but in no band row), and Studio does not apply that filter.

Each carrier's resource grid has 12·N_RB subcarriers. Row k (0 = lowest subcarrier) sits at FFT bin k − 6·N_RB, so subcarrier 6·N_RB is exactly at 0 Hz and is **used**: the DC subcarrier is not nulled and there is no half-subcarrier shift. This is the F_REF position of TS 38.104 clause 5.4.2.2 for a single carrier.

The FFT size is the smallest power of two N ≥ 128 with 12·N_RB / N ≤ 0.85, the rule of MATLAB's `nrOFDMInfo`. The native sample rate of a single carrier is N × SCS. The occupied fraction 12·N_RB / N therefore varies between 44.5 % and 84.4 % over the 40 table cells. The cyclic prefix is the normal CP of TS 38.211 clause 5.3.1: 144·N/2048 samples for every symbol, plus 16·2^μ·N/2048 samples for the symbols l = 0 and l = 7·2^μ of every subframe. There is no windowing and no filtering between symbols.

## Capture length, oversampling and the sample limit

A Studio capture holds at most **1,000,000 complex samples**. The sample rate is the native rate times the oversampling factor, and the capture is a whole number of 1 ms subframes.

* **Oversampling automatic** picks the largest of 4, 2, 1 at which the standard duration (10 or 20 ms, or your custom subframe count if longer) fits within 1,000,000 samples; it falls back to 1.
* If even at oversampling 1 the standard duration does not fit, the capture is **shortened** to the largest whole number of subframes that fits; the panel shows the **Shortened** chip and the signal analysis carries the warning. A narrower bandwidth or fewer carriers restore the full duration. You can also ask for fewer subframes yourself.
* If one subframe alone exceeds 1,000,000 samples, or the sample rate would exceed 2 GS/s, the setup is refused with a message that names the numbers.

| Setup | N_RB | Native rate | Automatic oversampling | Sample rate | Samples | Duration |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| FDD, 15 kHz, 3 MHz | 15 | 3.84 MS/s | 4 | 15.36 MS/s | 153,600 | 10 ms |
| FDD, 15 kHz, 5 MHz | 25 | 7.68 MS/s | 4 | 30.72 MS/s | 307,200 | 10 ms |
| FDD, 60 kHz, 10 MHz | 11 | 15.36 MS/s | 4 | 61.44 MS/s | 614,400 | 10 ms |
| FDD, 30 kHz, 20 MHz (default) | 51 | 30.72 MS/s | 2 | 61.44 MS/s | 614,400 | 10 ms |
| TDD, 15 kHz, 20 MHz | 106 | 30.72 MS/s | 1 | 30.72 MS/s | 614,400 | 20 ms |
| FDD, 30 kHz, 100 MHz | 273 | 122.88 MS/s | 1 | 122.88 MS/s | 983,040 | 8 ms (standard 10 ms, shortened) |
| TDD, 15 kHz, 50 MHz | 270 | 61.44 MS/s | 1 | 61.44 MS/s | 983,040 | 16 ms (standard 20 ms, shortened) |

These are the values the resolver returns today; **Defined by the standard** shows the same numbers for any setup. A capture shorter than the standard is still a valid stimulus for PA and DPD training but a measurement over the standard's 10 or 20 ms needs another route (smaller bandwidth, fewer carriers, or an external generator). A TDD capture shorter than 5 ms contains an incomplete downlink/uplink pattern and carries a warning.

### How many captures are shortened

**35 of the 80 combinations are shortened at default settings.** The 80 are the 40 (subcarrier spacing, bandwidth) cells of TS 38.104 table 5.3.2-1 in each of the two duplex modes, for one carrier with automatic oversampling and the standard capture length. The number is the same for each of the ten NR-FR1 models; NR-N-TM (FDD, 15 kHz, 3–50 MHz) is never shortened.

| Duplex | Shortened | Cells | Capture | Not shortened |
| --- | ---: | --- | --- | --- |
| FDD | 10 of 40 | 30 and 60 kHz, 60 to 100 MHz | 8 of 10 ms | the other 30 |
| TDD | 25 of 40 | 15, 30 and 60 kHz, 30 to 50 MHz | 16 of 20 ms | the other 15 |
| | | 30 and 60 kHz, 60 to 100 MHz | 8 of 20 ms | |

How to read this. It is a limitation of this release, documented rather than hidden: every shortened capture is marked by the **Shortened** chip and by a warning that appears in the summary and in the notes of the analysis. A shortened capture is a valid PA and DPD stimulus with the correct resource-block structure, but it is not the standard-length signal. An 8 ms TDD capture never reaches the second radio frame, so the frame-boundary restart of the PN sequence and of the TM2 slot classes does not occur in it (a 16 ms capture reaches it). Studio offers no route to the full duration for these combinations, because a capture is limited to 1,000,000 samples at every oversampling; a measurement over the standard's 10 or 20 ms needs a smaller bandwidth, fewer carriers or an external generator. Setups with several carriers reach the limit sooner: sixteen 10 MHz carriers at 15 kHz already shorten to 4 of 10 ms.

## Several carriers

Every carrier is its own test model with its own N_RB, cell identity, PN streams and resource grid, synthesised at its own FFT size and summed after a time-domain frequency shift. The rules follow TS 38.141-1 clauses 4.7 and 4.9.2.3 and TS 38.104 clause 5.4.1:

* **Order and cell identity.** Carriers are ordered by ascending frequency. N_ID^cell is 1 for the lowest, 2 for the second lowest and so on, unless you override it; an override is a Studio extension and adds a warning. Two carriers with the same numerology, bandwidth and cell ID would carry identical data; Studio warns.
* **Nominal channel spacing.** Adjacent carriers are separated, reference frequency to reference frequency, by the nominal channel spacing of TS 38.104 clause 5.4.1.2 for bands with a 100 kHz channel raster: `S = ⌊(BW₁ + BW₂ − 2·|GB₁ − GB₂|) / 0.6 MHz⌋ × 0.3 MHz`. GB is the minimum guard band of table 5.3.3-1, taken at the larger of the two carriers' subcarrier spacings if both bandwidths exist for it, else at the smaller one if both exist there, else (30 of the 1,600 ordered pairs, for example 3 MHz at 15 kHz next to 60 MHz at 30 kHz) for each carrier at its own subcarrier spacing. Examples: 20 + 20 MHz → 19.8 MHz; 100 + 100 MHz → 99.9 MHz; 5 + 100 MHz at 30 kHz → 51.9 MHz; 10 + 10 MHz → 9.9 MHz; 3 + 3 MHz → 3.0 MHz. Other rules of clause 5.4.1 and the operating-band dependence of the guard bands are not offered ([What you cannot set](#what-you-cannot-set)).
* **No overlap.** The occupied subcarriers of neighbours must not overlap; otherwise the setup is refused and says by how much they would. At the nominal spacing all 1,600 ordered pairs of table 5.3.2-1 cells (any bandwidths, any of the three spacings) are accepted, and a unit test asserts that the tightest pair, 3 MHz at 15 kHz next to 80 MHz at 60 kHz, still leaves 7.5 kHz between occupied subcarriers. The check only matters for gaps and rules that change the spacing.
* **Gap before a carrier.** `gap_before_mhz` adds to the spacing between a carrier and its lower neighbour, in multiples of 0.3 MHz (the least common multiple of the 100 kHz raster and the subcarrier spacings) up to 999.9 MHz. A gap models the sub-block gap of non-contiguous operation (NRTC3). It cannot reduce the spacing below nominal.
* **Placement.** The aggregate is centred at baseband DC by its outer edges (TS 38.104 clause 5.3A.2); a single carrier keeps its reference subcarrier at DC. Only relative offsets matter in baseband; the RF carrier frequency in Advanced is metadata.
* **Equal power spectral density.** Every carrier uses the same resource-element amplitude, so all carriers have equal power spectral density as NRTC1 and NRTC2 require. Each carrier is synthesised with the unitary scaling `√N_c · ifft(grid)` so this holds across subcarrier spacings. There is no per-carrier power offset; the equal *power* wording of NRTC3 for carriers of different bandwidth is not reproduced.
* **Mixed numerologies.** Carriers with different subcarrier spacings are allowed (each follows its own numerology; slot grids align at subframe boundaries). The standard text does not describe this case, the signals are unwindowed, and the ideal-demodulation EVM then includes the inter-numerology leakage of the other carriers (about 0.9 % overall for a 20 MHz / 15 kHz carrier next to a 100 MHz / 30 kHz carrier in a check run for this guide). Studio shows a warning.
* **Sample rate.** The common sample rate is the smallest 15 kHz·2^p at which each carrier gets its FFT size and the aggregate occupies at most 85 % of it, times the oversampling. Four 100 MHz carriers at 30 kHz work (491.52 MS/s, so only 2 ms fit in 1,000,000 samples). Five to eight of them still fit, as one 1 ms subframe at 983.04 MS/s (983,040 samples); nine are refused, because one subframe would need 1,966,080 samples and at most 1,000,000 can be generated. Carriers and gaps together can therefore span at most about 835 MHz (85 % of 983.04 MHz).

In the panel the number-of-carriers stepper clones the last row; the row buttons add or remove single carriers, and removing carrier 1 makes the next carrier the first one (its gap becomes 0). Sixteen carriers of 3 MHz at 15 kHz need 61.44 MS/s and fit the whole 10 ms. More than sixteen carriers are refused: 16 is a Studio limit, and the panel says so.

## TDD patterns

TDD models use the pattern of TS 38.141-1 table 4.9.2.2-1, written with the `TDD-UL-DL-ConfigCommon` fields of TS 38.331 and a 5 ms period that repeats for the 20 ms duration:

| Subcarrier spacing | Slots per period | Downlink slots | Special-slot symbols | Uplink slots | Slot sequence |
| --- | ---: | ---: | --- | ---: | --- |
| 15 kHz | 5 | 3 | 10 downlink, 2 uplink | 1 | D D D S U |
| 30 kHz | 10 | 7 | 6 downlink, 4 uplink | 2 | D×7 S U U |
| 60 kHz | 20 | 14 | 12 downlink, then 8 uplink in the next slot | 4 | D×14 S S U×4 |

The symbols between the downlink and the uplink symbols of the special slots are flexible. **Flexible and uplink symbols are not transmitted: those samples are exactly zero** (a switch without transient shaping). The PDCCH is present in every downlink and special slot; uplink slots carry nothing. In the special slot the PDSCH occupies the downlink symbols only (the DM-RS symbols follow TS 38.211 table 7.4.1.1.2-3 for that length: l = 2 and 11 for 14 symbols, 2 and 9 for 10 or 12, 2 for 6). The standard also defines a pattern for NB-IoT in-band operation (table 4.9.2.2-1a: 15 kHz, 5 ms, D S U U D with 10 downlink and 2 uplink symbols in the special slot); Studio does not generate it, because it does not implement NR-N-TM with TDD.

With carriers of different subcarrier spacings in one TDD recipe, each carrier follows the pattern of its own numerology. The summary's `tdd_pattern` lists the pattern of carrier 1 and a warning says so; the panel shows one labelled pattern for each distinct spacing.

## Data content and the two PN conventions

The data of every PDCCH and PDSCH are bits of the **PN23** sequence generator (ITU-T O.150), `s(i) = s(i−23) xor s(i−18)` with the seed `s(0…22) = 1`, scrambled with the Gold sequence of TS 38.211 clause 5.2.1 (PDSCH `c_init = n_RNTI·2^15 + N_ID^cell`; PDCCH `c_init = N_ID^cell` with n_RNTI = 0), QPSK/16/64/256/1024QAM mapped per clause 5.1 with unit average power, with no channel coding and no CRC. The PDSCH of each user carries `Q_m × (allocated resource elements − DM-RS resource elements)` bits per slot; the PDCCH carries 108 bits. DM-RS use the Gold sequences of clauses 7.4.1.1.1 and 7.4.1.3.1 with n_SCID = 0, on the even subcarriers (comb of DM-RS type 1); the odd subcarriers of a DM-RS symbol carry PDSCH data.

The standard says the PN generator is "initialized with a starting seed of all ones in the first allocated slot of each frame", continuous over slot boundaries, that for TDD it may run over all symbols or the downlink symbols only, and that for multi-user models it "can be generated per user". Two readings are implemented:

| Data convention | PN streams | Restart |
| --- | --- | --- |
| **Standard text** (default) | One stream per n_RNTI user and one for the PDCCH. TDD: downlink symbols only, or all symbols as if every slot were a full downlink slot (PN23 scope). | Restarted in the first allocated slot of every radio frame. |
| **MATLAB 5G Toolbox** | One stream per PDSCH allocation object (for TM2, three independent streams by slot number modulo 3; for TDD a separate stream for the special slot), one for the PDCCH. The scope option has no effect. | Started once, never restarted. |

In **one 10 ms FDD frame of TM1.1, TM1.2 and TM3.x the two conventions give identical grids** (checked for several bandwidths and numerologies). They differ in the random bits only, and only for TM2/2a/2b, TDD (the special slot, and the second frame of a standard-length capture) and captures longer than one frame; the structure and spectral shape do not depend on the choice. Choose *MATLAB 5G Toolbox* to compare against that product: the resource grids of the 438 configurations compared with MATLAB ([Validation evidence](#validation-evidence)) are those of this convention. The default *Standard text* convention is the reading of the standard's own wording; it agrees with an independent from-the-text implementation (58 cases) and differs from MATLAB's random bits where the list above says so.

**PN23 scope in the special slot.** With *All symbols*, every slot (uplink slots included) draws from each user's stream the bits of a full 14-symbol slot, so the stream advances as if every slot were a full downlink slot. In a special slot only the first bits of that draw are used, and they are mapped onto the resource elements of the shortened downlink allocation (for 10 downlink symbols the DM-RS are at symbols 2 and 9). The other possible reading maps the bits of the full slot onto the full slot's resource-element grid (DM-RS at 2 and 11) and transmits only the downlink symbols: it uses the same bits but places them differently from the symbol where the DM-RS positions differ. The standard does not decide between the two and MATLAB does not implement the scope; Studio uses the first.

**PN23 polarity.** The first 23 PN23 bits are the all-ones seed. ITU-T O.150 (05/96) gives a longest run of 23 zeros for its PRBS23 test signal, which it labels an inverted signal, whereas the literal "all ones" seed used here, like MATLAB, gives the non-inverted output (23 leading ones). No inversion option is offered, and which polarity a given instrument uses was not established. A single PN stream shared by all users, which the wording "can be generated per user" would also allow, is not offered either.

### Two layers (TM1.1)

With layers = 2, the PDSCH uses layer mapping `x^(i)(j) = d(2j+i)` (TS 38.211 clause 7.3.1.3), DM-RS of port 1001 are `r(m)·(−1)^(m mod 2)` on the same comb (clause 7.4.1.1.2, `w_f(k') = (+1, −1)`), there is no precoding and no 1/√2 scaling, and twice the bits per slot are drawn. The PDCCH exists only on its own port and is written to the port 1000 stream, so the exported port-1001 stream carries PDSCH and DM-RS but no PDCCH. The standard defines the two-layer model for the TAE test; the TAE requires both streams, and Studio exports one.

## NR-N-TM: NB-IoT operation in NR in-band

NR-N-TM (clause 4.9.2.2.9) is NR-FR1-TM1.1 with one NR resource block punctured by an NB-IoT carrier carrying the **N-TM of TS 36.141**. Studio implements it for **FDD at 15 kHz** (3–50 MHz):

* **Numerology.** An NB-IoT carrier is 180 kHz wide, which is one NR resource block only at 15 kHz. The test-model definition (the TDD configuration of table 4.9.2.2-1a and the N-TM of TS 36.141) and the NB-IoT receiver tables cover 15 kHz hosts of 3 to 50 MHz. The NB-IoT RB power dynamic range of TS 38.104 table 6.3.4.2-1 also lists host bandwidths of 60 to 100 MHz, which exist only at 30 and 60 kHz, and TS 38.104 clause 5.2 lists band n41, which supports 30 kHz, among the NB-IoT bands. No clause describes how an NB-IoT carrier would be built in a 30 or 60 kHz host, so the restriction to 15 kHz follows the better-supported reading; it is a Studio decision, not a statement that the standards exclude other hosts.
* **TDD.** The standard defines a TDD form of this test model (TS 38.141-1 table 4.9.2.2-1a: 15 kHz, 5 ms, D S U U D, with 10 downlink and 2 uplink symbols in the special slot; the NB-IoT test model of TS 36.141 clause 6.1.3 uses uplink/downlink configuration 1 and special-subframe configuration 7 over two frames). Studio does not implement it in this release, and there is no reference implementation to check it against (MATLAB has no NR-N-TM); a TDD recipe is refused and the message says so.
* **NB-IoT content** per TS 36.211 clause 10.2 and TS 36.141 clauses 6.1.3–6.1.6: NPSS (subframe 5), NSSS (subframe 9), NRS on port 2000 with the port-2001 positions reserved, NPBCH (subframe 0), NPDCCH format 1 (subframe 1) and NPDSCH (the remaining subframes, n_RNTI 1000), QPSK, all-zero data, one 10 ms frame that repeats. N_ID^Ncell = 97 + 6n + max(0, m−1) for the n-th NB-IoT host and host cell identity m, i.e. 103 for the lowest host with cell ID 1 and 110 for the second; the lower and upper NB-IoT carrier of one host share it. With several NR carriers the lower NB-IoT carrier sits in the lowest NR carrier and the upper one in the highest, and only these hosts carry NB-IoT.
* **Position.** The resource block is the *eligible in-band position closest to the NR minimum guard band*. The eligible positions form a comb with a spacing of 5 RB derived from the NB-IoT anchor-carrier raster (TS 36.104 clause 5.7.3), assuming the NR reference frequency is a multiple of 100 kHz; the standards define no explicit table. The lowest/highest eligible NR RB: 3 MHz 2/12, 5 MHz 2/22, 10 MHz 0/50, 15 MHz 4/74, 20 MHz 2/102, 25 MHz 1/131, 30 MHz 4/159, 35 MHz 3/183, 40 MHz 2/212, 45 MHz 0/240, 50 MHz 4/269. For each host carrier the panel shows how many positions are eligible and from which RB to which (carrier 1 hosts the lower NB-IoT carrier, the last carrier the upper one).
* **Power** (clause 4.9.2.2.9): average power per RE `P_avg = P_rated / (12·N_RB)`; a boosted NB-IoT RE has `P_avg·10^(X/10)`; the NR REs share the remainder. The generator balances the energy over the whole 10 ms frame (every RE of the grid loaded at unit amplitude), so the average power of a host carrier equals that of a carrier without NB-IoT. X is the declared NB-IoT RB power dynamic range; TS 38.104 table 6.3.4.2-1 sets a minimum of +6 dB (+3 dB at other positions), which for the lowest and highest eligible RB is 6 dB for hosts of 3 to 20 MHz and 3 dB for hosts of 25 MHz and wider. Studio enforces that minimum and accepts X up to 12 dB: the upper limit is a Studio limit, the standard sets none. A boost that would leave no power for the NR resource elements is refused as well; the only such limit below 12 dB is about 10.65 dB for two NB-IoT carriers in a 3 MHz host (13.66 dB for one). For one NB-IoT carrier and X = 6 dB, NR resource elements are about 0.17 dB below P_avg at 10 MHz and 0.63 dB below at 3 MHz. Balancing every OFDM symbol on its own would shift these by up to a few tenths of a dB.
* **Resolved ambiguities**, because the standards leave them open: resource elements of the punctured RB that carry no NB-IoT signal (the control region, host reference-signal positions, muted NRS port-2001 positions) keep the NR content; the 304/200-bit NPDCCH/NPDSCH budgets of TS 36.141 correspond to a two-port NRS budget, which is reproduced; each NPDSCH subframe is its own codeword; NPDSCH is QPSK only (the 16QAM variant exists only in TS 36.141's own EVM test); the NSSS is sent in every looped frame.

The NB-IoT symbols, resource-element sets and mapping order were compared with the MATLAB LTE Toolbox NB-IoT functions (see [Evidence](#validation-evidence)). A complete NR-N-TM waveform has no reference implementation to compare with.

## Reading the preview

The preview is the standard Signal Generator preview: time plot, Welch PSD, CCDF and PAPR, occupied bandwidth, constellation, EVM trends. For test models:

* The **allocation** plot shows each carrier in MHz around baseband DC with its subcarrier spacing and N_RB; per-carrier metrics (N_RB, cell ID, centre offset, EVM) are in the analysis.
* **Constellation and EVM** are computed on the PDSCH data resource elements of every carrier: the carrier's frequency shift is undone, every OFDM symbol's useful part is transformed, and the result is compared with the transmitted grid without equalisation, gain or phase fitting, EVM window or timing search. Without impairments this is a numerical floor of about 10⁻⁶ % (the exported samples are single-precision). It rises with the impairments you add and with filtering. EVM trend plots refer to the lowest carrier, and the symbol count shown next to the EVM is the number of OFDM symbols of that one carrier grid that carry PDSCH data, not carriers times symbols; the EVM and symbol count of each carrier are in its allocation entry.
* The coverage label reads **3GPP test model**. The analysis notes state the data convention, truncation, the TDD handling and the EVM definition.
* The level is normalised to `rms`; PAPR and CCDF are those of the signal at the chosen sample rate.

## Limitations

* **Content:** PDCCH, PDSCH and DM-RS only, as the clause defines the test models. No SSB, CSI-RS, PT-RS, PRACH, channel coding, CRC or HARQ. One exported antenna port.
* **Not a conformance tool:** no instrument or base-station comparison was made; no claim of TS 38.141-1 conformance is made; EVM is a diagnostic (no equaliser, no EVM window, no 10 ms measurement-period averaging as in clause 6.5.3).
* **Calibration:** RMS-normalised baseband; no calibrated power, no OSTP or total-power-dynamic-range measurement.
* **FR2 and other models:** the FR2 models of TS 38.141-2 and subcarrier spacings of 120 kHz and above are not included. Clause 4.9.2.2 of V18.14.0 defines exactly the eleven FR1 models above.
* **Open parameters that are not offered:** the channel-spacing rules of TS 38.104 clause 5.4.1 other than the 100 kHz-raster rule of 5.4.1.2 (and the operating-band μ₀), the NRTC1 to NRTC6 configurations as such and the NRTC6 ON-OFF patterns, NR-N-TM with TDD, PN23 polarity and a stream shared by all users, a second exported antenna port, and equal carrier power (instead of equal power spectral density) for NRTC3. See [What you cannot set](#what-you-cannot-set).
* **Studio extensions and limits:** a cell identity other than N_ID^cell = n is a Studio extension; 16 carriers, an NB-IoT boost of at most 12 dB, 1,000,000 samples per capture and 2 GS/s are Studio limits.
* **Frequency placement:** baseband only; no channel-raster alignment and no upconversion.
* **Releases:** definitions are those of TS 38.141-1 V18.14.0 (Release 18); other releases were not compared.
* **Duration:** 35 of the 80 combinations of duplex mode, subcarrier spacing and bandwidth are shortened at default settings (10 of 40 FDD, 25 of 40 TDD), and wide or multi-carrier setups are shortened sooner; see [how many captures are shortened](#how-many-captures-are-shortened). This is a documented limitation, not a defect that the panel hides.
* **TDD:** no transmit ON/OFF transient shaping; samples outside downlink symbols are exactly zero.
* **PN readings:** two readings of the PN rules are provided; the data bits of a given instrument may match neither.
* **Language:** the text that the server writes (model descriptions, the table of fixed parameters, validation messages, warnings and notes) is English only and is not translated into the other interface languages. The new panel labels in the eight non-English languages are translations that no native speaker has reviewed.

## Validation evidence

What was compared with what, and where the evidence is kept. Unless a line says "in the repository", the MATLAB data and scripts behind a number are **not** stored in the repository. MATLAB's test-model wrapper has no NR-N-TM, no two-layer TM1.1, no multi-carrier test model and no 3 MHz cell.

* **Resource grids against MATLAB R2026a, 5G Toolbox 26.1 (the author's run, 438 configurations).** MATLAB generated the ten FR1 models for FDD and TDD at 15, 30 and 60 kHz, with several bandwidths (TM1.2 and TM3.3 FDD at all 39 (SCS, bandwidth) cells MATLAB accepts), cell IDs from 1 to 1007 and 1 to 40 subframes: **438 configurations**. The package's resource grid for `data_convention="matlab"` was compared with MATLAB's in all 438: **414 are bit-identical and the other 24 (TM1.2 and TM3.3, whose boosted amplitudes are `10^(P/20)`) differ by at most 3.2×10⁻¹⁶**, one unit in the last place of a power computation. This comparison used the **MATLAB** data convention; the default **Standard text** convention differs from MATLAB's random bits for TDD, TM2/2a/2b and captures longer than one frame (see [Data content](#data-content-and-the-two-pn-conventions)).
* **Waveforms (the same run, 58 configurations).** For 58 of the 438 configurations MATLAB's time-domain waveform was saved as well. The package's OFDM modulator (IFFT and cyclic prefix, no windowing), given MATLAB's grid, reproduces them to 4.4×10⁻¹⁷ (absolute).
* **In the repository.** `tests/data/nr_tm/matlab_grid_digests.json` holds a SHA-256 digest of each of the 438 grids, rounded to 10⁻¹⁰ so that one-ulp differences do not matter, and `tests/unit/test_nr_tm_grid_parity.py` checks that the package reproduces every digest. Because of the rounding, this test establishes agreement to about 10⁻¹⁰; the bit-identical count and the 3.2×10⁻¹⁶ bound come from the raw grids of the author's run (491 MB), which are not stored. The repository holds no MathWorks code or data files: the digests are hashes of numerical output and `tests/data/nr_tm/nbiot_oracle.json` holds numbers produced by public toolbox calls.
* **Fresh MATLAB runs of the pre-release review.** A reviewer who had not written the generator ran new MATLAB R2026a / 5G Toolbox 26.1 references and compared them with the package. The scripts and data of these runs are not in the repository.
    * *106 random configurations* (all ten models; 57 FDD and 49 TDD; cell IDs 0–1000; 1–20 subframes; oversampling 1, 2, 4 and 8; 67 of them in a (model, duplex, SCS, bandwidth) cell that is not among the 438 digests; 12 of them two-layer TM1.1 with two layers set by hand on MATLAB's configuration, compared on both ports, so 118 streams in all). With the MATLAB data convention the largest error relative to the RMS is 2.7×10⁻¹⁵ for the raw baseband and 3.0×10⁻⁷ for the exported single-precision samples.
    * *117 resource-block-group references* (TM1.2, TM3.2 and TM3.3 at all 39 cells MATLAB accepts, cell ID 77, one subframe), matching the package to float rounding. Per-PRB levels of MATLAB's grids and of the package agree with the formulas read from the standard's page images in 183 configurations of the reviewer's runs (largest error 2.2×10⁻¹⁵ dB).
    * *20 configurations at 3 MHz* (ten models, FDD and TDD), which MATLAB's wrapper rejects: the reviewer built the 5 MHz model in MATLAB, resized its grid to 15 resource blocks and set the PRB sets and powers from independently derived formulas; the package matches to 1.8×10⁻¹⁵.
    * *10 multi-carrier compositions* (2 to 5 carriers; mixed 15 and 30 kHz; different bandwidths; a 0.6 MHz sub-block gap; explicit cell IDs; TDD; TM2 and TM2b), built from MATLAB single-carrier waveforms at the common sample rate, frequency-shifted and summed, with the carrier offsets recomputed from the TS 38.104 5.4.1.2 formula and the guard-band table: the offsets agree exactly and the waveforms to 1.4×10⁻¹⁵ – 2.7×10⁻¹⁵.
    * *A NumPy implementation written from the text* of TS 38.141-1 and TS 38.211 without reading the package's code, for the default **Standard text** convention (restart at every frame, one stream per user): 46 configurations (TM1.1, 1.2, 2, 2b, 3.1, 3.1b, 3.2, 3.3; FDD and TDD; 5 to 20 MHz; two frames) and 12 two-layer cases, 58 in all, agree with the package to 3.1×10⁻¹⁶ in the grid and 2.6×10⁻¹⁵ in the waveform. MATLAB is no reference for these, because it departs from the text there.
* **Specification double entry.** The resource-block-group selection of TM1.2, TM3.2 and TM3.3 for all 40 table cells (P, N_RBG, group indices, PRB counts, levels, power conservation to 10⁻⁹) was transcribed separately from the page images of TS 38.141-1 and compared with the package, together with closed forms written independently in the tests (`tests/unit/test_nr_tm_spec.py`, `tests/data/nr_tm/rbg_reference.json`). The N_RB table equals TS 38.104 table 5.3.2-1 in V18.10.0 and V18.15.0 (identical there, like tables 5.3.3-1, 5.4.2.2-1 and 6.3.4.2-1) and in V18.9.0, and the generic presets' table. PN23, Gold, QAM (all five constellations) and DM-RS sequences are checked against literal implementations of the defining equations, and the two-layer construction reproduces, on both ports, a two-layer TM1.1 grid (10 MHz, 15 kHz, FDD) obtained by setting two layers by hand on MATLAB's TM1.1 configuration (bit-identical; this case is not stored in the repository and was repeated by the review above).
* **Time domain.** Tests check cyclic-prefix lengths and positions, exact sample count, RMS, demodulation of the output back to the grid, exactly zero samples in TDD uplink, oversampling, carrier placement and equal power spectral density.
* **NB-IoT.** NPSS, NSSS, NRS, NPBCH, NPDCCH and NPDSCH symbols and resource-element sets (and their mapping order, for both the stand-alone and the in-band resource budget) match MATLAB R2026a LTE Toolbox NB-IoT functions to 10⁻¹² (NSSS 2×10⁻¹¹) for 13 combinations of N_ID, host cell ID and operation mode, and the NSSS sequence matches for 16 combinations of N_ID and frame number covering all four `b_q` ranges (`tests/data/nr_tm/nbiot_oracle.json`, `tests/unit/test_nr_tm_nbiot.py`; numbers produced by public toolbox calls). The puncturing, the power split and the placement are not compared with any reference.
* **Not covered by an independent reference:** NR-N-TM beyond its NB-IoT building blocks (puncturing, power split and placement; MATLAB has no NR-N-TM); the Standard text reading beyond what the from-the-text implementation above shows, because MATLAB departs from the text there; and everything about real instruments or base stations.

## Reproduce

Run the unit tests of the package:

```bash
python -m pytest tests/unit/test_nr_tm_*.py -q
```

Generate a signal from Python without Studio:

```python
from opendpd.core.waveforms.nr_tm import resolve, synthesize_nr_tm
from opendpd.schemas.nr_test_model import NrTestModelConfig, NrTmCarrier

recipe = NrTestModelConfig(
    model="NR-FR1-TM3.1a", duplex="fdd",
    carriers=[NrTmCarrier(scs_khz=30, bandwidth_mhz=100), NrTmCarrier(scs_khz=30, bandwidth_mhz=100)],
)
resolution = resolve(recipe)                      # recipe + full GeneratorConfig + per-carrier summary
iq, analysis = synthesize_nr_tm(resolution.config)    # complex64, exact length
print(resolution.summary.sample_rate_hz, len(iq), analysis.evm_percent)
```

The same recipe is `POST /signal-generator/nr-test-models/resolve` in the Studio API, and `GET /signal-generator/nr-test-models` returns the catalogue (models, N_RB table, TDD patterns, limits, fixed parameters). A configuration saved with **Save configuration** contains the recipe under `nr_test_model`; **Load configuration** re-derives and validates it, and refuses a file whose generator fields no longer match the recipe.

## References

* [ETSI / 3GPP TS 38.141-1 V18.14.0](https://www.etsi.org/deliver/etsi_ts/138100_138199/13814101/18.14.00_60/ts_13814101v181400p.pdf), clause 4.9.2 (test models), clauses 4.7 and 4.9.1 (test configurations).
* ETSI / 3GPP TS 38.104 [V18.10.0](https://www.etsi.org/deliver/etsi_ts/138100_138199/138104/18.10.00_60/ts_138104v181000p.pdf) and [V18.15.0](https://www.etsi.org/deliver/etsi_ts/138100_138199/138104/18.15.00_60/ts_138104v181500p.pdf), tables 5.3.2-1, 5.3.3-1, 5.4.2.2-1 and 6.3.4.2-1 (read in both releases; the tables are identical in them) and clauses 5.4.1 and 5.4.2.2. Table 5.3.2-1 is also the table of [V18.9.0](https://www.etsi.org/deliver/etsi_ts/138100_138199/138104/18.09.00_60/ts_138104v180900p.pdf), which the generic presets cite.
* [ETSI / 3GPP TS 38.211 V18.5.0](https://www.etsi.org/deliver/etsi_ts/138200_138299/138211/18.05.00_60/ts_138211v180500p.pdf), sequences, modulation mapping, DM-RS and OFDM generation.
* [ETSI / 3GPP TS 38.214 V18.5.0](https://www.etsi.org/deliver/etsi_ts/138200_138299/138214/18.05.00_60/ts_138214v180500p.pdf), table 5.1.2.2.1-1 (resource-block-group size).
* [ETSI / 3GPP TS 36.141 V19.1.0](https://www.etsi.org/deliver/etsi_ts/136100_136199/136141/19.01.00_60/ts_136141v190100p.pdf) and [TS 36.211 V19.3.0](https://www.etsi.org/deliver/etsi_ts/136200_136299/136211/19.03.00_60/ts_136211v190300p.pdf), the NB-IoT test model and physical channels (the NR specification cites TS 36.141 without a version).
* [ITU-T O.150 (05/96)](https://www.itu.int/rec/T-REC-O.150-199605-I/en), clause 5.6, the 2²³ − 1 pseudo-random test sequence.
