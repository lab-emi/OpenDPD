# NR standard test models

Studio 2.4.1 adds the NR FR1 base-station test models of **3GPP TS 38.141-1 V18.14.0, clause 4.9.2** to the [Signal Generator](signal-generator.md). The **5G NR** tab opens on **Standard test model (3GPP TS 38.141-1 §4.9.2)**: choose NR-FR1-TM1.1, 1.2, 2, 2a, 2b, 3.1, 3.1a, 3.1b, 3.2, 3.3 or NR-N-TM, then set every parameter the standard leaves open — duplex mode, number of carriers, subcarrier spacing and channel bandwidth of each carrier, cell identity, gaps between carriers, rank for the time-alignment model, the PN-sequence reading, NB-IoT placement and power, oversampling and capture length. Everything the standard fixes is shown read-only under **Defined by the standard**.

!!! warning "What this is, and is not"
    The output is a **complex-baseband stimulus** that follows the standard's definition of each test model. It is meant for PA and DPD experiments: a signal with the resource-block structure, boost levels, DM-RS and PN23 data that the standard prescribes for its tests. It is **not** a certified conformance waveform, it was not compared with an instrument or a base station, and the **EVM** Studio shows is an ideal-demodulation diagnostic on a known grid, not the receiver EVM of TS 38.141-1 clause 6.5.3. The level is RMS-normalised, not a calibrated power. See [Limitations](#limitations).

## Generate a test model

1. Open **Signal Generator → Generate** and choose **5G NR**. The first visit selects NR-FR1-TM1.1, FDD, one 30 kHz carrier of 20 MHz.
2. Pick a model card. Settings the new model does not allow are repaired when you switch: NR-N-TM forces FDD and 15 kHz, and layers return to 1 for every model but TM1.1.
3. Set duplex, the carriers and the options below. Studio sends the recipe to the server about 0.4 s after your last edit and shows the derived sample rate, sample count, duration and a **Shortened** chip when the standard duration does not fit the sample limit. **Generate & preview** stays disabled while the server has not accepted your latest edit; an unacceptable setup shows the reason.
4. **Defined by the standard** lists the fixed parameters of the model and, for your settings, the resolved numbers per carrier: N_RB, subcarriers, FFT size, cell ID, centre offset, RBG size P, number and indices of boosted or de-boosted RBGs, every user with modulation, PRB ranges, start symbol and level, and for TDD the slot pattern.
5. **Add another test model** appends a second entry (a different model by default). A batch holds up to 16 signals; they keep their own sample rates and lengths and are never concatenated. The selected chips read like "TM3.1a · 100 MHz · 30 kHz · FDD".
6. **Generate & preview**. The automatic dataset name carries the model: 100 MHz TM3.1a at 30 kHz gives `syn_pa_in_nrtm3p1a_bw98p31M_q256_c1_n1` (`nrntm` for NR-N-TM; the number after `c` counts carriers); see [dataset names](signal-presets.md#dataset-names-and-analysis).

The earlier preset **matrix** (940 NR presets by bandwidth, QAM and OFDMA channel count) is still available behind **Generic numerology matrix…** below the panel. It is labelled as generic engineering stimuli, selecting its cells works as before, and they join the same selection. See [Signal preset reference](signal-presets.md). Wi-Fi 6, Wi-Fi 7 and Custom are unchanged. If a hosted Studio's server does not yet provide the test-model catalogue, the NR tab falls back to the matrix.

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
| NR-N-TM | 4.9.2.2.9 | NR-FR1-TM1.1 on the NR carrier with one NR resource block punctured by an NB-IoT carrier (the N-TM of TS 36.141), boosted by X dB. FDD, 15 kHz only. | NB-IoT in-band tests: occupied bandwidth, ACLR, operating band unwanted emissions, spurious emissions, transmitter intermodulation, receiver spurious emissions, NB-IoT RB power dynamic range |

The rest of each model is derived from the clause; the **Defined by the standard** table of the panel prints the exact numbers for your bandwidth.

### Boosted and de-boosted groups (TM1.2, TM3.2, TM3.3)

With N = N_RB−3 PRBs from PRB 3, the nominal RBG size P comes from TS 38.214 table 5.1.2.2.1-1, configuration 1: 2, 4, 8 or 16 PRBs for N up to 36, 72, 144 or 275. The first RBG is the partial group that starts at PRB 3 and has P − (3 mod P) PRBs. The number of target groups is

`N_RBG = min( ⌊x·N / P⌋ , ½(⌊(N + (3 mod P) − P) / P⌋ − (⌊(N + (3 mod P) − P) / P⌋ mod 2)) + 1 )`

and the target groups are (0-based) `⌊(N + (3 mod P)) / P⌋ − 1` and, if N_RBG > 1, groups 1, 3, …, 2(N_RBG−2)+1. The complementary PRBs get the level `10·log10((N − 10^(L/10)·P·N_RBG) / (N − P·N_RBG))` dB, which keeps the PRB powers summing to N. The generator evaluates this with integer arithmetic. For TM1.2 the target groups are boosted by L = +3 dB (n_RNTI 0); for TM3.2 (L = −3 dB) and TM3.3 (L = −6 dB) the target groups are n_RNTI 1 and the complement is n_RNTI 0. DM-RS follow the PDSCH level of their allocation.

## What you can set

| Parameter | Allowed values | Notes |
| --- | --- | --- |
| Test model | the eleven models above | |
| Duplex | FDD or TDD | FDD lasts one radio frame (10 ms), TDD two (20 ms). NR-N-TM is FDD only. |
| Number of carriers | 1–16 | Each carrier is an independent copy of the chosen model with its own settings. |
| Subcarrier spacing (per carrier) | 15, 30 or 60 kHz | FR1 numerologies. NR-N-TM: 15 kHz. |
| Channel bandwidth (per carrier) | 15 kHz: 3, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50 MHz · 30 kHz: 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100 MHz · 60 kHz: 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100 MHz | The 40 cells of TS 38.104 table 5.3.2-1; the bandwidth sets N_RB (and with it the number of subcarriers, 12·N_RB). NR-N-TM: 3–50 MHz. |
| Cell identity (per carrier) | automatic, or 0–1007 | Automatic = N_ID^cell 1 for the lowest carrier, 2 for the next, … (TS 38.141-1 4.9.2.3). It drives scrambling and DM-RS. |
| Gap before carrier (carriers 2 and up) | 0, or a multiple of 0.3 MHz (up to 1000 MHz) | Added to the nominal channel spacing; > 0 starts a new sub-block (non-contiguous operation). |
| Layers | 1 or 2 | Two layers (rank 2, ports 1000 and 1001, no precoding) exist for NR-FR1-TM1.1 only, as the standard defines it for the TAE test. |
| Exported port | 1000 or 1001 | With two layers the generator exports one of the two streams. |
| PN23 data convention | Standard text or MATLAB 5G Toolbox | [Below](#data-content-and-the-two-pn-conventions). |
| PN23 scope (TDD) | downlink symbols only, or all symbols | The standard allows both. Standard convention only. |
| NB-IoT position (NR-N-TM) | lower, upper or both | The eligible in-band RB nearest the lower or upper minimum guard band, or one at each edge. |
| NB-IoT power boost X (NR-N-TM) | 3–12 dB, default 6 | The standard asks for a declared dynamic range of at least 6 dB (or 3 dB); the 12 dB upper bound is this generator's choice. |
| Oversampling | automatic, 1, 2, 4 or 8 | Not part of the standard; see [capture length](#capture-length-oversampling-and-the-sample-limit). |
| Capture length | standard duration, or 1–20 subframes | Standard: 10 ms FDD, 20 ms TDD. |

Through **Advanced** you can still set the generic impairment and level controls of any Signal Generator configuration: RMS (default 0.2), seed, RF carrier metadata, burst envelope, constant phase and phase jitter, frequency offset, I gain and Q phase mismatch, I/Q DC offsets, SNR (white Gaussian noise), clipping and the optional ideal band-edge filter (**off** for test models: the standard defines them unfiltered, and filtering changes ACLR and EVM). The FFT size, sample rate, bandwidth, channel list, pilots and length controls disappear for a test model because they are derived from the recipe. The seed only changes the added noise and phase jitter; the test-model data themselves are deterministic.

**Fixed by the standard** (shown, not editable): the modulation of every user, the boost target x (40 / 60 / 50 %), the boost and de-boost levels, the RBG size P, the number of users and the n_RNTI mapping of table 4.9.2.3.2-1, the PDCCH and PDSCH configuration of the common tables, DM-RS type and positions, PDSCH/PDCCH power ratios, normal CP and the model duration.

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

## Several carriers

Every carrier is its own test model with its own N_RB, cell identity, PN streams and resource grid, synthesised at its own FFT size and summed after a time-domain frequency shift. The rules follow TS 38.141-1 clauses 4.7 and 4.9.2.3 and TS 38.104 clause 5.4.1:

* **Order and cell identity.** Carriers are ordered by ascending frequency. N_ID^cell is 1 for the lowest, 2 for the second lowest and so on, unless you override it. Two carriers with the same numerology, bandwidth and cell ID would carry identical data; Studio warns.
* **Nominal channel spacing.** Adjacent carriers are separated, reference frequency to reference frequency, by the nominal channel spacing of TS 38.104 clause 5.4.1.2 for bands with a 100 kHz channel raster: `S = ⌊(BW₁ + BW₂ − 2·|GB₁ − GB₂|) / 0.6 MHz⌋ × 0.3 MHz`. GB is the minimum guard band of table 5.3.3-1, taken at the larger of the two subcarrier spacings if both bandwidths exist for it, else at the smaller one. Examples: 20 + 20 MHz → 19.8 MHz; 100 + 100 MHz → 99.9 MHz; 5 + 100 MHz at 30 kHz → 51.9 MHz; 10 + 10 MHz → 9.9 MHz; 3 + 3 MHz → 3.0 MHz.
* **No overlap.** The occupied subcarriers of neighbours must not overlap; otherwise the setup is refused and says by how much they would. At the nominal spacing all 1,600 ordered pairs of table 5.3.2-1 cells (any bandwidths, any of the three spacings) are accepted, and a unit test asserts that the tightest pair, 3 MHz at 15 kHz next to 80 MHz at 60 kHz, still leaves 7.5 kHz between occupied subcarriers. The check only matters for gaps and rules that change the spacing.
* **Gap before a carrier.** `gap_before_mhz` adds to the spacing between a carrier and its lower neighbour, in multiples of 0.3 MHz (the least common multiple of the 100 kHz raster and the subcarrier spacings). A gap models the sub-block gap of non-contiguous operation (NRTC3). It cannot reduce the spacing below nominal.
* **Placement.** The aggregate is centred at baseband DC by its outer edges (TS 38.104 clause 5.3A.2); a single carrier keeps its reference subcarrier at DC. Only relative offsets matter in baseband; the RF carrier frequency in Advanced is metadata.
* **Equal power spectral density.** Every carrier uses the same resource-element amplitude, so all carriers have equal power spectral density as NRTC1 and NRTC2 require. Each carrier is synthesised with the unitary scaling `√N_c · ifft(grid)` so this holds across subcarrier spacings. There is no per-carrier power offset; the equal *power* wording of NRTC3 for carriers of different bandwidth is not reproduced.
* **Mixed numerologies.** Carriers with different subcarrier spacings are allowed (each follows its own numerology; slot grids align at subframe boundaries). The standard text does not describe this case, the signals are unwindowed, and the ideal-demodulation EVM then includes the inter-numerology leakage of the other carriers (about 0.9 % overall for a 20 MHz / 15 kHz carrier next to a 100 MHz / 30 kHz carrier in a check run for this guide). Studio shows a warning.
* **Sample rate.** The common sample rate is the smallest 15 kHz·2^p at which each carrier gets its FFT size and the aggregate occupies at most 85 % of it, times the oversampling. Four 100 MHz carriers at 30 kHz work (491.52 MS/s, so only 2 ms fit in 1,000,000 samples). Five are refused: the description Studio keeps of a configuration is limited to 16,384 bins of the lowest subcarrier spacing, and the message says so.

In the panel the number-of-carriers stepper clones the last row; the row buttons add or remove single carriers. Sixteen carriers of 3 MHz at 15 kHz need 61.44 MS/s and fit the whole 10 ms.

## TDD patterns

TDD models use the pattern of TS 38.141-1 table 4.9.2.2-1, written with the `TDD-UL-DL-ConfigCommon` fields of TS 38.331 and a 5 ms period that repeats for the 20 ms duration:

| Subcarrier spacing | Slots per period | Downlink slots | Special-slot symbols | Uplink slots | Slot sequence |
| --- | ---: | ---: | --- | ---: | --- |
| 15 kHz | 5 | 3 | 10 downlink, 2 uplink | 1 | D D D S U |
| 30 kHz | 10 | 7 | 6 downlink, 4 uplink | 2 | D×7 S U U |
| 60 kHz | 20 | 14 | 12 downlink, then 8 uplink in the next slot | 4 | D×14 S S U×4 |

The symbols between the downlink and the uplink symbols of the special slots are flexible. **Flexible and uplink symbols are not transmitted: those samples are exactly zero** (a switch without transient shaping). The PDCCH is present in every downlink and special slot; uplink slots carry nothing. In the special slot the PDSCH occupies the downlink symbols only (the DM-RS symbols follow TS 38.211 table 7.4.1.1.2-3 for that length: l = 2 and 11 for 14 symbols, 2 and 9 for 10 or 12, 2 for 6). The pattern of NB-IoT in-band operation (table 4.9.2.2-1a) is documented in the standard but not generated; NR-N-TM is FDD only.

## Data content and the two PN conventions

The data of every PDCCH and PDSCH are bits of the **PN23** sequence generator (ITU-T O.150), `s(i) = s(i−23) xor s(i−18)` with the seed `s(0…22) = 1`, scrambled with the Gold sequence of TS 38.211 clause 5.2.1 (PDSCH `c_init = n_RNTI·2^15 + N_ID^cell`; PDCCH `c_init = N_ID^cell` with n_RNTI = 0), QPSK/16/64/256/1024QAM mapped per clause 5.1 with unit average power, with no channel coding and no CRC. The PDSCH of each user carries `Q_m × (allocated resource elements − DM-RS resource elements)` bits per slot; the PDCCH carries 108 bits. DM-RS use the Gold sequences of clauses 7.4.1.1.1 and 7.4.1.3.1 with n_SCID = 0, on the even subcarriers (comb of DM-RS type 1); the odd subcarriers of a DM-RS symbol carry PDSCH data.

The standard says the PN generator is "initialized with a starting seed of all ones in the first allocated slot of each frame", continuous over slot boundaries, that for TDD it may run over all symbols or the downlink symbols only, and that for multi-user models it "can be generated per user". Two readings are implemented:

| Data convention | PN streams | Restart |
| --- | --- | --- |
| **Standard text** (default) | One stream per n_RNTI user and one for the PDCCH. TDD: downlink symbols only, or all symbols as if every slot were a full downlink slot (PN23 scope). | Restarted in the first allocated slot of every radio frame. |
| **MATLAB 5G Toolbox** | One stream per PDSCH allocation object (for TM2, three independent streams by slot number modulo 3; for TDD a separate stream for the special slot), one for the PDCCH. The scope option has no effect. | Started once, never restarted. |

In **one 10 ms FDD frame of TM1.1, TM1.2 and TM3.x the two conventions give identical grids** (checked for several bandwidths and numerologies). They differ in the random bits only, and only for TM2/2a/2b, TDD, and captures longer than one frame. Choose *MATLAB 5G Toolbox* to compare against that product. Only the random bits differ, so the structure and spectral shape do not depend on the choice. The first 23 PN23 bits are the all-ones seed. ITU-T O.150 (05/96) describes its PRBS23 test signal as the inverted shift-register output, whereas the literal "all ones" seed used here, like MATLAB, gives the non-inverted output; no inversion option is offered, and which polarity a given instrument uses was not established.

### Two layers (TM1.1)

With layers = 2, the PDSCH uses layer mapping `x^(i)(j) = d(2j+i)` (TS 38.211 clause 7.3.1.3), DM-RS of port 1001 are `r(m)·(−1)^(m mod 2)` on the same comb (clause 7.4.1.1.2, `w_f(k') = (+1, −1)`), there is no precoding and no 1/√2 scaling, and twice the bits per slot are drawn. The PDCCH exists only on its own port and is written to the port 1000 stream, so the exported port-1001 stream carries PDSCH and DM-RS but no PDCCH. The standard defines the two-layer model for the TAE test; the TAE requires both streams, and Studio exports one.

## NR-N-TM: NB-IoT operation in NR in-band

NR-N-TM (clause 4.9.2.2.9) is NR-FR1-TM1.1 with one NR resource block punctured by an NB-IoT carrier carrying the **N-TM of TS 36.141**. Studio implements it for **FDD at 15 kHz** (3–50 MHz), the only numerology the standards define NB-IoT in-band operation for: an NB-IoT carrier is 180 kHz, which is one NR resource block only at 15 kHz, and no clause of TS 38.141-1 or TS 38.104 defines it for 30 or 60 kHz hosts. TDD is not generated either: the special-subframe resource budgets are not given by any standard that was read, and no reference implementation exists.

* **NB-IoT content** per TS 36.211 clause 10.2 and TS 36.141 clauses 6.1.3–6.1.6: NPSS (subframe 5), NSSS (subframe 9), NRS on port 2000 with the port-2001 positions reserved, NPBCH (subframe 0), NPDCCH format 1 (subframe 1) and NPDSCH (the remaining subframes, n_RNTI 1000), QPSK, all-zero data, one 10 ms frame that repeats. N_ID^Ncell = 97 + 6n + max(0, m−1) for the n-th NB-IoT host and host cell identity m, i.e. 103 for the lowest host with cell ID 1 and 110 for the second; the lower and upper NB-IoT carrier of one host share it. With several NR carriers the lower NB-IoT carrier sits in the lowest NR carrier and the upper one in the highest, and only these hosts carry NB-IoT.
* **Position.** The resource block is the *eligible in-band position closest to the NR minimum guard band*. The eligible positions form a comb with a spacing of 5 RB derived from the NB-IoT anchor-carrier raster (TS 36.104 clause 5.7.3), assuming the NR reference frequency is a multiple of 100 kHz; the standards define no explicit table. The lowest/highest eligible NR RB: 3 MHz 2/12, 5 MHz 2/22, 10 MHz 0/50, 15 MHz 4/74, 20 MHz 2/102, 25 MHz 1/131, 30 MHz 4/159, 35 MHz 3/183, 40 MHz 2/212, 45 MHz 0/240, 50 MHz 4/269. The panel shows the list for your bandwidth.
* **Power** (clause 4.9.2.2.9): average power per RE `P_avg = P_rated / (12·N_RB)`; a boosted NB-IoT RE has `P_avg·10^(X/10)`; the NR REs share the remainder. The generator balances the energy over the whole 10 ms frame (every RE of the grid loaded at unit amplitude), so the average power of a host carrier equals that of a carrier without NB-IoT. A boost that would leave no power for the NR resource elements is refused (the limit is 13.66 dB for one NB-IoT carrier in 3 MHz and 10.65 dB for two). For one NB-IoT carrier and X = 6 dB, NR resource elements are about 0.17 dB below P_avg at 10 MHz and 0.63 dB below at 3 MHz. Balancing every OFDM symbol on its own would shift these by up to a few tenths of a dB.
* **Resolved ambiguities**, because the standards leave them open: resource elements of the punctured RB that carry no NB-IoT signal (the control region, host reference-signal positions, muted NRS port-2001 positions) keep the NR content; the 304/200-bit NPDCCH/NPDSCH budgets of TS 36.141 correspond to a two-port NRS budget, which is reproduced; each NPDSCH subframe is its own codeword; NPDSCH is QPSK only (the 16QAM variant exists only in TS 36.141's own EVM test); the NSSS is sent in every looped frame.

The NB-IoT symbols, resource-element sets and mapping order were compared with the MATLAB LTE Toolbox NB-IoT functions (see [Evidence](#validation-evidence)). A complete NR-N-TM waveform has no reference implementation to compare with.

## Reading the preview

The preview is the standard Signal Generator preview: time plot, Welch PSD, CCDF and PAPR, occupied bandwidth, constellation, EVM trends. For test models:

* The **allocation** plot shows each carrier in MHz around baseband DC with its subcarrier spacing and N_RB; per-carrier metrics (N_RB, cell ID, centre offset, EVM) are in the analysis.
* **Constellation and EVM** are computed on the PDSCH data resource elements of every carrier: the carrier's frequency shift is undone, every OFDM symbol's useful part is transformed, and the result is compared with the transmitted grid without equalisation, gain or phase fitting, EVM window or timing search. Without impairments this is a numerical floor of about 10⁻⁶ % (the exported samples are single-precision). It rises with the impairments you add and with filtering. EVM trend plots refer to the lowest carrier.
* The coverage label reads **3GPP test model**. The analysis notes state the data convention, truncation, the TDD handling and the EVM definition.
* The level is normalised to `rms`; PAPR and CCDF are those of the signal at the chosen sample rate.

## Limitations

* **Content:** PDCCH, PDSCH and DM-RS only, as the clause defines the test models. No SSB, CSI-RS, PT-RS, PRACH, channel coding, CRC or HARQ. One exported antenna port.
* **Not a conformance tool:** no instrument or base-station comparison was made; no claim of TS 38.141-1 conformance is made; EVM is a diagnostic (no equaliser, no EVM window, no 10 ms measurement-period averaging as in clause 6.5.3).
* **Calibration:** RMS-normalised baseband; no calibrated power, no OSTP or total-power-dynamic-range measurement.
* **FR2 and other models:** the FR2 models of TS 38.141-2 and subcarrier spacings of 120 kHz and above are not included. Clause 4.9.2.2 of V18.14.0 defines exactly the eleven FR1 models above.
* **Test configurations:** NRTC1…NRTC6 are not generated as such. You choose the carriers; Studio does not pick the number of carriers that fit a declared Base Station RF Bandwidth, does not place them from the upper edge, and does not generate the ON-OFF patterns of NRTC6 (n46, n96, n102). The channel spacing is that of 100 kHz-raster bands; for 15 kHz-raster bands (for example n77, n78, n79) the nominal spacing differs by a fraction of a MHz (19.98 MHz where Studio uses 19.8 MHz for two 20 MHz carriers at 15 kHz), and that variant is not offered.
* **Frequency placement:** baseband only; no channel-raster alignment and no upconversion.
* **Releases:** definitions are those of TS 38.141-1 V18.14.0 (Release 18); other releases were not compared.
* **Duration:** wide or multi-carrier setups do not fit the standard duration into 1,000,000 samples and are shortened (see above).
* **TDD:** no transmit ON/OFF transient shaping; samples outside downlink symbols are exactly zero.
* **PN readings:** two readings of the PN rules are provided; the data bits of a given instrument may match neither.

## Validation evidence

* **MATLAB R2026a, 5G Toolbox.** All ten FR1 models were generated by MATLAB for FDD and TDD at 15, 30 and 60 kHz, with several bandwidths (TM1.2 and TM3.3 FDD at all 39 (SCS, bandwidth) cells MATLAB accepts), cell IDs from 1 to 1007 and 1 to 40 subframes: **438 configurations**. The package's resource grid for `data_convention="matlab"` was compared with MATLAB's in all 438: **414 are bit-identical and the other 24 (TM1.2 and TM3.3, whose boosted amplitudes are `10^(P/20)`) differ by at most 3.2×10⁻¹⁶**, one unit in the last place of a power computation. The package's OFDM modulator (IFFT and cyclic prefix, no windowing) reproduces the 58 MATLAB waveforms that were saved to 4.4×10⁻¹⁷ when it is given MATLAB's grid. MATLAB's 3 MHz is rejected by its test-model wrapper, so the 3 MHz cells, NR-N-TM and multi-carrier signals have no MATLAB model to compare against.
* **In the repository:** `tests/data/nr_tm/matlab_grid_digests.json` holds a SHA-256 digest of each of the 438 grids (rounded to 10⁻¹⁰, so one-ulp differences do not matter) and `tests/unit/test_nr_tm_grid_parity.py` checks that the package reproduces every digest. No MathWorks code or data is in the repository, and the MATLAB scripts that produced the digests are not either.
* **Specification double entry.** The resource-block-group selection of TM1.2, TM3.2 and TM3.3 for all 40 table cells (P, N_RBG, group indices, PRB counts, levels, power conservation to 10⁻⁹) was transcribed separately from the page images of TS 38.141-1 and compared with the package, together with closed forms written independently in the tests (`tests/unit/test_nr_tm_spec.py`, `tests/data/nr_tm/rbg_reference.json`). The N_RB table equals TS 38.104 table 5.3.2-1 (checked in V18.15.0) and the generic presets' table. PN23, Gold, QAM (all five constellations) and DM-RS sequences are checked against literal implementations of the defining equations, and the two-layer construction reproduces, on both ports, a two-layer TM1.1 grid (10 MHz, 15 kHz, FDD) obtained by setting two layers by hand on MATLAB's TM1.1 configuration (bit-identical; MATLAB's wrapper offers one layer only, and this one case is not stored in the repository).
* **Time domain.** Tests check cyclic-prefix lengths and positions, exact sample count, RMS, demodulation of the output back to the grid, exactly zero samples in TDD uplink, oversampling, carrier placement and equal power spectral density.
* **NB-IoT.** NPSS, NSSS, NRS, NPBCH, NPDCCH and NPDSCH symbols and resource-element sets (and their mapping order, for both the stand-alone and the in-band resource budget) match MATLAB R2026a LTE Toolbox NB-IoT functions to 10⁻¹² (NSSS 2×10⁻¹¹) for 13 combinations of N_ID, host cell ID and operation mode, and the NSSS sequence matches for 16 combinations of N_ID and frame number covering all four `b_q` ranges (`tests/data/nr_tm/nbiot_oracle.json`, `tests/unit/test_nr_tm_nbiot.py`; the data are numbers produced by public toolbox calls, no MathWorks code). The puncturing, the power split and the placement are not compared with any reference.
* **Not covered by an independent reference:** multi-carrier composition (checked against the spacing formula and by demodulating each carrier), the standard-text PN reading beyond one frame, the 3 MHz cells, TDD data bits beyond MATLAB's own conventions, and everything about real instruments or base stations.

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
* [ETSI / 3GPP TS 38.104 V18.9.0](https://www.etsi.org/deliver/etsi_ts/138100_138199/138104/18.09.00_60/ts_138104v180900p.pdf), tables 5.3.2-1 and 5.3.3-1, clauses 5.4.1.2 and 5.4.2.2.
* [ETSI / 3GPP TS 38.211 V18.5.0](https://www.etsi.org/deliver/etsi_ts/138200_138299/138211/18.05.00_60/ts_138211v180500p.pdf), sequences, modulation mapping, DM-RS and OFDM generation.
* [ETSI / 3GPP TS 38.214 V18.5.0](https://www.etsi.org/deliver/etsi_ts/138200_138299/138214/18.05.00_60/ts_138214v180500p.pdf), table 5.1.2.2.1-1 (resource-block-group size).
* [ETSI / 3GPP TS 36.141 V19.1.0](https://www.etsi.org/deliver/etsi_ts/136100_136199/136141/19.01.00_60/ts_136141v190100p.pdf) and [TS 36.211 V19.3.0](https://www.etsi.org/deliver/etsi_ts/136200_136299/136211/19.03.00_60/ts_136211v190300p.pdf), the NB-IoT test model and physical channels (the NR specification cites TS 36.141 without a version).
