# Studio Signal Generator

Open **Signal Generator** in the sidebar, or choose **Get Started → Signal Generator**.
The first visit generates a private 5G NR numerology preview. Choose one of the five
signal families, select a preset, and press **Generate & preview**. Parameter edits
mark the current plots stale and disable waveform export and the next step until
generation succeeds. Returning to the tab restores the selected input. The result
is explicitly a **PA Input Dataset (x)**, with no PA output.

The onboarding dialog has exactly one highlighted action: Signal Generator. Existing
datasets are the second choice, and CSV upload is third. PA and DPD each have one
model workspace with Training and Testing tabs. Existing task URLs remain valid.
Testing displays the selected dataset version's test split count and equivalent time,
before any model warm-up or evaluation-edge exclusions.

## Waveforms and coverage

The generator provides synthetic engineering stimuli with standard numerologies,
not complete protocol implementations or certified reference test models.

| Family | Presets | Implemented signal |
| --- | --- | --- |
| 5G NR | FR1 20 / 100 MHz; FR2 100 MHz | 30 / 120 kHz subcarrier spacing, normal CP, 51 / 273 / 66 resource-block-sized payload grids, generic pilots |
| Wi-Fi 6 | 20 / 40 / 80 / 160 MHz | 78.125 kHz spacing, 0.8 µs guard interval, up to 1024-QAM |
| Wi-Fi 7 | 20 / 40 / 80 / 160 / 320 MHz | 78.125 kHz spacing, up to 4096-QAM |
| Wi-Fi 8 | 80 / 160 / 320 MHz | Experimental OFDM numerology profile; generic allocations and pilots |
| Custom | OFDM/OFDMA, single-carrier QAM/PSK, single tone, multitone, chirp | Fully editable baseband parameters |

OFDM payload symbols are uncoded and continuous. Pilots are generic seeded BPSK,
including when their bin positions are explicitly specified. Synchronization, FEC,
NR physical/control-channel mapping, WLAN preambles, standard RU allocation bitmaps,
MAC packets, and draft-specific UHR mechanisms are **not implemented**. The GUI,
saved metadata, and exports all disclose this scope. Changing preset numerology
marks the waveform custom. Existing known-waveform evaluation bindings are not
assigned to these generated signals.

NR CP timing follows [TS 38.211, §5.3.1](https://www.etsi.org/deliver/etsi_ts/138200_138299/138211/15.02.00_60/ts_138211v150200p.pdf).
The long normal CP appears twice per subframe; extended CP requires 60 kHz spacing.
For WLAN background, see the [IEEE 802.11 working group](https://www.ieee802.org/11/).
As of 2026-09-13, 802.11bn remains a draft; its development status is tracked by
[IEEE TGbn](https://www.ieee802.org/11/Reports/tgbn_update.htm).

## Advanced parameters

- Output sample rate, nominal baseband bandwidth, RF carrier metadata, RMS and seed.
- Exact complex sample count, or equivalent duration rounded to the nearest sample.
  One complex sample contains I and Q. The current limit is 256–1,000,000 samples.
- FFT size, 1/2/4/8× oversampling, subcarrier spacing, fixed/NR CP and DC null.
- Up to 16 OFDMA channels with independent carrier counts, modulation and relative
  power. Counts include pilots. Channel gaps are specified in FFT bins. These are
  users within one RF band, not separate RF carriers for adjacent-channel metrics.
- Per-channel pilot comb, explicit signed carrier indices, or no pilots; pilot boost.
- RRC QAM/PSK pulse shaping, samples per symbol, roll-off and filter half-span.
- Tone frequency, multitone count, and a linear chirp over the declared bandwidth.
- Frequency offset, I gain mismatch, Q phase mismatch, I/Q DC offsets, envelope
  clipping and independent white Gaussian noise.

`SCS = output sample rate / (FFT size × oversampling)`. The FFT and fixed CP inputs
refer to the grid before oversampling; measured CP lengths are reported in output
samples. Changing SCS in the GUI updates the output sample rate. Carrier frequency
is saved as RF metadata and never digitally mixes a GHz signal into the baseband.

RMS normalization precedes impairments. I gain and Q phase mismatch are applied
first, followed by DC offset, frequency offset, clipping and noise. There is no
post-impairment normalization or hidden receiver equalization.

## Visualizations and measurements

Time plots display the first 2,048 contiguous samples without strided sampling.
Spectrum, RMS, peak, PAPR and CCDF statistics use the entire exported waveform.
Welch PSD uses a Hann window, 50% overlap and density scaling. Power is referenced
to unit complex RMS, not calibrated watts or dBm. Occupied bandwidth contains the
central 99% of integrated Welch power, including impairments.

PAPR is the maximum sample power divided by average sample power, in dB. It does
not estimate unsampled analog peaks. CCDF is the empirical exceedance probability
of power above its mean; zero-probability points are omitted on the logarithmic
plot, with empirical resolution 1/N.

OFDM constellation and diagnostic reference EVM use data carriers from up to the
first 16 complete symbols. The FFT receiver uses the known generation normalization;
it does not fit gain, phase, delay, or equalization. A capture with no complete symbol
has no reference EVM. Single-carrier constellation shows transmitted symbols before
RRC filtering and impairments, not a receiver estimate. Incomplete final symbols are
retained to preserve the exact requested sample count and disclosed in metadata.

## Export and training

**Save configuration** downloads JSON; **Load configuration** validates it against
the server contract before applying it. **Export I/Q + configuration** contains
`iq.csv` (I,Q), `iq.npy` (float32 N×2), configuration, measurements, source hash,
NumPy version and scope notes. CSV float32 values round-trip exactly. Seeded
regeneration requires the recorded configuration, generator implementation and
numeric environment. Generated records live privately under `signals/sg-<sha>/`
inside the workspace.

**Download PA input CSV** and **Download input metadata JSON** are separate actions.
The CSV has two columns, `I,Q`; metadata declares `signal_role: pa_input`,
`has_pa_output: false`, the sample rate/count, carrier metadata, generator parameters
and CSV/NPY hashes. The complete signal archive remains available as well.

The waveform alone has **no PA output**. **Choose Virtual PA** opens the
[PA Library](virtual-pa-library.md). Users select a mathematical Virtual PA,
adjust its formula parameters, choose a saved input and explicitly simulate y.
After reviewing the output, **Create paired dataset & train PA** pairs the exact
input and frozen output, then opens PA Training. It requires at least 8,192 input
samples and at least 256 samples per split after guards. Both x and y are marked
synthetic. The deprecated implicit-PA API remains available to older clients;
the Studio UI no longer uses it.

Get Started → existing dataset opens a paired-data selector and proceeds straight
to PA Training. The compact, expandable workflow diagram marks dataset making as
complete and explains the bypass. Model-training checks require successful runs
for the selected dataset/version and PA reference.

Generated datasets are private by default. Optional publication uses the existing
dataset contribution workflow: an explicit reviewed package, a separate branch,
push and PR, followed by human merge review. Contact: emi.lab@outlook.com.

Hosted sessions apply authentication, tenant isolation, size limits and generation
quotas. Dataset creation respects the host's custom-dataset capability. No RF
transmitter, GitHub submission, or email is activated by generating a waveform.

## Verification

Numerical tests cover all 20 presets, exact sample counts, one-millisecond NR CP
timing, FFT allocations, empty bins, deterministic impairments, analytic single-tone
PAPR, invalid parameters, export round trips and dataset provenance. API tests cover
authentication, CSRF, host feature gating and per-version test counts. Hosted tests
verify that one session cannot access another session's generated signals or data.

`scripts/verify_signal_generator.mjs` exercises the real local browser at 1366×768
and 1920×1080: onboarding order and highlight, generation, stale-result protection,
custom two-channel OFDMA with explicit pilots and noise, ZIP export, dataset creation,
and real CPU PA/DPD training and testing. These checks validate the software workflow;
they do not constitute independent standards conformance or physical RF validation.

## Studio 2.2.5 preview

![PA input waveform and its independent PSD](../../pics/studio-signal-generator.png)

The PSD here is labelled **PA Input**. It contains only the generated x signal; PA output appears after explicit simulation in PA Library. See [signal-chain spectra](signal-chain-spectra.md).

## Explicit generation and shared channels (2.2.5)

Opening the generator no longer creates a saved default waveform. Choose a preset and click **Generate & preview**. Generated sources remain PA Input Datasets until explicitly paired with an output.

**Use the same settings for all channels** is checked by default for equal allocations. One set of subcarrier-count, modulation and power fields then controls every channel; a newly added channel inherits it. Uncheck to edit those parameters independently. Importing unequal channel settings keeps them independent. Re-enabling sharing applies channel 1's settings to every channel. The number of channels remains visible. FFT timing, cyclic prefix and the global pilot-bin allocation share the OFDM grid.

Configuration JSON records the sharing choice together with every channel's resolved settings. Generated CSV/metadata downloads and the next-step PA Library button are above advanced parameters.
