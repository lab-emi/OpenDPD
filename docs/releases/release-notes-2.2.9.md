# OpenDPD Studio 2.2.9

Studio adds a reusable Signal Analyzer and a broader set of configurable signal-generation tools.

- **Signal Analyzer:** import real, complex or split I/Q CSV; open generated PA inputs, Virtual PA outputs and selected dataset versions directly.
- **Measurements:** independent PSD, interactive spectrogram, contiguous time traces, instantaneous frequency, raw I/Q, CCDF, amplitude distribution and manually configured eye diagrams. Export JSON reports and exact PSD CSV bins.
- **Explicit settings:** FFT/window/overlap, sample range, measurement and adjacent bands, optional DC removal/frequency shift and aligned reference waveform errors. LaTeX explains the metric definitions.
- **Signal generation:** Gray-labeled PSK/QAM, repeatable PRBS or editable bit patterns, continuous-phase FSK/GFSK, band-limited noise, per-channel DFT spreading, multitone phase choices, raised-cosine burst envelopes, constant phase and independent phase jitter.
- **Better diagnostics:** EVM by OFDM symbol and data subcarrier. Single-carrier plots now show symbols recovered through a matched RRC filter at known timing.
- **RRC correction:** span now denotes total filter length in symbol intervals: `span × samples_per_symbol + 1` taps. Existing captures remain readable; regenerating a configuration uses the updated implementation and a new content identity.
- **Bounded public analysis:** authenticated CSV quarantine and complete numeric validation; per-workspace isolation, file/sample limits, shared compute admission and upload quotas. Status and cancellation retain separate capacity.

Install with `uv pip install "opendpd==2.2.9" --torch-backend=auto`, or use [Studio on the web](https://opendpd.com/studio/).

NR/WLAN presets are continuous, uncoded engineering stimuli with generic pilots. They do not contain complete transport coding, synchronization/control channels or packet framing. Wi-Fi 8 remains experimental. Reference waveform error is not a conformance EVM result. The release does not change PA/DPD training algorithms or hosted workspace retention.

[Signal Generator](../guides/signal-generator.md) · [Signal Analyzer](../guides/signal-analyzer.md) · [Installation](../install.md)
