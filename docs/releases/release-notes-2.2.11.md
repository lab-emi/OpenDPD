# OpenDPD Studio 2.2.11

Signal Generator now puts the signal family, preset setup and next-step actions in one clear sequence. Compact matrices replace the preset dropdown: bandwidth runs horizontally, QAM vertically, and separate groups show OFDMA channel counts. The catalog includes 940 NR, 96 Wi-Fi 6, 140 Wi-Fi 7 and 10 custom engineering presets. Wi-Fi 8 generation is removed.

- Select up to 16 presets with independent sample rates, bandwidths and lengths. Each creates a separate capture; signals with different sampling clocks are not concatenated.
- An ideal PA-input filter is enabled by default, with a per-preset switch. It sharpens band edges, preserves sample count and RMS, and reports its effect on measured EVM/PAPR. PA output retains nonlinear spectral regrowth.
- **Simulate PA output** now saves a complete dataset and opens its details automatically. The separate pairing form and diagram node are removed.
- Dataset details offer a subdataset selector and top-level downloads. **Download CSV** uses the selected capture/version. **Download all · ZIP** contains all original captures, per-capture JSON metadata and one parameterized `simulate_pa.py` with frozen formulas, sample rates and model parameters.
- The replay script accepts input CSV and writes output CSV without an OpenDPD installation. Release tests reproduce float32 output exactly for all nine PA model families; numerical libraries/platforms may introduce roundoff differences.
- Batch APIs retain authentication, CSRF protection, tenant isolation, compute admission and aggregate storage quotas. Failed dataset registration rolls back only newly created datasets. Temporary download files are removed after response completion.

Presets remain continuous, uncoded engineering stimuli with generic pilot/allocation placement. Full NR/WLAN protocol frames and standards-conformance certification are outside this release. See the [preset reference](../guides/signal-presets.md) for exact table sources and validation limits.

Install with `uv pip install "opendpd==2.2.11" --torch-backend=auto`, or open [Studio on the web](https://opendpd.com/studio/?v=2.2.11).
