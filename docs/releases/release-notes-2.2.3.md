# OpenDPD 2.2.3 — Signal Generator and research workflows

Try [OpenDPD Studio](https://opendpd.com/studio/) or install `pip install -U "opendpd[gui]==2.2.3"`.

## Generate a signal and start training

- **Signal Generator** is the first and only highlighted action in Get Started, followed by existing datasets and CSV upload. It also has a dedicated Studio tab.
- Twenty presets cover 5G NR FR1/FR2 numerology, Wi-Fi 6/7, experimental Wi-Fi 8, and custom OFDM, pulse-shaped QAM/PSK, tones and chirps. Advanced controls include sample rate, bandwidth, RF carrier metadata, duration or exact I/Q count, OFDMA allocations, pilots, cyclic prefix and impairments.
- Inspect time traces, spectrum, constellation, resource allocation, PAPR/CCDF, RMS/peak and occupied bandwidth. Download exact I/Q with configuration and provenance, or create a clearly labelled synthetic PA dataset for training.
- **PA Model** and **DPD Model** each contain Training and Testing tabs. Testing displays the selected dataset version's exact complex I/Q sample count and equivalent duration.

The standard presets generate continuous, uncoded engineering stimuli with generic pilots. They do not implement complete protocol frames or certify standards compliance. Wi-Fi 8 is experimental. Constellation error is a generator diagnostic, not a released standards EVM measurement. See the [Signal Generator guide](../guides/signal-generator.md).

## Review and reproduce research locally

- RF facts, source hashes, fixed comparison references, condition differences and spectrum cursors make metric definitions and missing evidence visible.
- Save up to four publication panels, export plots and numerical data, and reproduce metrics and saved views from a private bundle containing the exact runtime source, data and model dependencies.
- Record measurement sessions, calibration, independent acquisitions and exclusions. Optional fractional alignment is a separately versioned processing protocol; the existing integer default is retained.
- Sweep Board manages conditions, seeds, budgets and ordinary training workers with cancellation and retries. QAT requires supported models and an explicitly bound float checkpoint; setup failures cannot silently become float runs.
- Hardware Costs distinguishes checkpoint storage and qualified operation estimates from uploaded implementation reports, fixed-point specifications and CPU reference timing.

These local workflows are documented in the [research review guide](../guides/research-review.md). Hardware reports, measurement sessions, sweep execution, saved publication figures and full reproduction bundles remain local Studio features; the hosted service retains its explicit route and resource limits.

## Synthetic datasets and optional contributions

Six reproducible datasets provide three synthetic drive conditions with two independent random realizations each. Synthetic origin is shown in the GUI and exports. CSV uploads and generated data remain private by default.

Local Studio can prepare a data-only package, obtain explicit public-disclosure consent, create a dedicated branch, push it and open a PR using GitHub CLI on the Studio host. Contributions require **human review before merge**; questions go to **emi.lab@outlook.com**. Hosted automatic submissions require a separately configured operator service and remain disabled by default. A private package preview does not publish data.

Synthetic captures and software checks do not replace physical repeats, independent operating conditions, hardware reports or independent EVM reference validation. Those acceptance items remain open. New research and generator controls have English and Chinese text; the other seven interface languages use English fallback for these additions.

## Validation

Release validation covers the Python and frontend suites, API contract, package contents, browser journeys and the deployed web session. The [implementation ledger](../design/studio-next-implementation.md) records the detailed software evidence and outstanding physical validation. Existing experiment task identifiers and saved links remain compatible.
