# OpenDPD Studio 2.2.13

PA output spectra now use larger, clearer legends above the plot. This makes DPD training and testing results easier to interpret on desktop and touch screens.

- Spectrum legends use 14 px text, with space for wrapped rows above the axes. Axis titles and tick labels are larger too. Adjacent signal-chain plots reserve the same space so their axes stay aligned.
- **With DPD · PA model** and **Without DPD · PA model** replace the corresponding “surrogate” legend labels. The PA output plot explains that a surrogate is the trained amplifier model: original input → PA model without DPD, and original input → DPD → the same PA model with DPD.
- Model predictions remain distinct from measured and synthetic dataset curves. The explanation and model labels are available in all nine interface languages.
- Touch layouts preserve each chart's legend size and spacing. Zoom, trace visibility, enlargement and native plot downloads remain available.

This release changes presentation only: stored PSD values, signal identities, training, metrics and exported source data are unchanged.

Install with `uv pip install "opendpd==2.2.13" --torch-backend=auto`, or open [Studio on the web](https://opendpd.com/studio/?v=2.2.13).
