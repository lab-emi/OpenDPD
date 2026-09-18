# OpenDPD Studio 2.2.14

Studio no longer mixes short validation/test preview scores with full-split
training metrics. New PA and DPD training starts with **TRes-GRU** (PA H27,
DPD H15), training/evaluation batch size **16**.

A real measurement defect caused part of the discrepancy: the historical
ACLR calculation included the final zero-padded evaluation segment. Its
artificial signal edge could dominate adjacent-channel power. For example,
the default filtered NR 20 MHz input read about −38/−35 dBc in the old
calculation despite having much lower actual leakage.

The new default `opendpd-spectral-v2` excludes padding before estimating
power. Training metrics, validation checkpoint selection and final test
results use the same calculation. Spectrum band overlays use the selected
carrier width. Historical `legacy-opendpd-v1` remains available unchanged for
reproduction; already stored results are not relabeled. Modern Studio/CLI
runs use the new default; historical `python main.py` and paper scripts keep
the legacy metric protocol unless explicitly overridden.

Carrier ACLR is adjacent-carrier power relative to the strongest in-band
carrier, in negative dBc. Full-band ACPR has different bands and reference
power. IBE means in-band error power ratio; it is not demodulated EVM.
See the [metric definitions](../protocols/metric-profiles.md).

Validation includes six synthetic NR/Wi-Fi conditions (5–320 MHz,
16–4096 QAM, six virtual PAs), measured DPA 160/200 MHz captures and APA
200 MHz. Each condition trained both PA and DPD. Independent FFT integration
agreed with stored ACLR within 1e-8 dB. APA TRes-GRU PA H27 / DPD H15,
batch 16, 300 epochs each achieved **−55.61 dBc** test carrier ACLR, using a
checkpoint selected only by validation ACLR. This is prediction through a
PA model trained on measured APA data, not a new RF hardware measurement.

[Experiment settings and results](validation-2.2.14.md)
