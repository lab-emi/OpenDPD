# OpenDPD Studio 2.2.16

Signal Generator now has **Generate** and **Preview** pages in an expanding sidebar submenu. The editable dataset name and **Generate & preview** button sit at the top of Signal setup. Generation opens Preview automatically, with Virtual PA and download actions above the plots and the member selector next to the visualizations.

Synthetic PA inputs, outputs and paired datasets have readable names beginning with `syn_pa_in_`, `syn_pa_out_` and `syn_pa_inout_`. Suggested names summarize signal specifications/ranges and the number of signals. Analyzer and PA Library use searchable dataset selection; Analyzer then lets you choose an individual member. Reopening a saved collection preserves all its members, and removing an input collection supports Undo.

Dataset pages emphasize **Train PA & DPD Models**, wrap long names, separate secondary tools, and show collection sample totals and sample-rate ranges. Saved experiments restore the relevant dataset and model lineage. Re-running a DPD configuration preserves its task, dataset and version.

Plots are clearer in light and dark modes, with larger legends, readable narrow-screen metrics, and enough space for dataset time-domain and I/Q charts. Comparison input plots start with visible traces. The current ACLR profile displays the correct carrier integration bands. AM/AM and AM/PM labels distinguish measured, synthetic and model-predicted output.

DPD setup now rejects evaluated splits too short for one PSD segment before queueing the run, with an explanation of how to fix the configuration. This prevents a late training failure; it does not pad measurements or change the metric definition. TRes-GRU, batch size 16 and the valid-sample ACLR protocol remain the defaults.

Local verification includes real synthetic generation and PA simulation, CSV and measured APA imports, PA/DPD training and testing, ILC, a small experiment matrix, cancellation, exports, desktop/mobile layouts, and accessibility checks. These short workflow runs do not claim converged DPD performance or new physical-PA measurement results.

[Open Studio](https://opendpd.com/studio/) · [Signal Generator guide](../guides/signal-generator.md) · [Signal Analyzer guide](../guides/signal-analyzer.md)
