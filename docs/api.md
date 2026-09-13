# API reference

`import opendpd` gives access to the functions and the trainer class below. They wrap the same training pipeline
as `python main.py`; keyword arguments that are not listed are forwarded to the command-line configuration
(`arguments.py`), so every `main.py` option is also available from Python. The `opendpd-cli` entry point is
`python main.py` under another name: run `opendpd-cli --help` for the options.

::: opendpd.api
    options:
      members:
        - train_pa
        - train_dpd
        - run_dpd
        - plot_dpd
        - load_dataset
        - create_dataset
        - OpenDPDTrainer
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 2

## Studio 2.2.4 signal services

The versioned OpenAPI contract includes `/api/v1/signal-generator/signals` for private input-only generation and separate input CSV/metadata downloads, and `/api/v1/pa-library/models` and `/api/v1/pa-library/simulations` for mathematical PA selection, simulation, exports and explicit paired-dataset creation. Spectrum traces carry `signal_node` (`dpd_input`, `pa_input`, `pa_output`, or `unknown`); this is display provenance and does not change PSD values or scoring. Figure preview normalizes mixed PSD panels into separate positions. See the [generator](guides/signal-generator.md), [Virtual PA](guides/virtual-pa-library.md) and [PSD](guides/signal-chain-spectra.md) guides.
