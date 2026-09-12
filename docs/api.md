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
