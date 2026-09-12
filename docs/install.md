# Installation

## From PyPI

```bash
pip install opendpd
```

The package provides the high-level functions `opendpd.train_pa`, `opendpd.train_dpd`, `opendpd.run_dpd`,
`opendpd.plot_dpd`, `opendpd.load_dataset` and `opendpd.create_dataset` (see the [API reference](api.md)) and the
`opendpd-cli` entry point, which mirrors `python main.py` of the repository.

## From source

The repository is the full research codebase: automation scripts, dataset tooling, quantization utilities and the
reproducible baselines. Clone it and install it in editable mode once the environment below is ready:

```bash
git clone https://github.com/lab-emi/OpenDPD.git
cd OpenDPD
pip install -e ".[dev]"
```

### Repository layout

--8<-- "README.md:layout"

--8<-- "README.md:environment"
