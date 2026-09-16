"""Import smoke tests: every core module must be importable.

Catches broken imports (missing modules, renamed symbols, syntax errors) at
collection speed, before any training runs. Importing a library never reads or writes a dataset.
"""

import importlib

import pytest

CORE_MODULES = [
    "main",
    "arguments",
    "project",
    "models",
    "opendpd",
    "opendpd.api",
    "opendpd.cli",
    "steps.train_pa",
    "steps.train_dpd",
    "steps.run_dpd",
    "steps.plot",
    "modules.data_collector",
    "modules.train_funcs",
    "modules.paths",
    "modules.loggers",
    "utils.metrics",
    "utils.util",
    "utils.plotting",
    "quant",
    "quant.quant_envs",
    "quant.qmodules",
    "datasets.demodulator",
    "backbones.gru",
    "backbones.dgru",
    "backbones.qgru",
    "backbones.qgru_amp1",
    "backbones.lstm",
    "backbones.vdlstm",
    "backbones.gmp",
    "backbones.tcn",
    "backbones.rvtdcnn",
    "backbones.mcldnn",
    "backbones.deltagru",
    "backbones.deltajanet",
    "backbones.tres_deltagru",
    "backbones.pgjanet",
    "backbones.dvrjanet",
    "backbones.bojanet",
    "backbones.apnrru",
]


@pytest.mark.parametrize("module_name", CORE_MODULES)
def test_module_imports(module_name):
    importlib.import_module(module_name)


def test_main_exposes_entry_point():
    """opendpd/cli.py and the opendpd-cli console script depend on main.main()."""
    import main

    assert callable(main.main)
