"""
OpenDPD - An Open-Source End-to-End Learning Framework for
Wideband Power Amplifier Modeling and Digital Pre-Distortion

Project Lead: Chang Gao
Core Developers: Yizhuo Wu, Ang Li
Lab of Efficient Machine Intelligence @ Delft University of Technology
Website: https://www.tudemi.com
"""

__version__ = "2.2.2"
__author__ = "Yizhuo Wu, Ang Li, Chang Gao"
__license__ = "Apache-2.0"
__email__ = "chang.gao@tudelft.nl"

# The high-level API (train_pa, train_dpd, ...) pulls in PyTorch and the whole
# training stack. It is resolved lazily so that ``import opendpd`` (and the
# contract package ``opendpd.schemas``, the CLI launcher and the GUI server)
# stay fast and dependency-light. ``opendpd.train_pa(...)`` keeps working.
_API_EXPORTS = (
    'train_pa',
    'train_dpd',
    'run_dpd',
    'plot_dpd',
    'load_dataset',
    'create_dataset',
    'OpenDPDTrainer',
)

__all__ = list(_API_EXPORTS)


def __getattr__(name):
    if name in _API_EXPORTS:
        from . import api
        return getattr(api, name)
    raise AttributeError(f"module 'opendpd' has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_API_EXPORTS))
