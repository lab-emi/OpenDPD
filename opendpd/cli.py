"""
Command-line entry points for OpenDPD.

``opendpd-cli`` (``main``) is the historical entry point mirroring
``python main.py``. ``opendpd`` (``studio_main``) is the Studio entry point
with sub-commands (``run``, ``validate``, ``gui`` ...). Both import their
implementation lazily so ``--help`` does not load PyTorch.
"""

__author__ = "Chang Gao, Yizhuo Wu, Ang Li"
__license__ = "Apache-2.0 License"
__email__ = "chang.gao@tudelft.nl, yizhuo.wu@tudelft.nl, a.li-2@tudelft.nl"

import sys
from pathlib import Path

# Add parent directory to path to import existing modules
_parent_dir = Path(__file__).parent.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))


def main():
    """Legacy entry point (``opendpd-cli``): identical to ``python main.py``."""
    from main import main as run_main
    run_main()


def studio_main(argv=None):
    """Studio entry point (``opendpd``)."""
    from opendpd.commands import main as commands_main
    return commands_main(argv)


if __name__ == '__main__':
    main()
