"""
Command-line interface for OpenDPD
"""

__author__ = "Chang Gao, Yizhuo Wu, Ang Li"
__license__ = "Apache-2.0 License"
__email__ = "chang.gao@tudelft.nl, yizhuo.wu@tudelft.nl, a.li-2@tudelft.nl"

import sys
import os
from pathlib import Path

# Add parent directory to path to import existing modules
_parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(_parent_dir))

from main import main as run_main


def main():
    """Main entry point for the CLI"""
    run_main()


if __name__ == '__main__':
    main()

