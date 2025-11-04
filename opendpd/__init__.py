"""
OpenDPD - An Open-Source End-to-End Learning Framework for 
Wideband Power Amplifier Modeling and Digital Pre-Distortion

Project Lead: Chang Gao
Core Developers: Yizhuo Wu, Ang Li
Lab of Efficient Machine Intelligence @ Delft University of Technology
Website: https://www.tudemi.com
"""

__version__ = "2.0.0"
__author__ = "Yizhuo Wu, Ang Li, Chang Gao"
__license__ = "Apache-2.0"
__email__ = "chang.gao@tudelft.nl"

# Import main API functions
from .api import train_pa, train_dpd, run_dpd, load_dataset, create_dataset

# Import core classes for advanced users
from .api import OpenDPDTrainer

# Define what gets imported with "from opendpd import *"
__all__ = [
    'train_pa',
    'train_dpd', 
    'run_dpd',
    'load_dataset',
    'create_dataset',
    'OpenDPDTrainer',
]

