"""
OpenDPD API - User-friendly interface for training PA and DPD models

This module provides high-level functions for easy use of OpenDPD functionality.
"""

__author__ = "Chang Gao, Yizhuo Wu, Ang Li"
__license__ = "Apache-2.0 License"
__email__ = "chang.gao@tudelft.nl, yizhuo.wu@tudelft.nl, a.li-2@tudelft.nl"

import os
import sys
from typing import Optional, Dict, Any, Union
from pathlib import Path

# Add parent directory to path to import existing modules
_parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(_parent_dir))

from project import Project
from steps import train_pa as train_pa_module
from steps import train_dpd as train_dpd_module
from steps import run_dpd as run_dpd_module
from arguments import get_arguments


def train_pa(
    dataset_name: Optional[str] = None,
    dataset_path: Optional[str] = None,
    PA_backbone: str = 'gru',
    PA_hidden_size: int = 23,
    n_epochs: int = 100,
    batch_size: int = 256,
    lr: float = 5e-4,
    accelerator: str = 'cpu',
    frame_length: int = 200,
    seed: int = 0,
    **kwargs
) -> Dict[str, Any]:
    """
    Train a Power Amplifier (PA) behavioral model.
    
    Args:
        dataset_name: Name of the dataset in the `datasets/` folder (e.g., 'DPA_200MHz')
        dataset_path: Deprecated. Specify `dataset_name` after importing the dataset.
        PA_backbone: Type of neural network backbone ('gru', 'lstm', 'dgru', 'deltagru', etc.)
        PA_hidden_size: Hidden size of the PA model
        n_epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        accelerator: Device to use ('cpu', 'cuda', or 'mps')
        frame_length: Length of signal frames
        seed: Random seed for reproducibility
        **kwargs: Additional arguments passed to the training configuration
        
    Returns:
        Dictionary containing training results and model path
        
    Examples:
        >>> import opendpd
        >>> results = opendpd.train_pa(dataset_name='DPA_200MHz', n_epochs=50)
        >>> print(f"Model saved at: {results['model_path']}")
        
        >>> # Train with custom dataset
        >>> results = opendpd.train_pa(dataset_path='/path/to/my_dataset.csv', PA_backbone='dgru')
    """
    # Prepare arguments
    sys.argv = ['opendpd']
    sys.argv.extend(['--step', 'train_pa'])
    
    if dataset_path:
        raise ValueError(
            "train_pa no longer accepts dataset_path. Please create an OpenDPD "
            "dataset (e.g., with create_dataset) and pass its dataset_name instead."
        )

    if dataset_name:
        sys.argv.extend(['--dataset_name', dataset_name])
    else:
        raise ValueError("train_pa requires dataset_name."
                         " Create a dataset first with create_dataset().")
    
    sys.argv.extend(['--PA_backbone', PA_backbone])
    sys.argv.extend(['--PA_hidden_size', str(PA_hidden_size)])
    sys.argv.extend(['--n_epochs', str(n_epochs)])
    sys.argv.extend(['--batch_size', str(batch_size)])
    sys.argv.extend(['--lr', str(lr)])
    sys.argv.extend(['--accelerator', accelerator])
    sys.argv.extend(['--frame_length', str(frame_length)])
    sys.argv.extend(['--seed', str(seed)])
    
    # Add any additional keyword arguments
    for key, value in kwargs.items():
        sys.argv.extend([f'--{key}', str(value)])
    
    # Create project and run training
    proj = Project()
    train_pa_module.main(proj)
    
    return {
        'status': 'completed',
        'model_path': proj.path_save_file_best,
        'log_path': proj.path_log_file_best,
    }


def train_dpd(
    dataset_name: Optional[str] = None,
    dataset_path: Optional[str] = None,
    DPD_backbone: str = 'deltagru_tcnskip',
    DPD_hidden_size: int = 15,
    PA_backbone: str = 'gru',
    PA_hidden_size: int = 23,
    n_epochs: int = 100,
    batch_size: int = 256,
    lr: float = 5e-4,
    accelerator: str = 'cpu',
    frame_length: int = 200,
    seed: int = 0,
    thx: float = 0.0,
    thh: float = 0.0,
    **kwargs
) -> Dict[str, Any]:
    """
    Train a Digital Pre-Distortion (DPD) model.
    
    This function trains a DPD model using a pre-trained PA model. The PA model
    must be trained first using train_pa().
    
    Args:
        dataset_name: Name of the dataset in the `datasets/` folder
        dataset_path: Deprecated. Specify `dataset_name` after importing the dataset.
        DPD_backbone: Type of DPD neural network backbone
        DPD_hidden_size: Hidden size of the DPD model
        PA_backbone: Type of PA backbone (must match the pre-trained PA model)
        PA_hidden_size: Hidden size of PA model (must match the pre-trained PA model)
        n_epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        accelerator: Device to use ('cpu', 'cuda', or 'mps')
        frame_length: Length of signal frames
        seed: Random seed for reproducibility
        thx: Threshold for input deltas (for delta-based models)
        thh: Threshold for hidden state deltas (for delta-based models)
        **kwargs: Additional arguments passed to the training configuration
        
    Returns:
        Dictionary containing training results and model path
        
    Examples:
        >>> import opendpd
        >>> # First train PA model
        >>> pa_results = opendpd.train_pa(dataset_name='DPA_200MHz')
        >>> # Then train DPD model
        >>> dpd_results = opendpd.train_dpd(dataset_name='DPA_200MHz', n_epochs=50)
        >>> print(f"DPD model saved at: {dpd_results['model_path']}")
    """
    # Prepare arguments
    sys.argv = ['opendpd']
    sys.argv.extend(['--step', 'train_dpd'])
    
    if dataset_path:
        raise ValueError(
            "train_dpd no longer accepts dataset_path. Please create an OpenDPD "
            "dataset (e.g., with create_dataset) and pass its dataset_name instead."
        )

    if dataset_name:
        sys.argv.extend(['--dataset_name', dataset_name])
    else:
        raise ValueError("train_dpd requires dataset_name."
                         " Create a dataset first with create_dataset().")
    
    sys.argv.extend(['--DPD_backbone', DPD_backbone])
    sys.argv.extend(['--DPD_hidden_size', str(DPD_hidden_size)])
    sys.argv.extend(['--PA_backbone', PA_backbone])
    sys.argv.extend(['--PA_hidden_size', str(PA_hidden_size)])
    sys.argv.extend(['--n_epochs', str(n_epochs)])
    sys.argv.extend(['--batch_size', str(batch_size)])
    sys.argv.extend(['--lr', str(lr)])
    sys.argv.extend(['--accelerator', accelerator])
    sys.argv.extend(['--frame_length', str(frame_length)])
    sys.argv.extend(['--seed', str(seed)])
    sys.argv.extend(['--thx', str(thx)])
    sys.argv.extend(['--thh', str(thh)])
    
    # Add any additional keyword arguments
    for key, value in kwargs.items():
        sys.argv.extend([f'--{key}', str(value)])
    
    # Create project and run training
    proj = Project()
    train_dpd_module.main(proj)
    
    return {
        'status': 'completed',
        'model_path': proj.path_save_file_best,
        'log_path': proj.path_log_file_best,
    }


def run_dpd(
    dataset_name: Optional[str] = None,
    dataset_path: Optional[str] = None,
    DPD_backbone: str = 'deltagru_tcnskip',
    DPD_hidden_size: int = 15,
    accelerator: str = 'cpu',
    **kwargs
) -> Dict[str, Any]:
    """
    Run the trained DPD model to generate pre-distorted signals.
    
    Args:
        dataset_name: Name of the dataset in the `datasets/` folder
        dataset_path: Deprecated. Specify `dataset_name` after importing the dataset.
        DPD_backbone: Type of DPD backbone (must match trained model)
        DPD_hidden_size: Hidden size of DPD model (must match trained model)
        accelerator: Device to use ('cpu', 'cuda', or 'mps')
        **kwargs: Additional arguments
        
    Returns:
        Dictionary containing output paths and results
        
    Examples:
        >>> import opendpd
        >>> results = opendpd.run_dpd(dataset_name='DPA_200MHz')
        >>> print(f"Output saved at: {results['output_path']}")
    """
    # Prepare arguments
    sys.argv = ['opendpd']
    sys.argv.extend(['--step', 'run_dpd'])
    
    if dataset_path:
        raise ValueError(
            "run_dpd no longer accepts dataset_path. Please create an OpenDPD "
            "dataset (e.g., with create_dataset) and pass its dataset_name instead."
        )

    if dataset_name:
        sys.argv.extend(['--dataset_name', dataset_name])
    else:
        raise ValueError("run_dpd requires dataset_name."
                         " Create a dataset first with create_dataset().")
    
    sys.argv.extend(['--DPD_backbone', DPD_backbone])
    sys.argv.extend(['--DPD_hidden_size', str(DPD_hidden_size)])
    sys.argv.extend(['--accelerator', accelerator])
    
    # Add any additional keyword arguments
    for key, value in kwargs.items():
        sys.argv.extend([f'--{key}', str(value)])
    
    # Create project and run DPD
    proj = Project()
    run_dpd_module.main(proj)
    
    return {
        'status': 'completed',
        'output_path': f'dpd_out/{dataset_name}' if dataset_name else 'dpd_out/',
    }


def load_dataset(dataset_path: str) -> Dict[str, Any]:
    """
    Load a dataset from a CSV file or directory.
    
    This function supports both formats:
    1. Split CSV files (train_input.csv, train_output.csv, etc.)
    2. Single CSV file with all data (I_in, Q_in, I_out, Q_out columns)
    
    Args:
        dataset_path: Path to the dataset directory or single CSV file
        
    Returns:
        Dictionary containing loaded data arrays
        
    Examples:
        >>> import opendpd
        >>> data = opendpd.load_dataset('datasets/DPA_200MHz')
        >>> print(data.keys())
        dict_keys(['X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test'])
        
        >>> # Load from single CSV
        >>> data = opendpd.load_dataset('my_data.csv')
    """
    from modules.data_collector import load_dataset as load_data

    path = Path(dataset_path).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
        if not path.exists():
            base_dir = Path(__file__).resolve().parent.parent
            candidate = (base_dir / dataset_path).resolve()
            if candidate.exists():
                path = candidate

    if path.is_file():
        data_arrays = load_data(dataset_path=str(path))
    elif path.is_dir():
        data_arrays = load_data(dataset_path=str(path))
    else:
        raise ValueError(f"Dataset path not found: {dataset_path}")
    
    X_train, y_train, X_val, y_val, X_test, y_test = data_arrays
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
    }


def create_dataset(
    csv_path: str,
    output_dir: str,
    dataset_name: str,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    dataset_format: str = 'single_csv',
    csv_filename: Optional[str] = None,
    **spec_kwargs
) -> str:
    """
    Create a dataset in OpenDPD format from a single CSV file.
    
    The input CSV should have 4 columns: I_in, Q_in, I_out, Q_out
    
    Args:
        csv_path: Path to the input CSV file
        output_dir: Directory where the dataset will be created
        dataset_name: Name of the dataset
        train_ratio: Ratio of data for training (default: 0.6)
        val_ratio: Ratio of data for validation (default: 0.2)
        test_ratio: Ratio of data for testing (default: 0.2)
        dataset_format: `'single_csv'` (default) to keep a single CSV or `'split_csv'` to
            generate the classic six-file layout
        csv_filename: Optional filename for the generated single CSV (defaults to `data.csv`)
        **spec_kwargs: Additional fields for spec.json (e.g., input_signal_fs, bw_main_ch)
        
    Returns:
        Path to the created dataset directory
        
    Examples:
        >>> import opendpd
        >>> dataset_path = opendpd.create_dataset(
        ...     csv_path='my_measurements.csv',
        ...     output_dir='datasets',
        ...     dataset_name='MyPA_Data',
        ...     input_signal_fs=800e6,
        ...     bw_main_ch=200e6
        ... )
        >>> print(f"Dataset created at: {dataset_path}")
    """
    import pandas as pd
    import json

    dataset_format_normalized = dataset_format.lower()
    if dataset_format_normalized not in {'single_csv', 'split_csv'}:
        raise ValueError("dataset_format must be 'single_csv' or 'split_csv'")

    # Resolve paths
    csv_path = Path(csv_path).expanduser()
    output_dir = Path(output_dir).expanduser()
    dataset_dir = (output_dir / dataset_name).resolve()
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Validate column names
    expected_cols = ['I_in', 'Q_in', 'I_out', 'Q_out']
    if not all(col in df.columns for col in expected_cols):
        raise ValueError(f"CSV must contain columns: {expected_cols}. Found: {df.columns.tolist()}")

    # Calculate split indices
    n_total = len(df)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    # Split the data
    train_data = df.iloc[:n_train]
    val_data = df.iloc[n_train:n_train + n_val]
    test_data = df.iloc[n_train + n_val:]

    spec = {
        'split_ratios': {
            'train': train_ratio,
            'val': val_ratio,
            'test': test_ratio
        },
        'nperseg': 2560,  # Default value
        'n_sub_ch': 1,
    }

    if dataset_format_normalized == 'split_csv':
        # Save split CSV files
        train_data[['I_in', 'Q_in']].to_csv(dataset_dir / 'train_input.csv', index=False)
        train_data[['I_out', 'Q_out']].to_csv(dataset_dir / 'train_output.csv', index=False)
        val_data[['I_in', 'Q_in']].to_csv(dataset_dir / 'val_input.csv', index=False)
        val_data[['I_out', 'Q_out']].to_csv(dataset_dir / 'val_output.csv', index=False)
        test_data[['I_in', 'Q_in']].to_csv(dataset_dir / 'test_input.csv', index=False)
        test_data[['I_out', 'Q_out']].to_csv(dataset_dir / 'test_output.csv', index=False)
    else:
        # Keep data in a single CSV
        if csv_filename is None:
            csv_filename = 'data.csv'
        concatenated = pd.concat([train_data, val_data, test_data], axis=0)
        concatenated.to_csv(dataset_dir / csv_filename, index=False)
        spec['csv_filename'] = csv_filename
        spec['split_indices'] = {
            'train_end': len(train_data),
            'val_end': len(train_data) + len(val_data)
        }

    # Update spec with caller-supplied fields (but keep enforced fields intact)
    spec.update(spec_kwargs)
    spec['dataset_format'] = dataset_format_normalized

    with open(dataset_dir / 'spec.json', 'w') as f:
        json.dump(spec, f, indent=4)

    print(f"Dataset created successfully at: {dataset_dir}")
    print(f"  - Training samples: {len(train_data)}")
    print(f"  - Validation samples: {len(val_data)}")
    print(f"  - Test samples: {len(test_data)}")

    return str(dataset_dir)


class OpenDPDTrainer:
    """
    Advanced trainer class for OpenDPD with more control over the training process.
    
    This class provides a more object-oriented interface for advanced users who need
    fine-grained control over the training process.
    
    Examples:
        >>> import opendpd
        >>> trainer = opendpd.OpenDPDTrainer(dataset_name='DPA_200MHz')
        >>> trainer.train_pa(n_epochs=50)
        >>> trainer.train_dpd(n_epochs=50)
        >>> trainer.evaluate()
    """
    
    def __init__(self, dataset_name: Optional[str] = None, dataset_path: Optional[str] = None, **kwargs):
        """
        Initialize the OpenDPD trainer.
        
        Args:
            dataset_name: Name of the dataset
            dataset_path: Path to custom dataset
            **kwargs: Additional configuration parameters
        """
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.config = kwargs
        self.pa_trained = False
        self.dpd_trained = False
        
    def train_pa(self, **kwargs):
        """Train PA model"""
        config = {**self.config, **kwargs}
        if self.dataset_name:
            config['dataset_name'] = self.dataset_name
        elif self.dataset_path:
            config['dataset_path'] = self.dataset_path
        
        result = train_pa(**config)
        self.pa_trained = True
        return result
        
    def train_dpd(self, **kwargs):
        """Train DPD model"""
        if not self.pa_trained:
            print("Warning: PA model not trained yet. Training PA model first...")
            self.train_pa()
        
        config = {**self.config, **kwargs}
        if self.dataset_name:
            config['dataset_name'] = self.dataset_name
        elif self.dataset_path:
            config['dataset_path'] = self.dataset_path
        
        result = train_dpd(**config)
        self.dpd_trained = True
        return result
        
    def run(self, **kwargs):
        """Run DPD model"""
        if not self.dpd_trained:
            raise RuntimeError("DPD model not trained yet. Call train_dpd() first.")
        
        config = {**self.config, **kwargs}
        if self.dataset_name:
            config['dataset_name'] = self.dataset_name
        elif self.dataset_path:
            config['dataset_path'] = self.dataset_path
        
        return run_dpd(**config)

