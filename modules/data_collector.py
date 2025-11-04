__author__ = "Yizhuo Wu, Chang Gao"
__license__ = "Apache-2.0 License"
__email__ = "yizhuo.wu@tudelft.nl, chang.gao@tudelft.nl"

import os
import json
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


BASE_DIR = Path(__file__).resolve().parent.parent


def load_dataset(dataset_name=None, dataset_path=None):
    """
    Load dataset from either split CSV files or a single CSV file.
    
    Supports two formats:
    1. Split CSV format: Separate train_input.csv, train_output.csv, etc.
    2. Single CSV format: One CSV with columns [I_in, Q_in, I_out, Q_out]
       and split ratios defined in spec.json
    
    Args:
        dataset_name: Name of dataset in datasets/ folder
        dataset_path: Direct path to dataset directory or CSV file
        
    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    # Determine dataset path
    if dataset_name:
        path_dataset = BASE_DIR / 'datasets' / dataset_name
    elif dataset_path:
        path_dataset = Path(dataset_path).expanduser()
        if not path_dataset.is_absolute():
            path_dataset = (Path.cwd() / path_dataset).resolve()
    else:
        raise ValueError("Either dataset_name or dataset_path must be provided")
    
    # Check if it's a single CSV file or a directory
    if path_dataset.is_file() and path_dataset.suffix.lower() == '.csv':
        # Single CSV file format
        return _load_single_csv(str(path_dataset))
    
    # Check for spec.json to determine format
    spec_path = path_dataset / 'spec.json'
    if spec_path.exists():
        with open(spec_path, 'r') as f:
            spec = json.load(f)
        dataset_format = spec.get('dataset_format', 'split_csv')
    else:
        # Default to split CSV format if no spec.json
        dataset_format = 'split_csv'
    
    if dataset_format == 'single_csv':
        # Single CSV with split info in spec.json
        csv_file = spec.get('csv_filename', 'data.csv')
        csv_path = path_dataset / csv_file
        return _load_single_csv_with_spec(str(csv_path), spec)
    else:
        # Split CSV format (original format)
        return _load_split_csv(path_dataset)


def _load_split_csv(path_dataset):
    """Load dataset from split CSV files (original format)"""
    path_dataset = Path(path_dataset)
    X_train = pd.read_csv(path_dataset / 'train_input.csv').to_numpy()
    y_train = pd.read_csv(path_dataset / 'train_output.csv').to_numpy()
    X_val = pd.read_csv(path_dataset / 'val_input.csv').to_numpy()
    y_val = pd.read_csv(path_dataset / 'val_output.csv').to_numpy()
    X_test = pd.read_csv(path_dataset / 'test_input.csv').to_numpy()
    y_test = pd.read_csv(path_dataset / 'test_output.csv').to_numpy()
    return X_train, y_train, X_val, y_val, X_test, y_test


def _load_single_csv(csv_path):
    """
    Load dataset from a single CSV file.
    Expected columns: I_in, Q_in, I_out, Q_out
    Uses default split ratios: 60% train, 20% val, 20% test
    """
    df = pd.read_csv(csv_path)
    
    # Validate columns
    required_cols = ['I_in', 'Q_in', 'I_out', 'Q_out']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"CSV must contain columns: {required_cols}. Found: {df.columns.tolist()}")
    
    # Default split ratios
    train_ratio = 0.6
    val_ratio = 0.2
    
    return _split_dataframe(df, train_ratio, val_ratio)


def _load_single_csv_with_spec(csv_path, spec):
    """
    Load dataset from a single CSV file with split ratios from spec.json
    """
    df = pd.read_csv(csv_path)
    
    # Validate columns
    required_cols = ['I_in', 'Q_in', 'I_out', 'Q_out']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"CSV must contain columns: {required_cols}. Found: {df.columns.tolist()}")
    
    # Get split ratios from spec (default: 60% train, 20% val, 20% test)
    split_ratios = spec.get('split_ratios', {"train": 0.6, "val": 0.2, "test": 0.2})
    train_ratio = split_ratios.get('train', 0.6)
    val_ratio = split_ratios.get('val', 0.2)
    
    return _split_dataframe(df, train_ratio, val_ratio)


def _split_dataframe(df, train_ratio, val_ratio):
    """
    Split a dataframe into train, validation, and test sets
    """
    n_total = len(df)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    # Split the data
    train_data = df.iloc[:n_train]
    val_data = df.iloc[n_train:n_train + n_val]
    test_data = df.iloc[n_train + n_val:]
    
    # Extract input (I, Q) and output (I, Q) as numpy arrays
    X_train = train_data[['I_in', 'Q_in']].to_numpy()
    y_train = train_data[['I_out', 'Q_out']].to_numpy()
    X_val = val_data[['I_in', 'Q_in']].to_numpy()
    y_val = val_data[['I_out', 'Q_out']].to_numpy()
    X_test = test_data[['I_in', 'Q_in']].to_numpy()
    y_test = test_data[['I_out', 'Q_out']].to_numpy()
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def prepare_segments(args):
    """
    Split the IQ_data into segments of size nperseg. Zero padding is applied
    if the last section is not of length nperseg.
    """
    nperseg = args.nperseg
    path_dataset = BASE_DIR / 'datasets' / args.dataset_name
    train_input = pd.read_csv(path_dataset / 'train_input.csv')
    train_output = pd.read_csv(path_dataset / 'train_output.csv')
    val_input = pd.read_csv(path_dataset / 'val_input.csv')
    val_output = pd.read_csv(path_dataset / 'val_output.csv')
    test_input = pd.read_csv(path_dataset / 'test_input.csv')
    test_output = pd.read_csv(path_dataset / 'test_output.csv')

    def split_segments(IQ_data):
        num_samples = IQ_data.shape[0]
        segments = []
        for i in range(0, num_samples, nperseg):
            segment = IQ_data[i:i + nperseg]
            segment = np.asarray(segment)
            if segment.shape[0] < nperseg:
                padding_shape = (nperseg - segment.shape[0], segment.shape[1])
                segment = np.vstack((segment, np.zeros(padding_shape, dtype=segment.dtype)))
            segments.append(segment)
        return np.array(segments)

    train_input_segments = split_segments(train_input)
    train_output_segments = split_segments(train_output)
    val_input_segments = split_segments(val_input)
    val_output_segments = split_segments(val_output)
    test_input_segments = split_segments(test_input)
    test_output_segments = split_segments(test_output)

    return train_input_segments, train_output_segments, val_input_segments, val_output_segments, test_input_segments, test_output_segments


def get_training_frames(segments, seq_len, stride=1):
    """
    For each segment, apply the framing operation.

    Args:
    - segments (3D array): The segments produced by get_ofdm_segments.
    - seq_len (int): The length of each frame.
    - stride_length (int, optional): The step between frames. Default is 1.

    Returns:
    - 3D array where the first dimension is the total number of frames,
      the second dimension is seq_len, and the third dimension is 2 (I and Q).
    """

    all_frames = []
    for segment in segments:
        num_frames = (segment.shape[0] - seq_len) // stride + 1
        for i in range(num_frames):
            frame = segment[i * stride: i * stride + seq_len]
            all_frames.append(frame)

    return np.array(all_frames)


class IQSegmentDataset(Dataset):
    def __init__(self, features, targets, nperseg=16384):
        self.nperseg = nperseg

        features = self.split_segments(features)
        targets = self.split_segments(targets)
        self.features = torch.Tensor(features)
        self.targets = torch.Tensor(targets)

    def split_segments(self, sequence):
        num_samples = len(sequence)
        segments = []
        for i in range(0, num_samples, self.nperseg):
            segment = sequence[i:i + self.nperseg]
            segment = np.asarray(segment)
            if segment.shape[0] < self.nperseg:
                padding_shape = (self.nperseg - segment.shape[0], segment.shape[1])
                segment = np.vstack((segment, np.zeros(padding_shape, dtype=segment.dtype)))
            segments.append(segment)
        return np.array(segments)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        features = self.features[idx, ...]
        targets = self.targets[idx, ...]
        return features, targets


class IQFrameDataset(Dataset):
    def __init__(self, features, targets, frame_length, stride=1):
        # Convert segments into frames
        self.features = torch.Tensor(self.get_frames(features, frame_length, stride))
        self.targets = torch.Tensor(self.get_frames(targets, frame_length, stride))

    @staticmethod
    def get_frames(sequence, frame_length, stride_length):
            frames = []
            sequence_length = len(sequence)
            num_frames = (sequence_length - frame_length) // stride_length + 1
            for i in range(num_frames):
                frame = sequence[i * stride_length: i * stride_length + frame_length]
                frames.append(frame)
            return np.stack(frames)
    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


def data_prepare(X, y, frame_length, degree):
    Input = []
    Output = []
    X = torch.Tensor(X)
    y = torch.Tensor(y)
    for k in range(X.shape[0]):
        Complex_In = torch.complex(X[k, :, 0], X[k, :, 1])
        Complex_Out = torch.complex(y[k, :, 0], y[k, :, 1])
        ulength = len(Complex_In) - frame_length
        Input_matrix = torch.complex(torch.zeros(ulength, frame_length),
                                     torch.zeros(ulength, frame_length))
        degree_matrix = torch.complex(torch.zeros(ulength - frame_length, frame_length * frame_length * degree),
                                      torch.zeros(ulength - frame_length, frame_length * frame_length * degree))
        for i in range(ulength):
            Input_matrix[i, :] = Complex_In[i:i + frame_length]
        for j in range(1, degree):
            for h in range(frame_length):
                degree_matrix[:,
                (j - 1) * frame_length * frame_length + h * frame_length:(j - 1) * frame_length * frame_length + (
                        h + 1) * frame_length] = Input_matrix[:ulength - frame_length] * torch.pow(
                    abs(Input_matrix[h:h + ulength - frame_length, :]), j)
        Input_matrix = torch.cat((Input_matrix[:ulength - frame_length], degree_matrix), dim=1)
        b_output = np.array(Complex_Out[:len(Complex_In) - 2 * frame_length])
        b_input = np.array(Input_matrix)
        Input.append(b_input)
        Output.append(b_output)

    return Input, Output


class IQFrameDataset_gmp(Dataset):
    def __init__(self, segment_dataset, frame_length, degree, stride_length=1):
        """
        Initialize the frame dataset using a subset of IQSegmentDataset.

        Args:
        - segment_dataset (IQSegmentDataset): The pre-split segment dataset.
        - seq_len (int): The length of each frame.
        - stride_length (int, optional): The step between frames. Default is 1.
        """

        # Extract segments from the segment_dataset
        IQ_in_segments = [item[0] for item in segment_dataset]
        IQ_out_segments = [item[1] for item in segment_dataset]

        # Convert the list of tensors to numpy arrays
        IQ_in_segments = torch.stack(IQ_in_segments).numpy()
        IQ_out_segments = torch.stack(IQ_out_segments).numpy()

        self.IQ_in_frames, self.IQ_out_frames = data_prepare(IQ_in_segments, IQ_out_segments, frame_length, degree)

    def __len__(self):
        return len(self.IQ_in_frames)

    def __getitem__(self, idx):
        return self.IQ_in_frames[idx], self.IQ_out_frames[idx]
