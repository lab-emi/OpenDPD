# OpenDPD Examples

This directory contains example scripts demonstrating how to use the OpenDPD Python API.

## Files

- `api_usage_example.py` - Comprehensive examples of all API functions
- `single_csv_format_example.csv` - Example CSV file in the correct format for single CSV datasets

## Running the Examples

### Prerequisites

Install OpenDPD first:
```bash
# From the root OpenDPD directory
pip install -e .
```

### Run the Example Script

```bash
python examples/api_usage_example.py
```

This script demonstrates:
1. Training a PA model
2. Training a DPD model
3. Running the trained DPD model
4. Using the OpenDPDTrainer class
5. Loading and inspecting datasets
6. Creating custom datasets from CSV files
7. Training with custom dataset paths

## Creating Your Own Dataset

### Single CSV Format

Create a CSV file with 4 columns: `I_in`, `Q_in`, `I_out`, `Q_out`

Example:
```csv
I_in,Q_in,I_out,Q_out
0.0123,-0.0456,0.0145,-0.0523
-0.0234,0.0567,-0.0267,0.0623
0.0345,-0.0678,0.0389,-0.0745
...
```

Then use it:
```python
import opendpd
opendpd.train_pa(dataset_path='my_measurements.csv', n_epochs=100)
```

### Split CSV Format

Create a directory with separate CSV files:
```
MyDataset/
├── spec.json
├── train_input.csv   (columns: I, Q)
├── train_output.csv  (columns: I, Q)
├── val_input.csv
├── val_output.csv
├── test_input.csv
└── test_output.csv
```

Then use it:
```python
import opendpd
opendpd.train_pa(dataset_path='path/to/MyDataset', n_epochs=100)
```

## Quick Examples

### Minimal PA Training
```python
import opendpd
opendpd.train_pa(dataset_name='DPA_200MHz', n_epochs=100)
```

### Minimal DPD Training
```python
import opendpd
opendpd.train_pa(dataset_name='DPA_200MHz', n_epochs=50)
opendpd.train_dpd(dataset_name='DPA_200MHz', n_epochs=50)
```

### Custom Dataset
```python
import opendpd
opendpd.train_pa(dataset_path='my_data.csv', n_epochs=50)
```

## More Information

- API Quick Start: See `../API_QUICKSTART.md`
- Full Documentation: See `../PACKAGE_INSTALLATION.md`
- Main README: See `../README.md`

