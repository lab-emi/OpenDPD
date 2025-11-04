"""
Example script demonstrating OpenDPD Python API usage

This script shows how to use OpenDPD after pip installation:
  pip install opendpd

or in development mode:
  pip install -e .
"""

import opendpd
from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parent
CUSTOM_DATASET_NAME = 'MyCustomPA'

# =============================================================================
# Example 1: Basic PA Training
# =============================================================================
print("=" * 80)
print("Example 1: Training a PA Model")
print("=" * 80)

pa_results = opendpd.train_pa(
    dataset_name='DPA_200MHz',
    PA_backbone='gru',
    PA_hidden_size=23,
    n_epochs=1,  # Use more epochs (e.g., 100) for actual training
    batch_size=256,
    lr=5e-4,
    accelerator='cpu',
    seed=0
)

print(f"\n✓ PA model saved at: {pa_results['model_path']}")
print(f"✓ Training log saved at: {pa_results['log_path']}")

# =============================================================================
# Example 2: Basic DPD Training
# =============================================================================
print("\n" + "=" * 80)
print("Example 2: Training a DPD Model")
print("=" * 80)

dpd_results = opendpd.train_dpd(
    dataset_name='DPA_200MHz',
    DPD_backbone='deltagru_tcnskip',
    DPD_hidden_size=15,
    PA_backbone='gru',  # Must match the PA model trained above
    PA_hidden_size=23,  # Must match the PA model trained above
    n_epochs=1,  # Use more epochs (e.g., 100) for actual training
    batch_size=256,
    lr=5e-4,
    accelerator='cpu',
    seed=0,
    thx=0.0,  # Threshold for input deltas (for delta-based models)
    thh=0.0   # Threshold for hidden state deltas
)

print(f"\n✓ DPD model saved at: {dpd_results['model_path']}")
print(f"✓ Training log saved at: {dpd_results['log_path']}")

# =============================================================================
# Example 3: Running DPD
# =============================================================================
print("\n" + "=" * 80)
print("Example 3: Running the Trained DPD Model")
print("=" * 80)

run_results = opendpd.run_dpd(
    dataset_name='DPA_200MHz',
    DPD_backbone='deltagru_tcnskip',
    DPD_hidden_size=15,
    accelerator='cpu'
)

print(f"\n✓ DPD output saved at: {run_results['output_path']}")

# =============================================================================
# Example 4: Using the Trainer Class (Advanced)
# =============================================================================
print("\n" + "=" * 80)
print("Example 4: Using OpenDPDTrainer Class")
print("=" * 80)

trainer = opendpd.OpenDPDTrainer(
    dataset_name='DPA_200MHz',
    accelerator='cpu',
    n_epochs=1,
    seed=0
)

# Train PA
print("\n→ Training PA model...")
trainer.train_pa(PA_backbone='gru', PA_hidden_size=23)

# Train DPD
print("\n→ Training DPD model...")
trainer.train_dpd(DPD_backbone='deltagru_tcnskip', DPD_hidden_size=15)

# Run DPD
print("\n→ Running DPD...")
trainer.run()

print("\n✓ All training completed!")

# =============================================================================
# Example 5: Loading a Dataset
# =============================================================================
print("\n" + "=" * 80)
print("Example 5: Loading and Inspecting a Dataset")
print("=" * 80)

data = opendpd.load_dataset('datasets/DPA_200MHz')

print(f"\nDataset statistics:")
print(f"  Training samples:   {len(data['X_train']):,}")
print(f"  Validation samples: {len(data['X_val']):,}")
print(f"  Test samples:       {len(data['X_test']):,}")
print(f"  Input shape:        {data['X_train'].shape}")
print(f"  Output shape:       {data['y_train'].shape}")

# =============================================================================
# Example 6: Creating a Custom Dataset (from single CSV)
# =============================================================================
print("\n" + "=" * 80)
print("Example 6: Creating a Custom Dataset")
print("=" * 80)

# Note: This example assumes you have a CSV file with columns: I_in, Q_in, I_out, Q_out
# If you don't have such a file, this section will be skipped

csv_example_path = EXAMPLES_DIR / 'single_csv_format_example.csv'
datasets_dir = (EXAMPLES_DIR.parent / 'datasets').resolve()

if csv_example_path.exists():
    dataset_path = opendpd.create_dataset(
        csv_path=str(csv_example_path),
        output_dir=str(datasets_dir),
        dataset_name=CUSTOM_DATASET_NAME,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        input_signal_fs=800e6,
        bw_main_ch=200e6,
        nperseg=2560
    )
    
    print(f"\n✓ Custom dataset created at: {dataset_path}")
    print(f"  Use dataset_name='{CUSTOM_DATASET_NAME}' in train_pa/train_dpd.")
else:
    print(f"\n⚠ Skipping: Example CSV file not found at {csv_example_path}")
    print("  To test this feature, create a CSV with columns: I_in, Q_in, I_out, Q_out")

# =============================================================================
# Example 7: Training with the Newly Created Dataset
# =============================================================================
print("\n" + "=" * 80)
print("Example 7: Training with the Newly Created Dataset")
print("=" * 80)

dataset_dir = datasets_dir / CUSTOM_DATASET_NAME

if dataset_dir.exists():
    results = opendpd.train_pa(
        dataset_name=CUSTOM_DATASET_NAME,
        PA_backbone='gru',
        PA_hidden_size=23,
        n_epochs=1,
        accelerator='cpu'
    )
    print(f"\n✓ Model trained with dataset '{CUSTOM_DATASET_NAME}': {results['model_path']}")
else:
    print(f"\n⚠ Skipping: Dataset '{CUSTOM_DATASET_NAME}' not found in {datasets_dir}")
    print("  Run Example 6 first to generate the dataset from a CSV file.")

print("\n" + "=" * 80)
print("All examples completed successfully!")
print("=" * 80)
print("\nNext steps:")
print("  1. Modify the hyperparameters (n_epochs, hidden_size, etc.)")
print("  2. Try different backbones (dgru, lstm, deltagru, etc.)")
print("  3. Use GPU acceleration by setting accelerator='cuda'")
print("  4. Create your own datasets and train custom models")
print("=" * 80)

