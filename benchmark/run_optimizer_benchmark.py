"""Benchmark all torch.optim optimizers on GRU DPD / APA_200MHz."""
import subprocess
import sys
import time
import json
import re
import os

OPTIMIZERS = [
    'adam', 'adamw', 'sgd', 'rmsprop',
    'adadelta', 'adafactor', 'adagrad', 'adamax',
    'asgd', 'nadam', 'radam', 'rprop',
    'lbfgs', 'sparseadam', 'muon',
]

BASE_CMD = [
    sys.executable, 'main.py',
    '--step', 'train_dpd',
    '--dataset_name', 'APA_200MHz',
    '--PA_backbone', 'gru',
    '--PA_hidden_size', '23',
    '--DPD_backbone', 'gru',
    '--DPD_hidden_size', '11',
    '--n_epochs', '100',
    '--lr', '5e-4',
    '--seed', '0',
    '--batch_size', '256',
]

RESULTS_DIR = 'benchmark/optimizer_results'
os.makedirs(RESULTS_DIR, exist_ok=True)


def parse_metrics(output: str):
    """Parse final metrics from training output."""
    metrics = {}
    # Parse table-style output: look for the last occurrence of metric values
    patterns = {
        'TEST_ACLR_AVG': r'TEST_ACLR_AVG\s*│\s*([-\d.]+)',
        'TEST_ACLR_L': r'TEST_ACLR_L\s*│\s*([-\d.]+)',
        'TEST_ACLR_R': r'TEST_ACLR_R\s*│\s*([-\d.]+)',
        'TEST_EVM': r'TEST_EVM\s*│\s*([-\d.]+)',
        'TEST_NMSE': r'TEST_NMSE\s*│\s*([-\d.]+)',
        'TEST_LOSS': r'TEST_LOSS\s*│\s*([-\d.]+)',
        'VAL_ACLR_AVG': r'VAL_ACLR_AVG\s*│\s*([-\d.]+)',
        'VAL_EVM': r'VAL_EVM\s*│\s*([-\d.]+)',
        'TRAIN_LOSS': r'TRAIN_LOSS\s*│\s*([-\d.]+)',
        'EPOCH': r'EPOCH\s*│\s*(\d+)',
        'TIME:': r'TIME:\s*│\s*([-\d.]+)',
    }
    for key, pattern in patterns.items():
        matches = re.findall(pattern, output)
        if matches:
            metrics[key] = float(matches[-1])  # Take last occurrence (final epoch)
    return metrics


def run_optimizer(opt_name):
    """Run a single optimizer benchmark."""
    cmd = BASE_CMD + ['--opt_type', opt_name]
    print(f"\n{'='*60}")
    print(f"  Running: {opt_name}")
    print(f"{'='*60}")

    start = time.time()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=3600,
            cwd='/home/cgao/git/OpenDPD'
        )
        elapsed = time.time() - start
        output = result.stdout + result.stderr

        # Save raw log
        with open(os.path.join(RESULTS_DIR, f'{opt_name}.log'), 'w') as f:
            f.write(output)

        if result.returncode != 0:
            print(f"  FAILED (exit code {result.returncode})")
            # Check for known errors
            if 'SparseAdam' in output and 'dense' in output.lower():
                error_msg = "SparseAdam requires sparse gradients (not applicable to dense GRU)"
            elif 'Muon' in output or 'muon' in output:
                error_msg = "Muon not available in this PyTorch version"
            else:
                # Get last few lines of error
                error_lines = [l for l in output.strip().split('\n') if l.strip()][-5:]
                error_msg = '\n'.join(error_lines)
            return {
                'optimizer': opt_name,
                'status': 'FAILED',
                'error': error_msg,
                'time_s': elapsed,
            }

        metrics = parse_metrics(output)
        print(f"  Completed in {elapsed:.1f}s")
        print(f"  Test ACLR_AVG: {metrics.get('TEST_ACLR_AVG', 'N/A')} dB")
        print(f"  Test EVM: {metrics.get('TEST_EVM', 'N/A')} dB")

        return {
            'optimizer': opt_name,
            'status': 'OK',
            'time_s': elapsed,
            **metrics,
        }

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        print(f"  TIMEOUT after {elapsed:.1f}s")
        return {
            'optimizer': opt_name,
            'status': 'TIMEOUT',
            'time_s': elapsed,
        }
    except Exception as e:
        elapsed = time.time() - start
        print(f"  ERROR: {e}")
        return {
            'optimizer': opt_name,
            'status': 'ERROR',
            'error': str(e),
            'time_s': elapsed,
        }


def main():
    print("=" * 60)
    print("  Optimizer Benchmark: GRU DPD on APA_200MHz")
    print("  Epochs: 100 | LR: 5e-4 | Batch: 256 | Seed: 0")
    print("=" * 60)

    results = []
    for opt in OPTIMIZERS:
        result = run_optimizer(opt)
        results.append(result)
        # Save intermediate results
        with open(os.path.join(RESULTS_DIR, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    print(f"{'Optimizer':<15} {'Status':<10} {'ACLR_AVG (dB)':<18} {'EVM (dB)':<15} {'Time (s)':<10}")
    print("-" * 80)
    for r in sorted(results, key=lambda x: x.get('TEST_ACLR_AVG', 0)):
        if r['status'] == 'OK':
            print(f"{r['optimizer']:<15} {r['status']:<10} {r.get('TEST_ACLR_AVG', 'N/A'):<18.2f} {r.get('TEST_EVM', 'N/A'):<15.2f} {r['time_s']:<10.1f}")
        else:
            print(f"{r['optimizer']:<15} {r['status']:<10} {'N/A':<18} {'N/A':<15} {r['time_s']:<10.1f}")

    with open(os.path.join(RESULTS_DIR, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_DIR}/results.json")


if __name__ == '__main__':
    main()
