"""Parse best-epoch results from optimizer benchmark logs."""
import re
import json
import os

RESULTS_DIR = 'benchmark/optimizer_results'

OPTIMIZERS = [
    'adam', 'adamw', 'sgd', 'rmsprop',
    'adadelta', 'adafactor', 'adagrad', 'adamax',
    'asgd', 'nadam', 'radam', 'rprop', 'lbfgs',
]

METRICS = ['EPOCH', 'TIME:', 'TRAIN_LOSS', 'VAL_ACLR_AVG', 'VAL_EVM', 'VAL_NMSE',
           'TEST_ACLR_AVG', 'TEST_ACLR_L', 'TEST_ACLR_R', 'TEST_EVM', 'TEST_NMSE', 'TEST_LOSS']


def parse_all_epochs(logfile):
    """Parse all epoch outputs from a log file. Return list of dicts."""
    with open(logfile) as f:
        text = f.read()

    # Split by epoch boundaries - each epoch prints a table
    # Find all EPOCH values and their surrounding metrics
    epochs = []
    # Find all table blocks
    blocks = text.split('Training Metrics')
    for block in blocks[1:]:  # skip preamble
        metrics = {}
        for key in METRICS:
            pattern = rf'{re.escape(key)}\s*│\s*([-\d.e+]+)'
            m = re.search(pattern, block)
            if m:
                metrics[key] = float(m.group(1))
        if metrics:
            epochs.append(metrics)
    return epochs


def main():
    results = []

    for opt in OPTIMIZERS:
        logfile = os.path.join(RESULTS_DIR, f'{opt}.log')
        if not os.path.exists(logfile):
            continue

        epochs = parse_all_epochs(logfile)
        if not epochs:
            continue

        # Find best epoch by validation ACLR_AVG (most negative = best)
        best = min(epochs, key=lambda e: e.get('VAL_ACLR_AVG', 0))
        best['optimizer'] = opt

        # Get total training time from last epoch
        if epochs:
            best['total_time_s'] = sum(e.get('TIME:', 0) for e in epochs)
            best['best_epoch'] = int(best.get('EPOCH', -1))

        results.append(best)

    # Sort by test ACLR_AVG
    results.sort(key=lambda r: r.get('TEST_ACLR_AVG', 0))

    # Print formatted table
    print(f"{'Rank':<5} {'Optimizer':<12} {'Best Ep':<8} {'Test ACLR_AVG':>14} {'Test EVM':>10} {'Test NMSE':>10} {'Val ACLR_AVG':>14} {'Time (min)':>10}")
    print("-" * 95)
    for i, r in enumerate(results, 1):
        print(f"{i:<5} {r['optimizer']:<12} {r['best_epoch']:<8} "
              f"{r.get('TEST_ACLR_AVG', 0):>14.2f} {r.get('TEST_EVM', 0):>10.2f} "
              f"{r.get('TEST_NMSE', 0):>10.2f} {r.get('VAL_ACLR_AVG', 0):>14.2f} "
              f"{r.get('total_time_s', 0)/60:>10.2f}")

    # Save detailed results
    with open(os.path.join(RESULTS_DIR, 'best_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {RESULTS_DIR}/best_results.json")


if __name__ == '__main__':
    main()
