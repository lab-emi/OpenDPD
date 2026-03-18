"""Parse convergence curves from optimizer benchmark logs."""
import re
import os
import json

RESULTS_DIR = 'benchmark/optimizer_results'
OPTIMIZERS = [
    'adamw', 'adam', 'adamax', 'radam', 'nadam',
    'adafactor', 'rprop', 'rmsprop',
    'sgd', 'adadelta', 'adagrad', 'asgd', 'lbfgs',
]


def parse_all_epochs(logfile):
    with open(logfile) as f:
        text = f.read()
    blocks = text.split('Training Metrics')
    epochs = []
    for block in blocks[1:]:
        metrics = {}
        for key in ['EPOCH', 'VAL_ACLR_AVG', 'TRAIN_LOSS']:
            pattern = rf'{re.escape(key)}\s*│\s*([-\d.e+]+)'
            m = re.search(pattern, block)
            if m:
                metrics[key] = float(m.group(1))
        if metrics:
            epochs.append(metrics)
    return epochs


# Print val ACLR at epochs 0, 9, 24, 49, 74, 99
checkpoints = [0, 9, 24, 49, 74, 99]

print(f"{'Optimizer':<12}", end="")
for cp in checkpoints:
    print(f" {'Ep'+str(cp+1):>8}", end="")
print()
print("-" * 72)

for opt in OPTIMIZERS:
    logfile = os.path.join(RESULTS_DIR, f'{opt}.log')
    if not os.path.exists(logfile):
        continue
    epochs = parse_all_epochs(logfile)
    if not epochs:
        continue
    print(f"{opt:<12}", end="")
    for cp in checkpoints:
        if cp < len(epochs):
            val = epochs[cp].get('VAL_ACLR_AVG', 0)
            print(f" {val:>8.2f}", end="")
        else:
            print(f" {'N/A':>8}", end="")
    print()
