#!/bin/bash
# Benchmark all torch.optim optimizers on GRU DPD / APA_200MHz
# GRU hidden_size=11, 100 epochs, lr=5e-4, seed=0

DATASET="APA_200MHz"
PA_BACKBONE="gru"
PA_HIDDEN=23
DPD_BACKBONE="gru"
DPD_HIDDEN=11
EPOCHS=100
LR="5e-4"
SEED=0
BATCH_SIZE=256

OPTIMIZERS="adam adamw sgd rmsprop adadelta adafactor adagrad adamax asgd nadam radam rprop lbfgs sparseadam muon"

RESULTS_DIR="benchmark/optimizer_results"
mkdir -p "$RESULTS_DIR"

echo "=== Optimizer Benchmark: GRU DPD on $DATASET ==="
echo "Epochs: $EPOCHS | LR: $LR | Batch Size: $BATCH_SIZE | Seed: $SEED"
echo ""

for OPT in $OPTIMIZERS; do
    echo ">>> Running optimizer: $OPT"
    LOG_FILE="$RESULTS_DIR/${OPT}.log"

    START_TIME=$(date +%s.%N)

    python main.py \
        --step train_dpd \
        --dataset_name "$DATASET" \
        --PA_backbone "$PA_BACKBONE" \
        --PA_hidden_size "$PA_HIDDEN" \
        --DPD_backbone "$DPD_BACKBONE" \
        --DPD_hidden_size "$DPD_HIDDEN" \
        --n_epochs "$EPOCHS" \
        --opt_type "$OPT" \
        --lr "$LR" \
        --seed "$SEED" \
        --batch_size "$BATCH_SIZE" \
        --filename "opt_bench_${OPT}" \
        2>&1 | tee "$LOG_FILE"

    END_TIME=$(date +%s.%N)
    ELAPSED=$(echo "$END_TIME - $START_TIME" | bc)
    echo ">>> $OPT completed in ${ELAPSED}s"
    echo "$ELAPSED" > "$RESULTS_DIR/${OPT}.time"
    echo ""
done

echo "=== All benchmarks complete ==="
