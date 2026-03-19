# MLP DPD Optimizer Benchmark: AdamW vs Levenberg-Marquardt

## 1. Overview

This report compares AdamW (first-order) against Levenberg-Marquardt (second-order) for training a fully-connected (MLP/FCN) DPD model on the APA_200MHz dataset. The experiment is designed to give LM optimal conditions: a small feedforward model (42 parameters) with no RNN components, eliminating the functorch compatibility issues and expensive sequential Jacobian computation encountered with GRU models.

## 2. Motivation

This experiment was prompted by Maarten's questions about the GRU-based LM benchmark:

> *"2nd order optimizers like LM get much of their advantage from a precise Jacobian calculation for which it might need a larger batch size (small batch-size leads to noisy Jacobian), might be worth trying again with a larger batch size. Some works suggest that LM can achieve good results with smaller networks."*

This experiment addresses these concerns by:
1. **Larger batch size**: batch_size=256 (same as AdamW), up from 32 in the GRU experiment
2. **Smaller network**: 42 parameters (FCN), down from 519 (GRU)
3. **Same frame_length**: 200 for both optimizers (vs. 20 for GRU-LM)
4. **No RNN bottleneck**: Fully feedforward pipeline (FCN DPD + FCN PA)

## 3. Answers to Maarten's Questions

### Q: Was the 91 minutes to reach 19 or 100 epochs?

**91 minutes was for all 100 epochs.** The best result (epoch 19, val ACLR = -51.08 dB) was reached at approximately minute 17.4 (19 epochs x 55s/epoch). After epoch 20, the model destabilized and never recovered.

### Q: What starting value of mu was chosen?

The `torch-levenberg-marquardt` library uses a `StandardDampingStrategy` with:

| Parameter | Value |
|-----------|-------|
| **mu_init (starting_value)** | **1e-3** |
| dec_factor | 0.1 (multiply mu by 0.1 on successful step) |
| inc_factor | 10.0 (multiply mu by 10 on failed step) |
| min_value | 1e-10 |
| max_value | 1e10 (triggers early stopping) |

The LM update rule: **x_{k+1} = x_k - [J^T J + mu * I]^{-1} J^T e**

- Small mu -> Newton-like (aggressive, fast convergence near solution)
- Large mu -> Gradient descent-like (conservative, stable far from solution)

The library adaptively adjusts mu each step: decrease by 10x on success, increase by 10x on failure, up to 10 attempts per batch.

## 4. Experimental Setup

### 4.1 Model Architecture

| Component | Architecture | Parameters |
|-----------|-------------|-----------|
| PA Surrogate | FCN (h=64, L=1) | 322 |
| DPD Model | FCN (h=8, L=1) | 42 |
| Cascade | DPD -> PA (PA frozen) | 42 trainable |

The FCN (Fully Connected Network) is a simple MLP: `Linear(2->h) + ReLU + Linear(h->2)`. It is memoryless — each time step is processed independently.

**PA surrogate quality:** -22.2 dB NMSE (limited by lack of memory modeling). Both optimizers use the same frozen PA model, ensuring a fair comparison.

### 4.2 Training Configuration

| Parameter | AdamW | Levenberg-Marquardt |
|-----------|-------|-------------------|
| Epochs | 100 | 100 |
| Batch Size | 256 | 256 |
| Frame Length | 200 | 200 |
| Learning Rate | 5e-4 | 1.0 |
| Loss Function | MSE | MSE |
| Gradient Clipping | 200 | N/A (LM uses damping) |
| mu_init | N/A | 1e-3 |
| Solve Method | N/A | QR |
| Attempts/Step | N/A | 10 |
| Seed | 0 | 0 |

### 4.3 Dataset

| Property | Value |
|----------|-------|
| Dataset | APA_200MHz |
| Signal | 5-carrier LTE TM3.1a, 200 MHz BW, 256-QAM |
| Samples | 98,304 (60/20/20 split) |

## 5. Results

### 5.1 Summary

| Optimizer | Best Epoch | ACLR_AVG (dB) | EVM (dB) | NMSE (dB) | Time (min) | Speedup |
|-----------|-----------|---------------|----------|-----------|------------|---------|
| **AdamW** | 98 | -27.52 | **-22.08** | -22.21 | **0.58** | **38x faster** |
| **LM** | 39 | **-27.54** | -22.07 | -22.21 | 22.00 | 1x |

### 5.2 Convergence

| Epoch | AdamW ACLR (dB) | LM ACLR (dB) |
|-------|----------------|--------------|
| 0 | -13.24 | -27.34 |
| 10 | -27.17 | -27.26 |
| 20 | -27.28 | -27.32 |
| 30 | -27.34 | -27.30 |
| 50 | -27.34 | -27.31 |
| 100 | -27.34 | -27.31 |

## 6. Analysis

### Both optimizers converge to the same solution

AdamW and LM achieve virtually identical final performance: -27.52 vs -27.54 dB ACLR, -22.08 vs -22.07 dB EVM, -22.21 dB NMSE. This ~-27.5 dB ACLR ceiling is the **model capacity limit** of a memoryless 42-parameter FCN — not an optimizer limitation. Neither optimizer can push beyond this because the FCN cannot model the PA's memory effects.

### LM converges faster in epochs, but slower in wall time

LM reaches near-optimal ACLR in just **1 epoch** (-27.34 dB, epoch 0) while AdamW needs ~20 epochs. However, each LM epoch takes **13.2s** (vs. 0.35s for AdamW) due to Jacobian computation. Over 100 epochs:
- AdamW: 0.58 min total
- LM: 22.0 min total (**38x slower**)

### LM's Jacobian cost dominates

Even with a tiny 42-parameter model, LM is 38x slower. The Jacobian matrix is (batch_size * frame_length * output_dim) x params = (256 * 200 * 2) x 42 = 102,400 x 42. While this is manageable in memory, the per-sample vmap computation through the cascaded FCN DPD + FCN PA pipeline remains expensive.

### No instability with FCN (unlike GRU)

Unlike the GRU experiment where LM destabilized at epoch 20, the FCN pipeline shows stable training throughout all 100 epochs. This confirms that the GRU instability was caused by the non-convex RNN loss landscape, not a fundamental LM issue.

### The DPD problem is capacity-limited, not optimizer-limited

For this model size and architecture, the optimizer choice doesn't matter — both converge to the same global optimum. The DPD performance is bounded by:
1. **Model capacity**: A memoryless 42-parameter MLP cannot correct memory effects in the PA
2. **PA surrogate quality**: The FCN PA has -22.2 dB NMSE (vs. -43.5 dB for GRU PA)

To see a meaningful optimizer comparison, the model would need enough capacity that convergence quality differs between optimizers (e.g., a multi-layer FCN or larger hidden size where AdamW might get stuck in local minima).

## 7. Conclusions

1. **LM is not beneficial for FCN-based DPD at this scale.** It reaches the same solution as AdamW but 38x slower.
2. **LM's fast epoch-wise convergence** (1 epoch vs. 20 for AdamW) is real but overwhelmed by per-epoch computational cost.
3. **LM is stable on feedforward models** — the GRU instability was architecture-specific, not inherent to LM.
4. **The mu_init=1e-3 default** worked well; the optimizer did not diverge or need manual tuning.
5. **Larger batch sizes** (256 vs. 32 in GRU experiment) did not unlock LM advantages — the problem is too small for second-order methods to differentiate themselves.
6. **Future work**: Test LM on a larger FCN (e.g., 3+ layers, 200+ params) where AdamW might struggle with local minima, or on a problem where first-order methods converge slowly.

## 8. Reproducing These Results

### Prerequisites

```bash
pip install torch-levenberg-marquardt

# Train FCN PA surrogate
python main.py --step train_pa --dataset_name APA_200MHz --PA_backbone fcn --PA_hidden_size 64 --n_epochs 100
```

### Running the Benchmark

```bash
python benchmark/run_mlp_optimizer_benchmark.py \
    --n_epochs 100 \
    --pa_backbone fcn --pa_hidden_size 64 \
    --dpd_hidden_size 8 \
    --batch_size 256 --lr 5e-4 \
    --lm_batch_size 256 --lm_lr 1.0 --lm_attempts 10 --lm_solve qr
```
