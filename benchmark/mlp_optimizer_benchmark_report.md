# MLP DPD Optimizer Benchmark: AdamW vs Levenberg-Marquardt

## 1. Overview

This report compares AdamW (first-order) against Levenberg-Marquardt (second-order) for training a fully-connected (MLP/FCN) DPD model on the APA_200MHz dataset. The experiment uses a small feedforward DPD (42 parameters) with the standard GRU PA surrogate model (1,911 parameters, frozen), addressing concerns about LM's performance with proper Jacobian computation on smaller networks.

## 2. Motivation

This experiment was prompted by Maarten's questions about the GRU-based LM benchmark:

> *"2nd order optimizers like LM get much of their advantage from a precise Jacobian calculation for which it might need a larger batch size (small batch-size leads to noisy Jacobian), might be worth trying again with a larger batch size. Some works suggest that LM can achieve good results with smaller networks."*

This experiment addresses these concerns by:
1. **Smaller network**: 42 parameters (FCN DPD), down from 519 (GRU DPD)
2. **Same GRU PA surrogate**: frozen GRU PA (h=23, 1,911 params, -43.5 dB NMSE)
3. **Same frame_length**: 200 for both optimizers
4. **LM batch_size=32**: limited by the GRU PA in the Jacobian forward pass (see Section 7)

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
| PA Surrogate | GRU (h=23, L=1) | 1,911 (frozen) |
| DPD Model | FCN (h=8, L=1) | 42 (trainable) |
| Cascade | DPD -> PA (ILA) | 42 trainable |

The FCN (Fully Connected Network) is a simple MLP: `Linear(2->8) + ReLU + Linear(8->2)`. It is memoryless — each time step is processed independently.

**PA surrogate quality:** -43.5 dB NMSE (GRU with memory modeling). Both optimizers use the same frozen PA model, ensuring a fair comparison.

### 4.2 Training Configuration

| Parameter | AdamW | Levenberg-Marquardt |
|-----------|-------|-------------------|
| Epochs | 100 | 100 |
| Batch Size | 256 | 32 |
| Frame Length | 200 | 200 |
| Learning Rate | 5e-4 | 1.0 |
| Loss Function | MSE | MSE |
| Gradient Clipping | 200 | N/A (LM uses damping) |
| mu_init | N/A | 1e-3 |
| Solve Method | N/A | QR |
| Attempts/Step | N/A | 10 |
| Seed | 0 | 0 |

**Note on LM batch size:** LM uses batch_size=32 (vs. 256 for AdamW) because the Jacobian computation requires forward+backward passes through the full cascaded model (including the GRU PA's 200-timestep unrolling) via functorch's `vmap`. Larger batch sizes make this prohibitively slow (see Section 7).

### 4.3 Dataset

| Property | Value |
|----------|-------|
| Dataset | APA_200MHz |
| Signal | 5-carrier LTE TM3.1a, 200 MHz BW, 256-QAM |
| PA | GaN Doherty PA (Ampleon AR211132), 3.5 GHz, 41.2 dBm |
| Samples | 98,304 (60/20/20 split) |

## 5. Results

### 5.1 Summary

| Optimizer | Best Epoch | ACLR_AVG (dB) | EVM (dB) | NMSE (dB) | Time (min) |
|-----------|-----------|---------------|----------|-----------|------------|
| AdamW | 14 | -27.65 | -29.74 | -29.65 | **0.67** |
| **LM** | 69 | **-27.79** | **-31.93** | **-31.38** | 411.0 |

### 5.2 Convergence (Validation ACLR_AVG)

| Epoch | AdamW (dB) | LM (dB) |
|-------|-----------|---------|
| 0 | -11.10 | -27.39 |
| 10 | -27.29 | -27.38 |
| 20 | -27.48 | -27.37 |
| 30 | -27.41 | -27.42 |
| 50 | -27.40 | -27.41 |
| 70 | -27.40 | -27.42 |
| 100 | -27.36 | -27.40 |

### 5.3 Training Loss

| Epoch | AdamW | LM |
|-------|-------|-----|
| 0 | 0.048815 | 0.000101 |
| 10 | 0.000158 | 0.000054 |
| 50 | 0.000055 | 0.000054 |
| 99 | 0.000055 | 0.000054 |

## 6. Analysis

### LM converges dramatically faster in epoch count

LM reaches near-optimal performance in **1 epoch** (val ACLR -27.39 dB, train loss 1.01e-4) while AdamW needs ~15 epochs to converge (val ACLR -27.48 dB, train loss 5.5e-5). LM's first epoch loss (1.01e-4) is already lower than AdamW's epoch 5 loss. This demonstrates the theoretical advantage of second-order methods: they use curvature information to take near-optimal steps.

### LM achieves marginally better final metrics

LM edges out AdamW on all three metrics:
- **ACLR:** -27.79 vs -27.65 dB (+0.14 dB)
- **EVM:** -31.93 vs -29.74 dB (+2.19 dB)
- **NMSE:** -31.38 vs -29.65 dB (+1.73 dB)

While the ACLR difference is small, the EVM and NMSE improvements (~2 dB) are meaningful. LM finds a slightly better minimum in the loss landscape, likely because the Gauss-Newton Hessian approximation provides more precise step directions than Adam's diagonal approximation.

### LM is 613x slower in wall time

Each LM epoch takes ~210s vs. 0.4s for AdamW. Over 100 epochs:
- AdamW: 0.67 min
- LM: 411.0 min (**613x slower**)

The bottleneck is the Jacobian computation via `vmap`, which must forward+backward propagate through the GRU PA's 200-timestep unrolling for each sample.

### LM is stable with the GRU PA (unlike GRU DPD)

Unlike the previous experiment where LM destabilized a GRU DPD at epoch 20, the FCN DPD + GRU PA configuration is stable throughout all 100 epochs. This confirms that the instability was caused by the GRU DPD's non-convex parameter landscape, not the GRU PA in the forward path. An FCN's loss surface is smoother, which suits LM's quadratic approximation.

### The ~-27.5 dB ACLR ceiling is a model capacity limit

Both optimizers converge to approximately -27.5 dB ACLR. This is the inherent limit of a memoryless 42-parameter FCN DPD — it cannot correct the PA's memory effects. For comparison, a GRU DPD (519 params) with AdamW achieves -49.4 dB ACLR on the same dataset.

## 7. LM Computational Cost Analysis

The Jacobian matrix dimensions for each batch:
- **Rows (residuals):** batch_size x frame_length x output_dim = 32 x 200 x 2 = 12,800
- **Columns (parameters):** 42 (FCN DPD trainable params)
- **Jacobian size:** 12,800 x 42 = 537,600 elements

However, the cost is dominated by the `vmap` computation, which runs 32 parallel forward+backward passes through the cascaded model. Each pass unrolls the GRU PA through 200 timesteps using `PureTensorGRUCell` (required because PyTorch's fused `nn.GRU` kernel is incompatible with functorch). This sequential unrolling is the primary bottleneck.

**Scaling considerations:**
- batch_size=256 with GRU PA was intractable (>74 min per epoch estimated)
- batch_size=32 yields ~210s/epoch
- A fully-FCN PA would reduce per-epoch time to ~13s (as shown in preliminary FCN PA experiment)
- The GRU PA adds ~16x overhead due to 200-step sequential unrolling

## 8. Conclusions

1. **LM finds a marginally better solution** than AdamW (+0.14 dB ACLR, +2.19 dB EVM) on FCN DPD with GRU PA surrogate.
2. **LM converges in 1 epoch** vs. ~15 for AdamW, demonstrating the power of second-order curvature information.
3. **LM is 613x slower** due to Jacobian computation through the GRU PA's temporal unrolling.
4. **LM is stable on FCN DPD** — the GRU instability in the previous experiment was DPD-architecture-specific.
5. **The mu_init=1e-3 default** worked well without manual tuning.
6. **The DPD performance is model-capacity-limited** (-27.5 dB ACLR for memoryless FCN vs. -49.4 dB for GRU), not optimizer-limited.
7. **LM may be better suited for offline DPD identification** where training time is not critical and solution quality matters, or for architectures where first-order methods struggle with local minima.

## 9. Reproducing These Results

### Prerequisites

```bash
pip install torch-levenberg-marquardt

# Train GRU PA surrogate (if not already available)
python main.py --step train_pa --dataset_name APA_200MHz --PA_backbone gru --PA_hidden_size 23 --n_epochs 100
```

### Running the Benchmark

```bash
python benchmark/run_mlp_optimizer_benchmark.py \
    --n_epochs 100 \
    --pa_backbone gru --pa_hidden_size 23 \
    --dpd_hidden_size 8 \
    --batch_size 256 --lr 5e-4 \
    --lm_batch_size 32 --lm_lr 1.0 --lm_attempts 10 --lm_solve qr
```
