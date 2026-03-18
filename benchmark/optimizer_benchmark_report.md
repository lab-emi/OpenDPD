# Optimizer Benchmark Report: All torch.optim Optimizers for GRU DPD

## 1. Overview

This report benchmarks all 15 optimizers available in `torch.optim` for training a GRU-based Digital Pre-Distortion (DPD) model on the APA_200MHz dataset. The goal is to identify which optimizers are most effective for DPD training under identical hyperparameter conditions.

All optimizers use the same learning rate (5e-4) and training configuration for a fair comparison. No optimizer-specific hyperparameter tuning was performed — each optimizer uses PyTorch defaults for all parameters except learning rate.

## 2. Test Signal

| Property | APA_200MHz |
|----------|-----------|
| Standard | LTE TM3.1a |
| Configuration | 5-carrier x 40 MHz |
| Total Bandwidth | 200 MHz |
| Modulation | 256-QAM |
| PAPR | 10.01 dB (at CCDF of 0.001%) |
| Sampling Rate | 983.04 MHz |
| Dataset Size | 98,304 samples |
| Segment Size (nperseg) | 19,662 |
| Dataset Split | 60% / 20% / 20% (train / val / test) |

## 3. Power Amplifier Device Under Test (DUT)

| Property | APA_200MHz |
|----------|-----------|
| PA Type | GaN Doherty PA |
| Part / Technology | Ampleon AR211132 (evaluation board) |
| Carrier Frequency | 3.5 GHz |
| Average Output Power | 41.2 dBm |
| P1dB Compression Point | 46.5 dBm |
| P3dB Compression Point | 50 dBm |
| Flat Gain Bandwidth | 200 MHz |
| Reference | Wu et al., "OpenDPDv2," arXiv:2507.06849 |

## 4. PA Surrogate Model

| Property | Value |
|----------|-------|
| PA Backbone | GRU |
| PA Hidden Size | 23 |
| PA Parameters | 1,911 |
| PA Training | 100 epochs, AdamW, lr=5e-4 |
| Best PA NMSE | -43.52 dB |

## 5. DPD Model

| Property | Value |
|----------|-------|
| DPD Backbone | GRU (standard PyTorch GRU cell + linear output layer) |
| Hidden Size | 11 |
| Num Layers | 1 |
| Parameters | 519 real-valued |
| Input / Output | I/Q samples (2D) → I/Q samples (2D) |
| Training Architecture | Indirect Learning Architecture (ILA) with frozen PA model |

## 6. Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 100 |
| Learning Rate | 5e-4 (uniform across all optimizers) |
| Batch Size | 256 |
| Loss Function | MSE (L2) |
| LR Schedule | None |
| Gradient Clipping | 200 |
| Best Model Selection | Validation ACLR_AVG |
| Seed | 0 |
| Device | CUDA |
| PyTorch Version | 2.10.0+cu128 |

## 7. Optimizers Tested

### 7.1 Successfully Benchmarked (13/15)

| Optimizer | Type | Key Characteristics |
|-----------|------|-------------------|
| **Adam** | Adaptive | Momentum + adaptive learning rates (beta1=0.9, beta2=0.999) |
| **AdamW** | Adaptive | Adam with decoupled weight decay |
| **Adamax** | Adaptive | Adam variant based on infinity norm |
| **NAdam** | Adaptive | Adam with Nesterov momentum |
| **RAdam** | Adaptive | Rectified Adam with variance-adaptive learning rate |
| **Adafactor** | Adaptive | Memory-efficient Adam alternative with factored second moments |
| **RMSprop** | Adaptive | Adaptive learning rates via running average of squared gradients |
| **Rprop** | Sign-based | Resilient backpropagation; uses only gradient signs |
| **Adagrad** | Adaptive | Adapts learning rate per-parameter; accumulates squared gradients |
| **SGD** | First-order | Stochastic gradient descent with momentum=0.9 |
| **Adadelta** | Adaptive | Extension of Adagrad that reduces aggressive LR decay |
| **ASGD** | Averaged | Averaged SGD; averages parameters over iterations |
| **LBFGS** | Quasi-Newton | Limited-memory BFGS; uses closure-based step |

### 7.2 External Optimizers

| Optimizer | Package | Type | Key Characteristics |
|-----------|---------|------|-------------------|
| **Levenberg-Marquardt** | `torch-levenberg-marquardt` | Second-order | Damped Gauss-Newton; computes full Jacobian, solves normal equations via QR |

**Note:** LM is not a `torch.optim.Optimizer`. It wraps the model via `LevenbergMarquardtModule` and manages optimization internally. PyTorch's fused `nn.GRU` kernel is incompatible with functorch's `functional_call` (required for Jacobian computation), so a pure-tensor GRU cell implementation is used. Due to the O(N*P) Jacobian memory cost, LM was run with `frame_length=20` and `batch_size=32` (vs. `frame_length=200` and `batch_size=256` for torch.optim optimizers).

### 7.3 Failed (2/15)

| Optimizer | Reason |
|-----------|--------|
| **SparseAdam** | Requires sparse gradients; GRU parameters are dense tensors |
| **Muon** | Requires all parameters to be 2D; GRU has 1D bias vectors |

## 8. Results

### 8.1 Ranking by Test ACLR_AVG

| Rank | Optimizer | Best Epoch | ACLR_AVG (dB) | EVM (dB) | NMSE (dB) | Time (min) |
|------|-----------|-----------|---------------|----------|-----------|------------|
| 1 | **AdamW** | 99 | **-49.43** | **-44.10** | **-43.73** | 0.65 |
| 2 | Adam | 99 | -49.27 | -43.92 | -43.59 | 0.66 |
| 3 | Adamax | 99 | -49.04 | -42.84 | -42.65 | 0.65 |
| 4 | RAdam | 99 | -48.61 | -43.02 | -42.85 | 0.66 |
| 5 | NAdam | 99 | -48.09 | -42.25 | -41.78 | 0.67 |
| 6 | Adafactor | 99 | -45.40 | -39.67 | -39.33 | 0.81 |
| 7 | Rprop | 99 | -44.05 | -38.88 | -38.27 | 0.74 |
| 8 | RMSprop | 97 | -43.75 | -38.18 | -33.66 | 0.65 |
| 9 | Adagrad | 0 | -29.07 | 0.73 | 0.93 | 0.66 |
| 10 | Adadelta | 0 | -29.92 | 2.03 | 2.32 | 0.65 |
| 11 | ASGD | 0 | -29.85 | 1.80 | 2.09 | 0.72 |
| 12 | SGD | 99 | -27.41 | -16.71 | -17.71 | 0.64 |
| 13 | LBFGS | 0 | -27.32 | -6.03 | -6.13 | 2.71 |

### 8.2 Levenberg-Marquardt (External, Different Training Config)

| Optimizer | Best Epoch | ACLR_AVG (dB) | EVM (dB) | NMSE (dB) | Time (min) | Config |
|-----------|-----------|---------------|----------|-----------|------------|--------|
| Levenberg-Marquardt | 19 | -46.73 | 0.84 | 14.04 | 91.01 | frame=20, batch=32, lr=1.0 |

**Caveat:** LM used `frame_length=20` (vs. 200 for all other optimizers) due to the O(N*P) Jacobian memory constraint. This reduces temporal context per sample and makes direct comparison imprecise. Additionally, LM briefly reached -51.08 dB validation ACLR at epoch 19, but the model destabilized at epoch 20 (training loss jumped 18x) and never recovered, settling at ~-27 dB for the remaining 80 epochs. The positive EVM (+0.84 dB) and NMSE (+14.04 dB) at the best ACLR epoch indicate overfitting to spectral metrics at the expense of signal fidelity.

### 8.3 Convergence: Validation ACLR_AVG (dB) at Key Epochs

| Optimizer | Ep 1 | Ep 10 | Ep 25 | Ep 50 | Ep 75 | Ep 100 |
|-----------|------|-------|-------|-------|-------|--------|
| AdamW | -26.93 | -29.52 | -33.90 | -44.21 | -47.94 | -49.37 |
| Adam | -26.93 | -29.52 | -33.92 | -44.22 | -47.88 | -49.27 |
| Adamax | -26.83 | -29.05 | -35.82 | -45.21 | -47.84 | -48.91 |
| RAdam | -27.13 | -29.06 | -31.92 | -42.79 | -46.89 | -48.53 |
| NAdam | -26.85 | -29.50 | -32.82 | -42.57 | -46.39 | -48.19 |
| Adafactor | -27.40 | -29.43 | -37.98 | -42.78 | -44.03 | -44.94 |
| Rprop | -36.91 | -42.15 | -42.80 | -43.43 | -43.82 | -44.13 |
| RMSprop | -27.06 | -30.30 | -37.67 | -41.94 | -43.61 | -44.22 |
| SGD | -26.64 | -26.88 | -27.10 | -27.37 | -27.45 | -27.48 |
| Adadelta | -29.54 | -28.94 | -26.35 | -25.61 | -26.61 | -26.79 |
| Adagrad | -28.43 | -25.68 | -26.55 | -27.05 | -27.51 | -27.71 |
| ASGD | -29.42 | -26.57 | -25.00 | -26.60 | -26.90 | -26.89 |
| LBFGS | -27.19 | -27.19 | -27.19 | -27.19 | -27.19 | -27.19 |

## 9. Analysis

### Tier 1: Adam Family Dominates (ACLR < -48 dB)

The top 5 optimizers are all Adam variants: **AdamW > Adam > Adamax > RAdam > NAdam**. These share the same core mechanism — adaptive per-parameter learning rates via first and second moment estimates. The differences between them are small (1.3 dB spread) and all are still improving at epoch 100, suggesting further gains with more epochs.

- **AdamW** leads with -49.43 dB ACLR, benefiting from decoupled weight decay that prevents the regularization from interfering with the adaptive learning rate.
- **Adam** is a close second at -49.27 dB, only 0.16 dB behind AdamW.
- **Adamax** uses the infinity norm instead of the L2 norm for second moments, achieving -49.04 dB.
- **RAdam** rectifies the variance of the adaptive learning rate in early training, reaching -48.61 dB.
- **NAdam** incorporates Nesterov momentum into Adam but slightly underperforms vanilla Adam here (-48.09 dB).

### Tier 2: Competitive Alternatives (ACLR -43 to -46 dB)

- **Adafactor** (-45.40 dB) is a memory-efficient alternative to Adam that factorizes the second moment. It is 4 dB behind AdamW but could be useful for models with very large parameter counts.
- **Rprop** (-44.05 dB) shows remarkably fast early convergence (-36.91 dB after just 1 epoch, vs ~-27 dB for Adam variants) by using only gradient signs. However, it plateaus early and the final performance is 5 dB below AdamW.
- **RMSprop** (-43.75 dB) is the precursor to Adam (without momentum estimates). Its 5.7 dB gap to AdamW confirms the value of first-moment estimation.

### Tier 3: Poor Convergence (ACLR > -30 dB)

Five optimizers failed to meaningfully train the GRU DPD with lr=5e-4:

- **SGD with momentum** (-27.41 dB): Converges extremely slowly without adaptive learning rates. The uniform lr=5e-4 is too small for some parameters and too large for others. SGD typically requires careful per-layer learning rate tuning and warmup schedules.
- **Adadelta** (-29.92 dB at epoch 0, degrading to -26.79 dB by epoch 100): This optimizer was designed to eliminate the need for a manual learning rate, but the provided lr=5e-4 likely conflicts with its internal adaptation mechanism. Adadelta actually got worse during training.
- **Adagrad** (-29.07 dB at epoch 0, barely improving): Accumulates squared gradients indefinitely, causing the effective learning rate to decay to near-zero early in training. This is a known limitation for non-convex problems.
- **ASGD** (-29.85 dB at epoch 0, degrading): Parameter averaging over iterations hurts when the optimizer hasn't converged, as early bad iterates corrupt the average.
- **LBFGS** (-27.32 dB, flat across all epochs): Despite being a quasi-Newton method, LBFGS struggles in the mini-batch setting. The curvature estimates from stochastic gradients are noisy and misleading. LBFGS also took 4x longer per epoch due to its closure-based multi-evaluation step.

### Levenberg-Marquardt (Second-Order)

Levenberg-Marquardt achieves competitive ACLR (-46.73 dB test, -51.08 dB val peak) in only 19 epochs, demonstrating the fast convergence potential of second-order methods. However, it has critical drawbacks for RNN-based DPD:

1. **Instability**: The model destabilized after epoch 20, with training loss jumping 18x. LM's Gauss-Newton Hessian approximation assumes a nearly linear residual structure, which breaks down on the non-convex RNN loss surface.
2. **Poor signal fidelity**: At the best ACLR epoch, EVM (+0.84 dB) and NMSE (+14.04 dB) are both positive, meaning the predistorted signal is spectrally clean but the time-domain waveform is severely distorted. This is not useful for practical DPD.
3. **Extreme computational cost**: 91 min for 100 epochs (140x slower than AdamW) due to per-batch Jacobian computation. This was with reduced `frame_length=20` — the standard `frame_length=200` was intractable.
4. **Architecture constraints**: Required a custom pure-tensor GRU implementation because PyTorch's fused GRU kernel is incompatible with functorch's `functional_call`.

### Key Takeaways

1. **Use AdamW (default) or Adam** for GRU DPD training. They consistently achieve the best linearization performance.
2. **The Adam family is robust to default hyperparameters**: All five Adam variants work well with lr=5e-4, no warmup, no scheduling.
3. **Rprop converges fastest in early epochs** but plateaus 5 dB below Adam. It could be useful for rapid prototyping or when training time is severely constrained.
4. **SGD, Adadelta, Adagrad, ASGD, and LBFGS are not recommended** for DPD training at this learning rate. They may perform better with extensive hyperparameter tuning, but the Adam family works well out of the box.
5. **Levenberg-Marquardt is not recommended** for RNN-based DPD despite fast early convergence. The instability, poor EVM/NMSE, and extreme computational cost make it impractical. It may work better for feedforward DPD models where the loss surface is more convex.
6. **SparseAdam and Muon are not applicable** to standard GRU models due to architecture constraints (sparse gradients and 2D-only parameters, respectively).

## 10. Reproducing These Results

### Prerequisites

A trained PA surrogate model is required:
```bash
python main.py --step train_pa --dataset_name APA_200MHz --PA_backbone gru --PA_hidden_size 23 --n_epochs 100
```

### Running Individual Optimizers

```bash
# Tier 1: Adam family
python main.py --step train_dpd --dataset_name APA_200MHz --PA_backbone gru --PA_hidden_size 23 --DPD_backbone gru --DPD_hidden_size 11 --n_epochs 100 --batch_size 256 --opt_type adamw
python main.py --step train_dpd --dataset_name APA_200MHz --PA_backbone gru --PA_hidden_size 23 --DPD_backbone gru --DPD_hidden_size 11 --n_epochs 100 --batch_size 256 --opt_type adam
python main.py --step train_dpd --dataset_name APA_200MHz --PA_backbone gru --PA_hidden_size 23 --DPD_backbone gru --DPD_hidden_size 11 --n_epochs 100 --batch_size 256 --opt_type adamax
python main.py --step train_dpd --dataset_name APA_200MHz --PA_backbone gru --PA_hidden_size 23 --DPD_backbone gru --DPD_hidden_size 11 --n_epochs 100 --batch_size 256 --opt_type radam
python main.py --step train_dpd --dataset_name APA_200MHz --PA_backbone gru --PA_hidden_size 23 --DPD_backbone gru --DPD_hidden_size 11 --n_epochs 100 --batch_size 256 --opt_type nadam

# Tier 2: Competitive alternatives
python main.py --step train_dpd --dataset_name APA_200MHz --PA_backbone gru --PA_hidden_size 23 --DPD_backbone gru --DPD_hidden_size 11 --n_epochs 100 --batch_size 256 --opt_type adafactor
python main.py --step train_dpd --dataset_name APA_200MHz --PA_backbone gru --PA_hidden_size 23 --DPD_backbone gru --DPD_hidden_size 11 --n_epochs 100 --batch_size 256 --opt_type rprop
python main.py --step train_dpd --dataset_name APA_200MHz --PA_backbone gru --PA_hidden_size 23 --DPD_backbone gru --DPD_hidden_size 11 --n_epochs 100 --batch_size 256 --opt_type rmsprop

# Tier 3: Poor convergence at lr=5e-4
python main.py --step train_dpd --dataset_name APA_200MHz --PA_backbone gru --PA_hidden_size 23 --DPD_backbone gru --DPD_hidden_size 11 --n_epochs 100 --batch_size 256 --opt_type sgd
python main.py --step train_dpd --dataset_name APA_200MHz --PA_backbone gru --PA_hidden_size 23 --DPD_backbone gru --DPD_hidden_size 11 --n_epochs 100 --batch_size 256 --opt_type adadelta
python main.py --step train_dpd --dataset_name APA_200MHz --PA_backbone gru --PA_hidden_size 23 --DPD_backbone gru --DPD_hidden_size 11 --n_epochs 100 --batch_size 256 --opt_type adagrad
python main.py --step train_dpd --dataset_name APA_200MHz --PA_backbone gru --PA_hidden_size 23 --DPD_backbone gru --DPD_hidden_size 11 --n_epochs 100 --batch_size 256 --opt_type asgd
python main.py --step train_dpd --dataset_name APA_200MHz --PA_backbone gru --PA_hidden_size 23 --DPD_backbone gru --DPD_hidden_size 11 --n_epochs 100 --batch_size 256 --opt_type lbfgs
```

### Levenberg-Marquardt (requires `pip install torch-levenberg-marquardt`)

```bash
python benchmark/run_lm_benchmark.py --n_epochs 100 --batch_size 32 --frame_length 20
```

### Running Full Benchmark

```bash
python benchmark/run_optimizer_benchmark.py
python benchmark/parse_best_results.py
```
