"""Benchmark Levenberg-Marquardt optimizer on GRU DPD / APA_200MHz.

Uses torch-levenberg-marquardt (tlm) which wraps the model with its own
training module instead of using torch.optim.Optimizer.
"""
import sys
import os
import time
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch_levenberg_marquardt as tlm
import models as model
from modules.data_collector import load_dataset, IQFrameDataset, IQSegmentDataset
from utils.util import count_net_params
from utils import metrics as metric_funcs


class PureTensorGRUCell(nn.Module):
    """GRU cell using only basic tensor ops (no fused C++ kernels).

    Fully compatible with functorch vmap — uses only matmul, sigmoid,
    tanh, and elementwise ops which all have proper batching rules.
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.randn(3 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.randn(3 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.zeros(3 * hidden_size))
        self.bias_hh = nn.Parameter(torch.zeros(3 * hidden_size))

    def forward(self, x, h):
        # x: (batch, input_size), h: (batch, hidden_size)
        gi = x @ self.weight_ih.t() + self.bias_ih
        gh = h @ self.weight_hh.t() + self.bias_hh

        i_r, i_z, i_n = gi.chunk(3, dim=-1)
        h_r, h_z, h_n = gh.chunk(3, dim=-1)

        r = torch.sigmoid(i_r + h_r)
        z = torch.sigmoid(i_z + h_z)
        n = torch.tanh(i_n + r * h_n)

        return (1 - z) * n + z * h


class FunctorchGRU(nn.Module):
    """GRU backbone using pure-tensor GRU cells (functorch-compatible).

    nn.GRU and nn.GRUCell both use fused C++ kernels that lack vmap
    batching rules. This version uses only basic tensor ops (matmul,
    sigmoid, tanh) which vmap can properly batch.
    """

    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cells = nn.ModuleList([
            PureTensorGRUCell(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, x, h_0):
        # x: (batch, seq_len, input_size), h_0: (num_layers, batch, hidden_size)
        batch, seq_len, _ = x.shape
        h = [h_0[i] for i in range(self.num_layers)]

        outputs = []
        for t in range(seq_len):
            inp = x[:, t, :]
            for i, cell in enumerate(self.cells):
                h[i] = cell(inp, h[i])
                inp = h[i]
            outputs.append(inp)

        out = torch.stack(outputs, dim=1)  # (batch, seq_len, hidden_size)
        return self.fc_out(out)


def copy_gru_weights(src_gru_backbone, dst_functorch_gru):
    """Copy weights from nn.GRU-based backbone to FunctorchGRU."""
    src_rnn = src_gru_backbone.rnn
    for i, cell in enumerate(dst_functorch_gru.cells):
        cell.weight_ih.data.copy_(getattr(src_rnn, f'weight_ih_l{i}').data)
        cell.weight_hh.data.copy_(getattr(src_rnn, f'weight_hh_l{i}').data)
        cell.bias_ih.data.copy_(getattr(src_rnn, f'bias_ih_l{i}').data)
        cell.bias_hh.data.copy_(getattr(src_rnn, f'bias_hh_l{i}').data)
    dst_functorch_gru.fc_out.weight.data.copy_(src_gru_backbone.fc_out.weight.data)
    dst_functorch_gru.fc_out.bias.data.copy_(src_gru_backbone.fc_out.bias.data)


class LMCascadedModel(nn.Module):
    """Cascaded DPD+PA model using FunctorchGRU for LM compatibility."""

    def __init__(self, dpd_backbone, pa_backbone, dpd_hidden_size, pa_hidden_size,
                 dpd_num_layers=1, pa_num_layers=1, input_size=2, output_size=2):
        super().__init__()
        self.dpd = FunctorchGRU(input_size, dpd_hidden_size, output_size, dpd_num_layers)
        self.pa = FunctorchGRU(input_size, pa_hidden_size, output_size, pa_num_layers)
        self.dpd_hidden_size = dpd_hidden_size
        self.pa_hidden_size = pa_hidden_size
        self.dpd_num_layers = dpd_num_layers
        self.pa_num_layers = pa_num_layers

    def freeze_pa(self):
        for p in self.pa.parameters():
            p.requires_grad = False

    def forward(self, x):
        batch = x.size(0)
        device = x.device
        h_dpd = torch.zeros(self.dpd_num_layers, batch, self.dpd_hidden_size, device=device)
        h_pa = torch.zeros(self.pa_num_layers, batch, self.pa_hidden_size, device=device)
        x = self.dpd(x, h_dpd)
        x = self.pa(x, h_pa)
        return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='APA_200MHz')
    parser.add_argument('--PA_hidden_size', type=int, default=23)
    parser.add_argument('--DPD_hidden_size', type=int, default=11)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--frame_length', type=int, default=20)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--learning_rate', type=float, default=1.0)
    parser.add_argument('--attempts_per_step', type=int, default=10)
    parser.add_argument('--solve_method', default='qr', choices=['qr', 'cholesky', 'solve'])
    args = parser.parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ---------------------------------------------------------------
    # Load dataset
    # ---------------------------------------------------------------
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(dataset_name=args.dataset_name)

    # Load spec for metrics
    spec_path = os.path.join('datasets', args.dataset_name, 'spec.json')
    with open(spec_path) as f:
        spec = json.load(f)
    nperseg = spec['nperseg']
    input_signal_fs = spec['input_signal_fs']
    bw_main_ch = spec['bw_main_ch']
    n_sub_ch = spec['n_sub_ch']

    # Build datasets
    train_set = IQFrameDataset(X_train, y_train, args.frame_length, stride=1)
    val_set = IQSegmentDataset(X_val, y_val, nperseg)
    test_set = IQSegmentDataset(X_test, y_test, nperseg)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    # ---------------------------------------------------------------
    # Build models
    # ---------------------------------------------------------------
    input_size = 2

    # Load original models to copy weights from
    net_pa_orig = model.CoreModel(input_size=input_size, hidden_size=args.PA_hidden_size,
                                  num_layers=1, backbone_type='gru')
    # PA model was trained with frame_length=200 regardless of LM frame_length
    pa_frame_length = 200
    pa_model_path = os.path.join('save', args.dataset_name, 'train_pa',
                                 f'PA_S_{args.seed}_M_GRU_H_{args.PA_hidden_size}_F_{pa_frame_length}_P_{count_net_params(net_pa_orig)}.pt')
    net_pa_orig.load_state_dict(torch.load(pa_model_path, map_location='cpu'))
    print(f"Loaded PA model from {pa_model_path}")

    net_dpd_orig = model.CoreModel(input_size=input_size, hidden_size=args.DPD_hidden_size,
                                   num_layers=1, backbone_type='gru')
    n_dpd_params = count_net_params(net_dpd_orig)
    print(f"DPD parameters: {n_dpd_params}")

    # Build functorch-compatible cascaded model and copy PA weights
    net_cas = LMCascadedModel(
        dpd_backbone=None, pa_backbone=None,
        dpd_hidden_size=args.DPD_hidden_size, pa_hidden_size=args.PA_hidden_size,
    )
    copy_gru_weights(net_pa_orig.backbone, net_cas.pa)
    # DPD weights are freshly initialized (same as standard training)
    net_cas.freeze_pa()
    net_cas = net_cas.to(device)

    lm_dpd_params = sum(p.numel() for p in net_cas.dpd.parameters())
    print(f"LM model DPD parameters: {lm_dpd_params}")

    # ---------------------------------------------------------------
    # LM training module
    # ---------------------------------------------------------------
    lm_module = tlm.training.LevenbergMarquardtModule(
        model=net_cas,
        loss_fn=tlm.loss.MSELoss(),
        learning_rate=args.learning_rate,
        attempts_per_step=args.attempts_per_step,
        solve_method=args.solve_method,
    )

    # ---------------------------------------------------------------
    # Training loop with per-epoch metrics
    # ---------------------------------------------------------------
    criterion = nn.MSELoss()
    best_val_aclr = 0.0
    best_epoch = -1
    best_test_metrics = {}

    results_dir = os.path.join('benchmark', 'optimizer_results')
    os.makedirs(results_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Levenberg-Marquardt Benchmark: GRU DPD on {args.dataset_name}")
    print(f"  Epochs: {args.n_epochs} | LR: {args.learning_rate} | Batch: {args.batch_size}")
    print(f"  Solve: {args.solve_method} | Attempts/step: {args.attempts_per_step}")
    print(f"{'='*60}\n")

    start_time = time.time()
    early_stopped = False

    for epoch in range(args.n_epochs):
        epoch_start = time.time()

        # Train one epoch using LM
        net_cas.train()
        epoch_losses = []
        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device)
            outputs, loss, stop, logs = lm_module.training_step(features, targets)
            epoch_losses.append(loss.item())
            if stop:
                print(f"  LM requested early stop at epoch {epoch}")
                early_stopped = True
                break

        train_loss = np.mean(epoch_losses)

        # Validation
        net_cas.eval()
        with torch.no_grad():
            val_preds, val_gts, val_losses = [], [], []
            for features, targets in val_loader:
                features = features.to(device)
                targets = targets.to(device)
                outputs = net_cas(features)
                loss = criterion(outputs, targets)
                val_preds.append(outputs.cpu())
                val_gts.append(targets.cpu())
                val_losses.append(loss.item())

        val_pred = torch.cat(val_preds, dim=0).numpy()
        val_gt = torch.cat(val_gts, dim=0).numpy()

        val_nmse = metric_funcs.NMSE(val_pred, val_gt)
        val_evm = metric_funcs.EVM(val_pred, val_gt, bw_main_ch=bw_main_ch,
                                   n_sub_ch=n_sub_ch, nperseg=nperseg)
        val_aclr_l, val_aclr_r = metric_funcs.ACLR(val_pred, fs=input_signal_fs,
                                                    nperseg=nperseg, bw_main_ch=bw_main_ch,
                                                    n_sub_ch=n_sub_ch)
        val_aclr_avg = (val_aclr_l + val_aclr_r) / 2

        # Test
        with torch.no_grad():
            test_preds, test_gts, test_losses = [], [], []
            for features, targets in test_loader:
                features = features.to(device)
                targets = targets.to(device)
                outputs = net_cas(features)
                loss = criterion(outputs, targets)
                test_preds.append(outputs.cpu())
                test_gts.append(targets.cpu())
                test_losses.append(loss.item())

        test_pred = torch.cat(test_preds, dim=0).numpy()
        test_gt = torch.cat(test_gts, dim=0).numpy()
        test_loss = np.mean(test_losses)

        test_nmse = metric_funcs.NMSE(test_pred, test_gt)
        test_evm = metric_funcs.EVM(test_pred, test_gt, bw_main_ch=bw_main_ch,
                                    n_sub_ch=n_sub_ch, nperseg=nperseg)
        test_aclr_l, test_aclr_r = metric_funcs.ACLR(test_pred, fs=input_signal_fs,
                                                      nperseg=nperseg, bw_main_ch=bw_main_ch,
                                                      n_sub_ch=n_sub_ch)
        test_aclr_avg = (test_aclr_l + test_aclr_r) / 2

        epoch_time = time.time() - epoch_start

        print(f"Epoch {epoch:3d}/{args.n_epochs} | "
              f"train_loss={train_loss:.6f} | "
              f"val_aclr={val_aclr_avg:.2f} dB | "
              f"test_aclr={test_aclr_avg:.2f} dB | "
              f"test_evm={test_evm:.2f} dB | "
              f"{epoch_time:.1f}s")

        # Track best by validation ACLR_AVG (most negative = best)
        if val_aclr_avg < best_val_aclr:
            best_val_aclr = val_aclr_avg
            best_epoch = epoch
            best_test_metrics = {
                'TEST_ACLR_AVG': test_aclr_avg,
                'TEST_ACLR_L': test_aclr_l,
                'TEST_ACLR_R': test_aclr_r,
                'TEST_EVM': test_evm,
                'TEST_NMSE': test_nmse,
                'TEST_LOSS': test_loss,
                'VAL_ACLR_AVG': val_aclr_avg,
                'VAL_EVM': val_evm,
                'TRAIN_LOSS': train_loss,
                'EPOCH': epoch,
            }

        if early_stopped:
            break

    total_time = time.time() - start_time

    # ---------------------------------------------------------------
    # Results
    # ---------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  RESULTS (Levenberg-Marquardt)")
    print(f"{'='*60}")
    print(f"  Best Epoch:     {best_epoch}")
    print(f"  Test ACLR_AVG:  {best_test_metrics.get('TEST_ACLR_AVG', 'N/A'):.4f} dB")
    print(f"  Test EVM:       {best_test_metrics.get('TEST_EVM', 'N/A'):.4f} dB")
    print(f"  Test NMSE:      {best_test_metrics.get('TEST_NMSE', 'N/A'):.4f} dB")
    print(f"  Val ACLR_AVG:   {best_test_metrics.get('VAL_ACLR_AVG', 'N/A'):.4f} dB")
    print(f"  Total Time:     {total_time:.1f}s ({total_time/60:.2f} min)")

    # Save results
    result = {
        'optimizer': 'levenberg_marquardt',
        'status': 'OK',
        'time_s': float(total_time),
        'best_epoch': int(best_epoch),
        'learning_rate': float(args.learning_rate),
        'solve_method': args.solve_method,
        'attempts_per_step': args.attempts_per_step,
        **{k: float(v) for k, v in best_test_metrics.items()},
    }
    with open(os.path.join(results_dir, 'levenberg_marquardt.json'), 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {results_dir}/levenberg_marquardt.json")


if __name__ == '__main__':
    main()
