"""Benchmark AdamW vs Levenberg-Marquardt on FCN (MLP) DPD / APA_200MHz.

Uses a small feedforward DPD model (42 params) to give LM the best
conditions: no RNN incompatibility, cheap Jacobian, large batch sizes.
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

RESULTS_DIR = 'benchmark/mlp_optimizer_results'

# Import FunctorchGRU and helpers from the GRU LM benchmark
from run_lm_benchmark import FunctorchGRU, copy_gru_weights


class LMCascadedFCN(nn.Module):
    """Cascaded FCN_DPD + FunctorchGRU_PA for LM compatibility.

    The FCN DPD is natively functorch-compatible.
    The PA uses PureTensorGRUCell to avoid nn.GRU's fused C++ kernel.
    """

    def __init__(self, dpd_model, pa_hidden_size, pa_num_layers=1, input_size=2):
        super().__init__()
        self.dpd_model = dpd_model  # CoreModel with FCN backbone
        self.pa = FunctorchGRU(input_size, pa_hidden_size, input_size, pa_num_layers)
        self.pa_hidden_size = pa_hidden_size
        self.pa_num_layers = pa_num_layers

    def freeze_pa(self):
        for p in self.pa.parameters():
            p.requires_grad = False

    def forward(self, x):
        x = self.dpd_model(x)
        batch = x.size(0)
        device = x.device
        h_pa = torch.zeros(self.pa_num_layers, batch, self.pa_hidden_size, device=device)
        x = self.pa(x, h_pa)
        return x


def load_data_and_spec(dataset_name, frame_length, batch_size_train, batch_size_eval):
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(dataset_name=dataset_name)
    spec_path = os.path.join('datasets', dataset_name, 'spec.json')
    with open(spec_path) as f:
        spec = json.load(f)

    train_set = IQFrameDataset(X_train, y_train, frame_length, stride=1)
    val_set = IQSegmentDataset(X_val, y_val, spec['nperseg'])
    test_set = IQSegmentDataset(X_test, y_test, spec['nperseg'])

    train_loader = DataLoader(train_set, batch_size=batch_size_train, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size_eval, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size_eval, shuffle=False)

    return train_loader, val_loader, test_loader, spec


def run_metrics(net, loader, criterion, spec, device):
    net.eval()
    with torch.no_grad():
        preds, gts, losses = [], [], []
        for features, targets in loader:
            features, targets = features.to(device), targets.to(device)
            outputs = net(features)
            loss = criterion(outputs, targets)
            preds.append(outputs.cpu())
            gts.append(targets.cpu())
            losses.append(loss.item())

    pred = torch.cat(preds, dim=0).numpy()
    gt = torch.cat(gts, dim=0).numpy()

    nmse = metric_funcs.NMSE(pred, gt)
    evm = metric_funcs.EVM(pred, gt, bw_main_ch=spec['bw_main_ch'],
                           n_sub_ch=spec['n_sub_ch'], nperseg=spec['nperseg'])
    aclr_l, aclr_r = metric_funcs.ACLR(pred, fs=spec['input_signal_fs'],
                                        nperseg=spec['nperseg'],
                                        bw_main_ch=spec['bw_main_ch'],
                                        n_sub_ch=spec['n_sub_ch'])
    return {
        'loss': float(np.mean(losses)),
        'NMSE': float(nmse),
        'EVM': float(evm),
        'ACLR_L': float(aclr_l),
        'ACLR_R': float(aclr_r),
        'ACLR_AVG': float((aclr_l + aclr_r) / 2),
    }


def build_cascaded_model(dataset_name, pa_hidden_size, dpd_hidden_size, seed, device,
                         pa_backbone='fcn'):
    input_size = 2

    net_pa = model.CoreModel(input_size=input_size, hidden_size=pa_hidden_size,
                             num_layers=1, backbone_type=pa_backbone)
    pa_params = count_net_params(net_pa)
    pa_tag = pa_backbone.upper()
    pa_path = os.path.join('save', dataset_name, 'train_pa',
                           f'PA_S_{seed}_M_{pa_tag}_H_{pa_hidden_size}_F_200_P_{pa_params}.pt')
    net_pa.load_state_dict(torch.load(pa_path, map_location='cpu'))
    print(f"  Loaded PA model: {pa_path}")

    net_dpd = model.CoreModel(input_size=input_size, hidden_size=dpd_hidden_size,
                              num_layers=1, backbone_type='fcn')
    dpd_params = count_net_params(net_dpd)
    print(f"  FCN DPD parameters: {dpd_params}")

    net_cas = model.CascadedModel(dpd_model=net_dpd, pa_model=net_pa)
    net_cas.freeze_pa_model()
    net_cas = net_cas.to(device)
    return net_cas, net_dpd, dpd_params


def run_adamw(args, train_loader, val_loader, test_loader, spec, device):
    print(f"\n{'='*60}")
    print(f"  AdamW: FCN DPD (h={args.dpd_hidden_size}) on {args.dataset_name}")
    print(f"  Epochs: {args.n_epochs} | LR: {args.lr} | Batch: {args.batch_size}")
    print(f"{'='*60}")

    net_cas, net_dpd, dpd_params = build_cascaded_model(
        args.dataset_name, args.pa_hidden_size, args.dpd_hidden_size, args.seed, device,
        pa_backbone=args.pa_backbone)

    optimizer = torch.optim.AdamW(net_cas.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    best_val_aclr = 0.0
    best_epoch = -1
    best_test = {}
    history = []

    start_time = time.time()
    for epoch in range(args.n_epochs):
        net_cas.train()
        epoch_losses = []
        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad()
            out = net_cas(features)
            loss = criterion(out, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(net_cas.parameters(), 200)
            optimizer.step()
            epoch_losses.append(loss.item())
        train_loss = float(np.mean(epoch_losses))

        val_m = run_metrics(net_cas, val_loader, criterion, spec, device)
        test_m = run_metrics(net_cas, test_loader, criterion, spec, device)
        elapsed = time.time() - start_time

        history.append({
            'epoch': epoch, 'train_loss': train_loss,
            'val_aclr': val_m['ACLR_AVG'], 'test_aclr': test_m['ACLR_AVG'],
            'test_evm': test_m['EVM'], 'test_nmse': test_m['NMSE'],
        })

        if epoch % 10 == 0 or epoch == args.n_epochs - 1:
            print(f"  Ep {epoch:3d} | loss={train_loss:.6f} | val_aclr={val_m['ACLR_AVG']:.2f} | "
                  f"test_aclr={test_m['ACLR_AVG']:.2f} | test_evm={test_m['EVM']:.2f} | {elapsed:.1f}s")

        if val_m['ACLR_AVG'] < best_val_aclr:
            best_val_aclr = val_m['ACLR_AVG']
            best_epoch = epoch
            best_test = {**test_m, 'val_ACLR_AVG': val_m['ACLR_AVG'], 'train_loss': train_loss}

    total_time = time.time() - start_time
    return {
        'optimizer': 'AdamW',
        'best_epoch': best_epoch,
        'time_s': total_time,
        'dpd_params': dpd_params,
        'batch_size': args.batch_size,
        'lr': args.lr,
        **{f'test_{k}': v for k, v in best_test.items() if k != 'loss'},
        'history': history,
    }


def run_lm(args, val_loader, test_loader, spec, device):
    print(f"\n{'='*60}")
    print(f"  LM: FCN DPD (h={args.dpd_hidden_size}) on {args.dataset_name}")
    print(f"  Epochs: {args.n_epochs} | LR: {args.lm_lr} | Batch: {args.lm_batch_size}")
    print(f"  Solve: {args.lm_solve} | Attempts: {args.lm_attempts}")
    print(f"{'='*60}")

    # Build LM-compatible cascaded model: FCN DPD + FunctorchGRU PA
    # (nn.GRU is incompatible with functorch's functional_call)
    input_size = 2
    net_pa_orig = model.CoreModel(input_size=input_size, hidden_size=args.pa_hidden_size,
                                  num_layers=1, backbone_type='gru')
    pa_params = count_net_params(net_pa_orig)
    pa_path = os.path.join('save', args.dataset_name, 'train_pa',
                           f'PA_S_{args.seed}_M_GRU_H_{args.pa_hidden_size}_F_200_P_{pa_params}.pt')
    net_pa_orig.load_state_dict(torch.load(pa_path, map_location='cpu'))
    print(f"  Loaded PA model: {pa_path}")

    net_dpd = model.CoreModel(input_size=input_size, hidden_size=args.dpd_hidden_size,
                              num_layers=1, backbone_type='fcn')
    dpd_params = count_net_params(net_dpd)
    print(f"  FCN DPD parameters: {dpd_params}")

    net_cas = LMCascadedFCN(dpd_model=net_dpd, pa_hidden_size=args.pa_hidden_size)
    copy_gru_weights(net_pa_orig.backbone, net_cas.pa)
    net_cas.freeze_pa()
    net_cas = net_cas.to(device)

    # Rebuild train loader with LM batch size
    X_train, y_train, _, _, _, _ = load_dataset(dataset_name=args.dataset_name)
    lm_train_set = IQFrameDataset(X_train, y_train, args.frame_length, stride=1)
    lm_train_loader = DataLoader(lm_train_set, batch_size=args.lm_batch_size,
                                 shuffle=True, drop_last=True)

    lm_module = tlm.training.LevenbergMarquardtModule(
        model=net_cas,
        loss_fn=tlm.loss.MSELoss(),
        learning_rate=args.lm_lr,
        attempts_per_step=args.lm_attempts,
        solve_method=args.lm_solve,
    )

    # Report damping parameters (answers Maarten's mu question)
    damping = lm_module.damping_strategy
    mu_init = float(damping.starting_value)
    mu_dec = float(damping.dec_factor)
    mu_inc = float(damping.inc_factor)
    print(f"  Damping: mu_init={mu_init}, dec_factor={mu_dec}, inc_factor={mu_inc}")

    criterion = nn.MSELoss()
    best_val_aclr = 0.0
    best_epoch = -1
    best_test = {}
    history = []
    early_stopped = False

    start_time = time.time()
    for epoch in range(args.n_epochs):
        net_cas.train()
        epoch_losses = []
        for features, targets in lm_train_loader:
            features, targets = features.to(device), targets.to(device)
            outputs, loss, stop, logs = lm_module.training_step(features, targets)
            epoch_losses.append(loss.item())
            if stop:
                print(f"  LM early stop at epoch {epoch}")
                early_stopped = True
                break
        train_loss = float(np.mean(epoch_losses))

        val_m = run_metrics(net_cas, val_loader, criterion, spec, device)
        test_m = run_metrics(net_cas, test_loader, criterion, spec, device)
        elapsed = time.time() - start_time

        history.append({
            'epoch': epoch, 'train_loss': train_loss,
            'val_aclr': val_m['ACLR_AVG'], 'test_aclr': test_m['ACLR_AVG'],
            'test_evm': test_m['EVM'], 'test_nmse': test_m['NMSE'],
        })

        if epoch % 10 == 0 or epoch == args.n_epochs - 1 or early_stopped:
            print(f"  Ep {epoch:3d} | loss={train_loss:.6f} | val_aclr={val_m['ACLR_AVG']:.2f} | "
                  f"test_aclr={test_m['ACLR_AVG']:.2f} | test_evm={test_m['EVM']:.2f} | {elapsed:.1f}s")

        if val_m['ACLR_AVG'] < best_val_aclr:
            best_val_aclr = val_m['ACLR_AVG']
            best_epoch = epoch
            best_test = {**test_m, 'val_ACLR_AVG': val_m['ACLR_AVG'], 'train_loss': train_loss}

        if early_stopped:
            break

    total_time = time.time() - start_time
    return {
        'optimizer': 'Levenberg-Marquardt',
        'best_epoch': best_epoch,
        'time_s': total_time,
        'dpd_params': dpd_params,
        'batch_size': args.lm_batch_size,
        'lr': args.lm_lr,
        'solve_method': args.lm_solve,
        'attempts_per_step': args.lm_attempts,
        'mu_init': mu_init,
        'mu_dec': mu_dec,
        'mu_inc': mu_inc,
        'early_stopped': early_stopped,
        **{f'test_{k}': v for k, v in best_test.items() if k != 'loss'},
        'history': history,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='APA_200MHz')
    parser.add_argument('--pa_hidden_size', type=int, default=23)
    parser.add_argument('--pa_backbone', default='gru', choices=['gru', 'fcn'])
    parser.add_argument('--dpd_hidden_size', type=int, default=8)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--frame_length', type=int, default=200)
    parser.add_argument('--seed', type=int, default=0)
    # AdamW params
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=5e-4)
    # LM params
    parser.add_argument('--lm_batch_size', type=int, default=256)
    parser.add_argument('--lm_lr', type=float, default=1.0)
    parser.add_argument('--lm_attempts', type=int, default=10)
    parser.add_argument('--lm_solve', default='qr', choices=['qr', 'cholesky', 'solve'])
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load data (shared eval loaders)
    train_loader, val_loader, test_loader, spec = load_data_and_spec(
        args.dataset_name, args.frame_length, args.batch_size, batch_size_eval=256)

    # Run AdamW
    torch.manual_seed(args.seed)
    adamw_result = run_adamw(args, train_loader, val_loader, test_loader, spec, device)

    # Run LM
    torch.manual_seed(args.seed)
    lm_result = run_lm(args, val_loader, test_loader, spec, device)

    # Summary
    print(f"\n{'='*70}")
    print(f"  SUMMARY: FCN DPD (h={args.dpd_hidden_size}) on {args.dataset_name}")
    print(f"{'='*70}")
    print(f"{'Optimizer':<25} {'Best Ep':<10} {'ACLR (dB)':<12} {'EVM (dB)':<12} {'NMSE (dB)':<12} {'Time':<10}")
    print("-" * 70)
    for r in [adamw_result, lm_result]:
        print(f"{r['optimizer']:<25} {r['best_epoch']:<10} "
              f"{r.get('test_ACLR_AVG', 'N/A'):<12.2f} "
              f"{r.get('test_EVM', 'N/A'):<12.2f} "
              f"{r.get('test_NMSE', 'N/A'):<12.2f} "
              f"{r['time_s']/60:<10.2f}min")

    # Save
    all_results = {'adamw': adamw_result, 'lm': lm_result, 'args': vars(args)}
    with open(os.path.join(RESULTS_DIR, 'results.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\nResults saved to {RESULTS_DIR}/results.json")


if __name__ == '__main__':
    main()
