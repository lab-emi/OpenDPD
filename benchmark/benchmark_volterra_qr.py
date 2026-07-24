"""
Benchmark: Volterra-series-based DPD identified via QR decomposition.

Supports two model structures:
  - MP  (Memory Polynomial): diagonal Volterra terms only
  - GMP (Generalized Memory Polynomial): aligned + lagging + leading cross-terms

Both identified in closed form via least squares (QR/SVD).

Indirect Learning Architecture (ILA):
  1. Build basis Phi from normalized PA output  z/G
  2. Solve  x = Phi * w  via QR  (postdistorter)
  3. Copy w to predistorter:  x_dpd = Phi(x) * w
  4. Pass through PA model:  PA(x_dpd) vs G*x
"""

import argparse
import json
import os
import numpy as np
import torch

# Project imports
from modules.data_collector import load_dataset
from utils.util import set_target_gain
from utils import metrics
import models as model


# ---------------------------------------------------------------------------
# Memory Polynomial basis
# ---------------------------------------------------------------------------

def build_mp_basis(x_complex, K, Q):
    """
    Build Memory Polynomial basis matrix.

    Basis functions:  phi_{k,q}(n) = x(n-q) * |x(n-q)|^k
    for k = 0..K-1,  q = 0..Q-1

    Parameters
    ----------
    x_complex : ndarray, shape (N,), complex
    K : int - number of nonlinearity orders
    Q : int - memory depth

    Returns
    -------
    Phi : ndarray, shape (N, K*Q), complex
    """
    N = len(x_complex)
    Phi = np.zeros((N, K * Q), dtype=np.complex128)
    col = 0
    for k in range(K):
        for q in range(Q):
            if q == 0:
                xd = x_complex
            else:
                xd = np.zeros(N, dtype=np.complex128)
                xd[q:] = x_complex[:N - q]
            Phi[:, col] = xd * np.abs(xd) ** k
            col += 1
    return Phi


# ---------------------------------------------------------------------------
# GMP basis (Morgan et al., IEEE TSP 2006)
# ---------------------------------------------------------------------------

def _delay(x, d):
    """Delay complex signal x by d samples (zero-pad)."""
    N = len(x)
    out = np.zeros(N, dtype=np.complex128)
    if d >= 0 and d < N:
        out[d:] = x[:N - d]
    elif d < 0 and -d < N:
        out[:N + d] = x[-d:]
    return out


def build_gmp_basis(x_complex, Ka, La, Kb, Lb, Mb, Kc, Lc, Mc):
    """
    Build Generalized Memory Polynomial basis matrix.

    Three components:
      Aligned:  x(n-q) * |x(n-q)|^k     k=0..Ka-1, q=0..La-1
      Lagging:  x(n-q) * |x(n-q-l)|^k   k=1..Kb,   q=0..Lb-1, l=1..Mb
      Leading:  x(n-q) * |x(n-q+l)|^k   k=1..Kc,   q=0..Lc-1, l=1..Mc

    Returns Phi of shape (N, Ka*La + Kb*Lb*Mb + Kc*Lc*Mc), complex.
    """
    N = len(x_complex)
    n_cols = Ka * La + Kb * Lb * Mb + Kc * Lc * Mc
    Phi = np.zeros((N, n_cols), dtype=np.complex128)
    col = 0

    # Aligned terms
    for k in range(Ka):
        for q in range(La):
            xq = _delay(x_complex, q)
            Phi[:, col] = xq * np.abs(xq) ** k
            col += 1

    # Lagging cross-terms
    for k in range(1, Kb + 1):
        for q in range(Lb):
            for l in range(1, Mb + 1):
                xq = _delay(x_complex, q)
                xql = _delay(x_complex, q + l)
                Phi[:, col] = xq * np.abs(xql) ** k
                col += 1

    # Leading cross-terms
    for k in range(1, Kc + 1):
        for q in range(Lc):
            for l in range(1, Mc + 1):
                xq = _delay(x_complex, q)
                xql = _delay(x_complex, q - l)
                Phi[:, col] = xq * np.abs(xql) ** k
                col += 1

    return Phi


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def iq_to_complex(iq):
    return iq[:, 0] + 1j * iq[:, 1]


def complex_to_iq(c):
    return np.stack([c.real, c.imag], axis=-1).astype(np.float32)


def run_pa_model(iq_signal, pa_net, device, nperseg):
    """Pass an IQ signal through the trained PA model segment-by-segment."""
    pa_net.eval()
    N = len(iq_signal)
    segments = []
    for i in range(0, N, nperseg):
        seg = iq_signal[i:i + nperseg]
        if len(seg) < nperseg:
            pad = np.zeros((nperseg - len(seg), 2), dtype=np.float32)
            seg = np.vstack([seg, pad])
        segments.append(seg)
    segments = np.array(segments)  # (n_seg, nperseg, 2)

    with torch.no_grad():
        out = pa_net(torch.tensor(segments, dtype=torch.float32).to(device))
        out = out.cpu().numpy()

    return out.reshape(-1, 2)[:N]


def compute_metrics(prediction_iq, ground_truth_iq, spec):
    """Compute NMSE, EVM, ACLR on segmented data (same as OpenDPD training)."""
    nperseg = int(spec['nperseg'])
    fs = spec['input_signal_fs']
    bw = spec['bw_main_ch']
    nsc = spec['n_sub_ch']

    # Segment prediction and ground truth
    N = len(prediction_iq)
    n_seg = N // nperseg
    pred_seg = prediction_iq[:n_seg * nperseg].reshape(n_seg, nperseg, 2)
    gt_seg = ground_truth_iq[:n_seg * nperseg].reshape(n_seg, nperseg, 2)

    nmse = metrics.NMSE(pred_seg, gt_seg)
    evm = metrics.EVM(pred_seg, gt_seg, sample_rate=int(fs),
                      bw_main_ch=bw, n_sub_ch=nsc, nperseg=nperseg)
    aclr_l, aclr_r = metrics.ACLR(pred_seg, fs=fs, nperseg=nperseg,
                                   bw_main_ch=bw, n_sub_ch=nsc)
    return nmse, evm, aclr_l, aclr_r


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--pa_hidden_size', type=int, default=24)
    parser.add_argument('--model', type=str, default='mp', choices=['mp', 'gmp'],
                        help='Model type: mp (Memory Polynomial) or gmp (Generalized MP)')
    # MP parameters
    parser.add_argument('--K', type=int, default=5,
                        help='Nonlinearity order (number of |x|^k terms)')
    parser.add_argument('--Q', type=int, default=50,
                        help='Memory depth (number of delay taps) for MP')
    # GMP parameters
    parser.add_argument('--Ka', type=int, default=5, help='GMP aligned nonlinearity orders')
    parser.add_argument('--La', type=int, default=15, help='GMP aligned memory depth')
    parser.add_argument('--Kb', type=int, default=4, help='GMP lagging nonlinearity orders')
    parser.add_argument('--Lb', type=int, default=15, help='GMP lagging memory depth')
    parser.add_argument('--Mb', type=int, default=2, help='GMP lagging cross-term depth')
    parser.add_argument('--Kc', type=int, default=4, help='GMP leading nonlinearity orders')
    parser.add_argument('--Lc', type=int, default=15, help='GMP leading memory depth')
    parser.add_argument('--Mc', type=int, default=1, help='GMP leading cross-term depth')
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()

    # Load spec
    spec_path = os.path.join('datasets', args.dataset_name, 'spec.json')
    with open(spec_path) as f:
        spec = json.load(f)
    nperseg = int(spec['nperseg'])

    # Load dataset
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(
        dataset_name=args.dataset_name)

    # Target gain (same as OpenDPD)
    target_gain = set_target_gain(X_train, y_train)

    # Load PA model
    device = torch.device(f'cuda:{args.device}')
    pa_net = model.CoreModel(input_size=2, hidden_size=args.pa_hidden_size,
                             num_layers=1, backbone_type='gru')
    from utils.util import count_net_params
    n_pa = count_net_params(pa_net)
    pa_path = (f'save/{args.dataset_name}/train_pa/'
               f'PA_S_0_M_GRU_H_{args.pa_hidden_size}_F_200_P_{n_pa}.pt')
    pa_net.load_state_dict(torch.load(pa_path, map_location='cpu'))
    pa_net = pa_net.to(device)

    # -----------------------------------------------------------------------
    # Step 1:  Identify DPD via ILA + QR
    # -----------------------------------------------------------------------
    K, Q = args.K, args.Q
    if args.model == 'gmp':
        def build_basis(x):
            return build_gmp_basis(x, args.Ka, args.La, args.Kb, args.Lb,
                                   args.Mb, args.Kc, args.Lc, args.Mc)
        n_complex_coeffs = (args.Ka * args.La + args.Kb * args.Lb * args.Mb
                            + args.Kc * args.Lc * args.Mc)
        print(f"\n=== Volterra GMP DPD via QR (Ka={args.Ka}, La={args.La}, "
              f"Kb={args.Kb}, Lb={args.Lb}, Mb={args.Mb}, "
              f"Kc={args.Kc}, Lc={args.Lc}, Mc={args.Mc}) ===")
    else:
        def build_basis(x):
            return build_mp_basis(x, K, Q)
        n_complex_coeffs = K * Q
        print(f"\n=== Volterra MP DPD via QR (K={K}, Q={Q}) ===")
    n_real_params = 2 * n_complex_coeffs
    print(f"Complex coefficients : {n_complex_coeffs}")
    print(f"Real parameters      : {n_real_params}")

    # Convert to complex
    x_train_c = iq_to_complex(X_train)
    z_train_c = iq_to_complex(y_train)  # measured PA output

    # Normalise PA output by target gain
    z_norm = z_train_c / target_gain

    # Build basis from normalised PA output (postdistorter input)
    Phi_train = build_basis(z_norm)

    # Solve  x_train = Phi_train * w  via least squares (uses QR/SVD internally)
    w, residuals, rank, sv = np.linalg.lstsq(Phi_train, x_train_c, rcond=None)
    print(f"Least-squares rank   : {rank} / {n_complex_coeffs}")

    # -----------------------------------------------------------------------
    # Step 2:  Apply DPD to test data
    # -----------------------------------------------------------------------
    x_test_c = iq_to_complex(X_test)

    # Build basis from test input (predistorter)
    Phi_test = build_basis(x_test_c)
    x_dpd_c = Phi_test @ w                     # predistorted signal
    x_dpd_iq = complex_to_iq(x_dpd_c)

    # -----------------------------------------------------------------------
    # Step 3:  Pass through PA model
    # -----------------------------------------------------------------------
    pa_output_iq = run_pa_model(x_dpd_iq, pa_net, device, nperseg)

    # Ground truth for DPD: target_gain * x_test
    gt_iq = (target_gain * X_test).astype(np.float32)

    # -----------------------------------------------------------------------
    # Step 4:  Metrics
    # -----------------------------------------------------------------------
    nmse, evm, aclr_l, aclr_r = compute_metrics(pa_output_iq, gt_iq, spec)
    aclr_avg = (aclr_l + aclr_r) / 2

    print(f"\nTest Results:")
    print(f"{'Metric':<12} {'Value':>12}")
    print("-" * 26)
    print(f"{'ACLR_L':<12} {aclr_l:>12.2f} dB")
    print(f"{'ACLR_R':<12} {aclr_r:>12.2f} dB")
    print(f"{'ACLR_AVG':<12} {aclr_avg:>12.2f} dB")
    print(f"{'EVM':<12} {evm:>12.2f} dB")
    print(f"{'NMSE':<12} {nmse:>12.2f} dB")

    # Also run on val set
    x_val_c = iq_to_complex(X_val)
    Phi_val = build_basis(x_val_c)
    x_dpd_val_c = Phi_val @ w
    x_dpd_val_iq = complex_to_iq(x_dpd_val_c)
    pa_val_iq = run_pa_model(x_dpd_val_iq, pa_net, device, nperseg)
    gt_val_iq = (target_gain * X_val).astype(np.float32)
    v_nmse, v_evm, v_aclr_l, v_aclr_r = compute_metrics(pa_val_iq, gt_val_iq, spec)
    v_aclr_avg = (v_aclr_l + v_aclr_r) / 2
    print(f"\nVal Results:")
    print(f"{'ACLR_AVG':<12} {v_aclr_avg:>12.2f} dB")
    print(f"{'EVM':<12} {v_evm:>12.2f} dB")
    print(f"{'NMSE':<12} {v_nmse:>12.2f} dB")


if __name__ == '__main__':
    main()
