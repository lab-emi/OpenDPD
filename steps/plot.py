__author__ = "Chang Gao"
__license__ = "Apache-2.0 License"
__email__ = "chang.gao@tudelft.nl"

import os
import numpy as np
import torch
import models as model
from project import Project
from utils.util import count_net_params
from modules.data_collector import load_dataset
from utils.metrics import NMSE, EVM, ACLR
from utils.plotting import get_plot_dir_compare, generate_plots_compare

import sys
sys.path.append('../..')
from quant import get_quant_model


def main(proj: Project):
    """
    Generate comparison plots: PA output without DPD vs PA output with DPD.

    This step loads the test data and the trained PA + DPD models, runs inference
    for both scenarios, and generates side-by-side comparison plots.
    """
    ###########################################################################################################
    # Initialization
    ###########################################################################################################
    proj.set_device()

    # Load Dataset (original PA input/output); a custom --dataset_path takes precedence
    if getattr(proj, 'dataset_path', None):
        X_train, y_train, _, _, X_test, y_test = load_dataset(dataset_path=proj.dataset_path)
    else:
        X_train, y_train, _, _, X_test, y_test = load_dataset(dataset_name=proj.dataset_name)

    ###########################################################################################################
    # Build PA Model and run inference without DPD
    ###########################################################################################################
    net_pa = model.CoreModel(input_size=2,
                             hidden_size=proj.PA_hidden_size,
                             num_layers=proj.PA_num_layers,
                             backbone_type=proj.PA_backbone,
                             num_dvr_units=proj.num_dvr_units)
    n_net_pa_params = count_net_params(net_pa)
    pa_model_id = proj.gen_pa_model_id(n_net_pa_params)

    # Load pretrained PA model
    path_pa_model = os.path.join('save', proj.dataset_name, 'train_pa', pa_model_id + '.pt')
    net_pa.load_state_dict(torch.load(path_pa_model, weights_only=True))
    net_pa = net_pa.to(proj.device)
    net_pa.eval()

    # PA output without DPD (use actual measured data y_test if available)
    # y_test is the actual PA output from the dataset
    pa_output_wo_dpd = y_test

    ###########################################################################################################
    # Build DPD Model and run inference with DPD
    ###########################################################################################################
    net_dpd = model.CoreModel(input_size=2,
                              hidden_size=proj.DPD_hidden_size,
                              num_layers=proj.DPD_num_layers,
                              backbone_type=proj.DPD_backbone)

    net_dpd = get_quant_model(proj, net_dpd)

    n_net_dpd_params = count_net_params(net_dpd)
    dpd_model_id = proj.gen_dpd_model_id(n_net_dpd_params)

    # Load pretrained DPD model
    path_dpd_model = os.path.join('save', proj.dataset_name, 'train_dpd',
                                  pa_model_id.split('_P_')[0], dpd_model_id + '.pt')
    if proj.args.quant:
        path_dpd_model = os.path.join('save', proj.dataset_name, 'train_dpd',
                                      pa_model_id.split('_P_')[0],
                                      proj.args.quant_dir_label, dpd_model_id + '.pt')
    net_dpd.load_state_dict(torch.load(path_dpd_model, weights_only=True))
    net_dpd = net_dpd.to(proj.device)
    net_dpd.eval()

    # Run DPD + PA cascade on test input
    with torch.no_grad():
        dpd_in = torch.Tensor(X_test).unsqueeze(dim=0).to(proj.device)
        dpd_out = net_dpd(dpd_in)
        pa_out_w_dpd = net_pa(dpd_out)
        pa_out_w_dpd = torch.squeeze(pa_out_w_dpd).cpu().numpy()

    ###########################################################################################################
    # Compute Metrics
    ###########################################################################################################
    fs = getattr(proj.args, 'input_signal_fs', 800e6)
    nperseg = getattr(proj.args, 'nperseg', 2560)
    bw_main_ch = getattr(proj.args, 'bw_main_ch', 200e6)
    n_sub_ch = getattr(proj.args, 'n_sub_ch', 10)

    # Compute target gain for reference
    from utils.util import set_target_gain
    target_gain = set_target_gain(X_train, y_train)
    linear_target = target_gain * X_test

    # Reshape for segment-based metrics: [n_seg, seg_len, 2]
    n_total = X_test.shape[0]
    n_seg = n_total // nperseg
    if n_seg > 0:
        seg_len = n_seg * nperseg
        wo_seg = pa_output_wo_dpd[:seg_len].reshape(n_seg, nperseg, 2)
        w_seg = pa_out_w_dpd[:seg_len].reshape(n_seg, nperseg, 2)
        ref_seg = linear_target[:seg_len].reshape(n_seg, nperseg, 2)
    else:
        wo_seg = pa_output_wo_dpd[np.newaxis, :]
        w_seg = pa_out_w_dpd[np.newaxis, :]
        ref_seg = linear_target[np.newaxis, :]

    metrics_wo = {
        'NMSE': NMSE(wo_seg, ref_seg),
        'EVM': EVM(wo_seg, ref_seg, sample_rate=int(fs), bw_main_ch=bw_main_ch,
                   n_sub_ch=n_sub_ch, nperseg=nperseg),
    }
    aclr_l, aclr_r = ACLR(wo_seg, fs=fs, nperseg=nperseg,
                           bw_main_ch=bw_main_ch, n_sub_ch=n_sub_ch)
    metrics_wo['ACLR_L'] = aclr_l
    metrics_wo['ACLR_R'] = aclr_r
    metrics_wo['ACLR_AVG'] = (aclr_l + aclr_r) / 2

    metrics_w = {
        'NMSE': NMSE(w_seg, ref_seg),
        'EVM': EVM(w_seg, ref_seg, sample_rate=int(fs), bw_main_ch=bw_main_ch,
                   n_sub_ch=n_sub_ch, nperseg=nperseg),
    }
    aclr_l, aclr_r = ACLR(w_seg, fs=fs, nperseg=nperseg,
                           bw_main_ch=bw_main_ch, n_sub_ch=n_sub_ch)
    metrics_w['ACLR_L'] = aclr_l
    metrics_w['ACLR_R'] = aclr_r
    metrics_w['ACLR_AVG'] = (aclr_l + aclr_r) / 2

    ###########################################################################################################
    # Print Metrics
    ###########################################################################################################
    print("\n" + "=" * 60)
    print("Metrics Comparison: Without DPD vs With DPD")
    print("=" * 60)
    print(f"{'Metric':<15} {'Without DPD':>15} {'With DPD':>15} {'Improvement':>15}")
    print("-" * 60)
    for m in ['NMSE', 'EVM', 'ACLR_L', 'ACLR_R', 'ACLR_AVG']:
        wo_val = metrics_wo[m]
        w_val = metrics_w[m]
        imp = wo_val - w_val
        print(f"{m:<15} {wo_val:>15.2f} {w_val:>15.2f} {imp:>15.2f} dB")
    print("=" * 60 + "\n")

    ###########################################################################################################
    # Generate Comparison Plots
    ###########################################################################################################
    spec = {'input_signal_fs': fs, 'nperseg': nperseg,
            'bw_main_ch': bw_main_ch, 'n_sub_ch': n_sub_ch}
    from datasets.demodulator import Demodulator
    from utils.plotting import needs_full_seq_constellation, load_full_dataset_iq
    try:
        demod = Demodulator.from_dataset(proj.dataset_name)
    except Exception:
        demod = None

    # For APA datasets, compute full-sequence constellation data
    full_const_data = None
    if needs_full_seq_constellation(proj.dataset_name):
        from project import Project
        full_input_iq, full_output_iq = load_full_dataset_iq(proj.dataset_name)
        full_input_c = full_input_iq[:, 0] + 1j * full_input_iq[:, 1]
        # PA output w/o DPD: use measured data from CSV
        # DPD+PA cascade: run full input through DPD then PA model (chunked)
        with torch.no_grad():
            full_dpd_out = Project._run_model_chunked(
                net_dpd, full_input_iq, proj.device)
            full_cas_out = Project._run_model_chunked(
                net_pa, full_dpd_out, proj.device)
        full_const_data = {
            'input_c': full_input_c,
            'wo_dpd_c': full_output_iq[:, 0] + 1j * full_output_iq[:, 1],
            'w_dpd_c': full_cas_out[:, 0] + 1j * full_cas_out[:, 1],
        }

    plot_dir = get_plot_dir_compare(proj.dataset_name, dpd_model_id)
    try:
        generate_plots_compare(plot_dir, X_test, pa_output_wo_dpd, pa_out_w_dpd,
                               spec, demod=demod,
                               metrics_wo=metrics_wo, metrics_w=metrics_w,
                               full_const_data=full_const_data)
        print(f"Comparison plots saved to {plot_dir}")
    except Exception as e:
        print(f"Warning: Comparison plot generation failed: {e}")
        import traceback
        traceback.print_exc()
