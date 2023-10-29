import argparse
import os
import pandas as pd
import torch
from steps import ofdm_generator, train_pa, train_dpd, ideal_input_gen
import matplotlib.pyplot as plt
import sys

if __name__ == '__main__':
    # Process Arguments
    parser = argparse.ArgumentParser(description='Train a GRU network.')
    # Initial Arguments
    parser.add_argument('--dataset_name', default=None, help='Dataset names')
    parser.add_argument('--input_signal_fs', default=800e6, help='Sampling Frequency of the input signal')
    parser.add_argument('--input_signal_bw', default=200e6, help='Bandwidth of the input signal')
    parser.add_argument('--input_signal_ch_bw', default=20e6, help='Bandwidth of each channel')
    parser.add_argument('--nperseg', default=2560, help='Bandwidth of the input signal')
    parser.add_argument('--norm', default=False, help='Whether normalize the data')
    parser.add_argument('--filename', default='', help='Filename to save model and log to.')
    parser.add_argument('--phase', default='seg', help='Phase of a step')
    parser.add_argument('--PA_model_phase', default='seg', help='Phase of a step')
    parser.add_argument('--log_precision', default=8, type=int, help='Number of decimals in the log.')
    # Feature Extraction
    parser.add_argument('--frame_length', default=20, type=int, help='Frame length of signals')
    parser.add_argument('--stride_length', default=1, type=int, help='stride length of signals')

    # Hyperparameters
    parser.add_argument('--batch_size', default=16, type=int, help='Batch size for training.')
    parser.add_argument('--batch_size_eval', default=128, type=int, help='Batch size for evaluation.')
    parser.add_argument('--n_epochs', default=350, type=int, help='Number of epochs to train for.')
    parser.add_argument('--lr_schedule', default=1, type=int, help='Whether enable learning rate scheduling')
    parser.add_argument('--lr', default=5e-3, type=float, help='Learning rate')
    parser.add_argument('--lr_end', default=1e-7, type=float, help='Learning rate')
    parser.add_argument('--decay_factor', default=0.5, type=float, help='Learning rate')
    parser.add_argument('--patience', default=10, type=float, help='Learning rate')

    # RNN Settings
    parser.add_argument('--degree', default=4, type=int, help='Degree of GMP model')
    # General Network Settings
    parser.add_argument('--PA_backbone', default='gru',
                             choices=['gmp', 'fc', 'gru', 'dgru', 'lstm', 'vdlstm', 'ligru', 'pgjanet', 'dvrjanet',
                                      'cnn1d', 'cnn2d'], help='Modeling PA Recurrent layer type')
    parser.add_argument('--PA_input_size', default=2, type=int, help='Size of PA model input features')
    parser.add_argument('--PA_hidden_size', default=16, type=int,
                             help='Size of PA model recurrent layers / kernel size in 1dCNN / kernel num in2dCNN')
    parser.add_argument('--DPD_backbone', default='gru',
                             choices=['gmp', 'fc', 'gru', 'dgru', 'lstm', 'vdlstm', 'ligru', 'pgjanet', 'cnn1d',
                                      'dvrjanet', 'cnn2d'], help='DPD model Recurrent layer type')
    parser.add_argument('--DPD_input_size', default=6, type=int, help='Size of DPD model input features')
    parser.add_argument('--DPD_hidden_size', default=40, type=int, help='Size of DPD model recurrent layers')
    # CNN Settings
    parser.add_argument('--pa_cnn_memory', default=5, type=int, help='Frame length of signals')
    parser.add_argument('--dpd_cnn_memory', default=6, type=int, help='Frame length of signals')
    parser.add_argument('--pa_output_len', default=5, type=int, help='Frame length of signals')
    parser.add_argument('--PA_CNN_H', default=5, type=int, help='kernel_height')
    parser.add_argument('--PA_CNN_W', default=5, type=int, help=' kernel_width')
    parser.add_argument('--DPD_CNN_H', default=5, type=int, help='kernel_height')
    parser.add_argument('--DPD_CNN_W', default=5, type=int, help='kernel_height')

    # Training Process
    parser.add_argument('--step', default='train_pa', help='Which step to start from')
    parser.add_argument('--eval_val', default=1, type=int, help='Whether eval val set during training')
    parser.add_argument('--eval_test', default=1, type=int, help='Whether eval test set during training')
    parser.add_argument('--eval_sp', default=1, type=int, help='Whether run through rest steps')
    parser.add_argument('--use_cuda', default=0, type=int, help='Use GPU yes/no')
    parser.add_argument('--gpu_device', default=0, type=int, help='Select GPU')

    # OFDM Generator
    parser.add_argument('--data_size', default=200000, help='The number of data pairs that ofdm generator output')
    parser.add_argument('--fftlength', default=4096, help='The number of subcarriers for each channel')
    parser.add_argument('--guardband', default=6, type=int, help='The length of guard band')
    parser.add_argument('--m_QAM', default=64, type=int, help='m QAM')
    parser.add_argument('--channel_BW', default=20e6, type=int, help='The bandwidth of each channel')
    parser.add_argument('--channel_num', default=1, type=int, help='The number of channel')
    parser.add_argument('--input_rms', default=0.075, type=int, help='PA input range')
    parser.add_argument('--oversampling_rate', default=4, type=int, help='Oversampling rate')
    parser.add_argument('--cylic_prefix', default=False, type=int, help='Whether add cylic prefix')
    parser.add_argument('--cplength', default=None, type=int, help='Cylic prefix length')
    parser.add_argument('--pilotnum', default=0, type=int, help='The number of pilot subcarriers.')
    parser.add_argument('--pilotindex', default=None, type=int, help='Pilot index')
    parser.add_argument('--pilot_value', default=None, type=int, help='Pilot value')
    args = parser.parse_args()

    # Set training precision
    torch.set_default_dtype(torch.float64)

    # Set training device
    if args.use_cuda:
        idx_gpu = torch.cuda.current_device()
        device = torch.device('cuda:' + str(idx_gpu))
    else:
        device = torch.device('cpu')

    # PA Modeling
    if args.step == 'train_pa':
        print("####################################################################################################")
        print("# Step: Train PA                                                                                 #")
        print("####################################################################################################")
        train_pa.main(args, device)
    # DPD Learning
    elif args.step == 'train_dpd':
        print("####################################################################################################")
        print("# Step: Train DPD                                                                                #")
        print("####################################################################################################")
        train_dpd.main(args, device)
    else:
        raise ValueError(f"The step '{args.step}' is not supported.")