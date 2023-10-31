__author__ = "Yizhuo Wu, Chang Gao"
__license__ = "MIT License"
__version__ = "1.0"
__email__ = "yizhuo.wu@tudelft.nl, chang.gao@tudelft.nl"

import argparse


def get_arguments():
    # Process Arguments
    parser = argparse.ArgumentParser(description='Train a GRU network.')
    # Dataset & Log
    parser.add_argument('--dataset_name', default=None, help='Dataset names')
    parser.add_argument('--filename', default='', help='Filename to save model and log to.')
    parser.add_argument('--log_precision', default=8, type=int, help='Number of decimals in the log files.')
    # Training Process
    parser.add_argument('--step', default='train_pa', help='Step to run.')
    parser.add_argument('--eval_val', default=1, type=int, help='Whether evaluate val set during training.')
    parser.add_argument('--eval_test', default=1, type=int, help='Whether evaluate test set during training.')
    parser.add_argument('--accelerator', default='cuda', choices=["cpu", "cuda", "mps"], help='Accelerator types.')
    parser.add_argument('--devices', default=0, type=int, help='Which accelerator to train on.')
    parser.add_argument('--re_level', default='soft', choices=['soft', 'hard'], help='Level of reproducibility.')
    parser.add_argument('--use_segments', action='store_true', default=False,
                        help='Whether partition training sequences into segments of length nperseg before doing the framing.')
    # Feature Extraction
    parser.add_argument('--frame_length', default=20, type=int, help='Frame length of signals')
    parser.add_argument('--stride_length', default=1, type=int, help='stride_length length of signals')
    # General Hyperparameters
    parser.add_argument('--seed', default=0, type=int, help='Global random number seed.')
    parser.add_argument('--batch_size', default=256, type=int, help='Batch size for training.')
    parser.add_argument('--batch_size_eval', default=128, type=int, help='Batch size for evaluation.')
    parser.add_argument('--n_epochs', default=100, type=int, help='Number of epochs to train for.')
    parser.add_argument('--lr_schedule', default=1, type=int, help='Whether enable learning rate scheduling')
    parser.add_argument('--lr', default=1e-2, type=float, help='Learning rate')
    parser.add_argument('--lr_end', default=1e-7, type=float, help='Learning rate')
    parser.add_argument('--decay_factor', default=0.5, type=float, help='Learning rate')
    parser.add_argument('--patience', default=10, type=float, help='Learning rate')
    # GMP Hyperparameters
    parser.add_argument('--degree', default=4, type=int, help='Degree of GMP model')
    # Power Amplifier Model Settings
    parser.add_argument('--PA_backbone', default='gru',
                        choices=['gmp', 'fc', 'gru', 'dgru', 'lstm', 'vdlstm', 'ligru', 'pgjanet', 'dvrjanet',
                                 'cnn1d', 'cnn2d'], help='Modeling PA Recurrent layer type')
    parser.add_argument('--PA_input_size', default=2, type=int, help='Size of PA model input features')
    parser.add_argument('--PA_hidden_size', default=16, type=int,
                        help='Size of PA model recurrent layers / kernel size in 1dCNN / kernel num in2dCNN')
    parser.add_argument('--PA_CNN_H', default=5, type=int, help='kernel_height')
    parser.add_argument('--PA_CNN_W', default=5, type=int, help=' kernel_width')
    parser.add_argument('--pa_output_len', default=5, type=int, help='Frame length of signals')
    parser.add_argument('--pa_cnn_memory', default=5, type=int, help='Frame length of signals')
    # Digital Predistortion Model Settings
    parser.add_argument('--DPD_backbone', default='gru',
                        choices=['gmp', 'fc', 'gru', 'dgru', 'lstm', 'vdlstm', 'ligru', 'pgjanet', 'cnn1d',
                                 'dvrjanet', 'cnn2d'], help='DPD model Recurrent layer type')
    parser.add_argument('--DPD_input_size', default=6, type=int, help='Size of DPD model input features')
    parser.add_argument('--DPD_hidden_size', default=10, type=int, help='Size of DPD model recurrent layers')
    parser.add_argument('--dpd_cnn_memory', default=6, type=int, help='Frame length of signals')
    # General Network Settings
    parser.add_argument('--DPD_CNN_H', default=5, type=int, help='kernel_height')
    parser.add_argument('--DPD_CNN_W', default=5, type=int, help='kernel_height')

    return parser.parse_args()
