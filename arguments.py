__author__ = "Yizhuo Wu, Chang Gao"
__license__ = "Apache-2.0 License"
__email__ = "yizhuo.wu@tudelft.nl, chang.gao@tudelft.nl"

import argparse


def get_arguments():
    # Process Arguments
    parser = argparse.ArgumentParser(description='Train a GRU network.')
    # Dataset & Log
    parser.add_argument('--dataset_name', default=None, help='Dataset names')
    parser.add_argument('--dataset_path', default=None, help='Path to custom dataset (CSV file or directory)')
    parser.add_argument('--filename', default='', help='Filename to save model and log to.')
    parser.add_argument('--log_precision', default=8, type=int, help='Number of decimals in the log files.')
    # Training Process
    parser.add_argument('--step', default='run_dpd', help='Step to run.')
    parser.add_argument('--eval_val', default=1, type=int, help='Whether evaluate val set during training.')
    parser.add_argument('--eval_test', default=1, type=int, help='Whether evaluate test set during training.')
    parser.add_argument('--accelerator', default='cuda', choices=["cpu", "cuda", "mps"], help='Accelerator types.')
    parser.add_argument('--devices', default=0, type=int, help='Which accelerator to train on.')
    parser.add_argument('--re_level', default='soft', choices=['soft', 'hard'], help='Level of reproducibility.')
    parser.add_argument('--use_segments', action='store_true', default=False,
                        help='Whether partition training sequences into segments of length nperseg before doing the framing.')
    # Feature Extraction
    parser.add_argument('--frame_length', default=200, type=int, help='Frame length of signals')
    parser.add_argument('--frame_stride', default=1, type=int, help='stride_length length of signals')
    # General Hyperparameters
    parser.add_argument('--seed', default=0, type=int, help='Global random number seed.')
    parser.add_argument('--loss_type', default='l2', choices=['l1', 'l2'], help='Type of loss function.')
    parser.add_argument('--opt_type', default='adamw', choices=['sgd', 'adam', 'adamw', 'adabound', 'rmsprop'], help='Type of optimizer.')
    parser.add_argument('--batch_size', default=256, type=int, help='Batch size for training.')
    parser.add_argument('--batch_size_eval', default=256, type=int, help='Batch size for evaluation.')
    parser.add_argument('--n_epochs', default=100, type=int, help='Number of epochs to train for.')
    parser.add_argument('--lr_schedule', default=0, type=int, help='Whether enable learning rate scheduling')
    parser.add_argument('--lr', default=5e-4, type=float, help='Learning rate')
    parser.add_argument('--lr_end', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--decay_factor', default=0.1, type=float, help='Learning rate')
    parser.add_argument('--patience', default=10, type=float, help='Learning rate')
    parser.add_argument('--grad_clip_val', default=200, type=float, help='Gradient clipping.')
    # GMP Hyperparameters
    parser.add_argument('--K', default=4, type=int, help='Degree of GMP model')
    # Power Amplifier Model Settings
    parser.add_argument('--PA_backbone', default='gru',
                        choices=['gmp','deltagru', 'deltajanet', 'janet', 'fcn', 'gru', 'dgru', 'qgru', 'qgru_amp1', 'lstm', 'vdlstm',
                                'rvtdcnn', 'mamba', 'tcn', 'pntdnn', 'pdgru', 'pgjanet', 'dvrjanet', 'bojanet', 'pnjanet', 'apnrnn', 'djanet',
                                'tcnn', 'neuraltx', 'mcldnn'],
                        help='Modeling PA Recurrent layer type')
    parser.add_argument('--PA_hidden_size', default=23, type=int,
                        help='Hidden size of PA backbone')
    parser.add_argument('--PA_num_layers', default=1, type=int,
                        help="Number of layers of the PA backbone.")
    # Digital Predistortion Model Settings
    parser.add_argument('--DPD_backbone', default='gru',
                        choices=['gmp', 'deltagru', 'deltajanet', 'janet', 'snn', 'fcn', 'gru', 'dgru', 'qgru', 'qgru_amp1', 'lstm', 'vdlstm',
                                'rvtdcnn', 'deltagru_tcnskip', 'tcn', 'pntdnn', 'pdgru', 'pgjanet', 'dvrjanet', 'bojanet', 'pnjanet', 'djanet',
                                'tcnn', 'neuraltx', 'mcldnn'],
                        help='DPD model Recurrent layer type')
    parser.add_argument('--DPD_hidden_size', default=15, type=int, help='Hidden size of DPD backbone.')
    parser.add_argument('--DPD_num_layers', default=1, type=int, help='Number of layers of the DPD backbone.')


    # quantization
    parser.add_argument('--quant', action='store_true', default=False, help='Whether to quantize the model')
    parser.add_argument('--n_bits_w', default=8, type=int, help='Number of bits for weights')
    parser.add_argument('--n_bits_a', default=8, type=int, help='Number of bits for activations')
    parser.add_argument('--pretrained_model', default='', help='Path to pretrained model')
    parser.add_argument('--quant_dir_label', default='', help='Directory label for quantization')
    parser.add_argument('--q_pretrain', default=False, type=bool, help='pretrain the model with \
                        self-implementation float models for quantization')


    # Add to model arguments
    parser.add_argument('--thx', type=float, default=0.0,
                        help='Threshold for input deltas')
    parser.add_argument('--thh', type=float, default=0.0,
                        help='Threshold for hidden state deltas')

    # Optionally, you might want to add DVR-specific arguments
    parser.add_argument('--num_dvr_units', default=3, type=int,
                        help='Number of DVR units in DVRJANET')


    # argument for PNJANET
    parser.add_argument('--window_size', default=4, type=int,
                        help='Window size for magnitude history in PNJANET')


    return parser.parse_args()
