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
    parser.add_argument('--step', default='run_dpd',
                        choices=['train_pa', 'train_dpd', 'run_dpd', 'plot'],
                        help='Step to run.')
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
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size for training.')
    parser.add_argument('--batch_size_eval', default=64, type=int, help='Batch size for evaluation.')
    parser.add_argument('--n_epochs', default=300, type=int, help='Number of epochs to train for.')
    parser.add_argument('--lr_schedule', default=1, type=int,
                        help='Whether to enable ReduceLROnPlateau learning-rate scheduling.')
    parser.add_argument('--lr', default=5e-3, type=float, help='Initial learning rate.')
    parser.add_argument('--lr_end', default=5e-5, type=float, help='Minimum learning rate.')
    parser.add_argument('--decay_factor', default=0.5, type=float,
                        help='Learning-rate reduction factor.')
    parser.add_argument('--patience', default=5, type=int,
                        help='Scheduler patience in epochs.')
    parser.add_argument('--grad_clip_val', default=200, type=float, help='Gradient clipping.')
    parser.add_argument(
        '--cuda_graph_training', action='store_true', default=False,
        help=(
            'Opt in to guarded whole-step CUDA-graph replay for supported '
            'TRes-DeltaGRU DPD training.'
        ),
    )
    # GMP Hyperparameters
    parser.add_argument('--K', default=5, type=int, help='Degree of GMP model')
    parser.add_argument('--gmp_memory_length', default=11, type=int, help='Memory length of GMP model')
    # Power Amplifier Model Settings
    parser.add_argument('--PA_backbone', default='gru',
                        choices=['gmp','deltagru', 'deltajanet', 'janet', 'fcn', 'gru', 'dgru', 'qgru', 'qgru_amp1', 'lstm', 'vdlstm',
                                'rvtdcnn', 'mamba', 'tcn', 'tres_deltagru', 'tres_gru', 'pntdnn', 'pdgru', 'pgjanet', 'dvrjanet', 'bojanet', 'pnjanet', 'apnrnn', 'djanet', 'mcldnn'],
                        help='Modeling PA Recurrent layer type')
    parser.add_argument('--PA_hidden_size', default=23, type=int,
                        help='Hidden size of PA backbone')
    parser.add_argument('--PA_num_layers', default=1, type=int,
                        help="Number of layers of the PA backbone.")
    # Digital Predistortion Model Settings
    parser.add_argument('--DPD_backbone', default='gru',
                        choices=['gmp', 'deltagru', 'deltajanet', 'janet', 'snn', 'fcn', 'gru', 'dgru', 'qgru', 'qgru_amp1', 'lstm', 'vdlstm',
                                'rvtdcnn', 'tres_deltagru', 'tres_gru', 'tcn', 'pntdnn', 'pdgru', 'pgjanet', 'dvrjanet', 'bojanet', 'pnjanet', 'djanet', 'mcldnn'],
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
    parser.add_argument('--collect_delta_stats', action='store_true', default=False,
                        help='Collect temporal delta sparsity diagnostics during training (adds overhead).')

    # Optionally, you might want to add DVR-specific arguments
    parser.add_argument('--num_dvr_units', default=3, type=int,
                        help='Number of DVR units in DVRJANET')


    # argument for PNJANET
    parser.add_argument('--window_size', default=4, type=int,
                        help='Window size for magnitude history in PNJANET')

    # Plotting
    parser.add_argument('--plot', action='store_true', default=False,
                        help='Enable plot generation during training and inference.')
    parser.add_argument('--plot_every', default=10, type=int,
                        help='Generate per-epoch plots every N epochs (default: 1).')
    parser.add_argument('--gif_duration', default=10.0, type=float,
                        help='Duration of GIF animations in seconds (default: 10.0).')

    return parser.parse_args()
