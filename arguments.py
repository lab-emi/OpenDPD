__author__ = "Chang Gao"
__copyright__ = "Copyright @ Chang Gao"
__credits__ = ["Chang Gao"]
__license__ = "Private"
__version__ = "0.0.1"
__maintainer__ = "Chang Gao"
__email__ = "gaochangw@outlook.com"
__status__ = "Prototype"

import importlib
import json
import os
import time
import typing
import argparse
import random as rnd
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.nn import CTCLoss
from tqdm import tqdm
from utils import pandaslogger, util
from thop import profile


class Project:
    def __init__(self):
        self.testfile = None
        self.devfile = None
        self.trainfile = None
        self.dataloader = None
        self.config = None
        self.hparams = None
        self.args = None
        self.num_cpu_threads = os.cpu_count()  # Hardware Info
        self.load_arguments()  # Load Arguments
        self.load_modules()  # Load Modules of a specific dataset
        self.load_config()  # Load Configurations
        self.update_args()  # Update arguments according to the current step

        # Define abbreviations of hparams
        self.args_to_abb = {
            'seed': 'S',
            'input_size': 'I',
            'rnn_size': 'H',
            'backbone_type': 'T',
            'rnn_layers': 'L',
            'num_classes': 'C',
            'ctxt_size': 'CT',
            'pred_size': 'PD',
            'qa': 'QA',
            'aqi': 'AQI',
            'aqf': 'AQF',
            'qw': 'QW',
            'wqi': 'WQI',
            'wqf': 'WQF',
        }
        self.abb_to_args = dict((v, k) for k, v in self.args_to_abb.items())
        self.experiment_key = None

        # Manage Steps
        self.list_steps = ['prepare', 'feature', 'pretrain', 'retrain']
        self.additem('step_idx', self.list_steps.index(self.step))

    def additem(self, key, value):
        setattr(self, key, value)
        setattr(self.args, key, value)
        self.hparams[key] = value

    def load_config(self):
        config_path = os.path.join('./config', self.dataset, self.cfg_feat + '.json')
        with open(config_path) as config_file:
            self.config = json.load(config_file)
        for k, v in self.config.items():
            setattr(self, k, v)
            self.hparams[k] = v

    def step_in(self):
        if self.run_through:
            self.additem('step_idx', self.step_idx + 1)
            self.additem('step', self.list_steps[self.step_idx])

    def gen_experiment_key(self, **kwargs) -> str:
        from operator import itemgetter

        # Add extra arguments if needed
        args_to_abb = {**self.args_to_abb, **kwargs}

        # Model ID
        list_args = list(args_to_abb.keys())
        list_abbs = list(itemgetter(*list_args)(args_to_abb))
        list_vals = list(itemgetter(*list_args)(self.hparams))
        list_vals_str = list(map(str, list_vals))
        experiment_key = list_abbs + list_vals_str
        experiment_key[::2] = list_abbs
        experiment_key[1::2] = list_vals_str
        experiment_key = '_'.join(experiment_key)
        self.experiment_key = experiment_key
        return experiment_key

    def decode_exp_id(self, exp_id: str):
        args = exp_id.split('_')
        vals = args[1::2]
        args = args[0::2]
        args = [self.abb_to_args[x] for x in args]
        dict_arg = dict(zip(args, vals))
        return dict_arg

    def reproducible(self, level='soft'):
        rnd.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        if level == 'soft':
            torch.use_deterministic_algorithms(mode=False)
            torch.backends.cudnn.benchmark = True
        else:  # level == 'hard'
            torch.use_deterministic_algorithms(mode=True)
            torch.backends.cudnn.benchmark = False
        torch.cuda.empty_cache()
        print("::: Are Deterministic Algorithms Enabled: ", torch.are_deterministic_algorithms_enabled())
        print("--------------------------------------------------------------------")

    def load_modules(self):
        # Load Modules
        print("### Loading modules for dataset: ", self.dataset)
        try:
            self.data_prep = importlib.import_module('modules.' + self.dataset + '.data_prep')
        except:
            raise RuntimeError('Please select a supported dataset.')
        try:
            self.dataloader = importlib.import_module('modules.' + self.dataset + '.dataloader')
        except:
            raise RuntimeError('Please select a supported dataset.')
        try:
            self.log = importlib.import_module('modules.' + self.dataset + '.log')
        except:
            raise RuntimeError('Please select a supported dataset.')
        try:
            self.train_func = importlib.import_module('modules.' + self.dataset + '.train_func')
        except:
            raise RuntimeError('Please select a supported dataset.')
        try:
            self.metric = importlib.import_module('modules.' + self.dataset + '.metric')
        except:
            raise RuntimeError('Please select a supported dataset.')
        try:
            self.net_pretrain = importlib.import_module('networks.models.' + self.model_pretrain)
        except:
            raise RuntimeError('Please select a supported dataset.')
        try:
            self.net_retrain = importlib.import_module('networks.models.' + self.model_retrain)
        except:
            raise RuntimeError('Please select a supported dataset.')

    def load_arguments(self):
        parser = argparse.ArgumentParser(description='Train a GRU network.')
        # Basic Setup
        args_basic = parser.add_argument_group("Basic Setup")
        args_basic.add_argument('--project_name', default='AMPRO', help='Useful for loggers like Comet')
        args_basic.add_argument('--data_dir', default='/data', help='Useful for loggers like Comet')
        args_basic.add_argument('--dataset', default='gscdv2', help='Useful for loggers like Comet')
        args_basic.add_argument('--cfg_feat', default='feat_fft', help='Useful for loggers like Comet')
        args_basic.add_argument('--path_net_pretrain', default=None, help='Useful for loggers like Comet')
        args_basic.add_argument('--trainfile', default=None, help='Training set feature path')
        args_basic.add_argument('--devfile', default=None, help='Development set feature path')
        args_basic.add_argument('--testfile', default=None, help='Test set feature path')
        args_basic.add_argument('--step', default='pretrain', help='A specific step to run.')
        args_basic.add_argument('--run_through', default=0, type=int, help='If true, run all following steps.')
        args_basic.add_argument('--eval_val', default=1, type=int, help='Whether eval val set during training')
        args_basic.add_argument('--score_val', default=1, type=int, help='Whether score val set during training')
        args_basic.add_argument('--eval_test', default=1, type=int, help='Whether eval test set during training')
        args_basic.add_argument('--score_test', default=1, type=int, help='Whether score test set during training')
        args_basic.add_argument('--debug', default=0, type=int, help='Log intermediate results for hardware debugging')
        args_basic.add_argument('--model_path', default='',
                                help='Model path to load. If empty, the experiment key will be used.')
        args_basic.add_argument('--use_cuda', default=1, type=int, help='Use GPU')
        args_basic.add_argument('--gpu_device', default=0, type=int, help='Select GPU')
        args_basic.add_argument('--save_every_epoch', default=0, type=int, help='Save model for every epoch.')
        # Dataset Processing/Feature Extraction
        args_feat = parser.add_argument_group("Dataset Processing/Feature Extraction")
        args_feat.add_argument('--augment_noise', default=0, type=int, help='Augment data with various SNRs')
        args_feat.add_argument('--target_snr', default=5, type=int, help='Signal-to-Noise ratio for test')
        args_feat.add_argument('--zero_padding', default='head',
                               help='Method of padding zeros to samples in a batch')
        args_feat.add_argument('--qf', default=0, type=int, help='Quantize features')
        args_feat.add_argument('--logf', default=None, help='Apply a log function on the feature; log - '
                                                            'floating-point log function; 2 - Look-Up '
                                                            'Table-based log function')
        # Training Hyperparameters
        args_hparam_t = parser.add_argument_group("Training Hyperparameters")
        args_hparam_t.add_argument('--seed', default=0, type=int, help='Random seed.')
        args_hparam_t.add_argument('--epochs_pretrain', default=50, type=int, help='Number of epochs to train for.')
        args_hparam_t.add_argument('--epochs_retrain', default=50, type=int, help='Number of epochs to train for.')
        args_hparam_t.add_argument('--batch_size', default=64, type=int, help='Batch size.')
        args_hparam_t.add_argument('--batch_size_eval', default=256, type=int,
                                   help='Batch size for test. Use larger values for faster test.')
        args_hparam_t.add_argument('--opt', default='ADAMW', help='Which optimizer to use (ADAM or SGD)')
        args_hparam_t.add_argument('--lr_schedule', default=1, type=int, help='Whether enable learning rate scheduling')
        args_hparam_t.add_argument('--lr', default=1e-3, type=float, help='Learning rate')  # 5e-4
        args_hparam_t.add_argument('--lr_end', default=3e-4, type=float, help='Learning rate')
        args_hparam_t.add_argument('--decay_factor', default=0.8, type=float, help='Learning rate')
        args_hparam_t.add_argument('--patience', default=4, type=float, help='Learning rate')
        args_hparam_t.add_argument('--beta', default=0, type=float,
                                   help='Best model used in testing, either "per", or "vloss"')
        args_hparam_t.add_argument('--loss', default='crossentropy', help='Loss function.')
        args_hparam_t.add_argument('--weight_decay', default=0.01, type=float, help='Weight decay')
        args_hparam_t.add_argument('--grad_clip_val', default=200, type=float, help='Gradient clipping')
        args_hparam_t.add_argument('--ctxt_size', default=100, type=int,
                                   help='The number of timesteps for RNN to look at')
        args_hparam_t.add_argument('--pred_size', default=1, type=int,
                                   help='The number of timesteps to predict in the future')
        # RNN Model Hyperparameters
        args_hparam_rnn = parser.add_argument_group("Model Hyperparameters")
        args_hparam_rnn.add_argument('--model_pretrain', default='cla-rnn', help='Network model for pretrain')
        args_hparam_rnn.add_argument('--model_retrain', default='cla-deltarnn', help='Network model for retrain')
        args_hparam_rnn.add_argument('--backbone_type', default='GRU', help='RNN layer type')
        args_hparam_rnn.add_argument('--rnn_layers', default=1, type=int, help='Number of RNN nnlayers')
        args_hparam_rnn.add_argument('--rnn_size', default=64, type=int,
                                   help='RNN Hidden layer size (must be a multiple of num_pe, see modules/edgedrnn.py)')
        args_hparam_rnn.add_argument('--rnn_dropout', default=0, type=float, help='RNN Hidden layer size')
        args_hparam_rnn.add_argument('--fc_extra_size', default=0, type=float, help='RNN Hidden layer size')
        args_hparam_rnn.add_argument('--fc_dropout', default=0, type=float, help='RNN Hidden layer size')
        args_hparam_rnn.add_argument('--use_hardsigmoid', default=0, type=int, help='Use hardsigmoid')
        args_hparam_rnn.add_argument('--use_hardtanh', default=0, type=int, help='Use hardtanh')
        # Quantization
        args_hparam_q = parser.add_argument_group("Quantization Hyperparameters")
        args_hparam_q.add_argument('--qa', default=0, type=int, help='Quantize the activations')
        args_hparam_q.add_argument('--qw', default=0, type=int, help='Quantize the weights')
        args_hparam_q.add_argument('--qc', default=0, type=int, help='Quantize the classification layer (CL)')
        args_hparam_q.add_argument('--qcw', default=0, type=int, help='Quantize the classification layer (CL) weights')
        args_hparam_q.add_argument('--aqi', default=3, type=int,
                                   help='Number of integer bits before decimal point for activation')
        args_hparam_q.add_argument('--aqf', default=5, type=int,
                                   help='Number of integer bits after decimal point for activation')
        args_hparam_q.add_argument('--wqi', default=1, type=int,
                                   help='Number of integer bits before decimal point for weight')
        args_hparam_q.add_argument('--wqf', default=7, type=int,
                                   help='Number of integer bits after decimal point for weight')
        args_hparam_q.add_argument('--bw_acc', default=32, type=int,
                                   help='Bit width of the MAC accumulator')
        args_hparam_q.add_argument('--nqi', default=2, type=int,
                                   help='Number of integer bits before decimal point for AF')
        args_hparam_q.add_argument('--nqf', default=6, type=int,
                                   help='Number of integer bits after decimal point for AF')
        args_hparam_q.add_argument('--cqi', default=3, type=int,
                                   help='Number of integer bits before decimal point for CL')
        args_hparam_q.add_argument('--cqf', default=5, type=int,
                                   help='Number of integer bits after decimal point for CL')
        args_hparam_q.add_argument('--cwqi', default=1, type=int,
                                   help='Number of integer bits before decimal point for CL')
        args_hparam_q.add_argument('--cwqf', default=7, type=int,
                                   help='Number of integer bits after decimal point for CL')
        # Delta Networks
        args_hparam_d = parser.add_argument_group("Delta Network Hyperparameters")
        args_hparam_d.add_argument('--thx', default=0, type=float, help='Delta threshold for inputs')
        args_hparam_d.add_argument('--thh', default=0, type=float, help='Delta threshold for hidden states')
        # Scoring Settings
        args_score = parser.add_argument_group("Scoring Hyperparameters")
        args_score.add_argument('--smooth', default=1, type=int, help='Whether smooth the posterior over time')
        args_score.add_argument('--smooth_window_size', default=60, type=int, help='Posterior smooth window size')
        args_score.add_argument('--confidence_window_size', default=80, type=int,
                                help='Confidence score window size')
        args_score.add_argument('--fire_threshold', default=0, type=float,
                                help='Threshold for train (1) firing a decision')
        # Get EdgeDRNN-Specific Arguments
        args_edgedrnn = parser.add_argument_group("EdgeDRNN Arguments")
        args_edgedrnn.add_argument('--stim_head', default=1000, type=int, help='Starting index of the HDL test stimuli')
        args_edgedrnn.add_argument('--stim_len', default=1000, type=int, help='#Timesteps of the HDL test stimuli')
        # CBTD
        args_cbtd = parser.add_argument_group("Column-Balanced Targeted Dropout Arguments")
        args_cbtd.add_argument('--cbtd', default=0, type=int,
                               help='Whether use Column-Balanced Weight Dropout')
        args_cbtd.add_argument('--gamma_rnn', default=0.7, type=float, help='Target sparsity of cbtd')
        args_cbtd.add_argument('--gamma_fc', default=0.75, type=float, help='Target sparsity of cbtd')
        args_cbtd.add_argument('--alpha_anneal_epoch', default=100, type=int, help='Target sparsity of cbtd')
        # Get Spartus-Specific Arguments
        args_spartus = parser.add_argument_group("Spartus Arguments")
        args_spartus.add_argument('--num_array', default=1, type=int, help='Number of MAC Arrays')
        args_spartus.add_argument('--num_array_pe', default=16, type=int, help='Number of PEs per MAC Array')
        args_spartus.add_argument('--num_array_pe_ext', default=8, type=int,
                                  help='Number of PEs per MAC Array for export')
        args_spartus.add_argument('--act_latency', default=8, type=int,
                                  help='Pipeline latency for calculating activations')
        args_spartus.add_argument('--act_interval', default=4, type=int,
                                  help='Pipeline latency for calculating activations')
        args_spartus.add_argument('--op_freq', default=2e8, type=int, help='Operation frequency of DeltaLSTM')
        args_spartus.add_argument('--w_sp_ext', default=0.9375, type=float, help='Weight sparsity for export')

        self.args = parser.parse_args()

        # Get Hyperparameter Dictionary
        self.hparams = vars(self.args)
        for k, v in self.hparams.items():
            setattr(self, k, v)

    def update_args(self):
        # Determine Arguments According to Steps
        if self.step == 'pretrain':
            self.additem('n_epochs', self.epochs_pretrain)
            self.additem('model_name', self.model_pretrain)
            self.additem('retrain', 0)
        elif self.step == 'retrain':
            self.additem('n_epochs', self.epochs_retrain)
            self.additem('model_name', self.model_retrain)
            self.additem('retrain', 1)
        elif self.step == 'test':
            self.additem('batch_size', self.batch_size_eval)

    def select_device(self):
        # Find Available GPUs
        if torch.cuda.is_available():
            torch.cuda.set_device(self.gpu_device)
            idx_gpu = torch.cuda.current_device()
            name_gpu = torch.cuda.get_device_name(idx_gpu)
            device = "cuda:" + str(idx_gpu)
            print("::: Available GPUs: %s" % (torch.cuda.device_count()))
            print("::: Using GPU %s:   %s" % (idx_gpu, name_gpu))
            print("--------------------------------------------------------------------")
        else:
            device = "cpu"
            print("::: Available GPUs: None")
            print("--------------------------------------------------------------------")
        self.additem("device", device)
        return device

    def build_model(self):
        from utils.util import count_net_params
        # Load Pretrained Model if Running Retrain
        if self.step == 'retrain':
            net = self.net_retrain.Model(self)  # Instantiate Retrain Model
            if self.path_net_pretrain is None:
                print('::: Loading pretrained model: ', self.default_path_net_pretrain)
                # net = util.load_model(self, net, self.default_path_net_pretrain)
                net.load_pretrain_model(self.default_path_net_pretrain)
            else:
                print('::: Loading pretrained model: ', self.path_net_pretrain)
                net = util.load_model(self, net, self.path_net_pretrain)
        else:
            net = self.net_pretrain.Model(self)  # Instantiate Pretrain Model

        # Get parameter count
        # n_param = count_net_params(net)
        # self.additem("n_param", n_param)
        # num_macs, num_params = net.get_model_size()
        num_params = net.get_model_size()
        print("::: Number of Parameters: ", num_params)
        # print("::: Number of MACs: ", num_macs)
        # self.additem("num_macs", num_macs)
        self.additem("num_params", num_params)

        # Cast net to the target device
        net.to(self.device)
        self.additem("net", net)


        return net

    def build_criterion(self):
        dict_loss = {'crossentropy': nn.CrossEntropyLoss(reduction='mean'),
                     'ctc': CTCLoss(blank=0, reduction='sum', zero_infinity=True),
                     'mse': nn.MSELoss(),
                     'l1': nn.L1Loss()
                     }
        loss_func_name = self.loss
        try:
            criterion = dict_loss[loss_func_name]
            self.additem("criterion", criterion)
            return criterion
        except AttributeError:
            raise AttributeError('Please use a valid loss function. See modules/argument.py.')


