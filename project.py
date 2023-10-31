__author__ = "Chang Gao"
__copyright__ = "Copyright @ Chang Gao"
__credits__ = ["Chang Gao"]
__license__ = "Private"
__version__ = "0.0.1"
__maintainer__ = "Chang Gao"
__email__ = "gaochangw@outlook.com"
__status__ = "Prototype"

import json
import os
import random as rnd
import numpy as np
import torch
import torch.nn as nn
from typing import Any
from torch import optim
from torch.nn import CTCLoss
from torch.utils.data import DataLoader
from arguments import get_arguments
from modules.paths import gen_model_id, create_folder, gen_paths
from utils import util
from modules.loggers import PandasLogger
from modules.data_collector import IQSegmentDataset, IQFrameDataset, IQFrameDataset_gmp, \
    prepare_segments, load_dataset


class Project:
    def __init__(self):
        ###########################################################################################################
        # Initialization
        ###########################################################################################################
        # Load Hyperparameters
        self.args = get_arguments()
        self.hparams = vars(self.args)
        for k, v in self.hparams.items():
            setattr(self, k, v)

        # Load Specifications
        self.load_spec()

        # Hardware Info
        self.num_cpu_threads = os.cpu_count()

        # Configure Reproducibility
        self.reproducible()

        ###########################################################################################################
        #  Model ID, Paths of folders and log files and Logger
        ###########################################################################################################
        # Model ID
        self.pa_model_id, self.dpdmodel_id = gen_model_id(self.args)

        # Create Folders
        paths_dir, paths_files, _ = gen_paths(self.args, model_id=self.pa_model_id)
        path_save_dir, path_log_dir_hist, path_log_dir_best, _ = paths_dir
        create_folder([path_save_dir, path_log_dir_hist, path_log_dir_best])

        # Get Save and Log Paths
        self.path_save_file_best, self.path_log_file_hist, self.path_log_file_best, _ = paths_files
        print("::: Best Model Save Path: ", self.path_save_file_best)
        print("::: Log-History     Path: ", self.path_log_file_hist)
        print("::: Log-Best        Path: ", self.path_log_file_best)

        # Instantiate Logger for Recording Training Statistics
        self.logger = PandasLogger(path_save_file_best=self.path_save_file_best,
                                   path_log_file_best=self.path_log_file_best,
                                   path_log_file_hist=self.path_log_file_hist,
                                   precision=self.log_precision)

    def reproducible(self):
        rnd.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        if self.re_level == 'soft':
            torch.use_deterministic_algorithms(mode=False)
            torch.backends.cudnn.benchmark = True
        else:  # re_level == 'hard'
            torch.use_deterministic_algorithms(mode=True)
            torch.backends.cudnn.benchmark = False
        torch.cuda.empty_cache()
        print("::: Are Deterministic Algorithms Enabled: ", torch.are_deterministic_algorithms_enabled())
        print("--------------------------------------------------------------------")

    def load_spec(self):
        # Get relative path to the spec file
        path_spec = os.path.join('datasets', self.dataset_name, 'spec.json')

        # Load the spec
        with open(path_spec) as config_file:
            spec = json.load(config_file)
        for k, v in spec.items():
            setattr(self, k, v)
            self.hparams[k] = v

    def add_arg(self, key: str, value: Any):
        setattr(self, key, value)
        setattr(self.args, key, value)
        self.hparams[key] = value

    def set_device(self):
        # Find Available GPUs
        if self.accelerator == 'cuda' and torch.cuda.is_available():
            idx_gpu = self.devices
            name_gpu = torch.cuda.get_device_name(idx_gpu)
            device = torch.device("cuda:" + str(idx_gpu))
            torch.cuda.set_device(device)
            print("::: Available GPUs: %s" % (torch.cuda.device_count()))
            print("::: Using GPU %s:   %s" % (idx_gpu, name_gpu))
            print("--------------------------------------------------------------------")
        elif self.accelerator == 'mps' and torch.backends.mps.is_available():
            device = torch.device("mps")
        elif self.accelerator == 'cpu':
            device = torch.device("cpu")
            print("::: Available GPUs: None")
            print("--------------------------------------------------------------------")
        else:
            raise ValueError(f"The select device {self.accelerator} is not supported.")
        self.add_arg("device", device)
        return device

    def tune_conditional_args(self):
        if self.PA_backbone == 'cnn1d':
            frame_length = self.pa_cnn_memory + self.PA_hidden_size + self.pa_output_len - 2
        elif self.PA_backbone == 'cnn2d':
            frame_length = self.pa_cnn_memory + self.PA_CNN_W + self.pa_output_len - 2
        else:
            frame_length = self.frame_length
        self.add_arg('frame_length', frame_length)

    def build_dataloaders(self):
        from modules.data_collector import IQSegmentDataset, IQFrameDataset, IQFrameDataset_gmp, load_dataset
        from modules.feat_ext import extract_feature

        # Load Dataset
        X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(dataset_name=self.dataset_name)

        # Extract Features
        X_train = extract_feature(X_train, self.PA_backbone)
        X_val = extract_feature(X_val, self.PA_backbone)
        X_test = extract_feature(X_test, self.PA_backbone)
        input_size = X_train.shape[-1]

        # Define PyTorch Datasets
        train_set = IQFrameDataset(X_train, y_train, frame_length=self.frame_length, stride=self.stride)
        val_set = IQSegmentDataset(X_val, y_val)
        test_set = IQSegmentDataset(X_test, y_test)

        # Define PyTorch Dataloaders
        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=self.batch_size_eval, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=self.batch_size_eval, shuffle=False)

        return (train_loader, val_loader, test_loader), input_size

    def build_model(self):
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

        # Cast net to the target device
        net.to(self.device)
        self.add_arg("net", net)

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
            self.add_arg("criterion", criterion)
            return criterion
        except AttributeError:
            raise AttributeError('Please use a valid loss function. See modules/argument.py.')

    # def build_dataloader(self):


    def build_structure(self):
        """
        Build project folder structure
        """
        dir_paths, file_paths, default_path_net_pretrain = self.log.gen_paths(self)
        self.add_arg('default_path_net_pretrain', default_path_net_pretrain)
        save_dir, log_dir_hist, log_dir_best, _ = dir_paths
        self.path_save_file_best, self.path_log_file_hist, self.path_log_file_best, _ = file_paths
        util.create_folder([save_dir, log_dir_hist, log_dir_best])
        print("::: Save Path: ", self.path_save_file_best)
        print("::: Log Path: ", self.path_log_file_hist)
        print("--------------------------------------------------------------------")
        self.add_arg('path_save_file_best', self.path_save_file_best)
        self.add_arg('path_log_file_hist', self.path_log_file_hist)
        self.add_arg('path_log_file_best', self.path_log_file_best)

    def build_optimizer(self, net=None):
        # Optimizer
        net = self.net if net is None else net
        if self.opt == 'ADAM':
            optimizer = optim.Adam(net.parameters(), lr=self.lr, amsgrad=False, weight_decay=self.weight_decay)
        elif self.opt == 'SGD':
            optimizer = optim.SGD(net.parameters(), lr=self.lr, momentum=0.9)
        elif self.opt == 'RMSPROP':
            optimizer = optim.RMSprop(net.parameters(), lr=0.0016, alpha=0.95, eps=1e-08, weight_decay=0, momentum=0,
                                      centered=False)
        elif self.opt == 'ADAMW':
            optimizer = optim.AdamW(net.parameters(), lr=self.lr, amsgrad=False, weight_decay=self.weight_decay)
        elif self.opt == 'AdaBound':
            import adabound  # Run pip install adabound (https://github.com/Luolc/AdaBound)
            optimizer = adabound.AdaBound(net.parameters(), lr=self.lr, final_lr=0.1)
        else:
            raise RuntimeError('Please use a valid optimizer.')
        self.add_arg("optimizer", optimizer)

        # Learning Rate Scheduler
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer,
                                                            mode='min',
                                                            factor=self.decay_factor,
                                                            patience=self.patience,
                                                            verbose=True,
                                                            threshold=1e-4,
                                                            min_lr=self.lr_end)
        self.add_arg("lr_scheduler", lr_scheduler)
        return optimizer, lr_scheduler
