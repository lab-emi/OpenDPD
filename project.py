__author__ = "Yizhuo Wu, Chang Gao"
__license__ = "Apache-2.0 License"
__email__ = "yizhuo.wu@tudelft.nl, chang.gao@tudelft.nl"

import json
import os
import random as rnd
import time
import numpy as np
import torch
import torch.nn as nn
from typing import Any, Callable
from torch import optim
from torch.utils.data import DataLoader
from arguments import get_arguments
from modules.paths import (
    create_folder,
    gen_log_stat,
    gen_dir_paths,
    gen_file_paths,
    warn_if_model_artifacts_exist,
)
from modules.train_funcs import net_train, net_eval, calculate_metrics
from utils import util
from modules.loggers import PandasLogger
from utils.util import set_target_gain


class Project:
    def __init__(self):
        ###########################################################################################################
        # Initialization
        ###########################################################################################################
        # Dictionary for Statistics Log
        self.log_all = {}
        self.log_train = {}
        self.log_val = {}
        self.log_test = {}

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
        # Create Folders
        dir_paths = gen_dir_paths(self.args)
        self.path_dir_save, self.path_dir_log_hist, self.path_dir_log_best = dir_paths
        create_folder([self.path_dir_save, self.path_dir_log_hist, self.path_dir_log_best])

    def gen_pa_model_id(self, n_net_params):
        dict_pa = {'S': f"{self.seed}",
                   'M': self.PA_backbone.upper(),
                   'H': f"{self.PA_hidden_size:d}",
                   'F': f"{self.frame_length:d}",
                   'P': f"{n_net_params:d}"
                   }
        dict_pamodel_id = dict(list(dict_pa.items()))

        # PA Model ID
        list_pamodel_id = []
        for item in list(dict_pamodel_id.items()):
            list_pamodel_id += list(item)
        pa_model_id = '_'.join(list_pamodel_id)
        pa_model_id = 'PA_' + pa_model_id
        return pa_model_id

    def gen_dpd_model_id(self, n_net_params):
        dict_dpd = {'S': f"{self.seed}",
                    'M': self.DPD_backbone.upper(),
                    'H': f"{self.DPD_hidden_size:d}",
                    'F': f"{self.frame_length:d}",
                    'P': f"{n_net_params:d}"
                    }
        if 'delta' in self.DPD_backbone:
            dict_dpd['THX'] = f"{self.thx:.3f}"
            dict_dpd['THH'] = f"{self.thh:.3f}"
        dict_dpdmodel_id = dict(list(dict_dpd.items()))

        # DPD Model ID
        list_dpdmodel_id = []
        for item in list(dict_dpdmodel_id.items()):
            list_dpdmodel_id += list(item)
        dpd_model_id = '_'.join(list_dpdmodel_id)
        dpd_model_id = 'DPD_' + dpd_model_id
        return dpd_model_id

    def build_logger(self, model_id: str):
        # Get Save and Log Paths
        file_paths = gen_file_paths(self.path_dir_save, self.path_dir_log_hist, self.path_dir_log_best, model_id)
        warn_if_model_artifacts_exist(model_id, file_paths)
        self.path_save_file_best, self.path_log_file_hist, self.path_log_file_best = file_paths
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
        # torch.autograd.set_detect_anomaly(True)

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
        base_dir = os.path.dirname(os.path.abspath(__file__))
        # Get relative path to the spec file
        spec = {}
        if hasattr(self, 'dataset_path') and self.dataset_path:
            # Custom dataset path
            dataset_path = os.path.abspath(self.dataset_path)
            self.dataset_path = dataset_path
            if os.path.isfile(dataset_path) and dataset_path.endswith('.csv'):
                # Single CSV file - create minimal spec
                spec = {
                    'dataset_format': 'single_csv',
                    'split_ratios': {
                        'train': 0.6,
                        'val': 0.2,
                        'test': 0.2
                    },
                    'nperseg': 2560  # Default value
                }
                path_spec = None
            elif os.path.isdir(dataset_path):
                path_spec = os.path.join(dataset_path, 'spec.json')
            else:
                raise ValueError(f"Invalid dataset path: {self.dataset_path}")
        else:
            # Standard dataset name
            path_spec = os.path.join(base_dir, 'datasets', self.dataset_name, 'spec.json')

        # Load the spec
        if path_spec and os.path.exists(path_spec):
            with open(path_spec) as config_file:
                spec = json.load(config_file)
        elif path_spec and hasattr(self, 'dataset_path') and self.dataset_path:
            raise FileNotFoundError(f"spec.json not found in dataset path: {self.dataset_path}")
        elif path_spec:
            # No spec file and no dataset_path - this shouldn't happen
            raise FileNotFoundError(f"spec.json not found for dataset: {self.dataset_name}")
        
        if spec:
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

    def build_dataloaders(self):
        from modules.data_collector import IQSegmentDataset, IQFrameDataset, load_dataset

        # Load Dataset
        if hasattr(self, 'dataset_path') and self.dataset_path:
            X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(dataset_path=self.dataset_path)
        else:
            X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(dataset_name=self.dataset_name)

        # Apply the PA Gain if training DPD
        self.target_gain = set_target_gain(X_train, y_train)
        if self.step == 'train_dpd':
            y_train = self.target_gain * X_train
            y_val = self.target_gain * X_val
            y_test = self.target_gain * X_test

        # Extract Features
        input_size = X_train.shape[-1]

        # Define PyTorch Datasets
        train_set = IQFrameDataset(X_train, y_train, frame_length=self.frame_length, stride=self.frame_stride)
        val_set = IQSegmentDataset(X_val, y_val, nperseg=self.args.nperseg)
        test_set = IQSegmentDataset(X_test, y_test, nperseg=self.args.nperseg)

        # Define PyTorch Dataloaders
        pin_memory = self.device.type == 'cuda'
        train_loader = DataLoader(
            train_set, batch_size=self.batch_size, shuffle=True,
            pin_memory=pin_memory
        )
        val_loader = DataLoader(
            val_set, batch_size=self.batch_size_eval, shuffle=False,
            pin_memory=pin_memory
        )
        test_loader = DataLoader(
            test_set, batch_size=self.batch_size_eval, shuffle=False,
            pin_memory=pin_memory
        )

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
        dict_loss = {'l2': nn.MSELoss(),
                     'l1': nn.L1Loss()
                     }
        loss_func_name = self.loss_type
        try:
            criterion = dict_loss[loss_func_name]
            self.add_arg("criterion", criterion)
            return criterion
        except AttributeError:
            raise AttributeError('Please use a valid loss function. Check argument.py.')

    def build_optimizer(self, net: nn.Module):
        # Frozen PA parameters in a cascaded DPD model never receive gradients;
        # excluding them avoids needless optimizer and clipping traversal while
        # leaving the DPD updates unchanged.
        trainable_params = tuple(
            parameter for parameter in net.parameters() if parameter.requires_grad
        )
        if not trainable_params:
            raise ValueError("Cannot build an optimizer without trainable parameters")

        # Optimizer
        if self.opt_type == 'adam':
            optimizer = optim.Adam(trainable_params, lr=self.lr)
        elif self.opt_type == 'sgd':
            optimizer = optim.SGD(trainable_params, lr=self.lr, momentum=0.9)
        elif self.opt_type == 'rmsprop':
            optimizer = optim.RMSprop(trainable_params, lr=self.lr)
        elif self.opt_type == 'adamw':
            optimizer = optim.AdamW(
                trainable_params,
                lr=self.lr,
                weight_decay=0.01,
                betas=(0.9, 0.999),
                eps=1e-8,
            )
        elif self.opt_type == 'adabound':
            import adabound  # Run pip install adabound (https://github.com/Luolc/AdaBound)
            optimizer = adabound.AdaBound(trainable_params, lr=self.lr, final_lr=0.1)
        else:
            raise RuntimeError('Please use a valid optimizer.')

        # Learning Rate Scheduler
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                            mode='min',
                                                            factor=self.decay_factor,
                                                            patience=self.patience,
                                                            threshold=1e-4,
                                                            threshold_mode='rel',
                                                            cooldown=0,
                                                            min_lr=self.lr_end,
                                                            eps=1e-8)
        return optimizer, lr_scheduler

    @staticmethod
    def _run_model_chunked(model, full_iq, device, chunk_size=16384):
        """Run a model on full IQ data in chunks to avoid GPU memory issues."""
        import numpy as np
        n_total = full_iq.shape[0]
        outputs = []
        for start in range(0, n_total, chunk_size):
            end = min(start + chunk_size, n_total)
            chunk = torch.Tensor(full_iq[start:end]).unsqueeze(0).to(device)
            out = model(chunk)
            outputs.append(torch.squeeze(out).cpu().numpy())
        return np.concatenate(outputs, axis=0)

    def _compute_full_const_data(self, net, full_input_iq, full_output_iq, full_pa_only_c):
        """Compute full-sequence constellation data for APA datasets.

        Runs the full dataset input through the current model to produce
        complex signals long enough for proper OFDM demodulation.

        Returns:
            dict with 'input_c' and model-specific complex signal arrays.
        """
        import numpy as np

        full_input_c = full_input_iq[:, 0] + 1j * full_input_iq[:, 1]
        result = {'input_c': full_input_c}

        was_training = net.training
        net.eval()
        with torch.no_grad():
            if self.step == 'train_pa':
                pred = self._run_model_chunked(net, full_input_iq, self.device)
                result['pred_c'] = pred[:, 0] + 1j * pred[:, 1]
                if full_output_iq is not None:
                    result['gt_c'] = full_output_iq[:, 0] + 1j * full_output_iq[:, 1]
            elif self.step == 'train_dpd':
                cascaded = self._run_model_chunked(net, full_input_iq, self.device)
                result['cascaded_c'] = cascaded[:, 0] + 1j * cascaded[:, 1]
                if full_pa_only_c is not None:
                    result['pa_only_c'] = full_pa_only_c
        if was_training:
            net.train()

        return result

    def train(self, net: nn.Module, criterion: Callable, optimizer: optim.Optimizer, lr_scheduler,
              train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader,
              best_model_metric: str, plot_dir: str = None, input_data_val=None,
              input_data_test=None, pa_only_data=None, metadata=None,
              full_input_iq=None, full_output_iq=None,
              full_pa_only_c=None) -> None:
        # Timer
        start_time = time.time()

        # Track history for training curve plots
        training_history = []

        # Cache epoch prediction data for fixed-axis re-rendering
        epoch_plot_cache = {}
        best_epoch_cache = {}

        # Epoch loop
        print("Starting training...")
        for epoch in range(self.n_epochs):
            # -----------
            # Train
            # -----------
            net = net_train(log=self.log_train,
                            net=net,
                            optimizer=optimizer,
                            criterion=criterion,
                            dataloader=train_loader,
                            grad_clip_val=self.grad_clip_val,
                            device=self.device,
                            cuda_graph_training=self.cuda_graph_training)

            # -----------
            # Validation
            # -----------
            val_prediction = None
            val_ground_truth = None
            if self.eval_val:
                _, val_prediction, val_ground_truth = net_eval(log=self.log_val,
                                                               net=net,
                                                               criterion=criterion,
                                                               dataloader=val_loader,
                                                               device=self.device)
                self.log_val = calculate_metrics(self.args, self.log_val, val_prediction, val_ground_truth)

            # -----------
            # Test
            # -----------
            test_prediction = None
            test_ground_truth = None
            if self.eval_test:
                _, test_prediction, test_ground_truth = net_eval(log=self.log_test,
                                                                 net=net,
                                                                 criterion=criterion,
                                                                 dataloader=test_loader,
                                                                 device=self.device)
                self.log_test = calculate_metrics(self.args, self.log_test, test_prediction, test_ground_truth)

            ###########################################################################################################
            # Logging & Saving
            ###########################################################################################################

            # Generate Log Dict
            end_time = time.time()
            elapsed_time_minutes = (end_time - start_time) / 60.0
            self.log_all = gen_log_stat(self.args, elapsed_time_minutes, net, optimizer, epoch, self.log_train,
                                        self.log_val, self.log_test)

            # Write Log
            self.logger.write_log(self.log_all)

            # Save best model and detect if a new best was found
            best_net = net.dpd_model if self.step == 'train_dpd' else net
            prev_best = self.logger.best_val_metric
            self.logger.save_best_model(net=best_net, epoch=epoch, val_stat=self.log_val, metric_name=best_model_metric)
            is_new_best = (prev_best is None) or (self.logger.best_val_metric != prev_best)

            ###########################################################################################################
            # Plotting
            ###########################################################################################################
            if self.plot and plot_dir is not None:
                from utils.plotting import generate_epoch_plots_train_pa, generate_epoch_plots_train_dpd
                from datasets.demodulator import Demodulator
                spec = {k: getattr(self.args, k, None) for k in
                        ['input_signal_fs', 'nperseg', 'n_sub_ch', 'bw_main_ch', 'bw_sub_ch', 'scs']}
                try:
                    demod = Demodulator.from_dataset(self.args.dataset_name)
                except Exception:
                    demod = None

                should_plot_epoch = (epoch % self.plot_every == 0 or epoch == self.n_epochs - 1)

                # Compute full-sequence data for plotting (PSD, AM/AM, AM/PM, constellation)
                full_const_data = None
                if full_input_iq is not None and (should_plot_epoch or is_new_best):
                    full_const_data = self._compute_full_const_data(
                        net, full_input_iq, full_output_iq, full_pa_only_c)

                # Determine which subdirs to write into
                plot_targets = []
                if should_plot_epoch:
                    plot_targets.append(('epoch', plot_dir))
                if is_new_best:
                    plot_targets.append(('best', plot_dir))

                for target_type, base_dir in plot_targets:
                    try:
                        if self.step == 'train_pa':
                            if self.eval_val and val_prediction is not None:
                                generate_epoch_plots_train_pa(
                                    base_dir, epoch, val_prediction, val_ground_truth,
                                    input_data_val, 'val', spec, demod=demod,
                                    subfolder='best' if target_type == 'best' else None,
                                    full_const_data=full_const_data)
                            if self.eval_test and test_prediction is not None:
                                generate_epoch_plots_train_pa(
                                    base_dir, epoch, test_prediction, test_ground_truth,
                                    input_data_test, 'test', spec, demod=demod,
                                    subfolder='best' if target_type == 'best' else None,
                                    full_const_data=full_const_data)
                        elif self.step == 'train_dpd':
                            if self.eval_val and val_prediction is not None:
                                generate_epoch_plots_train_dpd(
                                    base_dir, epoch, val_prediction, val_ground_truth,
                                    'val', spec, demod=demod,
                                    subfolder='best' if target_type == 'best' else None,
                                    pa_only_prediction=pa_only_data.get('val') if pa_only_data else None,
                                    full_const_data=full_const_data)
                            if self.eval_test and test_prediction is not None:
                                generate_epoch_plots_train_dpd(
                                    base_dir, epoch, test_prediction, test_ground_truth,
                                    'test', spec, demod=demod,
                                    subfolder='best' if target_type == 'best' else None,
                                    pa_only_prediction=pa_only_data.get('test') if pa_only_data else None,
                                    full_const_data=full_const_data)
                    except Exception as e:
                        print(f"Warning: Plot generation failed at epoch {epoch} ({target_type}): {e}")

                # Cache prediction data for fixed-axis re-rendering
                if should_plot_epoch:
                    cache_entry = {}
                    if self.eval_val and val_prediction is not None:
                        cache_entry['val'] = val_prediction.copy()
                    if self.eval_test and test_prediction is not None:
                        cache_entry['test'] = test_prediction.copy()
                    epoch_plot_cache[epoch] = cache_entry
                if is_new_best:
                    best_epoch_cache.clear()
                    if self.eval_val and val_prediction is not None:
                        best_epoch_cache['val'] = val_prediction.copy()
                    if self.eval_test and test_prediction is not None:
                        best_epoch_cache['test'] = test_prediction.copy()
                    best_epoch_cache['epoch'] = epoch

            # Track history for training curves
            training_history.append(dict(self.log_all))

            ###########################################################################################################
            # Learning Rate Schedule
            ###########################################################################################################
            # Schedule at the beginning of retrain
            lr_scheduler_criteria = self.log_val[best_model_metric]
            if self.lr_schedule:
                lr_scheduler.step(lr_scheduler_criteria)

        # Generate training curve plots at end of training
        if self.plot and plot_dir is not None and training_history:
            from utils.plotting import plot_training_curves
            curves_dir = os.path.join(plot_dir, 'training_curves')
            try:
                plot_training_curves(training_history, curves_dir)
            except Exception as e:
                print(f"Warning: Training curve plot generation failed: {e}")

        # Re-render epoch plots with fixed axis limits
        if self.plot and plot_dir is not None and epoch_plot_cache:
            from utils.plotting import compute_global_limits, rerender_epochs_fixed_axes
            spec = {k: getattr(self.args, k, None) for k in
                    ['input_signal_fs', 'nperseg', 'n_sub_ch', 'bw_main_ch', 'bw_sub_ch', 'scs']}
            # Build constants dict
            constants = {}
            if self.eval_val and val_ground_truth is not None:
                constants['val_ground_truth'] = val_ground_truth
            if self.eval_test and test_ground_truth is not None:
                constants['test_ground_truth'] = test_ground_truth
            if input_data_val is not None:
                constants['input_data_val'] = input_data_val
            if input_data_test is not None:
                constants['input_data_test'] = input_data_test
            if pa_only_data is not None:
                if 'val' in pa_only_data:
                    constants['pa_only_val'] = pa_only_data['val']
                if 'test' in pa_only_data:
                    constants['pa_only_test'] = pa_only_data['test']
            # Add best epoch to cache if not already there
            if best_epoch_cache and best_epoch_cache.get('epoch') is not None:
                best_ep = best_epoch_cache['epoch']
                if best_ep not in epoch_plot_cache:
                    epoch_plot_cache[best_ep] = {
                        k: v for k, v in best_epoch_cache.items() if k != 'epoch'
                    }
            try:
                from datasets.demodulator import Demodulator as _Demod
                try:
                    _demod = _Demod.from_dataset(self.args.dataset_name)
                except Exception:
                    _demod = None
                fixed_limits = compute_global_limits(epoch_plot_cache, constants, self.step, spec)
                # Compute full-sequence constellation data for re-rendering
                rerender_full_const = None
                if full_input_iq is not None:
                    rerender_full_const = self._compute_full_const_data(
                        net, full_input_iq, full_output_iq, full_pa_only_c)
                rerender_epochs_fixed_axes(plot_dir, epoch_plot_cache, constants,
                                           self.step, spec, fixed_limits, demod=_demod,
                                           full_const_data=rerender_full_const)
            except Exception as e:
                print(f"Warning: Fixed-axis re-rendering failed: {e}")
                import traceback
                traceback.print_exc()

        # Generate interactive dashboard
        if self.plot and plot_dir is not None and metadata is not None:
            from utils.plotting import generate_dashboard_html
            try:
                generate_dashboard_html(plot_dir, self.step, metadata)
            except Exception as e:
                print(f"Warning: Dashboard generation failed: {e}")

        # Generate GIF animations
        if self.plot and plot_dir is not None:
            from utils.plotting import generate_epoch_gifs
            try:
                generate_epoch_gifs(plot_dir, gif_duration=self.gif_duration)
            except Exception as e:
                print(f"Warning: GIF generation failed: {e}")

        print("Training Completed...")
        print(" ")
