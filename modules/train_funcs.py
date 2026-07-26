__author__ = "Yizhuo Wu, Chang Gao"
__license__ = "Apache-2.0 License"
__email__ = "yizhuo.wu@tudelft.nl, chang.gao@tudelft.nl"

import contextlib

import numpy as np
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import metrics
from typing import Dict, Any, Callable
import argparse


def _mean_losses(losses):
    """Return the historical NumPy mean with one device-to-host transfer.

    ``Tensor.item()`` used to synchronize the accelerator once per batch.  A
    detached scalar keeps exactly the same stored value, so we can transfer all
    batch losses together and still let NumPy perform the same float mean as
    before.
    """
    if not losses:
        # Preserve NumPy's historical empty-loader result (NaN).
        return np.mean([])
    return np.mean(torch.stack(losses).cpu().tolist())


def net_train(log: Dict[str, Any],
              net: nn.Module,
              dataloader: DataLoader,
              optimizer: Optimizer,
              criterion: Callable,
              grad_clip_val: float,
              device: torch.device,
              cuda_graph_training: bool = False):
    # Set Network to Training Mode
    net = net.train()
    # Statistics
    losses = []
    # The optimizer contains only trainable parameters in the standard
    # pipeline.  Materialize this tuple once per epoch rather than walking the
    # complete cascaded model for every batch.
    trainable_params = tuple(
        parameter
        for group in optimizer.param_groups
        for parameter in group['params']
        if parameter.requires_grad
    )
    non_blocking = device.type == 'cuda'
    if cuda_graph_training:
        from modules.cuda_graph_training import (
            force_clean_eager_tres,
            try_cuda_graph_training_step,
        )
    # Iterate through batches
    for features, targets in tqdm(dataloader):
        # Move features and targets to the proper device
        features = features.to(device, non_blocking=non_blocking)
        targets = targets.to(device, non_blocking=non_blocking)
        graph_result = None
        if cuda_graph_training:
            graph_result = try_cuda_graph_training_step(
                net, features, targets, criterion, trainable_params
            )
        if graph_result is None:
            # Strict opt-in fallback keeps the same clean-eager TRes
            # arithmetic as the captured body.  The original dispatch flag is
            # restored immediately afterward, including for evaluation.
            context = (
                force_clean_eager_tres(net)
                if cuda_graph_training else contextlib.nullcontext()
            )
            with context:
                optimizer.zero_grad(set_to_none=True)
                out = net(features)
                loss = criterion(out, targets)
                loss.backward()
        else:
            loss = graph_result.loss
        # Gradient clipping
        if grad_clip_val != 0:
            nn.utils.clip_grad_norm_(trainable_params, grad_clip_val)
        # Update parameters
        optimizer.step()
        # Keep only the scalar value, not its graph.  The complete vector is
        # copied to the host once after the epoch to avoid a sync per batch.
        losses.append(loss.detach())
    # Average loss
    loss = _mean_losses(losses)
    # Save Statistics
    log['loss'] = loss
    # End of Training Epoch
    return net


def net_eval(log: Dict,
             net: nn.Module,
             dataloader: DataLoader,
             criterion: Callable,
             device: torch.device):
    net = net.eval()
    with torch.inference_mode():
        # Statistics
        losses = []
        prediction = []
        ground_truth = []
        non_blocking = device.type == 'cuda'
        # Batch Iteration
        for features, targets in tqdm(dataloader):
            # Move features and targets to the proper device
            features = features.to(device, non_blocking=non_blocking)
            targets = targets.to(device, non_blocking=non_blocking)
            # Forward Propagation
            outputs = net(features)
            # Calculate loss function
            loss = criterion(outputs, targets)
            # Collect prediction and ground truth for metric calculation
            prediction.append(outputs)
            ground_truth.append(targets)
            # Collect losses to calculate the average loss per epoch
            losses.append(loss)
    # Average loss per epoch
    avg_loss = _mean_losses(losses)
    # Concatenate before copying so evaluation performs one host transfer per
    # result tensor instead of one transfer per batch.
    prediction = torch.cat(prediction, dim=0).cpu().numpy()
    ground_truth = torch.cat(ground_truth, dim=0).cpu().numpy()
    # Save Statistics
    log['loss'] = avg_loss
    # End of Evaluation Epoch
    return net, prediction, ground_truth


def calculate_metrics(args: argparse.Namespace, stat: Dict[str, Any], prediction: np.ndarray, ground_truth: np.ndarray):
    stat['NMSE'] = metrics.NMSE(prediction, ground_truth)
    stat['EVM'] = metrics.EVM(
        prediction,
        ground_truth,
        sample_rate=args.input_signal_fs,
        bw_main_ch=args.bw_main_ch,
        n_sub_ch=args.n_sub_ch,
        nperseg=args.nperseg,
    )
    ACLR_L = []
    ACLR_R = []
    ACLR_left, ACLR_right = metrics.ACLR(prediction, fs=args.input_signal_fs, nperseg=args.nperseg,
                                         bw_main_ch=args.bw_main_ch, n_sub_ch=args.n_sub_ch)
    ACLR_L.append(ACLR_left)
    ACLR_R.append(ACLR_right)
    stat['ACLR_L'] = np.mean(ACLR_L)
    stat['ACLR_R'] = np.mean(ACLR_R)
    stat['ACLR_AVG'] = (stat['ACLR_L'] + stat['ACLR_R']) / 2
    return stat
