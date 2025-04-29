__author__ = "Yizhuo Wu, Chang Gao"
__license__ = "Apache-2.0 License"
__email__ = "yizhuo.wu@tudelft.nl, chang.gao@tudelft.nl"

import numpy as np
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import metrics
from typing import Dict, Any, Callable
import argparse


def net_train(log: Dict[str, Any],
              net: nn.Module,
              dataloader: DataLoader,
              optimizer: Optimizer,
              criterion: Callable,
              grad_clip_val: float,
              device: torch.device):
    # Set Network to Training Mode
    net = net.train()
    # Statistics
    losses = []
    # Iterate through batches
    for features, targets in tqdm(dataloader):
        # Move features and targets to the proper device
        features = features.to(device)
        targets = targets.to(device)
        # Initialize all gradients to zero
        optimizer.zero_grad()
        # Forward Propagation
        out = net(features)
        # Calculate the Loss Function
        loss = criterion(out, targets)
        # Backward propagation
        loss.backward()
        # Gradient clipping
        if grad_clip_val != 0:
            nn.utils.clip_grad_norm_(net.parameters(), grad_clip_val)
        # Update parameters
        optimizer.step()
        # Detach loss from the graph indicating the end of forward propagation
        loss.detach()
        # Get losses
        losses.append(loss.item())
    # Average loss
    loss = np.mean(losses)
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
    with torch.no_grad():
        # Statistics
        losses = []
        prediction = []
        ground_truth = []
        # Batch Iteration
        for features, targets in tqdm(dataloader):
            # Move features and targets to the proper device
            features = features.to(device)
            targets = targets.to(device)
            # Forward Propagation
            outputs = net(features)
            # Calculate loss function
            loss = criterion(outputs, targets)
            # Collect prediction and ground truth for metric calculation
            prediction.append(outputs.cpu())
            ground_truth.append(targets.cpu())
            # Collect losses to calculate the average loss per epoch
            losses.append(loss.item())
    # Average loss per epoch
    avg_loss = np.mean(losses)
    # Prediction and Ground Truth
    prediction = torch.cat(prediction, dim=0).numpy()
    ground_truth = torch.cat(ground_truth, dim=0).numpy()
    # Save Statistics
    log['loss'] = avg_loss
    # End of Evaluation Epoch
    return net, prediction, ground_truth


def calculate_metrics(args: argparse.Namespace, stat: Dict[str, Any], prediction: np.ndarray, ground_truth: np.ndarray):
    stat['NMSE'] = metrics.NMSE(prediction, ground_truth)
    stat['EVM'] = metrics.EVM(prediction, ground_truth, bw_main_ch=args.bw_main_ch, n_sub_ch=args.n_sub_ch, nperseg=args.nperseg)
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
