import numpy as np
import torch
from tqdm import tqdm
from utils import metrics
from typing import Dict, Any
import argparse


def net_train(args,
              net,
              optimizer,
              criterion,
              dataloader,
              device):
    # Set Network Properties
    net = net.train()

    # Statistics
    losses = []

    # Iterate through batches
    for features, targets in tqdm(dataloader):
        features = features.to(device)
        targets = targets.to(device)

        # Optimization
        optimizer.zero_grad()

        # Forward Propagation
        out = net(features)

        # Calculate Loss
        if args.PA_backbone == 'gmp':
            targetY = torch.zeros(targets.size(1), 2)
            targetY[:, 0] = torch.real(targets)
            targetY[:, 1] = torch.imag(targets)
            loss = criterion(out, targetY)
        elif args.PA_backbone == 'cnn2d' or args.PA_backbone == 'cnn1d':
            loss = criterion(out, targets[:, :args.paoutput_len, :].to(device))
        else:
            loss = criterion(out, targets.to(device))
        # Backward propagation
        loss.backward()

        # Update parameters
        optimizer.step()

        # Increment monitoring variables
        loss.detach()

        # Get losses
        losses.append(loss.item())

    # Average loss
    loss = np.mean(losses)

    # Save Statistics
    stat = {'loss': loss}

    return net, stat


def net_eval(args,
             net,
             criterion,
             dataloader,
             device):
    net = net.eval()
    with torch.no_grad():
        # Statistics
        losses = []
        prediction = []
        ground_truth = []

        # Batch Iteration
        for features, targets in tqdm(dataloader):
            features = features.to(device)
            targets = targets.to(device)
            outputs = net(features)
            loss = criterion(outputs, targets)
            outputs = outputs.cpu()
            if args.norm == True:
                # Denormalization
                targets *= net.y_train_std.cpu()
                targets += net.y_train_mean.cpu()
                outputs *= net.y_train_std.cpu()
                outputs += net.y_train_mean.cpu()
            if args.PA_backbone == 'gmp':
                targetY = torch.zeros(targets.size(1), 2)
                targetY[:, 0] = torch.real(targets)
                targetY[:, 1] = torch.imag(targets)
                prediction.append(outputs)
                ground_truth.append(targetY)
            else:
                if args.PA_backbone == 'cnn2d' or args.PA_backbone == 'cnn1d':
                    ground_truth.append(targets[:, :outputs.size(1), :])
                    prediction.append(outputs)
                else:
                    ground_truth.append(targets)
                    prediction.append(outputs)
            losses.append(loss.item())

    avg_loss = np.mean(losses)
    prediction = torch.cat(prediction, dim=0).numpy()
    ground_truth = torch.cat(ground_truth, dim=0).numpy()

    # Save Statistics
    stat = {'loss': avg_loss}

    return net, stat, prediction, ground_truth


def calculate_metrics(args: argparse.Namespace, stat: Dict[str, Any], prediction: np.ndarray, ground_truth: np.ndarray):
    stat['NMSE'] = metrics.NMSE(prediction, ground_truth)
    stat['EVM'] = metrics.EVM(prediction, ground_truth, nperseg=args.nperseg)
    ACLR_L = []
    ACLR_R = []
    for segment in ground_truth:
        ACLR_left, ACLR_right = metrics.ACLR(segment, fs=args.input_signal_fs, nperseg=args.nperseg,
                                             bw_main_ch=args.input_signal_bw, bw_side_ch=args.input_signal_ch_bw)
        ACLR_L.append(ACLR_left)
        ACLR_R.append(ACLR_right)
    stat['ACLR_L'] = np.mean(ACLR_L)
    stat['ACLR_R'] = np.mean(ACLR_R)
    return stat
