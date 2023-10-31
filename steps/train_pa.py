import os
import time
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import models as model
from modules.data_collector import IQSegmentDataset, IQFrameDataset_gmp, IQFrameDataset, load_dataset
from modules.feat_ext import extract_feature
from modules.train_funcs import net_train, net_eval, calculate_metrics
from project import Project
from modules.paths import gen_model_id, gen_log_stat, gen_paths, create_folder
from utils.util import count_net_params


def main(proj: Project):
    ###########################################################################################################
    # Initialization
    ###########################################################################################################
    # Set Accelerator Device
    proj.set_device()

    # Update Conditional Arguments or Hyperparameters
    proj.tune_conditional_args()

    # Build Dataloaders
    (train_loader, val_loader, test_loader), input_size = proj.build_dataloaders()

    ###########################################################################################################
    # Network Settings
    ###########################################################################################################

    # Instantiate Model
    PA_CNN_setup = [proj.PA_CNN_H, proj.PA_CNN_W]
    net = model.CoreModel(input_size=input_size,
                          cnn_set=PA_CNN_setup,
                          cnn_memory=proj.pa_cnn_memory,
                          pa_output_len=proj.pa_output_len,
                          frame_len=proj.frame_length,
                          hidden_size=proj.PA_hidden_size,
                          num_layers=1,
                          degree=proj.degree,
                          backbone_type=proj.PA_backbone)
    # Get parameter count
    n_param = count_net_params(net)

    print("::: Number of Parameters: ", n_param)

    ###########################################################################################################
    # Settings
    ###########################################################################################################
    # Use CUDA
    net = net.to(proj.device)

    # Select Loss function
    criterion = nn.MSELoss()

    # Create Optimizer
    optimizer = torch.optim.AdamW(net.parameters(), lr=proj.lr)

    # Setup Learning Rate Scheduler
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                        mode='min',
                                                        factor=proj.decay_factor,
                                                        patience=proj.patience,
                                                        verbose=True,
                                                        threshold=1e-4,
                                                        min_lr=proj.lr_end)

    ###########################################################################################################
    # Training
    ###########################################################################################################

    # Timer
    start_time = time.time()
    # Epoch loop
    print("Starting training...")
    for epoch in range(proj.n_epochs):
        # -----------
        # Train
        # -----------
        net, train_stat = net_train(args=proj.args,
                                    net=net,
                                    optimizer=optimizer,
                                    criterion=criterion,
                                    dataloader=train_loader,
                                    device=proj.device)

        # -----------
        # Validation
        # -----------
        val_stat = None
        if proj.eval_val:
            _, val_stat, prediction, ground_truth = net_eval(args=proj.args,
                                                             net=net,
                                                             criterion=criterion,
                                                             dataloader=val_loader,
                                                             device=proj.device)
            val_stat = calculate_metrics(proj, val_stat, prediction, ground_truth)

        # -----------
        # Test
        # -----------
        test_stat = None
        if proj.eval_test:
            _, test_stat, prediction, ground_truth = net_eval(args=proj.args,
                                                              net=net,
                                                              criterion=criterion,
                                                              dataloader=test_loader,
                                                              device=proj.device)
            test_stat = calculate_metrics(proj, test_stat, prediction, ground_truth)

        ###########################################################################################################
        # Logging & Saving
        ###########################################################################################################

        # Generate Log Dict
        end_time = time.time()
        elapsed_time_minutes = (end_time - start_time) / 60.0
        log_stat = gen_log_stat(proj, elapsed_time_minutes, net, optimizer, epoch, train_stat, val_stat, test_stat)

        # Write Log
        proj.logger.write_log(log_stat)

        # Print
        print(log_stat)

        # Save best model
        proj.logger.save_best_model(net=net, epoch=epoch, val_stat=val_stat, metric_name='ACLR_L')

        ###########################################################################################################
        # Learning Rate Schedule
        ###########################################################################################################
        # Schedule at the beginning of retrain
        lr_scheduler_criteria = val_stat['ACLR_L']
        if proj.lr_schedule:
            lr_scheduler.step(lr_scheduler_criteria)

    result_graph = get_plotdata(plt_head_idx=50, plt_tail_idx=60, prediction=prediction, ground_truth=ground_truth)
    result_graph.savefig('save/figure/pa_results.png')
    print("Training Completed...                                               ")
    print(" ")


def get_phase(i, q):
    amp = np.sqrt(np.square(i) + np.square(q))
    arccos = np.arccos(np.divide(i, amp))
    phase = np.zeros_like(i)
    phase[q >= 0] = arccos[q >= 0]
    phase[q < 0] = (2 * np.pi - arccos)[q < 0]
    return phase


def get_plotdata(plt_head_idx, plt_tail_idx, prediction, ground_truth):
    length = plt_tail_idx - plt_head_idx
    prediction = prediction[plt_head_idx:plt_tail_idx, :]
    ground_truth = ground_truth[plt_head_idx:plt_tail_idx, :]
    i_pre = prediction[:, 0]
    i_true = ground_truth[:, 0]
    I = np.concatenate([i_pre, i_true], axis=0)
    q_pre = prediction[:, 1]
    q_true = ground_truth[:, 1]
    Q = np.concatenate([q_pre, q_true], axis=0)
    amp_pre = np.sqrt(np.square(i_pre) + np.square(q_pre))
    amp_true = np.sqrt(np.square(i_true) + np.square(q_true))
    Amp = np.concatenate([amp_pre, amp_true], axis=0)
    arcsin_pre = np.arcsin(np.divide(q_pre, amp_pre))
    arcsin_true = np.arcsin(np.divide(q_true, amp_true))
    arcsin = np.concatenate([arcsin_pre, arcsin_true], axis=0)
    arccos_pre = np.arccos(np.divide(i_pre, amp_pre))
    arccos_true = np.arccos(np.divide(i_true, amp_true))
    arccos = np.concatenate([arccos_pre, arccos_true], axis=0)
    phase_pre = get_phase(i_pre, q_pre)
    phase_true = get_phase(i_true, q_true)
    phase = np.concatenate([phase_pre, phase_true], axis=0)
    time = np.arange(plt_head_idx, plt_tail_idx, 1)
    time = np.concatenate((time, time), axis=0)
    group = ["pre"] * length
    group = np.concatenate((group, ["true"] * length), axis=0)
    df = {'I': I, 'Q': Q, 'Amp': Amp, 'arcsin': arcsin, 'arccos': arccos, 'phase': phase, 'time': time, 'group': group}
    labelsize = 14
    titlesize = 16
    result_graph, ax = plt.subplots(2, 3, figsize=(24, 18))
    sns.color_palette("Set1", 2)
    sns.lineplot(x=df["time"], y=df["I"], ax=ax[0][0], hue=df["group"], style=df["group"])
    # ax[0][0].set_ylim(-30, 30)
    ax[0][0].set_xlabel('Timestep', fontsize=labelsize)
    ax[0][0].set_ylabel('I', fontsize=labelsize)
    ax[0][0].set_title('I', fontsize=labelsize)
    sns.color_palette("Set1", 2)
    sns.lineplot(x=df["time"], y=df["Q"], ax=ax[0][1], hue=df["group"], style=df["group"])
    # ax[0][1].set_ylim(-30, 30)
    ax[0][1].set_xlabel('Timestep', fontsize=labelsize)
    ax[0][1].set_ylabel('Q', fontsize=labelsize)
    ax[0][1].set_title('Q', fontsize=titlesize)
    sns.color_palette("Set1", 2)
    sns.lineplot(x=df["time"], y=df["Amp"], ax=ax[0][2], hue=df["group"], style=df["group"])
    ax[0][2].set_ylim(0, 50)
    ax[0][2].set_xlabel('Timestep', fontsize=labelsize)
    ax[0][2].set_ylabel('Amplitude', fontsize=labelsize)
    ax[0][2].set_title('Amplitude', fontsize=titlesize)
    sns.color_palette("Set1", 2)
    sns.lineplot(x=df["time"], y=df["phase"], ax=ax[1][0], hue=df["group"], style=df["group"])
    ax[1][0].set_ylim(-1, 7)
    ax[1][0].set_xlabel('Timestep', fontsize=labelsize)
    ax[1][0].set_ylabel('$\Theta$', fontsize=labelsize)
    ax[1][0].set_title('Phase $\Theta$', fontsize=titlesize)
    sns.color_palette("Set1", 2)
    sns.lineplot(x=df["time"], y=df["arcsin"], ax=ax[1][1], hue=df["group"], style=df["group"])
    ax[1][1].set_ylim(-2, 2)
    ax[1][1].set_xlabel('Timestep', fontsize=labelsize)
    ax[1][1].set_ylabel('arcsin(Q/Amp)', fontsize=labelsize)
    ax[1][1].set_title('arcsin(Q/Amp)', fontsize=titlesize)
    sns.color_palette("Set1", 2)
    sns.lineplot(x=df["time"], y=df["arccos"], ax=ax[1][2], hue=df["group"], style=df["group"])
    ax[1][2].set_ylim(-1, 4)
    ax[1][2].set_xlabel('Timestep', fontsize=labelsize)
    ax[1][2].set_ylabel('arccos(I/Amp)', fontsize=labelsize)
    ax[1][2].set_title('arccos(I/Amp)', fontsize=titlesize)
    plt.subplots_adjust(top=0.95, bottom=0.1, left=0.09, right=0.95, wspace=0.2, hspace=0.4)
    plt.show()
    return result_graph
