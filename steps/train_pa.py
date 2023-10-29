import time
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import models as model
from modules.feature_extractor import extract_feature
from modules.train_funcs import net_train, net_eval, calculate_metrics
from utils import pandaslogger, util, metrics
from tqdm import tqdm
import importlib
from torch.utils.data import DataLoader
from modules.data_collector import IQSegmentDataset, IQFrameDataset, IQFrameDataset_gmp, \
    prepare_segments
from modules.log import gen_model_id, gen_log_stat, gen_paths, count_net_params, create_folder


def main(args, device):
    ###########################################################################################################
    # Overhead
    ###########################################################################################################
    # Create Dataset Iterators
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_segments(args)

    # Extract Features
    X_train = extract_feature(X_train, args.PA_backbone)
    X_val = extract_feature(X_val, args.PA_backbone)
    X_test = extract_feature(X_test, args.PA_backbone)

    y_train_mean = np.zeros_like(y_train.mean(axis=0))
    y_train_std = np.zeros_like(y_train.std(axis=0))

    if args.norm == True:
        # Normalize Training Data
        X_train_mean = torch.mean(X_train, dim=0)
        X_train_std = torch.std(X_train, dim=0)
        X_train -= X_train_mean
        X_train /= X_train_std
        X_val -= X_train_mean
        X_val /= X_train_std
        X_test -= X_train_mean
        X_test /= X_train_std

        # Normalize Labels
        y_train_mean = torch.mean(y_train, dim=0)
        y_train_std = torch.std(y_train, dim=0)
        y_train -= y_train_mean
        y_train /= y_train_std
        y_val -= y_train_mean
        y_val /= y_train_std
        y_test -= y_train_mean
        y_test /= y_train_std

    feat_size = X_train.shape[-1]

    if args.PA_backbone == 'cnn1d':
        frame_length = args.pa_cnn_memory + args.PA_hidden_size + args.paoutput_len - 2
    elif args.PA_backbone == 'cnn2d':
        frame_length = args.pa_cnn_memory + args.PA_CNN_W + args.paoutput_len - 2
    else:
        frame_length = args.frame_length

    train_segment_dataset = IQSegmentDataset(X_train, y_train)
    val_segment_dataset = IQSegmentDataset(X_val, y_val)
    test_segment_dataset = IQSegmentDataset(X_test, y_test)

    """# Data Preparation
    Now, we'll split the IQ_frame_dataset into training, validation, and testing sets with a 60-20-20 ratio.
    """
    if args.PA_backbone == 'gmp':
        train_frame_dataset = IQFrameDataset_gmp(train_segment_dataset, frame_length=frame_length, degree=args.degree,
                                                 stride=args.stride_length)
    else:
        train_frame_dataset = IQFrameDataset(train_segment_dataset, frame_length=frame_length,
                                             stride=args.stride_length)

    train_loader = DataLoader(train_frame_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_segment_dataset, batch_size=args.batch_size_eval, shuffle=False)
    test_loader = DataLoader(test_segment_dataset, batch_size=args.batch_size_eval, shuffle=False)

    ###########################################################################################################
    # Network Settings
    ###########################################################################################################
    # Instantiate Model
    PA_CNN_setup = [args.PA_CNN_H, args.PA_CNN_W]
    net = model.CoreModel(input_size=feat_size,
                          cnn_set=PA_CNN_setup,
                          cnn_memory=args.pa_cnn_memory,
                          pa_output_len=args.pa_output_len,
                          frame_len=args.frame_length,
                          hidden_size=args.PA_hidden_size,
                          num_layers=1,
                          degree=args.degree,
                          backbone_type=args.PA_backbone,
                          y_train_mean=y_train_mean,
                          y_train_std=y_train_std)
    # Get parameter count
    n_param = count_net_params(net)
    print("::: Number of Parameters: ", n_param)

    ###########################################################################################################
    # Save & Log Naming Convention
    ###########################################################################################################

    # Model ID
    pamodel_id, dpdmodel_id = gen_model_id(args)

    # Create Folders
    dir_paths, file_paths, _ = gen_paths(args, model_id=pamodel_id)
    save_dir, log_dir_hist, log_dir_best, _ = dir_paths
    save_file, logfile_hist, logfile_best, _ = file_paths
    create_folder([save_dir, log_dir_hist, log_dir_best])
    print("::: Save Path: ", save_file)
    print("::: Log Path: ", logfile_hist)

    # Logger
    logger = pandaslogger.PandasLogger(logfile_hist, precision=args.log_precision)

    ###########################################################################################################
    # Settings
    ###########################################################################################################
    # Use CUDA
    if args.use_cuda:
        net = net.cuda()

    # Select Loss function
    criterion = nn.MSELoss()

    # Create Optimizer
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr)

    # Setup Learning Rate Scheduler
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                        mode='min',
                                                        factor=args.decay_factor,
                                                        patience=args.patience,
                                                        verbose=True,
                                                        threshold=1e-4,
                                                        min_lr=args.lr_end)

    ###########################################################################################################
    # Training
    ###########################################################################################################
    # Value for Saving Best Model
    best_model = None
    # Timer
    start_time = time.time()
    # Epoch loop
    print("Starting training...")
    for epoch in range(args.n_epochs):
        # -----------
        # Train
        # -----------
        net, train_stat = net_train(args=args,
                                    net=net,
                                    optimizer=optimizer,
                                    criterion=criterion,
                                    dataloader=train_loader,
                                    device=device)

        # -----------
        # Validation
        # -----------
        val_stat = None
        if args.eval_val:
            _, val_stat, prediction, ground_truth = net_eval(args=args,
                                                             net=net,
                                                             criterion=criterion,
                                                             dataloader=val_loader,
                                                             device=device)
            val_stat = calculate_metrics(args, val_stat, prediction, ground_truth)

        # -----------
        # Test
        # -----------
        test_stat = None
        if args.eval_test:
            _, test_stat, prediction, ground_truth = net_eval(args=args,
                                                              net=net,
                                                              criterion=criterion,
                                                              dataloader=test_loader,
                                                              device=device)
            test_stat = calculate_metrics(args, test_stat, prediction, ground_truth)

        ###########################################################################################################
        # Logging & Saving
        ###########################################################################################################

        # Generate Log Dict
        end_time = time.time()
        elapsed_time_minutes = (end_time - start_time) / 60.0
        log_stat = gen_log_stat(args, elapsed_time_minutes, net, optimizer, epoch, train_stat, val_stat, test_stat)

        # Write Log
        logger.write_log(log_stat)

        # Print
        print(log_stat)

        # Save best model
        best_model = logger.save_best_model(best_val=best_model,
                                            net=net,
                                            save_file=save_file,
                                            logger=logger,
                                            logfile_best=logfile_best,
                                            epoch=epoch,
                                            val_stat=val_stat)

        ###########################################################################################################
        # Learning Rate Schedule
        ###########################################################################################################
        # Schedule at the beginning of retrain
        lr_scheduler_criteria = val_stat['NMSE']
        if args.lr_schedule:
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
