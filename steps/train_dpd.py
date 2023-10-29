import time
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import models as model
from utils import pandaslogger, util
from tqdm import tqdm
import importlib
from torch.utils.data import DataLoader
from modules.data_collector import IQSegmentDataset, IQFrameDataset, IQFrameDataset_gmp
from modules.log import gen_paths, count_net_params


def main(args, In, Out, Ideal, device):
    ###########################################################################################################
    # Overhead
    ###########################################################################################################
    # Load Modules
    module_log = importlib.import_module('modules.log')

    # Assign methods to be used
    gen_dataset_name = module_log.gen_dataset_name
    gen_model_id = module_log.gen_model_id

    print("::: Phase:   ", args.phase)

    def get_amplitude(IQ_signal):
        I = IQ_signal[:, 0]
        Q = IQ_signal[:, 1]
        power = I ** 2 + Q ** 2
        amplitude = np.sqrt(power)
        return amplitude

    def set_target_gain(input_IQ, output_IQ):
        """Calculate the total energy of the I-Q signal."""
        amp_in = get_amplitude(input_IQ)
        amp_out = get_amplitude(output_IQ)
        max_in_amp = np.max(amp_in)
        max_out_amp = np.max(amp_out)
        target_gain = np.mean(max_out_amp / max_in_amp)
        return target_gain

    gain = set_target_gain(In, Out)
    # Create Dataset Iterators
    PA_IQ_in_segments = get_segments(In, nperseg=args.nperseg)
    PA_IQ_out_segments = get_segments(gain * In, nperseg=args.nperseg)

    X_train, y_train, X_val, y_val, X_test, y_test = split(PA_IQ_in_segments, PA_IQ_out_segments)

    # Extract Features
    def extract_feature(X, PA_model_type):
        i_x = np.expand_dims(X[:, :, 0], axis=-1)
        q_x = np.expand_dims(X[:, :, 1], axis=-1)
        amp2 = np.power(i_x, 2) + np.power(q_x, 2)
        amp = np.sqrt(amp2)
        amp3 = np.power(amp, 3)
        angle = np.angle(i_x + 1j * q_x)
        cos = i_x / amp
        sin = q_x / amp
        if PA_model_type == 'lstm' or PA_model_type == 'gru' or PA_model_type == 'fc':
            # Feat = torch.hstack((X, amp, angle))
            Feat = X
        elif PA_model_type == 'pgjanet' or PA_model_type == 'dvrjanet':
            Feat = np.concatenate((amp, angle), axis=-1)
        elif PA_model_type == 'vdlstm':
            Feat = np.concatenate((amp, angle), axis=-1)
        elif PA_model_type == 'cnn2d':
            Feat = np.concatenate((i_x, q_x, amp, amp2, amp3), axis=-1)
        elif PA_model_type == 'rgru':
            Feat = np.concatenate((i_x, q_x, amp, amp3, sin, cos), axis=-1)
        else:
            Feat = X
        return Feat

    X_train = extract_feature(X_train, args.DPD_model)
    X_val = extract_feature(X_val, args.DPD_model)
    X_test = extract_feature(X_test, args.DPD_model)
    DPD_target = extract_feature(Ideal, args.DPD_model)

    DPD_target_mean = torch.zeros_like(torch.mean(DPD_target, dim=0))
    DPD_target_std = torch.zeros_like(torch.std(DPD_target, dim=0))
    y_train_mean = torch.zeros_like(torch.mean(y_train, dim=0))
    y_train_std = torch.zeros_like(torch.std(y_train, dim=0))

    if args.norm == True:
        # Normalize DPD target Data
        DPD_target_mean = torch.mean(DPD_target, dim=0)
        DPD_target_std = torch.std(DPD_target, dim=0)
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

    feat_size = X_train.size(-1)

    if args.DPD_model == 'cnn1d':
        dpdoutput_len = args.paoutput_len + args.pa_cnn_memory + args.PA_hidden_size - 2
        frame_length = args.dpd_cnn_memory + args.DPD_hidden_size + dpdoutput_len - 2
    elif args.DPD_model == 'cnn2d':
        dpdoutput_len = args.paoutput_len + args.pa_cnn_memory + args.PA_CNN_W - 2
        frame_length = args.dpd_cnn_memory + args.DPD_CNN_W + dpdoutput_len - 2
    else:
        dpdoutput_len = 0
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
        val_frame_dataset = IQFrameDataset_gmp(val_segment_dataset, frame_length=frame_length, degree=args.degree,
                                               stride=args.stride_length)
        test_frame_dataset = IQFrameDataset_gmp(test_segment_dataset, frame_length=frame_length, degree=args.degree,
                                                stride=args.stride_length)
    else:
        train_frame_dataset = IQFrameDataset(train_segment_dataset, frame_length=frame_length,
                                             stride=args.stride_length)
        val_frame_dataset = IQFrameDataset(val_segment_dataset, frame_length=frame_length, stride=args.stride_length)
        test_frame_dataset = IQFrameDataset(test_segment_dataset, frame_length=frame_length, stride=args.stride_length)

    train_loader = DataLoader(train_frame_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_frame_dataset, batch_size=args.batch_size_eval, shuffle=False)
    test_loader = DataLoader(test_frame_dataset, batch_size=args.batch_size_eval, shuffle=False)

    ###########################################################################################################
    # Network Settings
    ###########################################################################################################
    # Instantiate Model
    PA_CNN_setup = [args.PA_CNN_H, args.PA_CNN_W]
    DPD_CNN_setup = [args.DPD_CNN_H, args.DPD_CNN_W]
    pa = model.CoreModel(input_size=args.PA_input_size,
                         cnn_memory=args.pa_cnn_memory,
                         pa_output_len=args.paoutput_len,
                         cnn_set=PA_CNN_setup, frame_len=args.frame_length,
                         hidden_size=args.PA_hidden_size,
                         num_layers=1,
                         degree=args.degree,
                         backbone_type=args.PA_backbone, y_train_mean=y_train_mean, y_train_std=y_train_std)

    for weight in pa.parameters():
        weight.requires_grad = False
    net = model.DPD_MODEL(input_size=args.DPD_input_size,
                          cnn_memory=args.dpd_cnn_memory,
                          dpd_output_len=dpdoutput_len,
                          cnn_set=DPD_CNN_setup, frame_len=args.frame_length,
                          hidden_size=args.DPD_hidden_size,
                          num_layers=1,
                          degree=args.degree,
                          rnn_type=args.DPD_model, X_train_mean=DPD_target_mean,
                          X_train_std=DPD_target_std,
                          y_train_mean=y_train_mean, y_train_std=y_train_std)
    # dpd_dict_path = os.path.join('save', args.dataset_name, args.phase, args.DPD_model_file)
    # net.load_state_dict(torch.load(dpd_dict_path))

    # Get parameter count
    n_param = count_net_params(net)
    print("::: Number of Parameters: ", n_param)

    ###########################################################################################################
    # Save & Log Naming Convention
    ###########################################################################################################

    # Model ID
    pamodel_id, dpdmodel_id = gen_model_id(args)
    pa_dict_path = os.path.join('save', args.dataset_name, args.PA_model_phase, pamodel_id + '.pt')
    pa.load_state_dict(torch.load(pa_dict_path))
    # Create Folders
    dir_paths, file_paths, _ = gen_paths(args, model_id=dpdmodel_id, pretrain_model_id=pamodel_id)
    save_dir, log_dir_hist, log_dir_best, _ = dir_paths
    save_file, logfile_hist, logfile_best, _ = file_paths
    util.create_folder([save_dir, log_dir_hist, log_dir_best])
    print("::: Save Path: ", save_file)
    print("::: Log Path: ", logfile_hist)

    # Logger
    logger = pandaslogger.PandasLogger(logfile_hist)

    ###########################################################################################################
    # Settings
    ###########################################################################################################
    # Use CUDA
    if args.use_cuda:
        pa = pa.cuda()
        net = net.cuda()

    # Select Loss function
    criterion = nn.MSELoss()

    # Create Optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, amsgrad=False, weight_decay=0)

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
                                    pa_net=pa,
                                    optimizer=optimizer,
                                    criterion=criterion,
                                    dataloader=train_loader,
                                    device=device)

        # -----------
        # Validation
        # -----------
        val_stat = None
        if args.eval_val:
            _, val_stat, _, _ = net_eval(args=args,
                                         net=net,
                                         pa_net=pa,
                                         criterion=criterion,
                                         dataloader=val_loader,
                                         device=device)

        # -----------
        # Test
        # -----------
        test_stat = None
        if args.eval_test:
            _, test_stat, prediction, ground_truth = net_eval(args=args,
                                                              net=net,
                                                              pa_net=pa,
                                                              criterion=criterion,
                                                              dataloader=test_loader,
                                                              device=device)

        ###########################################################################################################
        # Logging & Saving
        ###########################################################################################################

        # Generate Log Dict
        end_time = time.time()
        elapsed_time_minutes = (end_time - start_time) / 60.0

        log_stat = gen_log_stat(args, net, optimizer, epoch, train_stat, val_stat, test_stat)

        # Write Log
        logger.write_log(log_stat)

        # Print
        print(log_stat)
        print("time for this epoch: %1.5f" % (stop_time - start_time))

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
        if args.lr_schedule:
            lr_scheduler.step(val_stat)

    print("Training Completed...                                               ")
    print(" ")

def net_train(args,
              net,
              pa_net,
              optimizer,
              criterion,
              dataloader,
              device):
    # Set Network Properties
    net = net.train()

    # Stat
    epoch_loss = 0
    n_batches = 0

    # Iterate through batches
    for data in tqdm(dataloader):
        trainX, trainY = data

        # Optimization
        optimizer.zero_grad()

        out = net(trainX.to(device))

        # Calculate Loss
        if args.DPD_model == 'gmp':
            targetY, _ = gmp_input(out, out, args.frame_length, args.degree)
            out = pa_net(targetY.to(device))
            targetY = torch.zeros(trainY.size(0), 2)
            targetY[:, 0] = torch.real(trainY)
            targetY[:, 1] = torch.imag(trainY)
            loss = criterion(out, targetY[:len(out), :])
        else:
            i_x = out[:, :, 0].unsqueeze(-1)
            q_x = out[:, :, 1].unsqueeze(-1)
            amp2 = torch.pow(i_x, 2) + torch.pow(q_x, 2)
            amp = torch.sqrt(amp2)
            amp3 = torch.pow(amp, 3)
            angle = torch.angle(i_x + 1j * q_x)
            cos = torch.div(i_x, amp)
            sin = torch.div(q_x, amp)
            if args.PA_backbone == 'lstm' or args.PA_backbone == 'gru' or args.PA_backbone == 'fc':
                Feat = torch.cat((out, amp, amp3, sin, cos), dim=-1)
            elif args.PA_backbone == 'pgjanet' or args.PA_backbone == 'dvrjanet':
                Feat = torch.cat((amp, angle), dim=-1)
            elif args.PA_backbone == 'vdlstm':
                Feat = torch.cat((amp, angle), dim=-1)
            elif args.PA_backbone == 'cnn2d':
                Feat = torch.cat((out, amp, amp2, amp3), dim=-1)
            elif args.PA_backbone == 'rgru':
                Feat = torch.cat((out, amp, amp3, sin, cos), dim=-1)
            else:
                Feat = out
            if args.norm == True:
                Feat -= net.X_train_mean.to(device)
                Feat /= net.X_train_std.to(device)
            out = pa_net(Feat.to(device))
            if args.DPD_model == 'cnn2d' or args.PA_backbone == 'cnn1d':
                loss = criterion(out, trainY[:, :args.paoutput_len, :].to(device))
            else:
                loss = criterion(out, trainY.to(device))
        # Backward propagation
        loss.backward()

        # Update parameters
        optimizer.step()

        # Increment monitoring variables
        loss.detach()
        batch_loss = loss.item()
        epoch_loss += batch_loss  # Accumulate loss
        n_batches += 1
    # Average loss and regularizer values across batches

    epoch_loss /= n_batches
    epoch_loss = epoch_loss
    return net, epoch_loss


def net_eval(args,
             net,
             pa_net,
             criterion,
             dataloader,
             device):
    with torch.no_grad():
        # Set Network Properties
        net = net.eval()

        # Statistics
        prediction = []
        ground_truth = []

        # Batch Iteration
        for data in tqdm(dataloader):
            testX, testY = data
            if args.DPD_model == 'gmp':
                testout = net(testX.to(device))
                targetY, _ = gmp_input(testout, testout, args.frame_length, args.degree)
                testout = pa_net(targetY.to(device))
                targetY = torch.zeros(testY.size(0), 2)
                targetY[:, 0] = torch.real(testY)
                targetY[:, 1] = torch.imag(testY)
                prediction.append(testout)
                ground_truth.append(targetY[:len(testout), :])
            else:
                testout = net(testX.to(device))
                i_x = testout[:, :, 0].unsqueeze(-1)
                q_x = testout[:, :, 1].unsqueeze(-1)
                amp2 = torch.pow(i_x, 2) + torch.pow(q_x, 2)
                amp = torch.sqrt(amp2)
                amp3 = torch.pow(amp, 3)
                angle = torch.angle(i_x + 1j * q_x)
                cos = torch.div(i_x, amp)
                sin = torch.div(q_x, amp)
                if args.PA_backbone == 'lstm' or args.PA_backbone == 'gru' or args.PA_backbone == 'fc':
                    # Feat = torch.cat((testout, amp, angle), dim=-1)
                    Feat = testout
                    # Feat = torch.cat((testout, amp, amp3, sin, cos), dim=-1)
                elif args.PA_backbone == 'pgjanet' or args.PA_backbone == 'dvrjanet':
                    Feat = torch.cat((amp, angle), dim=-1)
                elif args.PA_backbone == 'vdlstm':
                    Feat = torch.cat((amp, angle), dim=-1)
                elif args.PA_backbone == 'cnn2d':
                    Feat = torch.cat((testout, amp, amp2, amp3), dim=-1)
                elif args.PA_backbone == 'rgru':
                    Feat = torch.cat((testout, amp, amp3, sin, cos), dim=-1)
                else:
                    Feat = testout
                if args.norm == True:
                    Feat -= net.X_train_mean.to(device)
                    Feat /= net.X_train_std.to(device)
                testout = pa_net(Feat.to(device))
                testout = testout.cpu()
                if args.norm == True:
                    testY *= net.y_train_std.cpu()
                    testY += net.y_train_mean.cpu()
                    testout *= net.y_train_std.cpu()
                    testout += net.y_train_mean.cpu()

                if args.DPD_model == 'cnn2d' or args.PA_backbone == 'cnn1d':
                    ground_truth.append(testY[:, args.paoutput_len - 1, :])
                    prediction.append(testout[:, args.paoutput_len - 1, :])
                else:
                    ground_truth.append(testY)
                    prediction.append(testout)

    prediction = torch.cat(prediction, dim=0).numpy()
    ground_truth = torch.cat(ground_truth, dim=0).numpy()

    i_pre = prediction[:, 0]
    i_true = ground_truth[:, 0]
    q_pre = prediction[:, 1]
    q_true = ground_truth[:, 1]
    amp_true = np.sqrt(np.square(i_true) + np.square(q_true))
    # amp_pre = np.sqrt(np.square(i_pre) + np.square(q_pre))
    # phase_pre = get_phase(i_pre, q_pre)
    # phase_true = get_phase(ground_truth[:, 0], ground_truth[:, 1])

    MSE = (np.square(i_true - i_pre) + np.square(q_true - q_pre)).mean()

    NMSE = 10 * np.log10(MSE / np.square(amp_true).mean())
    return net, NMSE, prediction, ground_truth


def gen_log_stat(args, net, optimizer, epoch, train_stat=None, val_stat=None, test_stat=None):
    # Get Epoch & Batch Size
    n_epochs = args.n_epochs
    batch_size = args.batch_size

    # Get current learning rate
    lr_curr = 0
    if optimizer is not None:
        for param_group in optimizer.param_groups:
            lr_curr = param_group['lr']

    # Get parameter count
    n_param = 0
    for name, param in net.named_parameters():
        sizes = 1
        for el in param.size():
            sizes = sizes * el
        n_param += sizes

    # Create log dictionary
    log_stat = {'Epoch': epoch,
                ''
                'val_NMSE': val_stat,
                'test_NMSE': test_stat,
                'train_loss_epoch': train_stat,
                'N_EPOCH': n_epochs,
                'BATCH_SIZE': batch_size,
                'N_PARAM': n_param,
                'LR': lr_curr,
                'Frame_Length': args.frame_length,
                'DPD_model': args.DPD_model,
                'DPD_input_size': args.DPD_input_size,
                'DPD_hidden_size': args.DPD_hidden_size,
                }

    return log_stat


def get_phase(i, q):
    amp = np.sqrt(np.square(i) + np.square(q))
    arccos = np.arccos(np.divide(i, amp))
    phase = np.zeros_like(i)
    phase[q >= 0] = arccos[q >= 0]
    phase[q < 0] = (2 * np.pi - arccos)[q < 0]
    return phase


def gmp_input(X, y, frame_length, degree):
    Complex_In = X[:, 0] + 1j * X[:, 1]
    Complex_Out = y[:, 0] + 1j * y[:, 1]
    Input_matrix = torch.complex(torch.zeros(len(y) - frame_length, frame_length),
                                 torch.zeros(len(y) - frame_length, frame_length))
    degree_matrix = torch.complex(torch.zeros(len(y) - frame_length, frame_length * frame_length),
                                  torch.zeros(len(y) - frame_length, frame_length * frame_length))
    for i in range(0, len(y) - frame_length):
        Input_matrix[i, :] = Complex_In[i:i + frame_length]
    for j in range(1, degree):
        for h in range(frame_length):
            for k in range(frame_length):
                degree_matrix[:, k * frame_length + h] = Input_matrix[:, k] * torch.pow(abs(Input_matrix[:, h]), j)
        Input_matrix = torch.cat((Input_matrix, degree_matrix), dim=1)
    b_output = Complex_Out[:len(y) - frame_length]
    b_input = Input_matrix
    return b_input, b_output
