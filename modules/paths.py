import argparse
import os
import typing


def gen_log_stat(args: argparse.Namespace, elapsed_time, net, optimizer, epoch, train_stat=None, val_stat=None,
                 test_stat=None):
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

    if args.step == 'train_pa':
        backbone = args.PA_backbone
        hidden_size = args.PA_hidden_size
    elif args.step == 'train_dpd':
        backbone = args.DPD_backbone
        hidden_size = args.DPD_hidden_size

    # Create log dictionary
    log_stat = {'EPOCH': epoch,
                'N_EPOCH': n_epochs,
                'TIME:': elapsed_time,
                'LR': lr_curr,
                'BATCH_SIZE': batch_size,
                'N_PARAM': n_param,
                'FRAME_LENGTH': args.frame_length,
                'BACKBONE': backbone,
                'HIDDEN_SIZE': hidden_size,
                }

    # Merge stat dicts into the log dict
    if train_stat is not None:
        train_stat_log = {f'TRAIN_{k.upper()}': v for k, v in train_stat.items()}
        log_stat = {**log_stat, **train_stat_log}
    if val_stat is not None:
        val_stat_log = {f'VAL_{k.upper()}': v for k, v in val_stat.items()}
        log_stat = {**log_stat, **val_stat_log}
    if test_stat is not None:
        test_stat_log = {f'TEST_{k.upper()}': v for k, v in test_stat.items()}
        log_stat = {**log_stat, **test_stat_log}

    return log_stat


def gen_dir_paths(args: argparse.Namespace):
    path_dir_save = os.path.join('./save', args.dataset_name, args.step)  # Best model save dir
    path_dir_log_hist = os.path.join('./log', args.dataset_name, args.step, 'history')  # Log dir to save training history
    path_dir_log_best = os.path.join('./log', args.dataset_name, args.step, 'best')  # Log dir to save info of the best epoch
    dir_paths = (path_dir_save, path_dir_log_hist, path_dir_log_best)
    return dir_paths


def gen_file_paths(path_dir_save: str, path_dir_log_hist: str, path_dir_log_best: str, model_id: str):
    # File Paths
    path_file_save = os.path.join(path_dir_save, model_id + '.pt')
    path_file_log_hist = os.path.join(path_dir_log_hist, model_id + '.csv')  # .csv path_log_file_hist
    path_file_log_best = os.path.join(path_dir_log_best, model_id + '.csv')  # .csv path_log_file_hist
    file_paths = (path_file_save, path_file_log_hist, path_file_log_best)
    return file_paths


def create_folder(folder_list):
    for folder in folder_list:
        try:
            os.makedirs(folder)
        except:
            pass
