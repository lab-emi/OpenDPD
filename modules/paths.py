import argparse
import os
import typing


def gen_model_id(args: argparse.Namespace):
    dict_pa = {'M': args.PA_backbone.upper(),
               'I': f"{args.PA_input_size:d}",
               'H': f"{args.PA_hidden_size:d}",
               'F': f"{args.frame_length:d}"
               }
    dict_dpd = {'M': args.DPD_backbone.upper(),
                'I': f"{args.DPD_input_size:d}",
                'H': f"{args.DPD_hidden_size:d}",
                'F': f"{args.frame_length:d}"
                }
    dict_pamodel_id = dict(list(dict_pa.items()))
    dict_dpdmodel_id = dict(list(dict_dpd.items()))

    # PA Model ID
    list_pamodel_id = []
    for item in list(dict_pamodel_id.items()):
        list_pamodel_id += list(item)
    pa_model_id = '_'.join(list_pamodel_id)

    # DPD Model ID
    list_dpdmodel_id = []
    for item in list(dict_dpdmodel_id.items()):
        list_dpdmodel_id += list(item)
    dpd_model_id = '_'.join(list_dpdmodel_id)

    pa_model_id = 'PA_' + pa_model_id
    dpd_model_id = 'DPD_' + dpd_model_id

    return pa_model_id, dpd_model_id


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
        input_size = args.PA_input_size
        hidden_size = args.PA_hidden_size
    elif args.step == 'train_dpd':
        backbone = args.DPD_backbone
        input_size = args.DPD_input_size
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
                'INPUT_SIZE': input_size,
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


def gen_paths(args: argparse.Namespace,
              model_id: typing.AnyStr = None,
              pretrain_model_id: typing.AnyStr = None):
    save_dir = os.path.join('./save', args.dataset_name, args.step)  # Best model save dir
    log_dir_hist = os.path.join('./log', args.dataset_name, args.step, 'history')  # Log dir to save training history
    log_dir_best = os.path.join('./log', args.dataset_name, args.step, 'best')  # Log dir to save info of the best epoch
    log_dir_test = os.path.join('./log', args.dataset_name, args.step, 'test')  # Log dir to save info of the best epoch
    dir_paths = (save_dir, log_dir_hist, log_dir_best, log_dir_test)

    # File Paths
    if model_id is not None:
        logfile_hist = os.path.join(log_dir_hist, model_id + '.csv')  # .csv path_log_file_hist
        logfile_best = os.path.join(log_dir_best, model_id + '.csv')  # .csv path_log_file_hist
        logfile_test = os.path.join(log_dir_test, model_id + '.csv')  # .csv path_log_file_hist
        save_file = os.path.join(save_dir, model_id + '.pt')
    if model_id is not None:
        file_paths = (save_file, logfile_hist, logfile_best, logfile_test)
    else:
        file_paths = None

    # Pretrain Model Path
    if pretrain_model_id is not None:
        pretrain_file = os.path.join('./save', args.dataset_name, 'pretrain', pretrain_model_id + '.pt')
    else:
        pretrain_file = None

    return dir_paths, file_paths, pretrain_file


def create_folder(folder_list):
    for folder in folder_list:
        try:
            os.makedirs(folder)
        except:
            pass
