__author__ = "Yizhuo Wu, Chang Gao"
__license__ = "Apache-2.0 License"
__email__ = "yizhuo.wu@tudelft.nl, chang.gao@tudelft.nl"

import argparse
import os
import warnings


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
    
    # Record delta thresholds for both PA and DPD training.  They are runtime
    # model settings rather than checkpoint tensors, so the CSV evidence is
    # what lets a benchmark prove which thresholds were actually used.
    delta_model = None
    if args.step == 'train_pa' and 'delta' in net.backbone_type:
        delta_model = net
    elif args.step == 'train_dpd' and 'delta' in net.dpd_model.backbone_type:
        delta_model = net.dpd_model

    if delta_model is not None:
        log_stat['THX'] = delta_model.backbone.thx
        log_stat['THH'] = delta_model.backbone.thh
            
        # Statistics are opt-in because collecting them in an eager recurrent
        # cell adds reductions at every timestep.
        if (
            getattr(delta_model.backbone.rnn, 'debug', 0)
            and hasattr(delta_model.backbone, 'get_temporal_sparsity')
            and hasattr(delta_model.backbone, 'set_debug')
        ):
            sparsity_metrics = delta_model.backbone.get_temporal_sparsity()
            sparsity_log = {f'{k}': v for k, v in sparsity_metrics.items()}
            log_stat.update(sparsity_log)
            delta_model.backbone.set_debug(1)

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
    if args.step == 'train_pa':
        path_dir_save = os.path.join('./save', args.dataset_name, args.step, args.quant_dir_label)
        path_dir_log_hist = os.path.join('./log', args.dataset_name, args.step, args.quant_dir_label, 'history')
        path_dir_log_best = os.path.join('./log', args.dataset_name, args.step, args.quant_dir_label, 'best')
    elif args.step == 'train_dpd' or 'run_dpd':
        # Organize DPD files under PA model directory
        path_dir_save = os.path.join('./save', args.dataset_name, args.step, gen_pa_model_id(args), args.quant_dir_label)
        path_dir_log_hist = os.path.join('./log', args.dataset_name, args.step, gen_pa_model_id(args), args.quant_dir_label, 'history')
        path_dir_log_best = os.path.join('./log', args.dataset_name, args.step, gen_pa_model_id(args),args.quant_dir_label, 'best')
    dir_paths = (path_dir_save, path_dir_log_hist, path_dir_log_best)
    return dir_paths


def gen_file_paths(path_dir_save: str, path_dir_log_hist: str, path_dir_log_best: str, model_id: str):
    # File Paths
    path_file_save = os.path.join(path_dir_save, model_id + '.pt')
    path_file_log_hist = os.path.join(path_dir_log_hist, model_id + '.csv')  # .csv path_log_file_hist
    path_file_log_best = os.path.join(path_dir_log_best, model_id + '.csv')  # .csv path_log_file_hist
    file_paths = (path_file_save, path_file_log_hist, path_file_log_best)
    return file_paths


def warn_if_model_artifacts_exist(model_id: str, file_paths):
    """Warn before an architecture-only model ID reuses existing artifacts.

    Model IDs are part of the public checkpoint lookup convention, so changing
    their format would break existing consumers.  They do not, however, encode
    the full training recipe.  An explicit warning keeps that convention while
    making a potentially destructive recipe collision visible before training.
    """
    existing_paths = [path for path in file_paths if os.path.exists(path)]
    if not existing_paths:
        return

    formatted_paths = ", ".join(os.path.normpath(path) for path in existing_paths)
    warnings.warn(
        (
            f"Existing artifact(s) for model ID '{model_id}' will be reused or "
            f"overwritten: {formatted_paths}. This model ID does not encode the "
            "full training recipe (for example optimizer, learning rate, "
            "schedule, batch size, and epoch count), so the existing artifacts "
            "may come from a different recipe. Move or rename them before "
            "continuing if they must be preserved."
        ),
        RuntimeWarning,
        stacklevel=2,
    )


def create_folder(folder_list):
    for folder in folder_list:
        try:
            os.makedirs(folder)
        except:
            pass
def gen_pa_model_id(args):
    dict_pa = {'S': f"{args.seed}",
               'M': args.PA_backbone.upper(),
               'H': f"{args.PA_hidden_size:d}",
               'F': f"{args.frame_length:d}",
               }
    dict_pamodel_id = dict(list(dict_pa.items()))

    # PA Model ID
    list_pamodel_id = []
    for item in list(dict_pamodel_id.items()):
        list_pamodel_id += list(item)
    pa_model_id = '_'.join(list_pamodel_id)
    pa_model_id = 'PA_' + pa_model_id
    return pa_model_id
