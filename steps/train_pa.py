__author__ = "Yizhuo Wu, Chang Gao"
__license__ = "Apache-2.0 License"
__email__ = "yizhuo.wu@tudelft.nl, chang.gao@tudelft.nl"

import models as model
from project import Project
from utils.util import count_net_params


def main(proj: Project):
    ###########################################################################################################
    # Initialization
    ###########################################################################################################
    # Set Accelerator Device
    proj.set_device()

    # Build Dataloaders
    (train_loader, val_loader, test_loader), input_size = proj.build_dataloaders()

    ###########################################################################################################
    # Network Settings
    ###########################################################################################################
    # Instantiate Model
    net = model.CoreModel(input_size=input_size,
                          hidden_size=proj.PA_hidden_size,
                          num_layers=proj.PA_num_layers,
                          backbone_type=proj.PA_backbone,
                          window_size=proj.window_size,
                          num_dvr_units=proj.num_dvr_units,
                          thx=proj.thx,
                          thh=proj.thh)
    n_net_pa_params = count_net_params(net)
    print("::: Number of PA Model Parameters: ", n_net_pa_params)
    pa_model_id = proj.gen_pa_model_id(n_net_pa_params)

    # Move the network to the proper device
    net = net.to(proj.device)

    ###########################################################################################################
    # Logger, Loss and Optimizer Settings
    ###########################################################################################################
    # Build Logger
    proj.build_logger(model_id=pa_model_id)

    # Select Loss function
    criterion = proj.build_criterion()

    # Create Optimizer and Learning Rate Scheduler
    optimizer, lr_scheduler = proj.build_optimizer(net=net)

    ###########################################################################################################
    # Plotting Setup
    ###########################################################################################################
    plot_dir = None
    input_data_val = None
    input_data_test = None
    full_input_iq = None
    full_output_iq = None
    if proj.plot:
        from utils.plotting import get_plot_dir_train_pa, needs_full_seq_constellation, load_full_dataset_iq
        plot_dir = get_plot_dir_train_pa(proj.dataset_name, pa_model_id)
        # Extract raw input data from val/test loaders for plotting
        import torch
        import numpy as np
        if proj.eval_val:
            input_data_val = np.concatenate([b[0].numpy() for b in val_loader], axis=0)
        if proj.eval_test:
            input_data_test = np.concatenate([b[0].numpy() for b in test_loader], axis=0)
        # Load full dataset for APA constellation plotting
        if needs_full_seq_constellation(proj.dataset_name):
            full_input_iq, full_output_iq = load_full_dataset_iq(proj.dataset_name)

    # Build metadata for dashboard
    metadata = {
        'dataset': proj.dataset_name,
        'step': 'train_pa',
        'backbone': proj.PA_backbone.upper(),
        'hidden_size': proj.PA_hidden_size,
        'n_params': n_net_pa_params,
        'model_id': pa_model_id,
    }

    ###########################################################################################################
    # Training
    ###########################################################################################################
    proj.train(net=net,
               criterion=criterion,
               optimizer=optimizer,
               lr_scheduler=lr_scheduler,
               train_loader=train_loader,
               val_loader=val_loader,
               test_loader=test_loader,
               best_model_metric='NMSE',
               plot_dir=plot_dir,
               input_data_val=input_data_val,
               input_data_test=input_data_test,
               metadata=metadata,
               full_input_iq=full_input_iq,
               full_output_iq=full_output_iq)
