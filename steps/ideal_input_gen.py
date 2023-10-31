# import os
# import numpy as np
# import pandas as pd
# import torch
# from utils import fft_wrappers
# import models as model
# import matplotlib.pyplot as plt
# import importlib
# from modules.paths import gen_paths, count_net_params
#
#
# def main(args, In, Out):
#     module_log = importlib.import_module('modules.log')
#
#     # Assign methods to be used
#     gen_model_id = module_log.gen_model_id
#     ###########################################################################################################
#     # Network Settings
#     ###########################################################################################################
#     # Instantiate Model
#     def extract_feature(X,PA_model_type):
#         i_x = X[:, 0].unsqueeze(-1)
#         q_x = X[:, 1].unsqueeze(-1)
#         amp2 = torch.pow(i_x, 2) + torch.pow(q_x, 2)
#         amp = torch.sqrt(amp2)
#         amp3 = torch.pow(amp, 3)
#         angle = torch.angle(i_x + 1j * q_x)
#         cos = torch.div(i_x, amp)
#         sin = torch.div(q_x, amp)
#         if PA_model_type == 'lstm' or PA_model_type == 'gru' or PA_model_type == 'fc':
#             Feat = X
#         elif PA_model_type == 'pgjanet' or PA_model_type == 'dvrjanet':
#             Feat = torch.cat((amp, angle), dim=-1)
#         elif PA_model_type == 'vdlstm':
#             Feat = torch.cat((amp, angle), dim=-1)
#         elif PA_model_type == 'cnn2d':
#             Feat = torch.cat((X, amp, amp2, amp3),dim=-1)
#         elif PA_model_type == 'rgru':
#             Feat = torch.cat((X, amp, amp3, sin, cos), dim=-1)
#         else:
#             Feat = X
#         return Feat
#     frame = len(In)
#     if args.PA_RNN_type == 'gmp':
#         Input,_ = gmp_in(In, In, args.frame_length,args.degree)
#         frame = args.frame_length
#     else:
#         Input = extract_feature(In, args.DPD_RNN_type)
#         Out = Out.unsqueeze(0)
#         Input = Input.unsqueeze(0)
#
#     y_train_mean = torch.zeros_like(torch.mean(args.gain * In, dim=0))
#     y_train_std = torch.ones_like(torch.std(args.gain * In, dim=0))
#     DPD_target_mean = torch.zeros_like(torch.mean(Input, dim=0))
#     DPD_target_std = torch.ones_like(torch.std(Input, dim=0))
#
#     if args.norm == True:
#         y_train_mean = torch.mean(args.gain * In, dim=1)
#         y_train_std = torch.std(args.gain * In, dim=1)
#         DPD_target_mean = torch.mean(In, dim=1).unsqueeze(0)
#         DPD_target_std = torch.std(In, dim=1).unsqueeze(0)
#         # Normalize Training Data
#         X_train_mean = torch.mean(Input, dim=1).unsqueeze(0)
#         X_train_std = torch.std(Input, dim=1).unsqueeze(0)
#         In -= X_train_mean
#         In /= X_train_std
#
#     if args.DPD_RNN_type == 'cnn1d':
#         dpdoutput_len = args.paoutput_len + args.pa_cnn_memory + args.PA_hidden_size - 2
#     elif args.DPD_RNN_type == 'cnn2d':
#         dpdoutput_len = args.paoutput_len + args.pa_cnn_memory + args.DPD_CNN_W - 2
#     else:
#         dpdoutput_len = 0
#
#     PA_CNN_setup = [args.PA_CNN_H, args.PA_CNN_W]
#     DPD_CNN_setup = [args.DPD_CNN_H, args.DPD_CNN_W]
#     pa = model.CoreModel(input_size=args.PA_input_size,
#                          cnn_memory=args.pa_cnn_memory,
#                          pa_output_len=args.paoutput_len,
#                          cnn_set=PA_CNN_setup,
#                          frame_len=args.frame_length,
#                          hidden_size=args.PA_hidden_size,
#                          num_layers=1,
#                          degree=args.degree,
#                          backbone_type=args.PA_RNN_type, y_train_mean=y_train_mean, y_train_std=y_train_std)
#
#
#     net = model.DPD_MODEL(input_size=args.DPD_input_size,
#                           cnn_memory=args.dpd_cnn_memory,
#                           dpdoutput_len=dpdoutput_len,
#                           cnn_set=DPD_CNN_setup,
#                           frame_len=args.frame_length,
#                           hidden_size=args.DPD_hidden_size,
#                           num_layers=1,
#                           degree=args.degree,
#                           rnn_type=args.DPD_RNN_type, X_train_mean=DPD_target_mean,
#                           X_train_std=DPD_target_std,
#                           y_train_mean=y_train_mean, y_train_std=y_train_std)
#
#     pamodel_id, dpdmodel_id = gen_model_id(args)
#     pa_dict_path = os.path.join('save', args.dataset_name, args.PA_model_phase, pamodel_id+'.pt')
#     pa.load_state_dict(torch.load(pa_dict_path))
#     for weight in pa.parameters():
#         weight.requires_grad = False
#     dpd_dict_path = os.path.join('save', args.dataset_name, args.phase, dpdmodel_id+'.pt')
#     net.load_state_dict(torch.load(dpd_dict_path))
#     for weight in net.parameters():
#         weight.requires_grad = False
#     ii = net(Input)
#     if args.PA_RNN_type == 'gmp':
#         i_input = pd.DataFrame({'I': ii[:, 0], 'Q': ii[:, 1]})
#         i_dict = os.path.join('datasets/', dpdmodel_id+'.csv')
#         i_input.to_csv(i_dict, index=False)
#         ideal_input, _ = gmp_in(ii, ii, args.frame_length, args.degree)
#         out = pa(ideal_input)
#     else:
#         ideal_input = pd.DataFrame({'I': ii[0, :, 0], 'Q': -ii[0, :, 1]})
#         i_dict = os.path.join('datasets', dpdmodel_id+'.csv')
#         ideal_input.to_csv(i_dict, index=False)
#         i_x = ii[:, :, 0].unsqueeze(-1)
#         q_x = ii[:, :, 1].unsqueeze(-1)
#         amp2 = torch.pow(i_x, 2) + torch.pow(q_x, 2)
#         amp = torch.sqrt(amp2)
#         amp3 = torch.pow(amp, 3)
#         angle = torch.angle(i_x + 1j * q_x)
#         cos = torch.div(i_x, amp)
#         sin = torch.div(q_x, amp)
#         if args.PA_RNN_type == 'lstm' or args.PA_RNN_type == 'gru' or args.PA_RNN_type == 'fc':
#             Feat = ii
#         elif args.PA_RNN_type == 'pgjanet' or args.PA_RNN_type == 'dvrjanet':
#             Feat = torch.cat((amp, angle), dim=-1)
#         elif args.PA_RNN_type == 'vdlstm':
#             Feat = torch.cat((amp, angle), dim=-1)
#         elif args.PA_RNN_type == 'cnn2d':
#             Feat = torch.cat((ii, amp, amp2, amp3), dim=-1)
#         elif args.PA_RNN_type == 'rgru':
#             Feat = torch.cat((ii, amp, amp3, sin, cos), dim=-1)
#         else:
#             Feat = ii
#     if args.norm == True:
#         Feat -= net.X_train_mean
#         Feat /= net.X_train_std
#     out = pa(Feat)
#     out = out.squeeze(0)
#     if args.norm == True:
#         out *= pa.y_train_std
#         out += pa.y_train_mean
#     ofdm_time = out[:, 0] + 1j * out[:, 1]
#     h_rrc_f_freq, h_rrc_f = fft_wrappers.fft_wrapper(ofdm_time, args.input_signal_fs, shift=True)
#     h_rrc_f_mag = abs(h_rrc_f)
#     h_rrc_f_norm = h_rrc_f_mag / np.max(h_rrc_f_mag)
#     psd = 20 * np.log10(h_rrc_f_norm)
#     plt.plot(h_rrc_f_freq, psd)
#     plt.show()
#     index_left1 = np.min(np.where(h_rrc_f_freq >= -args.input_signal_bw/2))
#     index_left2 = np.max(np.where(h_rrc_f_freq <= -args.input_signal_bw/2+args.input_signal_bw/10))
#     index_right1 = np.min(np.where(h_rrc_f_freq >= args.input_signal_bw / 2-args.input_signal_bw / 10))
#     index_right2 = np.max(np.where(h_rrc_f_freq <= args.input_signal_bw / 2 ))
#     channel_pow1 = np.sum(h_rrc_f_mag[index_left1:index_left2])
#     channel_pow2 = np.sum(h_rrc_f_mag[index_right1:index_right2])
#     long1 = index_left2 - index_left1
#     long2 = index_right2 - index_right1
#     left_pow = np.sum(h_rrc_f_mag[index_left1 - long1:index_left1])
#     ACPRL = 20 * np.log10(left_pow / channel_pow1)
#     right_pow = np.sum(h_rrc_f_mag[index_right2:index_right2 + long2])
#     ACPRR = 20 * np.log10(right_pow / channel_pow2)
#
#     return ACPRL,ACPRR
#
# def gmp_in(X, y, frame_length, degree):
#     Complex_In = X[:, 0] + 1j * X[:, 1]
#     Complex_Out = y[:, 0] + 1j * y[:, 1]
#     Input_matrix = torch.complex(torch.zeros(len(y) - frame_length, frame_length),
#                                     torch.zeros(len(y) - frame_length, frame_length))
#     degree_matrix = torch.complex(torch.zeros(len(y) - frame_length, frame_length * frame_length),
#                                     torch.zeros(len(y) - frame_length, frame_length * frame_length))
#     for i in range(0, len(y) - frame_length):
#         Input_matrix[i, :] = Complex_In[i:i + frame_length]
#     for j in range(1, degree):
#         for h in range(frame_length):
#             for k in range(frame_length):
#                 degree_matrix[:, k * frame_length + h] = Input_matrix[:, k] * torch.pow(abs(Input_matrix[:, h]), j)
#         Input_matrix = torch.cat((Input_matrix, degree_matrix), dim=1)
#     b_output = Complex_Out[:len(y) - frame_length]
#     b_input = Input_matrix
#     return b_input, b_output