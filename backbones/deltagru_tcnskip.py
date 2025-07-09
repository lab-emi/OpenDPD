import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import csv
import os
from quant.modules.ops import Add, Mul
from torch import nn

class DeltaGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, thx=0, thh=0,
                 bias=True):
        super(DeltaGRU, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = 6
        self.output_size = output_size
        self.num_layers = num_layers
        self.thh = thh
        self.thx = thx
        self.bias = bias

        # Instantiate NN Layers
        self.rnn =  DeltaGRULayer(input_size=self.input_size,
                          hidden_size=self.hidden_size,
                          num_layers=num_layers,
                          thx=self.thx,
                          thh=self.thh)
        self.fc_out = nn.Linear(in_features=self.hidden_size,
                                out_features=self.output_size,
                                bias=False)
        self.tcn = nn.Sequential(
            nn.Conv1d(in_channels=2,
                      out_channels=3,
                      kernel_size=3,
                      padding=16,
                      stride=1,
                      dilation=16,
                      bias=False),
            nn.Hardswish(),
            nn.Conv1d(in_channels=3,
                      out_channels=2,
                      kernel_size=1,
                      padding=0,
                      stride=1,
                      dilation=1,
                      bias=False),
            nn.Hardswish(),
        )
        self.set_debug(1)

    def set_debug(self, value):
        setattr(self, "debug", value)
        self.rnn.statistics = {
            "num_dx_zeros": 0,
            "num_dx_numel": 0,
            "num_dh_zeros": 0,
            "num_dh_numel": 0
        }

    def enable_neuron_stats_saving(self, enable=True):
        """Enable or disable saving of neuron statistics to CSV file."""
        setattr(self, "save_neuron_stats", enable)

    def reset_parameters(self):
        for name, param in self.tcn.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            if 'bias' in name:
                nn.init.constant_(param, 0)
        for name, param in self.rnn.named_parameters():
            num_gates = int(param.shape[0] / self.hidden_size)
            if 'bias' in name:
                nn.init.constant_(param, 0)
            if 'weight' in name:
                for i in range(0, num_gates):
                    nn.init.orthogonal_(param[i * self.hidden_size:(i + 1) * self.hidden_size, :])
            if 'x2h.weight' in name:
                for i in range(0, num_gates):
                    nn.init.xavier_uniform_(param[i * self.hidden_size:(i + 1) * self.hidden_size, :])

        for name, param in self.fc_out.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            if 'bias' in name:
                nn.init.constant_(param, 0)
        

    def forward(self, x, h_0):
        skip_x = self.tcn(x.transpose(1,2)).transpose(1,2)
        last_step = torch.roll(x, shifts=-1, dims=1)
        # Feature Extraction
        i_x = x[..., 0].unsqueeze(-1)
        q_x = x[..., 1].unsqueeze(-1)
        i_last = last_step[..., 0].unsqueeze(-1)
        q_last = last_step[..., 1].unsqueeze(-1)
        amp2 = torch.pow(i_x, 2) + torch.pow(q_x, 2)
        amp = torch.sqrt(amp2)
        amp3 = torch.pow(amp, 3)
        x = torch.cat((i_x, q_x, amp, amp3, i_last, q_last), dim=-1)
        out = self.rnn(x, h_0)   
        out = self.fc_out(out)+skip_x
        return out

    def get_temporal_sparsity(self):
        temporal_sparsity = {}
        if self.rnn.debug:
            # Get RNN layer sparsity
            num_dx_zeros = self.rnn.statistics["num_dx_zeros"]
            num_dx_numel = self.rnn.statistics["num_dx_numel"]
            num_dh_zeros = self.rnn.statistics["num_dh_zeros"]
            num_dh_numel = self.rnn.statistics["num_dh_numel"]
            
            # Calculate RNN parameters by summing weights from all layers
            rnn_numel = sum(p.numel() for name, p in self.rnn.named_parameters() if 'weight' in name)
            rnn_bias_numel = sum(p.numel() for name, p in self.rnn.named_parameters() if 'bias' in name)
            fc_numel = sum(p.numel() for name, p in self.fc_out.named_parameters())+sum(p.numel() for name, p in self.tcn.named_parameters())
            total_numel = num_dx_numel + num_dh_numel 
            total_zeros = num_dx_zeros + num_dh_zeros 
            
            temporal_sparsity["SP_T_DX"] = float(num_dx_zeros / num_dx_numel)
            temporal_sparsity["SP_T_DH"] = float(num_dh_zeros / num_dh_numel)
            temporal_sparsity["SP_T_DV"] = float(total_zeros / total_numel)
            temporal_sparsity["HW_PARAM"] = float(fc_numel + rnn_numel * (1-float(total_zeros / total_numel))+rnn_bias_numel)
            
        return temporal_sparsity


class DeltaGRULayer(nn.Module):
    def __init__(self,
                 input_size=16,
                 hidden_size=256,
                 num_layers=1,
                 thx=0.1,
                 thh=0):
        super(DeltaGRULayer, self).__init__()

        # Hyperparameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.th_x = thx
        self.th_h = thh
        self.weight_ih_height = 3 * self.hidden_size
        self.weight_ih_width = self.input_size
        self.weight_hh_width = self.hidden_size
        self.x_p_length = max(self.input_size, self.hidden_size)
        self.batch_first = True
        self.debug = 1
        # Statistics
        self.abs_sum_delta_hid = torch.zeros(1)
        self.sp_dx = 0
        self.sp_dh = 0

        # Replace weight matrices with Linear layers
        self.x2h = nn.Linear(self.weight_ih_width , self.weight_ih_height, bias=False)
        self.h2h = nn.Linear(self.weight_hh_width, self.weight_ih_height, bias=False)

        self.add = Add()
        self.mul = Mul()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()



    def add_to_debug(self, x, i_layer, name):
        if self.debug:
            if isinstance(x, Tensor):
                variable = np.squeeze(x.cpu().numpy())
            else:
                variable = np.squeeze(np.asarray(x))
            variable_name = '_'.join(['l' + str(i_layer), name])
            if variable_name not in self.statistics.keys():
                self.statistics[variable_name] = []
            self.statistics[variable_name].append(variable)

    def reset_parameters(self):
        for name, param in self.rnn.named_parameters():
            num_gates = int(param.shape[0] / self.hidden_size)
            if 'bias' in name:
                nn.init.constant_(param, 0)
            if 'weight' in name:
                for i in range(0, num_gates):
                    nn.init.orthogonal_(param[i * self.hidden_size:(i + 1) * self.hidden_size, :])
            if 'x2h.weight' in name:
                for i in range(0, num_gates):
                    nn.init.xavier_uniform_(param[i * self.hidden_size:(i + 1) * self.hidden_size, :])
            
        for name, param in self.fc_out.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)


    def process_inputs_first(self, x: Tensor, x_p_0: Tensor = None, h_0: Tensor = None, h_p_0: Tensor = None,
                       dm_ch_0: Tensor = None, dm_0: Tensor = None):
        if self.batch_first:
            x = x.transpose(0, 1)
            setattr(self, 'batch_size', int(x.size()[0]))
        else:
            setattr(self, 'batch_size', int(x.size()[1]))
        batch_size = x.size()[1]

        if x_p_0 is None or h_0 is None or h_p_0 is None or dm_ch_0 is None or dm_0 is None:
            x_p_0 = torch.zeros(self.num_layers, batch_size, self.x_p_length, dtype=x.dtype, device=x.device)
            h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, dtype=x.dtype, device=x.device)
            h_p_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, dtype=x.dtype, device=x.device)
            dm_ch_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, dtype=x.dtype, device=x.device)
            dm_0 = torch.zeros(self.num_layers, batch_size, self.weight_ih_height, dtype=x.dtype, device=x.device)
        return x, x_p_0, h_0, h_p_0, dm_ch_0, dm_0

    @staticmethod
    def compute_deltas(x: Tensor, x_p: Tensor, h: Tensor, h_p: Tensor, th_x: Tensor, th_h: Tensor):
        delta_x = x - x_p
        delta_h = h - h_p

        delta_x_abs = torch.abs(delta_x)
        delta_x = delta_x.masked_fill(delta_x_abs < th_x, 0)

        delta_h_abs = torch.abs(delta_h)
        delta_h = delta_h.masked_fill(delta_h_abs < th_h, 0)

        return delta_x, delta_h, delta_x_abs, delta_h_abs

    @staticmethod
    def update_states(delta_x_abs, delta_h_abs, x, h, x_p, h_p, th_x, th_h):
        x_p = torch.where(delta_x_abs >= th_x, x, x_p)
        h_p = torch.where(delta_h_abs >= th_h, h, h_p)
        return x_p, h_p

    def compute_gates(self, delta_x: Tensor, delta_h: Tensor, dm: Tensor, dm_nh: Tensor):
        mac_x = self.x2h(delta_x) + dm
        mac_h = self.h2h(delta_h)
        mac_x_chunks = mac_x.chunk(3, dim=1)
        mac_h_chunks = mac_h.chunk(3, dim=1)
        dm_r = mac_x_chunks[0] + mac_h_chunks[0]
        dm_z = mac_x_chunks[1] + mac_h_chunks[1]
        dm_n = mac_x_chunks[2]
        dm_nh = mac_h_chunks[2] + dm_nh
        dm = torch.cat((dm_r, dm_z, dm_n), 1)
        return dm, dm_r, dm_z, dm_n, dm_nh

    def layer_forward(self, input: Tensor, l: int, x_p_0: Tensor = None, h_0: Tensor = None,
                      h_p_0: Tensor = None, dm_nh_0: Tensor = None, dm_0: Tensor = None):

        input_size = input.size(-1)

        th_x = torch.tensor(self.th_x, dtype=input.dtype)
        th_h = torch.tensor(self.th_h, dtype=input.dtype)

        inputs = input.unbind(0)

        output = []

        reg = torch.zeros(1, dtype=input.dtype, device=input.device).squeeze()

        x_p = x_p_0[:, :input_size]
        h = h_0
        h_p = h_p_0
        dm_nh = dm_nh_0
        dm = dm_0
        l1_norm_delta_h = torch.zeros(1, dtype=input.dtype, device=input.device)

        seq_len = len(inputs)
        for t in range(seq_len):
            x = inputs[t]

            delta_x, delta_h, delta_x_abs, delta_h_abs = self.compute_deltas(x, x_p, h, h_p, th_x, th_h)
            reg += torch.sum(torch.abs(delta_h))

            if self.debug:
                zero_mask_delta_x = torch.as_tensor(delta_x == 0, dtype=x.dtype)
                zero_mask_delta_h = torch.as_tensor(delta_h == 0, dtype=x.dtype)
                self.statistics["num_dx_zeros"] += torch.sum(zero_mask_delta_x)
                self.statistics["num_dh_zeros"] += torch.sum(zero_mask_delta_h)
                self.statistics["num_dx_numel"] += torch.numel(delta_x)
                self.statistics["num_dh_numel"] += torch.numel(delta_h)

            x_p, h_p = self.update_states(delta_x_abs, delta_h_abs, x, h, x_p, h_p, th_x, th_h)

            l1_norm_delta_h += torch.sum(torch.abs(delta_h))

            dm, dm_r, dm_z, dm_n, dm_nh = self.compute_gates(delta_x, delta_h, dm, dm_nh)

            gate_r = self.sigmoid(dm_r)
            gate_z = self.sigmoid(dm_z)

            gate_n = self.tanh(self.add(dm_n, self.mul(gate_r, dm_nh)))
            h = self.add(self.mul(self.add(1, -gate_z), gate_n), self.mul(gate_z, h))
            output += [h]

        output = torch.stack(output)
        return output

    def forward(self, input: Tensor, x_p_0: Tensor = None, h_0: Tensor = None, h_p_0: Tensor = None,
                dm_nh_0: Tensor = None, dm_0: Tensor = None):
        x, x_p_0, h_0, h_p_0, dm_nh_0, dm_0 = self.process_inputs_first(input, x_p_0, h_0, h_p_0, dm_nh_0, dm_0)

        for l in range(self.num_layers):
            x = self.layer_forward(x, l, x_p_0[l], h_0[l],  h_p_0[l], dm_nh_0[l], dm_0[l])

        x = x.transpose(0, 1)
        return x