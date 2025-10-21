"""
Reproduced from: https://ieeexplore.ieee.org/document/10896855/
"""

import torch
from torch import nn


class MCLDNN(nn.Module):
    def __init__(self,hidden_size=8):
        super(MCLDNN, self).__init__()
        self.memory_length = 5
        self.order = 3
        self.input_height = 2+self.order
        self.input_width = self.memory_length
        self.channels = hidden_size
        self.kernel_size = 3



        self.conv2d_1 = nn.Conv2d(1,self.channels,kernel_size=self.kernel_size,padding=1)
        self.conv1d = nn.Conv1d(self.input_height,self.input_height*self.channels,kernel_size=self.kernel_size,padding=1,groups=self.input_height)
        
        self.conv2d_2 = nn.Conv2d(2*self.input_height,1,kernel_size=self.kernel_size,padding=1)
        self.lstm = nn.LSTM(input_size=self.channels*self.memory_length,hidden_size=8,num_layers=1,batch_first=True)
        self.fc_out = nn.Linear(in_features=8,out_features=16)
        self.fc_out_2 = nn.Linear(in_features=16,out_features=2)

        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def count_flops(self, input_shape):
        """
        Calculate FLOPs for MCLDNN model per timestep.
        
        Args:
            input_shape (tuple): Input tensor shape (batch_size, seq_len, 2)
            
        Returns:
            int: Total FLOPs per timestep
        """
        batch_size, seq_len, _ = input_shape
        
        # 1. Feature Extraction
        # i_x, q_x extraction (no FLOPs, just indexing)
        # amp2 = i_x^2 + q_x^2
        amp2_ops = 3  # 2 multiplications + 1 addition
        # amp = sqrt(amp2)
        amp_ops = 1
        # amp3 = amp^3
        amp3_ops = 2  # 2 multiplications
        feature_ops = amp2_ops + amp_ops + amp3_ops
        
        # 2. Conv2d_1: Conv2d(1 -> channels, kernel_size=3, padding=1)
        # Input: (batch * seq_len, 1, input_height=5, memory_length=3)
        # FLOPs per output pixel = in_channels * kernel_h * kernel_w * out_channels
        conv2d_1_flops_per_pixel = 1 * self.kernel_size * self.kernel_size * self.channels
        # Output size: (channels, input_height=5, memory_length=3)
        conv2d_1_output_size = self.input_height * self.memory_length
        conv2d_1_ops = conv2d_1_flops_per_pixel * conv2d_1_output_size
        
        # 3. Conv1d: Conv1d(input_height -> input_height*channels, kernel_size=3, groups=input_height)
        # For grouped convolution: FLOPs = (in_channels/groups) * kernel_size * (out_channels/groups) * groups * output_length
        # = (input_height/input_height) * kernel_size * channels * input_height * memory_length
        # = kernel_size * channels * input_height * memory_length
        conv1d_ops = self.kernel_size * self.channels * self.input_height * self.memory_length
        
        # 4. Conv2d_2: Conv2d(input_height*2 -> 1, kernel_size=3, padding=1)
        # Input after concat: (input_height*2, channels, memory_length)
        # Output: (1, channels, memory_length)
        conv2d_2_input_channels = self.input_height * 2
        conv2d_2_flops_per_pixel = conv2d_2_input_channels * self.kernel_size * self.kernel_size * 1
        conv2d_2_output_size = self.channels * self.memory_length
        conv2d_2_ops = conv2d_2_flops_per_pixel * conv2d_2_output_size
        
        # 5. LSTM: LSTM(input_size=channels*memory_length, hidden_size=8, num_layers=1)
        # LSTM FLOPs per timestep = 4 * (input_size * hidden_size + hidden_size * hidden_size + hidden_size)
        # The 4 comes from the 4 gates (input, forget, cell, output)
        lstm_input_size = self.channels * self.memory_length
        lstm_hidden_size = 8
        lstm_ops = 4 * (lstm_input_size * lstm_hidden_size + lstm_hidden_size * lstm_hidden_size + lstm_hidden_size)
        
        # 6. Linear layers
        # fc_out: Linear(8 -> 16)
        fc_out_ops = 8 * 16
        # fc_out_2: Linear(16 -> 2)
        fc_out_2_ops = 16 * 2
        
        total_flops = (feature_ops + conv2d_1_ops + conv1d_ops + 
                      conv2d_2_ops)*self.memory_length + lstm_ops + fc_out_ops + fc_out_2_ops
        
        return total_flops

    def forward(self, x, h_0):
        batch_size = x.size(0)
        frame_length = x.size(1)
        # x Dim: (batch_size, frame_length, input_size)

        # Feature Extraction
        i_x = torch.unsqueeze(x[..., 0], dim=-1)
        q_x = torch.unsqueeze(x[..., 1], dim=-1)
        amp2 = torch.pow(i_x, 2) + torch.pow(q_x, 2)
        amp = torch.sqrt(amp2)
        amp3 = torch.pow(amp, 3)
        x = torch.cat((i_x, q_x, amp, amp2, amp3), dim=-1)
        feature_size = x.size(2)

        # Split a frame into memory windows
        # zero_pad = torch.zeros((batch_size, self.window_size - 1, feature_size))
        pad = x[:, -(self.memory_length - 1):, :]
        x = torch.cat((pad, x), dim=1)
        windows = x.unfold(dimension=1, size=self.memory_length, step=1)
        windows = windows.contiguous().view(-1, 1, feature_size, self.memory_length)

        # Forward Propagation
        out_conv2d = self.conv2d_1(windows)  # Dim: (batch_size * n_windows, channels, feature_size, memory_length)
        out_conv1d = self.conv1d(windows.squeeze(1))  # Dim: (batch_size * n_windows, feature_size, memory_length)
        out_conv1d = out_conv1d.view(-1, self.channels,self.input_height, self.memory_length)
        out = torch.cat((out_conv2d, out_conv1d), dim=2)

        out = self.conv2d_2(out.transpose(1,2))
        out = out.view(batch_size, frame_length, -1)  # Dim: (batch_size * n_windows, 1, window_size, feature_size_new)
        out,_ = self.lstm(out)  # Dim: (batch_size * n_windows, 1, window_size, feature_size_new)
        out = self.fc_out(out)  # Dim: (batch_size * n_windows, 1, window_size, feature_size_new)
        out = self.fc_out_2(out)  # Dim: (batch_size * n_windows, 2)
        out = out.view(batch_size, frame_length, 2)
        return out
