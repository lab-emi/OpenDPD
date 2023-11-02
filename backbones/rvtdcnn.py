"""
Reproduced from: https://ieeexplore.ieee.org/document/9352501
"""

import torch
from torch import nn


class RVTDCNN(nn.Module):
    def __init__(self, window_size=4, out_channels=3, kernel_size=3, stride=1, padding=(1, 0),
                 dilation=1, fc_hid_size=6):
        super(RVTDCNN, self).__init__()
        self.out_channels = out_channels
        self.window_size = window_size
        self.stride = stride
        self.feature_size_new = 3
        self.fc_in_features = self.out_channels * self.feature_size_new * self.window_size
        self.fc_hid_size = fc_hid_size
        self.Conv2d = torch.nn.Conv2d(in_channels=1,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=padding,
                                      dilation=dilation,
                                      bias=True,
                                      padding_mode='zeros')
        self.fc_hid = nn.Linear(in_features=self.fc_in_features,
                                out_features=fc_hid_size,
                                bias=True)
        self.fc_out = nn.Linear(in_features=self.fc_hid_size,
                                out_features=2,
                                bias=True)

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
        pad = x[:, -(self.window_size - 1):, :]
        x = torch.cat((pad, x), dim=1)
        windows = x.unfold(dimension=1, size=4, step=1).transpose(2, 3)
        windows = torch.unsqueeze(windows, dim=2)  # Dim: (batch_size, n_windows, 1, window_size, feature_size)
        windows = windows.contiguous().view(-1, 1, self.window_size, feature_size)

        # Forward Propagation
        out = torch.tanh(self.Conv2d(windows))  # Dim: (batch_size * n_windows, 1, window_size, feature_size_new)
        out = out.view(-1, self.fc_in_features)  # Dim: (batch_size * n_windows, window_size*feature_size_new)
        out = torch.tanh(self.fc_hid(out))
        out = self.fc_out(out)  # Dim: (batch_size * n_windows, 2)
        out = out.view(batch_size, frame_length, 2)
        return out
