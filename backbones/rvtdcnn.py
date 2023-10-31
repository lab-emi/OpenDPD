"""
Reproduced from: https://ieeexplore.ieee.org/document/9352501
"""

import torch
from torch import nn


class RVTDCNN(nn.Module):
    def __init__(self, input_size, windows_length=4, out_channels=3, kernel_size=3, stride=1, padding=(0, 1),
                 dilation=1, fc_hid_size=6):
        super(RVTDCNN, self).__init__()
        self.out_channels = out_channels
        self.H = input_size
        self.W = windows_length
        self.fc_in_features = self.out_channels * self.H * self.W
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

    @staticmethod
    def get_memory_window(sequence, windows_length=4, stride_length=1):
        frames = []
        padding = windows_length - 1
        pad = torch.zeros((padding, sequence.size(1)), device=sequence.device)
        sequence = torch.vstack((pad, sequence))
        sequence_length = len(sequence)
        num_frames = (sequence_length - windows_length) // stride_length + 1
        for i in range(num_frames):
            frame = sequence[i * stride_length: i * stride_length + windows_length]
            frames.append(frame)
        return torch.stack(frames)

    def forward(self, x):
        batch_size = x.size(0)
        frame_length = x.size(1)
        # x Dim: (batch_size, frame_length, input_size)

        # Split a frame into memory windows
        windows = []
        for sample in x:  # sample Dim: (frame_length, H)
            window = self.get_memory_window(sample)  # Dim: (n_windows, W, H)
            windows.append(window)
        windows = torch.stack(window)  # Dim: (batch_size, n_windows, W, H)
        windows = torch.unsqueeze(windows, dim=2)  # Dim: (batch_size, n_windows, 1, W, H)
        windows = windows.view(-1, 1, self.W, self.H)

        # Forward Propagation
        out = torch.tanh(self.Conv2d(windows))  # Dim: (batch_size * n_windows, 1, W, H)
        W_new = out.size(2)
        out = out.view(-1, W_new * self.H)  # Dim: (batch_size * n_windows, W_new*H)
        out = torch.tanh(self.fc_hid(out))
        out = self.fc_out(out)  # Dim: (batch_size * n_windows, 2)
        out = out.view(batch_size, frame_length, 2)
        return out
