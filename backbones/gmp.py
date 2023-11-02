"""
Reproduced from: https://ieeexplore.ieee.org/document/1703853
"""

import torch
from torch import nn


class GMP(nn.Module):
    def __init__(self, memory_length, degree=3):
        super(GMP, self).__init__()
        self.memory_length = memory_length
        self.degree = degree
        self.W = 1+degree*memory_length
        self.Weight = nn.Parameter(torch.Tensor(memory_length, self.W))

    def reset_parameters(self):
        for name, param in self.named_parameters():
            if 'W' in name:
                nn.init.xavier_uniform_(param)


    def forward(self, x, h_0):
        batch_size = x.size(0)
        frame_length = x.size(1)
        out = torch.zeros(batch_size,frame_length,2)
        # x Dim: (batch_size, frame_length, input_size)

        # Split a frame into memory windows
        x = torch.complex(x[..., 0], x[..., 1]) #Dim: (batch_size, frame_length)
        zero_pad = torch.zeros((batch_size, self.memory_length-1)).to(x.device)
        x = torch.cat((zero_pad, x), dim=1)
        amp = torch.abs(torch.cat((zero_pad,x), dim=1))
        x_degree=[]
        for i in range(1, self.degree+1):
            x_degree.append(torch.pow(amp.unsqueeze(1), i))
        x_degree=torch.cat(x_degree, dim=1)             #Dim: (batch_size, degree, frame_length)
        windows_x = x.unfold(dimension=-1, size=self.memory_length, step=1)  #Dim: (batch_size, n_windows, memory_length)
        windows_x=windows_x.transpose(0, 1)
        windows_degree = x_degree.unfold(dimension=-1, size=self.memory_length, step=1)
        windows_degree = windows_degree.transpose(0, 1).transpose(1, 2)
        # Dim: (batch_size, degree+1, n_windows, window_size, feature_size)


        for j in range(frame_length):
            x_input=windows_x[:, j, :].unsqueeze(-1)
            for m in range(self.memory_length):
                for d in range(self.degree):
                    x_input = torch.cat((x_input, windows_x[:, j, :].unsqueeze(-1) * windows_degree[:, d, j+m, :].unsqueeze(-1)), dim=2)
            # Forward Propagation
            complex_out = torch.sum(torch.sum(x_input * self.Weight, dim=-1), dim=-1)
            out[:, j, 0] = torch.real(complex_out)
            out[:, j, 1] = torch.imag(complex_out)
        return out
