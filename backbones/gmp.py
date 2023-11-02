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
        self.Wreal = torch.Tensor(memory_length, self.W)
        self.Wimag = torch.Tensor(memory_length, self.W)
        self.Weight = nn.Parameter(torch.complex(self.Wreal, self.Wimag))

    @staticmethod
    def get_memory_window(sequence, memory_length):
        frames = []
        sequence_length = sequence.shape[2]
        num_frames = (sequence_length - memory_length) // 1 + 1
        for i in range(num_frames):
            frame = sequence[:, :, i * 1: i * 1 + memory_length, :]
            frames.append(frame)
        return torch.stack(frames)

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
        x = torch.complex(x[..., 0], x[..., 1]).unsqueeze(-1)  #Dim: (batch_size, frame_length, 1)
        zero_pad = torch.zeros((batch_size, 2*self.memory_length-1, 1)).to(x.device)
        x = torch.cat((zero_pad, x), dim=1)
        amp = torch.abs(x)
        x_degree=[]
        for i in range(self.degree):
            x_degree.append(torch.pow(amp.unsqueeze(1), i))
        x_degree=torch.cat(x_degree, dim=1)             #Dim: (batch_size, degree, frame_length, 1)
        x = torch.cat((x.unsqueeze(1), x_degree), dim=1) #Dim: (batch_size, degree+1, frame_length, 1)
        windows = self.get_memory_window(x, self.memory_length)
        windows=windows.transpose(0, 1).transpose(1, 2)     # Dim: (batch_size, degree, n_windows, window_size, feature_size)


        for j in range(frame_length):
            x_input=windows[:, 0, j, :, :]
            for m in range(self.memory_length):
                for d in range(self.degree):
                    x_input = torch.cat((x_input, windows[:, 0, j, :, :] * windows[:, d, j+m, :, :]), dim=2)
            # Forward Propagation
            complex_out = torch.sum(x_input * self.Weight)
            out[:, j, 0] = torch.real(complex_out)
            out[:, j, 1] = torch.imag(complex_out)
        return out
