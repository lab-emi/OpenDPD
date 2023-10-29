import torch
from torch import nn


class GMP(nn.Module):
    def __init__(self, memory_depth, degree):
        super(GMP, self).__init__()
        self.filter_length = memory_depth + memory_depth * memory_depth * (degree)
        self.WI = torch.DoubleTensor(self.filter_length, 1)
        self.WQ = torch.DoubleTensor(self.filter_length, 1)
        self.W = nn.Parameter(torch.complex(self.WI, self.WQ))

    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)
        out = torch.zeros(batch_size, seq_len, 2)
        output = torch.squeeze(torch.matmul(x, self.W))
        real = torch.real(output)
        out[:, :, 0] = torch.real(output)
        out[:, :, 1] = torch.imag(output)
        return out
