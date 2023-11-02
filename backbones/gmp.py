import torch
from torch import nn


class GMP(nn.Module):
    def __init__(self, memory_length=11, degree=5):
        super(GMP, self).__init__()
        self.memory_length = memory_length
        self.degree = degree
        self.W = 1 + (degree - 1) * memory_length
        self.Weight = nn.Parameter(torch.Tensor(1, memory_length * self.W))

    def reset_parameters(self):
        for name, param in self.named_parameters():
            if 'W' in name:
                nn.init.xavier_uniform_(param)

    def forward(self, x, h_0):
        batch_size = x.size(0)
        frame_length = x.size(1)
        out = torch.zeros((batch_size, frame_length, 2), device=x.device)
        # x Dim: (batch_size, frame_length, input_size)

        # Split a frame into memory windows
        x = torch.complex(x[..., 0], x[..., 1])  # Dim: (batch_size, frame_length)
        zero_pad = torch.zeros((batch_size, self.memory_length - 1)).to(x.device)
        x = torch.cat((zero_pad, x), dim=1)
        windows_x = x.unfold(dimension=-1, size=self.memory_length,
                             step=1)
        windows_x = windows_x.unsqueeze(1).unsqueeze(1)
        windows_x = windows_x.repeat(1, self.degree - 1, self.memory_length, 1,
                                     1)  # Dim: (batch_size, n_windows, memory_length)
        amp = torch.abs(torch.cat((zero_pad, x), dim=1))
        x_degree = []
        for i in range(1, self.degree):
            x_degree.append(torch.pow(amp.unsqueeze(1), i))
        x_degree = torch.cat(x_degree, dim=1)  # Dim: (batch_size, degree, frame_length)

        windows_degree = x_degree.unfold(dimension=-1, size=self.memory_length, step=1)
        # Dim: (batch_size, degree+1, n_windows, window_size, feature_size)

        for j in range(frame_length):
            x_input = windows_x[:, 0, 0, j, :]
            mul_term = torch.mul(windows_x[:, :, :, j, :], windows_degree[:, :, j:j + self.memory_length, :])
            mul_term = mul_term.reshape(batch_size, -1)
            x_input = torch.cat((x_input, mul_term), dim=-1)
            # Forward Propagation
            complex_out = torch.sum(x_input * self.Weight, dim=-1)
            out[:, j, 0] = torch.real(complex_out)
            out[:, j, 1] = torch.imag(complex_out)
        return out