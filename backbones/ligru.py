import torch
from torch import nn


class LiGRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LiGRU, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size

        # z_n
        self.Wz = nn.Parameter(torch.DoubleTensor(self.input_size, self.hidden_size))
        self.Uz = nn.Parameter(torch.DoubleTensor(self.hidden_size, self.hidden_size))
        self.bz = nn.Parameter(torch.DoubleTensor(self.hidden_size))
        # h_n
        self.Wh = nn.Parameter(torch.DoubleTensor(self.input_size, self.hidden_size))
        self.Uh = nn.Parameter(torch.DoubleTensor(self.hidden_size, self.hidden_size))
        self.bh = nn.Parameter(torch.DoubleTensor(self.hidden_size))

        self.fc_out = nn.Linear(in_features=hidden_size,
                                out_features=self.output_size,
                                bias=True)

    def forward(self, x, h_0):
        batch_size = x.size(0)
        seq_len = x.size(1)
        out = torch.zeros(batch_size, seq_len, self.hidden_size).to(x.device)
        h_n = h_0.squeeze(0)
        for t in range(seq_len):
            z_n = torch.sigmoid(torch.matmul(x[:, t, :], self.Wz) + torch.matmul(h_n, self.Uz) + self.bz)
            h_n_1 = torch.tanh(torch.matmul(x[:, t, :], self.Wh) + torch.matmul(h_n, self.Uh) + self.bh)
            h_n = z_n * h_n + (1 - z_n) * h_n_1
            out[:, t, :] = h_n
        out = self.fc_out(out)

        return out