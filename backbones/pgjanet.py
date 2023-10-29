import torch
from torch import nn


class PGJANET(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(PGJANET, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size

        # a_n
        self.Wa = nn.Parameter(torch.DoubleTensor(1 + self.hidden_size, self.hidden_size))
        self.ba = nn.Parameter(torch.DoubleTensor(self.hidden_size))

        # p1_n
        self.Wp1 = nn.Parameter(torch.DoubleTensor(1 + self.hidden_size, self.hidden_size))
        self.bp1 = nn.Parameter(torch.DoubleTensor(self.hidden_size))

        # p2_n
        self.Wp2 = nn.Parameter(torch.DoubleTensor(1 + self.hidden_size, self.hidden_size))
        self.bp2 = nn.Parameter(torch.DoubleTensor(self.hidden_size))
        # z_n
        self.Wz = nn.Parameter(torch.DoubleTensor(2 * self.hidden_size, self.hidden_size))
        self.bz = nn.Parameter(torch.DoubleTensor(self.hidden_size))

        # h_n
        self.Wh = nn.Parameter(torch.DoubleTensor(2 * self.hidden_size, self.hidden_size))
        self.bh = nn.Parameter(torch.DoubleTensor(self.hidden_size))

        # Output Layer
        self.fc_out = nn.Linear(in_features=hidden_size,
                                out_features=self.output_size,
                                bias=True)

    def forward(self, x, h_0):
        batch_size = x.size(0)
        seq_len = x.size(1)
        out = torch.zeros(batch_size, seq_len, self.hidden_size).to(x.device)
        h_n = h_0.squeeze(0)
        for t in range(seq_len):
            a_n = torch.tanh(torch.matmul((torch.cat((x[:, t, 0].unsqueeze(-1), h_n), dim=-1)), self.Wa) + self.ba)
            p1_n = torch.tanh(
                torch.matmul((torch.cat((torch.cos(x[:, t, 1].unsqueeze(-1)), h_n), dim=-1)), self.Wp1) + self.bp1)
            p2_n = torch.tanh(
                torch.matmul((torch.cat((torch.sin(x[:, t, 1].unsqueeze(-1)), h_n), dim=-1)), self.Wp2) + self.bp2)
            u_n = a_n * p1_n * p2_n * (1 - a_n) * (1 - p1_n) * (1 - p2_n)
            newin = torch.cat((u_n, h_n), dim=-1)
            z_n = torch.sigmoid(torch.matmul(newin, self.Wz) + self.bz)
            h_n_1 = torch.tanh(torch.matmul(newin, self.Wh) + self.bh)
            h_n = z_n * h_n + (1 - z_n) * h_n_1
            out[:, t, :] = h_n
        out = self.fc_out(out)
        return out, h_n