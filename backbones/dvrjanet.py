import torch
from torch import nn


class DVRJANET(nn.Module):
    def __init__(self, hidden_size):
        super(DVRJANET, self).__init__()
        self.hidden_size = hidden_size

        # c_k
        self.c1 = nn.Parameter(torch.DoubleTensor(1))
        self.c2 = nn.Parameter(torch.DoubleTensor(1))
        self.c3 = nn.Parameter(torch.DoubleTensor(1))

        # a_n
        self.Wa = nn.Parameter(torch.DoubleTensor(1, self.hidden_size))
        self.Wah = nn.Parameter(torch.DoubleTensor(self.hidden_size, self.hidden_size))
        # theta_n
        self.Wp1 = nn.Parameter(torch.DoubleTensor(1, self.hidden_size))
        self.Wph = nn.Parameter(torch.DoubleTensor(self.hidden_size, self.hidden_size))
        # f_n
        self.Wf = nn.Parameter(torch.DoubleTensor(self.hidden_size, self.hidden_size))
        self.bf = nn.Parameter(torch.DoubleTensor(self.hidden_size))
        # gcos_n
        self.Wgc = nn.Parameter(torch.DoubleTensor(2 * self.hidden_size, self.hidden_size))
        self.bgc = nn.Parameter(torch.DoubleTensor(self.hidden_size))
        # gsin_n
        self.Wgs = nn.Parameter(torch.DoubleTensor(2 * self.hidden_size, self.hidden_size))
        self.bgs = nn.Parameter(torch.DoubleTensor(self.hidden_size))

        # Output Layers
        self.fc_I = nn.Linear(in_features=hidden_size,
                              out_features=1,
                              bias=True)
        self.fc_Q = nn.Linear(in_features=hidden_size,
                              out_features=1,
                              bias=True)

    def forward(self, x, hI_0, hQ_0):
        device = x.device
        batch_size = x.size(0)  # NOTE: dim of x must be (batch, time, feat)/(N, T, F)
        seq_len = x.size(1)
        out = torch.zeros(batch_size, seq_len, 2).to(device)
        batch_size = x.size(0)
        out_I = torch.zeros(batch_size, seq_len, self.hidden_size).to(device)
        out_Q = torch.zeros(batch_size, seq_len, self.hidden_size).to(device)
        hI_n = hI_0.squeeze(0)
        hQ_n = hQ_0.squeeze(0)
        for t in range(seq_len):
            a_n = torch.matmul(x[:, t, 0].unsqueeze(-1), self.Wa) + torch.matmul(hI_n * hQ_n, self.Wah)
            anew_n = torch.mul(torch.add(a_n, -1 / 3), self.c1) + torch.mul(torch.add(a_n, -2 / 3),
                                                                            self.c2) + torch.mul(torch.add(a_n, -1),
                                                                                                 self.c3)
            theta_n = torch.matmul(x[:, t, 1].unsqueeze(-1), self.Wp1) + torch.matmul(hI_n * hQ_n, self.Wph)
            f_n = torch.sigmoid(torch.matmul(hI_n * hQ_n, self.Wf) + self.bf)
            cos_in = torch.cat((hI_n, torch.cos(theta_n) * anew_n), dim=-1)
            sin_in = torch.cat((hQ_n, torch.sin(theta_n) * anew_n), dim=-1)
            gcos_n = torch.tanh(torch.matmul(cos_in, self.Wgc) + self.bgc)
            gsin_n = torch.tanh(torch.matmul(sin_in, self.Wgs) + self.bgs)
            hI_n = f_n * hI_n + (1 - f_n) * gcos_n
            hQ_n = f_n * hQ_n + (1 - f_n) * gsin_n
            out_I[:, t, :] = hI_n
            out_Q[:, t, :] = hQ_n

        I = self.fc_I(out_I)
        Q = self.fc_Q(out_Q)
        out[:, :, 0] = I.squeeze(-1)
        out[:, :, 1] = Q.squeeze(-1)
        return out