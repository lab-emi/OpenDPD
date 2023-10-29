import torch
from torch import nn


class VDLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, bidirectional=False, batch_first=True,
                 bias=True):
        super(VDLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        self.bias = bias

        # Instantiate NN Layers
        self.rnn = nn.LSTM(input_size=1,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           bidirectional=False,
                           batch_first=self.batch_first,
                           bias=True)
        self.fc_I = nn.Linear(in_features=2 * hidden_size,
                              out_features=1,
                              bias=True)
        self.fc_Q = nn.Linear(in_features=2 * hidden_size,
                              out_features=1,
                              bias=True)

    def reset_parameters(self):
        for name, param in self.rnn.named_parameters():
            num_gates = int(param.shape[0] / self.hidden_size)
            if 'bias' in name:
                nn.init.constant_(param, 0)
            if 'weight' in name:
                for i in range(0, num_gates):
                    nn.init.orthogonal_(param[i * self.hidden_size:(i + 1) * self.hidden_size, :])
            if 'weight_ih_l0' in name:
                for i in range(0, num_gates):
                    nn.init.xavier_uniform_(param[i * self.hidden_size:(i + 1) * self.hidden_size, :])

        for name, param in self.fc_I.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            if 'bias' in name:
                nn.init.constant_(param, 0)

        for name, param in self.fc_Q.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            if 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, x, h_0):
        batch_size = x.size(0)  # NOTE: dim of x must be (batch, time, feat)/(N, T, F)
        seq_len = x.size(1)
        out = torch.zeros(batch_size, seq_len, 2).to(x.device)
        Aout, (h_n, _) = self.rnn(x[:, :, 0].unsqueeze(-1), (h_0, h_0))
        PRB = torch.cat(
            (Aout * torch.cos(x[:, :, 1].unsqueeze(-1)), Aout * torch.sin(x[:, :, 1].unsqueeze(-1))), dim=-1)
        out[:, :, 0] = (self.fc_I(PRB)).squeeze(-1)
        out[:, :, 1] = (self.fc_Q(PRB)).squeeze(-1)
        return out
