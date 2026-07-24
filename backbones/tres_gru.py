import torch
import torch.nn as nn


class TResGRU(nn.Module):
    """TRes-DeltaGRU with the DeltaGRU cell replaced by a dense nn.GRU.

    Same 6-feature extraction and TCN residual skip path as TResDeltaGRU;
    the recurrent core is a standard cuDNN GRU. Bias-free like the DeltaGRU
    layer's x2h/h2h projections, so parameter counts match at equal hidden
    size (e.g. H=10 -> 524 params).
    """

    def __init__(self, input_size, hidden_size, output_size, num_layers, bias=True):
        super(TResGRU, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = 6
        self.output_size = output_size
        self.num_layers = num_layers
        self.bias = bias

        self.rnn = nn.GRU(input_size=self.input_size,
                          hidden_size=self.hidden_size,
                          num_layers=num_layers,
                          batch_first=True,
                          bias=False)
        self.fc_out = nn.Linear(in_features=self.hidden_size,
                                out_features=self.output_size,
                                bias=False)
        self.tcn = nn.Sequential(
            nn.Conv1d(in_channels=2,
                      out_channels=3,
                      kernel_size=3,
                      padding=16,
                      stride=1,
                      dilation=16,
                      bias=False),
            nn.Hardswish(),
            nn.Conv1d(in_channels=3,
                      out_channels=2,
                      kernel_size=1,
                      padding=0,
                      stride=1,
                      dilation=1,
                      bias=False),
            nn.Hardswish(),
        )

    def reset_parameters(self):
        for name, param in self.tcn.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            if 'bias' in name:
                nn.init.constant_(param, 0)
        for name, param in self.rnn.named_parameters():
            num_gates = int(param.shape[0] / self.hidden_size)
            if 'bias' in name:
                nn.init.constant_(param, 0)
            if 'weight_hh' in name:
                for i in range(0, num_gates):
                    nn.init.orthogonal_(param[i * self.hidden_size:(i + 1) * self.hidden_size, :])
            if 'weight_ih' in name:
                for i in range(0, num_gates):
                    nn.init.xavier_uniform_(param[i * self.hidden_size:(i + 1) * self.hidden_size, :])
        for name, param in self.fc_out.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            if 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, x, h_0=None):
        skip_x = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        last_step = torch.roll(x, shifts=-1, dims=1)
        # Feature Extraction
        i_x = x[..., 0].unsqueeze(-1)
        q_x = x[..., 1].unsqueeze(-1)
        i_last = last_step[..., 0].unsqueeze(-1)
        q_last = last_step[..., 1].unsqueeze(-1)
        amp2 = torch.pow(i_x, 2) + torch.pow(q_x, 2)
        amp = torch.sqrt(amp2)
        amp3 = torch.pow(amp, 3)
        feat = torch.cat((i_x, q_x, amp, amp3, i_last, q_last), dim=-1)
        out, _ = self.rnn(feat, h_0)
        out = self.fc_out(out) + skip_x
        return out
