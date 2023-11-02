import torch
from torch import nn


class VDLSTM(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size,
                 num_layers,
                 window_length=4,
                 stride=1,
                 bidirectional=False,
                 batch_first=True,
                 bias=True):
        super(VDLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = window_length
        self.output_size = output_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        self.window_length = window_length
        self.stride = stride
        self.bias = bias
        self.pad_size = self.window_length - 1

        # Instantiate NN Layers
        self.rnn = nn.LSTM(input_size=window_length,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           bidirectional=False,
                           batch_first=self.batch_first,
                           bias=True)
        self.fc_lambda_1 = nn.Linear(in_features=hidden_size,
                                     out_features=window_length,
                                     bias=True)
        self.fc_lambda_2 = nn.Linear(in_features=hidden_size,
                                     out_features=window_length,
                                     bias=True)
        self.fc_out = nn.Linear(in_features=2 * window_length,
                                out_features=2,
                                bias=True)

    def get_memory_window(self, sequence):
        windows = []
        padding = self.window_length - 1
        # pad = torch.zeros((padding, sequence.size(1)), device=sequence.device)
        pad = sequence[-padding:, :]
        sequence = torch.vstack((pad, sequence))
        sequence_length = len(sequence)
        num_windows = (sequence_length - self.window_length) // self.stride + 1
        for i in range(num_windows):
            window = sequence[i * self.stride: i * self.stride + self.window_length]
            windows.append(window)
        return torch.stack(windows)

    def forward(self, x, h_0):
        # Get amplitude |x|
        i_x = x[..., 0]
        q_x = x[..., 1]
        amp2 = torch.pow(i_x, 2) + torch.pow(q_x, 2)
        amp = torch.sqrt(amp2)  # Dim: (batch_size, frame_length, 1)

        # for (i_x_sample, q_x_sample, amp_sample) in zip(i_x, q_x, amp):  # sample Dim: (frame_length, 1)
        pad = i_x[:, -self.pad_size:]
        i_x = torch.cat((pad, i_x), dim=1)
        i_x = i_x.unfold(dimension=1, size=self.window_length, step=self.stride)
        pad = q_x[:, -self.pad_size:]
        q_x = torch.cat((pad, q_x), dim=1)
        q_x = q_x.unfold(dimension=1, size=self.window_length, step=self.stride)
        pad = amp[:, -self.pad_size:]
        amp = torch.cat((pad, amp), dim=1)
        amp = amp.unfold(dimension=1, size=self.window_length, step=self.stride)
        cos = i_x / amp  # Dim: (batch, n_windows, window_length)
        sin = q_x / amp  # Dim: (batch, n_windows, window_length)
        rnn_out, _ = self.rnn(amp)  # Dim: (batch, n_windows, hidden_size)
        lambda_1 = self.fc_lambda_1(rnn_out)  # Dim: (batch, n_windows, hidden_size)
        lambda_2 = self.fc_lambda_2(rnn_out)  # Dim: (batch, n_windows, hidden_size)
        out = self.fc_out(torch.cat((lambda_1 * cos, lambda_2 * sin), dim=-1))
        return out

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

        for name, param in self.fc_lambda_1.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            if 'bias' in name:
                nn.init.constant_(param, 0)

        for name, param in self.fc_lambda_2.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            if 'bias' in name:
                nn.init.constant_(param, 0)

        for name, param in self.fc_out.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            if 'bias' in name:
                nn.init.constant_(param, 0)
