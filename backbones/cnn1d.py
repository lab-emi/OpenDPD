import torch
import torch.nn as nn


class CNN1D(nn.Module):
    def __init__(self, cnn_memory, output_length, kernel_size):
        super(CNN1D, self).__init__()
        self.W = cnn_memory + kernel_size - 1
        self.outW = (self.W - kernel_size) + 1
        self.width = kernel_size
        self.outlen = output_length
        self.Conv1d = torch.nn.Conv1d(2, 2, kernel_size, stride=1, padding=0, dilation=1, groups=2, bias=True,
                                      padding_mode='zeros', device=None, dtype=None)

        self.fc = nn.Linear(in_features=self.outW * 2,
                            out_features=2,
                            bias=True)

    def forward(self, x):
        batch_size = x.size(0)
        out = torch.zeros(batch_size, self.outlen, 2).to(x.device)
        x = x.transpose(1, 2)
        for t in range(self.outlen):
            Input = x[:, :, t:t + self.W]
            Convout = torch.tanh(self.Conv1d(Input))
            fcin = torch.reshape(Convout, [batch_size, 2 * self.outW])
            out[:, t, :] = self.fc(fcin)
        return out
