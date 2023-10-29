import torch
from torch import nn


class RVTDCNN(nn.Module):
    def __init__(self, feature_num, cnn_memory, output_length, kernel_num, kernel_size):
        super(RVTDCNN, self).__init__()
        self.C = kernel_num
        self.H = feature_num
        self.W = cnn_memory + kernel_size[1] - 1
        self.outH = (self.H - kernel_size[0]) + 1
        self.outW = (self.W - kernel_size[1]) + 1
        self.width = kernel_size[1]
        self.outlen = output_length
        self.Conv2d = torch.nn.Conv2d(kernel_num, kernel_num, kernel_size, stride=1, padding=0, dilation=1,
                                      groups=kernel_num, bias=True,
                                      padding_mode='zeros', device=None, dtype=None)

        self.fc = nn.Linear(in_features=self.outH * self.outW * self.C,
                            out_features=2,
                            bias=True)

    def forward(self, x):
        batch_size = x.size(0)
        out = torch.zeros(batch_size, self.outlen, 2).to(x.device)
        for t in range(self.outlen):
            Input = x[:, t:t + self.W, :].unsqueeze(1)
            Input = Input.repeat(1, self.C, 1, 1)
            Input = Input.transpose(2, 3)
            Convout = torch.tanh(self.Conv2d(Input))
            fcin = torch.reshape(Convout, [batch_size, self.C * self.outH * self.outW])
            out[:, t, :] = self.fc(fcin)
        return out