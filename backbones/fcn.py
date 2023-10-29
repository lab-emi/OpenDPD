import torch
import torch.nn as nn


class FCN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, bidirectional=False, batch_first=True,
                 bias=True):
        super(FCN, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        self.bias = bias

        # Instantiate NN Layers
        self.fc_in = nn.Linear(in_features=input_size,
                               out_features=hidden_size,
                               bias=self.bias)

        # Create the hidden layers
        self.fc_hidden = nn.ModuleList()
        for _ in range(num_layers - 1):  # -1 because we already have one layer (fc_in)
            self.fc_hidden.append(nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=self.bias))

        self.fc_out = nn.Linear(in_features=hidden_size,
                                out_features=self.output_size,
                                bias=self.bias)

    def reset_parameters(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            if 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, x):
        out = torch.relu(self.fc_in(x))
        for layer in self.fc_hidden:
            out = torch.relu(layer(out))
        out = self.fc_out(out)
        return out