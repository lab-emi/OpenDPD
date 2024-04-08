import torch
import torch.nn as nn
import numpy as np
import torch.nn.init as init
from torch.autograd import Variable
from .ops import *

class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
        self.add = Add()
        self.mul = Mul()

        self.reset_parameters()


    def reset_parameters(self):
        std = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            init.uniform_(w, -std, std)

    def forward(self, input, hx=None):

        # Inputs:
        #       input: of shape (batch_size, input_size)
        #       hx: of shape (batch_size, hidden_size)
        # Output:
        #       hy: of shape (batch_size, hidden_size)

        if hx is None:
            hx = Variable(input.new_zeros(input.size(0), self.hidden_size))

        x_t = self.x2h(input)
        h_t = self.h2h(hx)


        x_reset, x_upd, x_new = torch.split(x_t, self.hidden_size, dim=1)
        h_reset, h_upd, h_new = torch.split(h_t, self.hidden_size, dim=1)

        # reset_gate = torch.sigmoid(x_reset + h_reset)
        # update_gate = torch.sigmoid(x_upd + h_upd)
        # new_gate = torch.tanh(x_new + (reset_gate * h_new))
        
        reset_gate = self.sigmoid(self.add(x_reset, h_reset))
        update_gate = self.sigmoid(self.add(x_upd, h_upd))
        new_gate = self.tanh(self.add(x_new, self.mul(reset_gate, h_new)))

        # hy = update_gate * hx + (1 - update_gate) * new_gate
        hy = self.add(self.mul(update_gate, hx), self.mul((1 - update_gate), new_gate))

        return hy

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_first=True, bias=True):
        super(GRU, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        
        self.batch_first = batch_first
        
        self.rnn_cell_list = nn.ModuleList()

        self.rnn_cell_list.append(GRUCell(self.input_size,
                                          self.hidden_size,
                                          self.bias))
        for l in range(1, self.num_layers):
            self.rnn_cell_list.append(GRUCell(self.hidden_size,
                                              self.hidden_size,
                                              self.bias))


    def forward(self, input, hx=None):

        # Input of shape (batch_size, seqence length, input_size) or (seqence length, batch_size, input_size)
        # Turn (seqence length, batch_size, input_size) to (batch_size, seqence length, input_size) if not batch_first
        if not self.batch_first:
            input = input.permute(1, 0, 2)

        if hx is None:
            if torch.cuda.is_available():
                h0 = Variable(torch.zeros(self.num_layers, input.size(0), self.hidden_size).cuda())
            else:
                h0 = Variable(torch.zeros(self.num_layers, input.size(0), self.hidden_size))

        else:
             h0 = hx

        outs = []

        hidden = list()
        for layer in range(self.num_layers):
            hidden.append(h0[layer, :, :])

        for t in range(input.size(1)):

            for layer in range(self.num_layers):

                if layer == 0:
                    hidden_l = self.rnn_cell_list[layer](input[:, t, :], hidden[layer])
                else:
                    hidden_l = self.rnn_cell_list[layer](hidden[layer - 1],hidden[layer])
                hidden[layer] = hidden_l

                hidden[layer] = hidden_l

            outs.append(hidden_l)

        # out shape [batch_size, seq_length, hidden_size]
        out = torch.stack(outs, dim=1)

        return out, hidden_l