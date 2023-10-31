import torch
import math
import torch.nn as nn
from backbones.pgjanet import PGJANET
from backbones.dvrjanet import DVRJANET
from backbones.gmp import GMP
from backbones.ligru import LiGRU
from backbones.cnn1d import CNN1D
from backbones.rvtdcnn import RVTDCNN


class CoreModel(nn.Module):
    def __init__(self, input_size, cnn_set, cnn_memory, pa_output_len, frame_len, hidden_size, num_layers, degree, backbone_type):
        super(CoreModel, self).__init__()
        self.output_size = 2  # PA outputs: I & Q
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.backbone_type = backbone_type
        self.batch_first = True  # Force batch first
        self.bidirectional = False
        self.bias = True

        if backbone_type == 'gmp':
            self.backbone = GMP(memory_depth=frame_len,
                                degree=degree)
        elif backbone_type == 'fc':
            from backbones.fcn import FCN
            self.backbone = FCN(input_size=self.input_size,
                                hidden_size=self.hidden_size,
                                output_size=self.output_size,
                                num_layers=self.num_layers,
                                bias=self.bias)
        elif backbone_type == 'gru':
            from backbones.gru import GRU
            self.backbone = GRU(input_size=self.input_size,
                                hidden_size=self.hidden_size,
                                output_size=self.output_size,
                                num_layers=self.num_layers,
                                bidirectional=self.bidirectional,
                                batch_first=self.batch_first,
                                bias=self.bias)
        elif backbone_type == 'dgru':
            from backbones.dgru import DGRU
            self.backbone = DGRU(input_size=self.input_size,
                                 hidden_size=self.hidden_size,
                                 output_size=self.output_size,
                                 num_layers=self.num_layers,
                                 bidirectional=self.bidirectional,
                                 batch_first=self.batch_first,
                                 bias=self.bias)
        elif backbone_type == 'lstm':
            from backbones.lstm import LSTM
            self.backbone = LSTM(input_size=self.input_size,
                                 hidden_size=self.hidden_size,
                                 output_size=self.output_size,
                                 num_layers=self.num_layers,
                                 bidirectional=self.bidirectional,
                                 batch_first=self.batch_first,
                                 bias=self.bias)
        elif backbone_type == 'vdlstm':
            from backbones.vdlstm import VDLSTM
            self.backbone = VDLSTM(input_size=self.input_size,
                                   hidden_size=self.hidden_size,
                                   output_size=self.output_size,
                                   num_layers=self.num_layers,
                                   bidirectional=self.bidirectional,
                                   batch_first=self.batch_first,
                                   bias=self.bias)
        elif backbone_type == 'ligru':
            self.backbone = LiGRU(input_size=input_size,
                                  hidden_size=hidden_size)
        elif backbone_type == 'pgjanet':
            self.backbone = PGJANET(input_size=input_size,
                                    hidden_size=hidden_size)
        elif backbone_type == 'dvrjanet':
            self.backbone = DVRJANET(hidden_size=hidden_size)
        elif backbone_type == 'rvtdcnn':
            self.backbone = RVTDCNN(input_size=input_size,
                                    fc_hid_size=hidden_size)

        # Initialize backbone parameters
        try:
            self.backbone.reset_parameters()
        except AttributeError:
            pass


    def forward(self, x, h_0=None):
        device = x.device
        batch_size = x.size(0)  # NOTE: dim of x must be (batch, time, feat)/(N, T, F)

        if h_0 is None:  # Create initial hidden states if necessary
            h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)

        # Forward Propagate through the RNN
        if self.backbone_type in ('gmp', 'fcn', 'cnn1d', 'rvtdcnn'):
            out = self.backbone(x)
        elif self.backbone_type in ('lstm', 'gru', 'dgru', 'ligru', 'vdlstm', 'pgjanet', 'dvrjanet'):
            out = self.backbone(x, h_0)
        else:
            raise ValueError(f"The backbone type '{self.backbone_type}' is not supported. Please add your own "
                             f"backbone under ./backbones and update models.py accordingly.")
        return out


class CascadedModel(nn.Module):
    def __init__(self, dpd_model, pa_model):
        super(CascadedModel, self).__init__()
        self.dpd_model = dpd_model
        self.pa_model = pa_model

    def freeze_pa_model(self):
        for param in self.pa_model.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.dpd_model(x)
        x = self.pa_model(x)
        return x
