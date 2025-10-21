__author__ = "Yizhuo Wu, Chang Gao"
__license__ = "Apache-2.0 License"
__email__ = "yizhuo.wu@tudelft.nl, chang.gao@tudelft.nl"

import torch
import torch.nn as nn
from backbones.rvtdcnn import RVTDCNN


class CoreModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, backbone_type, window_size=None, num_dvr_units=None, thx=0, thh=0):
        super(CoreModel, self).__init__()
        self.output_size = 2  # PA outputs: I & Q
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.backbone_type = backbone_type
        self.thx = thx
        self.thh = thh
        self.window_size = window_size
        self.num_dvr_units = num_dvr_units
        self.batch_first = True  # Force batch first
        self.bidirectional = False
        self.bias = True

        if backbone_type == 'gmp':
            from backbones.gmp import GMP
            self.backbone = GMP()
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
            self.backbone = DGRU(hidden_size=self.hidden_size,
                                 output_size=self.output_size,
                                 num_layers=self.num_layers,
                                 bidirectional=self.bidirectional,
                                 batch_first=self.batch_first,
                                 bias=self.bias)
        elif backbone_type == 'qgru':
            from backbones.qgru import QGRU
            self.backbone = QGRU(hidden_size=self.hidden_size,
                                 output_size=self.output_size,
                                 num_layers=self.num_layers,
                                 bidirectional=self.bidirectional,
                                 batch_first=self.batch_first,
                                 bias=self.bias)
        elif backbone_type == 'qgru_amp1':
            from backbones.qgru_amp1 import QGRU
            self.backbone = QGRU(hidden_size=self.hidden_size,
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
        elif backbone_type == 'rvtdcnn':
            self.backbone = RVTDCNN(fc_hid_size=hidden_size)
        elif backbone_type == 'apnrru':
            from backbones.apnrru import APNRRU
            self.backbone = APNRRU(hidden_size=self.hidden_size,
                                   bias=self.bias)
        elif backbone_type == 'bojanet':
            from backbones.bojanet import BOJANET
            self.backbone = BOJANET(hidden_size=self.hidden_size,
                                   output_size=self.output_size,
                                   bias=self.bias)
        elif backbone_type == 'deltagru':
            from backbones.deltagru import DeltaGRU
            self.backbone = DeltaGRU(input_size=6,
                                     hidden_size=self.hidden_size,
                                     output_size=self.output_size,
                                     num_layers=self.num_layers,
                                     thx=self.thx,
                                     thh=self.thh,
                                     bias=self.bias)  
        elif backbone_type == 'deltajanet':
            from backbones.deltajanet import DeltaJANET
            self.backbone = DeltaJANET(input_size=6,
                                     hidden_size=self.hidden_size,
                                     output_size=self.output_size,
                                     num_layers=self.num_layers,
                                     thx=self.thx,
                                     thh=self.thh,
                                     bias=self.bias)
        elif backbone_type == 'pgjanet':
            from backbones.pgjanet import PGJANET
            self.backbone = PGJANET(hidden_size=self.hidden_size,
                                  output_size=self.output_size,
                                  bias=self.bias,
                                  window_size=self.window_size)
        elif backbone_type == 'dvrjanet':
            from backbones.dvrjanet import DVRJANET
            self.backbone = DVRJANET(hidden_size=self.hidden_size,
                                   output_size=self.output_size,
                                   num_dvr_units=self.num_dvr_units,
                                   bias=self.bias)
        elif backbone_type == 'deltagru_tcnskip':
            from backbones.deltagru_tcnskip import DeltaGRU
            self.backbone = DeltaGRU(input_size=6,
                                             hidden_size=self.hidden_size,
                                             output_size=self.output_size,
                                             num_layers=self.num_layers,
                                             thx=self.thx,
                                             thh=self.thh,
                                             bias=self.bias)
        elif backbone_type == 'tcnn':
            from backbones.tcnn import TCNN
            self.backbone = TCNN(hidden_channels=self.hidden_size)
        elif backbone_type == 'neuraltx':
            from backbones.neuraltx import NeuralTX
            self.backbone = NeuralTX(hidden_channels=self.hidden_size)
        elif backbone_type == 'mcldnn':
            from backbones.mcldnn import MCLDNN
            self.backbone = MCLDNN(hidden_size=self.hidden_size)
        else:
            raise ValueError(f"The backbone type '{self.backbone_type}' is not supported. Please add your own "
                             f"backbone under ./backbones and update models.py accordingly.")

        # Initialize backbone parameters
        try:
            self.backbone.reset_parameters()
            print("Backbone Initialized...")
        except AttributeError:
            pass

    def forward(self, x, h_0=None):
        device = x.device
        batch_size = x.size(0)  # NOTE: dim of x must be (batch, time, feat)/(N, T, F)

        if h_0 is None:  # Create initial hidden states if necessary
            h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)

        # Forward Propagate through the RNN
        out = self.backbone(x, h_0)

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
