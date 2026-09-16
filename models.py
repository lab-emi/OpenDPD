__author__ = "Yizhuo Wu, Chang Gao"
__license__ = "Apache-2.0 License"
__email__ = "yizhuo.wu@tudelft.nl, chang.gao@tudelft.nl"

import torch
import torch.nn as nn


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

        from opendpd.core.backbone_builders import build_backbone
        self.backbone = build_backbone(backbone_type, vars(self))
        if hasattr(self.backbone, 'reset_parameters'):
            self.backbone.reset_parameters()

    def forward(self, x, h_0=None):
        batch_size = x.size(0)  # NOTE: dim of x must be (batch, time, feat)/(N, T, F)

        if h_0 is None and self.backbone_type != 'tres_deltagru':
            # Create directly on the input device.  TRes-DeltaGRU owns five
            # recurrent states internally and historically discarded this one.
            h_0 = torch.zeros(
                self.num_layers, batch_size, self.hidden_size, device=x.device
            )

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
