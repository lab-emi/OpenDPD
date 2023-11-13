#!/usr/bin/env python
# Copyright (c) 2022 Qualcomm Technologies, Inc.
# All Rights Reserved.

from torch import nn



class QuantizerBase(nn.Module):
    def __init__(self, n_bits, per_channel=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_bits = n_bits
        self.per_channel = per_channel
        self.state = None
        self.x_min_fp32 = self.x_max_fp32 = None

    @property
    def is_initialized(self):
        raise NotImplementedError()

    @property
    def x_max(self):
        raise NotImplementedError()

    @property
    def symmetric(self):
        raise NotImplementedError()

    @property
    def x_min(self):
        raise NotImplementedError()

    def forward(self, x_float):
        raise NotImplementedError()

    def _adjust_params_per_channel(self, x):
        raise NotImplementedError()

    def set_quant_range(self, x_min, x_max):
        raise NotImplementedError()

    def extra_repr(self):
        return "n_bits={}, per_channel={}, is_initalized={}".format(
            self.n_bits, self.per_channel, self.is_initialized
        )

    def reset(self):
        self._delta = None
