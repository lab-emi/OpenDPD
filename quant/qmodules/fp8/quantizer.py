###############################################################################
# Quantized FP8
###############################################################################
import torch.nn as nn
from .quantizers import FPQuantizer

class FP8_Quantizer(nn.Module):
    def __init__(self, bits, all_positive=False):
        super().__init__()
        self.bits = bits
        self.quantizer = FPQuantizer(n_bits=self.bits, mantissa_bits=5, set_maxval=True)
        self.quantizer.make_range_trainable()
        
    def init_step_size(self, x):
        pass
    
    def init_params(self):
        pass
    
    def init_act_params(self):
        pass
    
    def forward(self, x):
        return self.quantizer(x)


    def __repr__(self):
        return super().__repr__() + '(fp bits={})'.format(self.bits)
    