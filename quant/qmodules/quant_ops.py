"""
Name: quant_ops.py
Des: Contains the quantization operators
"""

import torch.nn as nn

from .quantizers import OP_INT_Quantizer

import sys
sys.path.append('../')
from ..modules.ops import Mul

class Quant_sigmoid(nn.Module):
    def __init__(self, quantizer=OP_INT_Quantizer(8, all_positive=True)):
        super(Quant_sigmoid, self).__init__()
        self.quantizer = quantizer
        self.quantizer.init_params()
    
    def forward(self, x):
        x = self.quantizer(nn.Sigmoid()(x))
        
        return x
    def __repr__(self):
        return self.__class__.__name__ + '({})'.format(self.quantizer)
    
class Quant_tanh(nn.Module):
    def __init__(self, quantizer=OP_INT_Quantizer(8, all_positive=True)):
        super(Quant_tanh, self).__init__()
        self.quantizer = quantizer
        self.quantizer.init_params()
        
    def forward(self, x): 
        x = self.quantizer(nn.Tanh()(x))
        
        return x
    
    def __repr__(self):
        return self.__class__.__name__ + '({})'.format(self.quantizer)
    
class Quant_mult(nn.Module):
    def __init__(self, quantizer=OP_INT_Quantizer(8, all_positive=False)):
        super(Quant_mult, self).__init__()
        self.quantizer = quantizer
        self.quantizer.init_params()

    def forward(self, x, y):
        x = self.quantizer(Mul()(x, y))
        return x
    
    def __repr__(self):
        return self.__class__.__name__ + '({})'.format(self.quantizer)

class Quant_add(nn.Module):
    def __init__(self, quantizer=OP_INT_Quantizer(8, all_positive=False)):
        super(Quant_add, self).__init__()
        self.quantizer = quantizer
        self.quantizer.init_params()

    def forward(self, x, y):
        x = self.quantizer(x + y)
        return x
    
    def __repr__(self):
        return self.__class__.__name__ + '({})'.format(self.quantizer)