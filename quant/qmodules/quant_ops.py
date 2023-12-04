"""
Name: quant_ops.py
Des: Contains the quantization operators
"""

import torch.nn as nn

from .quantizers import OP_INT_Quantizer

import sys
sys.path.append('../')
from ..modules import Mul, Sqrt, Pow

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
    
    
class Quant_sqrt(nn.Module):
    def __init__(self, quantizer=OP_INT_Quantizer(8, all_positive=False)):
        super(Quant_sqrt, self).__init__()
        self.quantizer = quantizer
        self.quantizer.init_params()
        
    def forward(self, x):
        x = self.quantizer(Sqrt()(x))
        
        return x
    
    def __repr__(self):
        return self.__class__.__name__ + '({})'.format(self.quantizer)
    
class Quant_pow(nn.Module):
    def __init__(self, m: nn.Module, quantizer=OP_INT_Quantizer(8, all_positive=False)):
        super(Quant_pow, self).__init__()
        self.quantizer = quantizer
        self.quantizer.init_params()
        self.power = m.power
        self.pow = Pow(self.power)
    
    def forward(self, x):
        powx = self.pow(x)
        if self.training:
            x = powx
        else:
            x = self.quantizer(powx)
        
        return x
    
    def __repr__(self):
        return self.__class__.__name__ + '({}, pow = {})'.format(self.quantizer, self.power)