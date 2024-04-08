import torch
import torch.nn.functional as F
from .quantizers import INT_Quantizer

def calc_similarity(tensor1, tensor2):
    tensor1 = tensor1.flatten()
    tensor2 = tensor2.flatten()
    return torch.dot(tensor1, tensor2) / (torch.norm(tensor1) * torch.norm(tensor2))

class INT_Conv2D(torch.nn.Conv2d):
    def __init__(self, m: torch.nn.Conv2d, weight_quantizer=INT_Quantizer(bits=8, all_positive=False), act_quantizer=INT_Quantizer(bits=8, all_positive=True)):
        super(INT_Conv2D, self).__init__(
            in_channels=m.in_channels,
            out_channels=m.out_channels,
            kernel_size=m.kernel_size,
            stride=m.stride,
            padding=m.padding,
            dilation=m.dilation,
            groups=m.groups,
            bias=True if m.bias is not None else False,
            padding_mode=m.padding_mode
        )

        self.weight = torch.nn.Parameter(m.weight.detach())

        self.weight_quantizer = weight_quantizer
        self.weight_quantizer.init_step_size(m.weight)

        self.act_quantizer = act_quantizer
        self.act_quantizer.init_act_params()
            
        # save the parameters 
        self.register_buffer('n_bits_w', torch.Tensor([self.weight_quantizer.bits]))
        self.register_buffer('n_bits_a', torch.Tensor([self.act_quantizer.bits]))
        
    def forward(self, x):
        quantized_weight = self.weight_quantizer(self.weight)
        # self.register_buffer('quantized_weight', quantized_weight)
        
        quantized_act = self.act_quantizer(x)
        # self.register_buffer('quantized_act', quantized_act)

        # quantized_act = x

        return F.conv2d(quantized_act, quantized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class INT_Linear(torch.nn.Linear):
    def __init__(self, m: torch.nn.Conv2d, weight_quantizer=INT_Quantizer(bits=8, all_positive=False), act_quantizer=INT_Quantizer(bits=8, all_positive=True)):
        super(INT_Linear, self).__init__(
            in_features=m.in_features,
            out_features=m.out_features,
            bias=True if m.bias is not None else False)

        self.weight = torch.nn.Parameter(m.weight.detach())        
        self.weight_quantizer = weight_quantizer
        self.weight_quantizer.init_act_params()
        self.act_quantizer = act_quantizer
        self.act_quantizer.init_act_params()
        
        self.out_quantizer = INT_Quantizer(bits=16, all_positive=False)
        self.out_quantizer.init_act_params()
        
        self.out_quant = False
        # save the parameters 
        self.register_buffer('n_bits_w', torch.Tensor([self.weight_quantizer.bits]))
        self.register_buffer('n_bits_a', torch.Tensor([self.act_quantizer.bits]))
        

    def forward(self, x):    
        quantized_weight = self.weight_quantizer(self.weight)
        # self.register_buffer('quantized_weight', quantized_weight)

        quantized_act = self.act_quantizer(x)
        # self.register_buffer('quantized_act', quantized_act)
        q_out = F.linear(quantized_act, quantized_weight, self.bias)
        if self.out_quant and not self.training:
            out = self.out_quantizer(q_out)
        else:
            out = q_out

        return out
    
    def __repr__(self):
        return super().__repr__() + '(out_quant={})'.format(self.out_quant)

class INT_Pass(torch.nn.Module):
    """ Quantization pass for activation; can be added to the model as a layer
    """
    def __init__(self, act_quantizer=INT_Quantizer(bits=8, all_positive=False)):
        super(INT_Pass, self).__init__()
        
        self.act_quantizer = act_quantizer
        
    def forward(self, x):
        quantized_act = self.act_quantizer(x)
        return quantized_act
            
            