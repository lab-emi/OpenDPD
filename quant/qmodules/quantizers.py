import torch

def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return (y - y_grad).detach() + y_grad


def round_pass(x):
    y = x.round()
    y_grad = x
    return (y - y_grad).detach() + y_grad


class INT_Quantizer(torch.nn.Module):
    def __init__(self, bits, all_positive=False):
        super().__init__()
    
        self.bits = bits
        self.all_positive = all_positive
        
        if(all_positive):
            self.Qn = 0
            self.Qp = 2 ** bits - 1
        else:
            self.Qn = -2**(bits - 1)
            self.Qp = 2 ** (bits - 1) - 1
            
        self.scale = torch.nn.Parameter(torch.Tensor([1.0]))
        # self.zp = torch.nn.Parameter(torch.Tensor([0.0]))
        
        self.register_params()
        
    def register_params(self):
        self.register_buffer('pow2_scale', torch.Tensor([0.0]))
        self.register_buffer('decimal_num', torch.Tensor([1.0]))
        self.register_buffer('integer_num', self.bits - 1 - self.decimal_num)
   
    def update_params(self, pow2_scale, dec_num=None):
        self.pow2_scale.copy_(pow2_scale)
        self.decimal_num.copy_(dec_num)
        self.integer_num.copy_(self.bits - 1 - dec_num)
    
    def init_act_params(self):
        integer_num = 2
        init_scale = 2 ** (integer_num - self.bits)
        
        self.scale = torch.nn.Parameter(torch.Tensor([init_scale]))
        
    def init_step_size(self, x):
        self.scale = torch.nn.Parameter(
            x.detach().abs().mean() * 2 / (self.Qp) ** 0.5)
        # self.zp = torch.nn.Parameter(
        #     x.detach().mean())
        
    def round_scale2pow2(self, scale):
        # get the nearest power of 2
        scale = scale.abs()
        scale_log2 = scale.log2()
        scale_log2_round = scale_log2.round()
        scale = 2 ** scale_log2_round
        
        dec_num = scale_log2_round.abs().int()
        
        return scale, dec_num

    def forward(self, x):
        # scale = self.scale
        pow2_scale, dec_num = self.round_scale2pow2(self.scale)
        if dec_num != self.decimal_num:
            self.update_params(pow2_scale, dec_num)
        
        x = x / pow2_scale
        x = x.clamp(self.Qn, self.Qp)
        
        # zp = round_pass(self.zp)
        x_bar = round_pass(x)

        x_hat = x_bar * pow2_scale
                
        return x_hat
        
    def __repr__(self):
        return super().__repr__() + '(bits={})'.format(self.bits)


class OP_INT_Quantizer(INT_Quantizer):
    """ Quantizer for the quantized operators
    Des: Different from the quantizer for the layers, the quantizer for the operators should initialize the scale factor
    """
    def __init__(self, bits, all_positive=False):
        super().__init__(bits, all_positive)
        
        
    def init_params(self):
        init_scale = 2 ** (2 - self.bits)
        self.scale = torch.nn.Parameter(torch.Tensor([init_scale]))

class Identity_Quantizer(torch.nn.Module):
    def __init__(self, bits=0, all_positive=False):
        super().__init__()
    
        self.bits = bits
        self.all_positive = all_positive

    def init_step_size(self, x):
        pass

    def init_params(self):
        pass
    
    def init_act_params(self):
        pass

    def forward(self, x):
        return x
    
