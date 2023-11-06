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
        self.round_scale, self.dec_num = self.round_scale2pow2(self.scale)
        self.int_num = self.bits - self.dec_num - 1
        self.register_buffer('pow2_scale', self.round_scale)
        self.register_buffer('integer_num', self.int_num)
        self.register_buffer('decimal_num', self.dec_num)
    
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
        scale, _ = self.round_scale2pow2(self.scale)
        scale_factor = 1 / (x.numel() * self.Qp) ** 0.5
        
        scale = grad_scale(scale, scale_factor)
        x = x / scale
        x = x.clamp(self.Qn, self.Qp)
        
        # zp = round_pass(self.zp)
        x_bar = round_pass(x)

        x_hat = x_bar * scale
        
        self.register_params()
        
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
        init_scale = 2 ** (1 - self.bits)
        self.scale = torch.nn.Parameter(torch.Tensor([init_scale]))