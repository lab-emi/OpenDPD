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
    
    def update_params(self):
        self.round_scale, self.dec_num = self.round_scale2pow2(self.scale)
        self.int_num = self.bits - self.dec_num - 1
        self.pow2_scale.copy_(self.round_scale)
        self.integer_num.copy_(self.int_num)
        self.decimal_num.copy_(self.dec_num)
    
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
        if self.training:
            scale_factor = 1 / (x.numel() * self.Qp) ** 0.5
            scale = grad_scale(self.scale, scale_factor)
        else:
            scale = self.scale
            scale, dec_num = self.round_scale2pow2(scale)
            if scale != self.pow2_scale:
                self.update_params()
        
        x = x / scale
        x = x.clamp(self.Qn, self.Qp)
        
        # zp = round_pass(self.zp)
        x_bar = round_pass(x)

        x_hat = x_bar * scale
                
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
    
##############################################
## DOREFA Quantizer
##############################################
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class GRound(Function):
    @staticmethod
    def forward(self, input):
        sign = torch.sign(input)
        output = sign * torch.floor(torch.abs(input) + 0.5)
        return output

    @staticmethod
    def backward(self, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class Drf_Act_Quantizer(nn.Module):
    def __init__(self, bits, all_positive=True):
        super(Drf_Act_Quantizer, self).__init__()
        self.bits = bits
        
        
    def round(self, input):
        output = GRound.apply(input)
        return output
    
    def init_step_size(self, x):
        pass
    
    def init_act_params(self):
        pass

    def init_params(self):
        pass
        
    def forward(self, input):
        if self.bits == 32:
            output = input
        elif self.bits == 1:
            return torch.sign(input)
        else:
            output = torch.clamp(input * 0.1, 0, 1)  # clamp input to [0,1], and scale it by 0.1 first
            scale = 1 / float(2 ** self.bits - 1)  # scale
            output = self.round(output / scale) * scale # quantize / dequantize
        return output


class Drf_Weight_Quantizer(nn.Module):
    def __init__(self, bits, all_positive=True):
        super(Drf_Weight_Quantizer, self).__init__()
        self.bits = bits

    def round(self, input):
        output = GRound.apply(input)
        return output

    def init_step_size(self, x):
        pass
    
    def init_act_params(self):
        pass

    def forward(self, input):
        if self.bits == 32:
            output = input
        elif self.bits == 1:
            return torch.sign(input)
        else:
            output = torch.tanh(input)
            output = output / 2 / torch.max(torch.abs(output)) + 0.5 
            scale = 1 / float(2 ** self.bits - 1)  # scale
            output = self.round(output / scale) * scale  # quantize / dequantize
            output = 2 * output - 1
        return output
    
##############################################
## IAO Quantizer
##############################################

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributed
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.autograd import Function


class ObserverBase(nn.Module):
    def __init__(self, q_level):
        super(ObserverBase, self).__init__()
        self.q_level = q_level

    def update_range(self, min_val, max_val):
        raise NotImplementedError

    @torch.no_grad()
    def forward(self, input):
        if self.q_level == "L":  
            # layer level
            min_val = torch.min(input)
            max_val = torch.max(input)
        elif self.q_level == "C":  
            # channel level(conv_weight)
            input = torch.flatten(input, start_dim=1)
            min_val = torch.min(input, 1)[0]
            max_val = torch.max(input, 1)[0]
        elif self.q_level == "FC":  
            # channel level(fc_weight)
            min_val = torch.min(input, 1, keepdim=True)[0]
            max_val = torch.max(input, 1, keepdim=True)[0]

        self.update_range(min_val, max_val)


class MinMaxObserver(ObserverBase):
    def __init__(self, q_level, out_channels):
        super(MinMaxObserver, self).__init__(q_level)
        self.num_flag = 0
        self.out_channels = out_channels
        if self.q_level == "L":
            self.register_buffer("min_val", torch.zeros((1), dtype=torch.float32))
            self.register_buffer("max_val", torch.zeros((1), dtype=torch.float32))
        elif self.q_level == "C":
            self.register_buffer(
                "min_val", torch.zeros((out_channels, 1, 1, 1), dtype=torch.float32)
            )
            self.register_buffer(
                "max_val", torch.zeros((out_channels, 1, 1, 1), dtype=torch.float32)
            )
        elif self.q_level == "FC":
            self.register_buffer(
                "min_val", torch.zeros((out_channels, 1), dtype=torch.float32)
            )
            self.register_buffer(
                "max_val", torch.zeros((out_channels, 1), dtype=torch.float32)
            )

    def update_range(self, min_val_cur, max_val_cur):
        if self.q_level == "C":
            min_val_cur.resize_(self.min_val.shape)
            max_val_cur.resize_(self.max_val.shape)
        if self.num_flag == 0:
            self.num_flag += 1
            min_val = min_val_cur
            max_val = max_val_cur
        else:
            min_val = torch.min(min_val_cur, self.min_val)
            max_val = torch.max(max_val_cur, self.max_val)
        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)


class MovingAverageMinMaxObserver(ObserverBase):
    def __init__(self, q_level, out_channels, momentum=0.1):
        super(MovingAverageMinMaxObserver, self).__init__(q_level)
        self.momentum = momentum
        self.num_flag = 0
        self.out_channels = out_channels
        if self.q_level == "L":
            self.register_buffer("min_val", torch.zeros((1), dtype=torch.float32))
            self.register_buffer("max_val", torch.zeros((1), dtype=torch.float32))
        elif self.q_level == "C":
            self.register_buffer(
                "min_val", torch.zeros((out_channels, 1, 1, 1), dtype=torch.float32)
            )
            self.register_buffer(
                "max_val", torch.zeros((out_channels, 1, 1, 1), dtype=torch.float32)
            )
        elif self.q_level == "FC":
            self.register_buffer(
                "min_val", torch.zeros((out_channels, 1), dtype=torch.float32)
            )
            self.register_buffer(
                "max_val", torch.zeros((out_channels, 1), dtype=torch.float32)
            )

    def update_range(self, min_val_cur, max_val_cur):
        if self.q_level == "C":
            min_val_cur.resize_(self.min_val.shape)
            max_val_cur.resize_(self.max_val.shape)
        if self.num_flag == 0:
            self.num_flag += 1
            min_val = min_val_cur
            max_val = max_val_cur
        else:
            min_val = (1 - self.momentum) * self.min_val + self.momentum * min_val_cur
            max_val = (1 - self.momentum) * self.max_val + self.momentum * max_val_cur
        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)


class HistogramObserver(nn.Module):
    def __init__(self, q_level, momentum=0.1, percentile=0.9999):
        super(HistogramObserver, self).__init__()
        self.q_level = q_level
        self.momentum = momentum
        self.percentile = percentile
        self.num_flag = 0
        self.register_buffer("min_val", torch.zeros((1), dtype=torch.float32))
        self.register_buffer("max_val", torch.zeros((1), dtype=torch.float32))

    @torch.no_grad()
    def forward(self, input):
        # MovingAveragePercentileCalibrator
        # PercentileCalibrator
        max_val_cur = torch.kthvalue(
            input.abs().view(-1), int(self.percentile * input.view(-1).size(0)), dim=0
        )[0]
        # MovingAverage
        if self.num_flag == 0:
            self.num_flag += 1
            max_val = max_val_cur
        else:
            max_val = (1 - self.momentum) * self.max_val + self.momentum * max_val_cur
        self.max_val.copy_(max_val)


class Round(Function):
    @staticmethod
    def forward(self, input, observer_min_val, observer_max_val, q_type):
        # symmetric
        if q_type == 0:
            max_val = torch.max(
                torch.abs(observer_min_val), torch.abs(observer_max_val)
            )
            min_val = -max_val
        # asymmetric
        else:
            max_val = observer_max_val
            min_val = observer_min_val
        self.save_for_backward(input, min_val, max_val)
        sign = torch.sign(input)
        output = sign * torch.floor(torch.abs(input) + 0.5)
        return output

    @staticmethod
    def backward(self, grad_output):
        input, min_val, max_val = self.saved_tensors
        grad_input = grad_output.clone()
        # clamp gradient 
        grad_input[input.gt(max_val)] = 0
        grad_input[input.lt(min_val)] = 0
        return grad_input, None, None, None


class Quantizer(nn.Module):
    def __init__(self, bits, observer, activation_weight_flag, qaft=False, union=False):
        super(Quantizer, self).__init__()
        self.bits = bits
        self.observer = observer
        self.activation_weight_flag = activation_weight_flag
        self.qaft = qaft
        self.union = union
        self.q_type = 0
        # scale/zero_point/eps
        if self.observer.q_level == "L":
            self.register_buffer("scale", torch.ones((1), dtype=torch.float32))
            self.register_buffer("zero_point", torch.zeros((1), dtype=torch.float32))
        elif self.observer.q_level == "C":
            self.register_buffer(
                "scale",
                torch.ones((self.observer.out_channels, 1, 1, 1), dtype=torch.float32),
            )
            self.register_buffer(
                "zero_point",
                torch.zeros((self.observer.out_channels, 1, 1, 1), dtype=torch.float32),
            )
        elif self.observer.q_level == "FC":
            self.register_buffer(
                "scale",
                torch.ones((self.observer.out_channels, 1), dtype=torch.float32),
            )
            self.register_buffer(
                "zero_point",
                torch.zeros((self.observer.out_channels, 1), dtype=torch.float32),
            )
        self.register_buffer(
            "eps", torch.tensor((torch.finfo(torch.float32).eps), dtype=torch.float32)
        )

    def update_qparams(self):
        raise NotImplementedError

    # round function
    def round(self, input, observer_min_val, observer_max_val, q_type):
        output = Round.apply(input, observer_min_val, observer_max_val, q_type)
        return output

    def forward(self, input):
        if self.bits == 32:
            output = input
        elif self.bits == 1:
            return torch.sign(input)
        else:
            if not self.qaft:
                # qat, update quant_para
                if self.training:
                    if not self.union:
                        self.observer(input)  # update observer_min and observer_max
                    self.update_qparams()  # update scale and zero_point
            output = (
                torch.clamp(
                    self.round(
                        input / self.scale.clone() - self.zero_point,
                        self.observer.min_val / self.scale - self.zero_point,
                        self.observer.max_val / self.scale - self.zero_point,
                        self.q_type,
                    ),
                    self.quant_min_val,
                    self.quant_max_val,
                )
                + self.zero_point
            ) * self.scale.clone()
        return output


class SignedQuantizer(Quantizer):
    def __init__(self, *args, **kwargs):
        super(SignedQuantizer, self).__init__(*args, **kwargs)
        if self.activation_weight_flag == 0:  # weight
            self.register_buffer(
                "quant_min_val",
                torch.tensor((-((1 << (self.bits - 1)) - 1)), dtype=torch.float32),
            )
            self.register_buffer(
                "quant_max_val",
                torch.tensor(((1 << (self.bits - 1)) - 1), dtype=torch.float32),
            )
        elif self.activation_weight_flag == 1:  # activation
            self.register_buffer(
                "quant_min_val",
                torch.tensor((-(1 << (self.bits - 1))), dtype=torch.float32),
            )
            self.register_buffer(
                "quant_max_val",
                torch.tensor(((1 << (self.bits - 1)) - 1), dtype=torch.float32),
            )
        else:
            print("activation_weight_flag error")


class UnsignedQuantizer(Quantizer):
    def __init__(self, *args, **kwargs):
        super(UnsignedQuantizer, self).__init__(*args, **kwargs)
        if self.activation_weight_flag == 0:  # weight
            self.register_buffer(
                "quant_min_val", torch.tensor((0), dtype=torch.float32)
            )
            self.register_buffer(
                "quant_max_val",
                torch.tensor(((1 << self.bits) - 2), dtype=torch.float32),
            )
        elif self.activation_weight_flag == 1:  # activation
            self.register_buffer(
                "quant_min_val", torch.tensor((0), dtype=torch.float32)
            )
            self.register_buffer(
                "quant_max_val",
                torch.tensor(((1 << self.bits) - 1), dtype=torch.float32),
            )
        else:
            print("activation_weight_flag error")


# symmetric quantization
class SymmetricQuantizer(SignedQuantizer):
    def update_qparams(self):
        self.q_type = 0
        quant_range = (
            float(self.quant_max_val - self.quant_min_val) / 2
        )  # quantized_range
        float_range = torch.max(
            torch.abs(self.observer.min_val), torch.abs(self.observer.max_val)
        )  # float_range
        scale = float_range / quant_range  # scale
        scale = torch.max(scale, self.eps)  # processing for very small scale
        zero_point = torch.zeros_like(scale)  # zero_point
        self.scale.copy_(scale)
        self.zero_point.copy_(zero_point)


# asymmetric quantization
class AsymmetricQuantizer(UnsignedQuantizer):
    def update_qparams(self):
        self.q_type = 1
        quant_range = float(self.quant_max_val - self.quant_min_val)  # quantized_range
        float_range = self.observer.max_val - self.observer.min_val  # float_range
        scale = float_range / quant_range  # scale
        scale = torch.max(scale, self.eps)  # processing for very small scale
        sign = torch.sign(self.observer.min_val)
        zero_point = sign * torch.floor(
            torch.abs(self.observer.min_val / scale) + 0.5
        )  # zero_point
        self.scale.copy_(scale)
        self.zero_point.copy_(zero_point)

class IAO_Quantizer(nn.Module):
    def __init__(self, bits, all_positive=True, act_or_weight='weight'):
        super(IAO_Quantizer, self).__init__()
        self.bits = bits
        self.all_positive = all_positive
        self.act_or_weight = act_or_weight
        
        self.q_type = 0 if not self.all_positive else 1
        self.activation_weight_flag = 0 if self.act_or_weight == 'weight' else 1
        
        
        if self.q_type == 0:
            self.quantizer = SymmetricQuantizer(
                bits=self.bits,
                observer=MovingAverageMinMaxObserver(
                    q_level="L", out_channels=None
                ),
                activation_weight_flag=self.activation_weight_flag,
                qaft=False,
            )
        else:
            self.quantizer = AsymmetricQuantizer(
                bits=self.bits,
                observer=MovingAverageMinMaxObserver(
                    q_level="L", out_channels=None
                ),
                activation_weight_flag=self.activation_weight_flag,
                qaft=False,
            )
    
    
    def init_step_size(self, x):
        pass
    
    def init_params(self):
        pass

    def init_act_params(self):
        pass
    
    def forward(self, input):
        quant_input = self.quantizer(input)
        return quant_input
        
    def __repr__(self):
        return super().__repr__() + ' (quant bits={}) '.format(self.bits)

##############################################
## PACT
##############################################

class PACT_Quantizer(nn.Module):
    def __init__(self, bits, all_positive=False):
        super().__init__()
        self.bits = bits
        self.all_positive = all_positive
        
        self.quantizer = ActFn.apply
        
        self.alpha = nn.Parameter(torch.tensor(10.))
        
     
    def init_step_size(self, x):
        pass
    
    def init_params(self):
        pass

    def init_act_params(self):
        pass
           
    def forward(self, x):
        if self.training:
            x = self.quantizer(x, self.alpha, self.bits-1)
        else:
            x = self.quantizer(x, self.alpha.detach(), self.bits-1)
        
        return x 

    def __repr__(self):
        return super().__repr__() + '(bits={})'.format(self.bits)
    
# k = 8
class ActFn(Function):
	@staticmethod
	def forward(ctx, x, alpha, k):
		ctx.save_for_backward(x, alpha)
		# y_1 = 0.5 * ( torch.abs(x).detach() - torch.abs(x - alpha).detach() + alpha.item() )
		y = torch.clamp(x, min = 0, max = alpha.item())
		scale = (2**k - 1) / alpha
		y_q = torch.round( y * scale) / scale
		return y_q

	@staticmethod
	def backward(ctx, dLdy_q):
		# Backward function, I borrowed code from
		# https://github.com/obilaniu/GradOverride/blob/master/functional.py
		# We get dL / dy_q as a gradient
		x, alpha, = ctx.saved_tensors
		# Weight gradient is only valid when [0, alpha]
		# Actual gradient for alpha,
		# By applying Chain Rule, we get dL / dy_q * dy_q / dy * dy / dalpha
		# dL / dy_q = argument,  dy_q / dy * dy / dalpha = 0, 1 with x value range 
		lower_bound      = x < 0
		upper_bound      = x > alpha
		# x_range       = 1.0-lower_bound-upper_bound
		x_range = ~(lower_bound|upper_bound)
		grad_alpha = torch.sum(dLdy_q * torch.ge(x, alpha).float()).view(-1)
		return dLdy_q * x_range.float(), grad_alpha, None