#!/usr/bin/env python
# Copyright (c) 2022 Qualcomm Technologies, Inc.
# All Rights Reserved.
import torch
import torch.nn as nn
from .base_quantizers import QuantizerBase
import numpy as np
from itertools import product
from torch.autograd import Function
from .round_utils import round_ste_func


def generate_all_values_fp(
    num_total_bits: int = 8, num_exponent_bits: int = 4, bias: int = 8
) -> list:
    num_fraction_bits = num_total_bits - 1 - num_exponent_bits

    all_values = []
    exp_lower = -bias
    for S in [-1.0, 1.0]:
        for E_str_iter in product(*[[0, 1]] * num_exponent_bits):
            for F_str_iter in product(*[[0, 1]] * num_fraction_bits):
                E_str = "".join(str(i) for i in E_str_iter)
                F_str = "".join(str(i) for i in F_str_iter)

                # encoded exponent
                E_enc = decode_binary_str(E_str)
                E_eff = E_enc - bias
                if E_eff == exp_lower:
                    is_subnormal = 1
                else:
                    is_subnormal = 0

                F_enc = decode_binary_str(F_str) * 2**-num_fraction_bits
                F_eff = F_enc + 1 - is_subnormal

                fp8_val = S * 2.0 ** (E_enc - bias + is_subnormal) * F_eff
                all_values.append(fp8_val)
    res = np.array(all_values)
    res = np.sort(res)
    return res


def generate_all_float_values_scaled(num_total_bits, num_exp_bits, exp_bias, range_limit_fp):
    grid = generate_all_values_fp(num_total_bits, num_exp_bits, exp_bias)
    float_max_abs_val = np.max(np.abs(grid))

    float_scale = float_max_abs_val / range_limit_fp
    floats_all = grid / float_scale
    return floats_all


def decode_float8(S, E, F, bias=16):
    sign = -2 * int(S) + 1
    exponent = int(E, 2) if E else 0
    # Normal FP8   : exponent > 0 : result = 2^(exponent-bias) * 1.F
    # Subnormal FP8: exponent == 0: result = 2^(-bias+1)       * 0.F
    # Lowest quantization bin: 2^(-bias+1)       * {0.0 ... 1 + (2^mantissa-1)/2^mantissa}
    # All other bins         : 2^(exponent-bias) * {1.0 ... 1 + (2^mantissa-1)/2^mantissa}; exponent > 0
    A = int(exponent != 0)
    fraction = A + sum([2 ** -(i + 1) * int(a) for i, a in enumerate(F)])
    exponent += int(exponent == 0)
    return sign * fraction * 2.0 ** (exponent - bias)


def i(x):
    return np.array([x]).astype(np.int32)


def gen(n_bits, exponent_bits, bias):
    all_values = []
    for s in product(*[[0, 1]] * 1):
        for e in product(*[[0, 1]] * exponent_bits):
            for m in product(*[[0, 1]] * (n_bits - 1 - exponent_bits)):
                s = str(s[0])
                e = "".join(str(i) for i in e)
                m = "".join(str(i) for i in m)
                all_values.append(decode_float8(s, e, m, bias=bias))
    return sorted(all_values)


def get_max_value(num_exponent_bits: int = 4, bias: int = 8):
    num_fraction_bits = 7 - num_exponent_bits
    scale = 2**-num_fraction_bits
    max_frac = 1 - scale
    max_value = 2 ** (2**num_exponent_bits - 1 - bias) * (1 + max_frac)

    return max_value


def quantize_to_fp8_ste_MM(
    x_float: torch.Tensor,
    n_bits: int,
    maxval: torch.Tensor,
    num_mantissa_bits: torch.Tensor,
    sign_bits: int,
) -> torch.Tensor:
    """
    Simpler FP8 quantizer that exploits the fact that FP quantization is just INT quantization with
    scales that depend on the input.

    This allows to define FP8 quantization using STE rounding functions and thus learn the bias

    """
    M = torch.clamp(round_ste_func(num_mantissa_bits), 1, n_bits - sign_bits)
    E = n_bits - sign_bits - M

    if maxval.shape[0] != 1 and len(maxval.shape) != len(x_float.shape):
        maxval = maxval.view([-1] + [1] * (len(x_float.shape) - 1))
    bias = 2**E - torch.log2(maxval) + torch.log2(2 - 2 ** (-M)) - 1

    minval = -maxval if sign_bits == 1 else torch.zeros_like(maxval)
    xc = torch.min(torch.max(x_float, minval), maxval)

    """
    2 notes here:
    1: Shifting by bias to ensure data is aligned to the scaled grid in case bias not in Z.
       Recall that implicitly bias := bias' - log2(alpha), where bias' in Z. If we assume 
       alpha in (0.5, 1], then alpha contracts the grid, which is equivalent to translate the
       data 'to the right' relative to the grid, which is what the subtraction of log2(alpha) 
       (which is negative) accomplishes. 
    2: Ideally this procedure doesn't affect gradients wrt the input (we want to use the STE).
       We can achieve this by detaching log2 of the (absolute) input.

    """

    # log_scales = torch.max((torch.floor(torch.log2(torch.abs(xc)) + bias)).detach(), 1.)
    log_scales = torch.clamp((torch.floor(torch.log2(torch.abs(xc)) + bias)).detach(), 1.0)

    scales = 2.0 ** (log_scales - M - bias)

    result = round_ste_func(xc / scales) * scales
    return result


class FP8QuantizerFunc(Function):
    @staticmethod
    def forward(ctx, x_float, bias, num_exponent_bits):
        return quantize_to_fp8_ste_MM(x_float, bias, num_exponent_bits)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


def decode_binary_str(F_str):
    F = sum([2 ** -(i + 1) * int(a) for i, a in enumerate(F_str)]) * 2 ** len(F_str)
    return F


class FPQuantizer(QuantizerBase):
    """
    8-bit Floating Point Quantizer
    """

    def __init__(
        self,
        *args,
        scale_domain=None,
        mantissa_bits=4,
        maxval=3,
        set_maxval=False,
        learn_maxval=False,
        learn_mantissa_bits=False,
        mse_include_mantissa_bits=True,
        allow_unsigned=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.mantissa_bits = mantissa_bits

        self.ebits = self.n_bits - self.mantissa_bits - 1
        self.default_bias = 2 ** (self.ebits - 1)

        # assume signed, correct when range setting turns out to be unsigned
        default_maxval = (2 - 2 ** (-self.mantissa_bits)) * 2 ** (
            2**self.ebits - 1 - self.default_bias
        )

        self.maxval = maxval if maxval is not None else default_maxval

        self.maxval = torch.Tensor([self.maxval])
        self.mantissa_bits = torch.Tensor([float(self.mantissa_bits)])

        self.set_maxval = set_maxval
        self.learning_maxval = learn_maxval
        self.learning_mantissa_bits = learn_mantissa_bits
        self.mse_include_mantissa_bits = mse_include_mantissa_bits

        self.allow_unsigned = allow_unsigned
        self.sign_bits = 1

    def forward(self, x_float):
        if self.maxval.device != x_float.device:
            self.maxval = self.maxval.to(x_float.device)
        if self.mantissa_bits.device != x_float.device:
            self.mantissa_bits = self.mantissa_bits.to(x_float.device)

        res = quantize_to_fp8_ste_MM(
            x_float, self.n_bits, self.maxval, self.mantissa_bits, self.sign_bits
        )

        ebits = self.n_bits - self.mantissa_bits - 1
        return res

    def is_initialized(self):
        return True

    def symmetric(self):
        return False

    def effective_bit_width(self):
        return None

    def _make_unsigned(self, x_min):
        if isinstance(x_min, torch.Tensor):
            return self.allow_unsigned and torch.all(x_min >= 0)
        else:
            return self.allow_unsigned and x_min >= 0

    def set_quant_range(self, x_min, x_max):

        if self._make_unsigned(x_min):
            self.sign_bits = 0

        if self.set_maxval:
            if not isinstance(x_max, torch.Tensor):
                x_max = torch.Tensor([x_max]).to(self.maxval.device)
                x_min = torch.Tensor([x_min]).to(self.maxval.device)
            if self.maxval.device != x_max.device:
                self.maxval = self.maxval.to(x_max.device)
            if self.mantissa_bits.device != x_max.device:
                self.mantissa_bits = self.mantissa_bits.to(x_max.device)

            mx = torch.abs(torch.max(torch.abs(x_min), x_max))
            self.maxval = mx

            if not isinstance(self.maxval, torch.Tensor) or len(self.maxval.shape) == 0:
                self.maxval = torch.Tensor([self.maxval])

    def make_range_trainable(self):
        if self.learning_maxval:
            self.learn_maxval()
        if self.learning_mantissa_bits:
            self.learn_mantissa_bits()

    def learn_maxval(self):
        self.learning_maxval = True
        self.maxval = torch.nn.Parameter(self.maxval)

    def learn_mantissa_bits(self):
        self.learning_mantissa_bits = True
        self.mantissa_bits = torch.nn.Parameter(self.mantissa_bits)

    def fix_ranges(self):
        if isinstance(self.maxval, nn.Parameter):
            self.parameter_to_fixed("maxval")
        if isinstance(self.mantissa_bits, nn.Parameter):
            self.parameter_to_fixed("mantissa_bits")

    def extra_repr(self):
        maxval = self.maxval

        M = torch.clamp(torch.round(self.mantissa_bits), 1, 7)
        E = 7 - M
        maxval = 2**E - torch.log2(self.maxval) + torch.log2(2 - 2 ** (-M)) - 1
        if maxval.shape[0] > 1:
            bstr = "[per_channel]"
        else:
            bstr = f"{maxval.item()}"
        return f"Exponent: {E.item()} bits; mode: ; bias: {bstr}"
