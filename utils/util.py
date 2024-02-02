__author__ = "Yizhuo Wu, Chang Gao"
__license__ = "Apache-2.0 License"
__email__ = "yizhuo.wu@tudelft.nl, chang.gao@tudelft.nl"

import numpy as np


def count_net_params(net):
    n_param = 0
    for name, param in net.named_parameters():
        sizes = 1
        for el in param.size():
            sizes = sizes * el
        n_param += sizes
    return n_param


def get_amplitude(IQ_signal):
    I = IQ_signal[:, 0]
    Q = IQ_signal[:, 1]
    power = I ** 2 + Q ** 2
    amplitude = np.sqrt(power)
    return amplitude


def set_target_gain(input_IQ, output_IQ):
    """Calculate the total energy of the I-Q signal."""
    amp_in = get_amplitude(input_IQ)
    amp_out = get_amplitude(output_IQ)
    max_in_amp = np.max(amp_in)
    max_out_amp = np.max(amp_out)
    target_gain = np.mean(max_out_amp / max_in_amp)
    return target_gain
