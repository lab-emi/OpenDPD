#!/usr/bin/env python
# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.

import collections
import os
import random
from collections import namedtuple
from enum import Flag, auto
from functools import partial

import numpy as np
import torch
import torch.nn as nn


class DotDict(dict):
    """
    A dictionary that allows attribute-style access.
    Examples
    --------
    >>> config = DotDict(a=None)
    >>> config.a = 42
    >>> config.b = 'egg'
    >>> config  # can be used as dict
    {'a': 42, 'b': 'egg'}
    """

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __delattr__(self, key):
        self.__delitem__(key)

    def __getattr__(self, key):
        if key in self:
            return self.__getitem__(key)
        raise AttributeError(f"DotDict instance has no key '{key}' ({self.keys()})")


def relu(x):
    x = np.array(x)
    return x * (x > 0)


def get_all_layer_names(model, subtypes=None):
    if subtypes is None:
        return [name for name, module in model.named_modules()][1:]
    return [name for name, module in model.named_modules() if isinstance(module, subtypes)]


def get_layer_name_to_module_dict(model):
    return {name: module for name, module in model.named_modules() if name}


def get_module_to_layer_name_dict(model):
    modules_to_names = collections.OrderedDict()
    for name, module in model.named_modules():
        modules_to_names[module] = name
    return modules_to_names


def get_layer_name(model, layer):
    for name, module in model.named_modules():
        if module == layer:
            return name
    return None


def get_layer_by_name(model, layer_name):
    for name, module in model.named_modules():
        if name == layer_name:
            return module
    return None


def create_conv_layer_list(cls, model: nn.Module) -> list:
    """
    Function finds all prunable layers in the provided model

    Parameters
    ----------
    cls: SVD class
    model : torch.nn.Module
    A pytorch model.

    Returns
    -------
    conv_layer_list : list
    List of all prunable layers in the given model.

    """
    conv_layer_list = []

    def fill_list(mod):
        if isinstance(mod, tuple(cls.supported_layer_types)):
            conv_layer_list.append(mod)

    model.apply(fill_list)
    return conv_layer_list


def create_linear_layer_list(cls, model: nn.Module) -> list:
    """
    Function finds all prunable layers in the provided model

    Parameters
    ----------
    model : torch.nn.Module
        A pytorch model.

    Returns
    -------
    conv_layer_list : list
        List of all prunable layers in the given model.

    """
    conv_layer_list = []

    def fill_list(mod):
        if isinstance(mod, tuple(cls.supported_layer_types)):
            conv_layer_list.append(mod)

    model.apply(fill_list)
    return conv_layer_list


def to_numpy(tensor):
    """
    Helper function that turns the given tensor into a numpy array

    Parameters
    ----------
    tensor : torch.Tensor

    Returns
    -------
    tensor : float or np.array

    """
    if isinstance(tensor, np.ndarray):
        return tensor
    if hasattr(tensor, "is_cuda"):
        if tensor.is_cuda:
            return tensor.cpu().detach().numpy()
    if hasattr(tensor, "detach"):
        return tensor.detach().numpy()
    if hasattr(tensor, "numpy"):
        return tensor.numpy()

    return np.array(tensor)


def set_module_attr(model, layer_name, value):
    split = layer_name.split(".")

    this_module = model
    for mod_name in split[:-1]:
        if mod_name.isdigit():
            this_module = this_module[int(mod_name)]
        else:
            this_module = getattr(this_module, mod_name)

    last_mod_name = split[-1]
    if last_mod_name.isdigit():
        this_module[int(last_mod_name)] = value
    else:
        setattr(this_module, last_mod_name, value)


def search_for_zero_planes(model: torch.nn.Module):
    """If list of modules to winnow is empty to start with, search through all modules to check
    if any
    planes have been zeroed out. Update self._list_of_modules_to_winnow with any findings.
    :param model: torch model to search through modules for zeroed parameters
    """

    list_of_modules_to_winnow = []
    for _, module in model.named_modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.modules.conv.Conv2d)):
            in_channels_to_winnow = _assess_weight_and_bias(module.weight, module.bias)
            if in_channels_to_winnow:
                list_of_modules_to_winnow.append((module, in_channels_to_winnow))
    return list_of_modules_to_winnow


def _assess_weight_and_bias(weight: torch.nn.Parameter, _bias: torch.nn.Parameter):
    """4-dim weights [CH-out, CH-in, H, W] and 1-dim bias [CH-out]"""
    if len(weight.shape) > 2:
        input_channels_to_ignore = (weight.sum((0, 2, 3)) == 0).nonzero().squeeze().tolist()
    else:
        input_channels_to_ignore = (weight.sum(0) == 0).nonzero().squeeze().tolist()

    if type(input_channels_to_ignore) != list:
        input_channels_to_ignore = [input_channels_to_ignore]

    return input_channels_to_ignore


def seed_all(seed: int = 1029, deterministic: bool = False):
    """
    This is our attempt to make experiments reproducible by seeding all known RNGs and setting
    appropriate torch directives.
    For a general discussion of reproducibility in Pytorch and CUDA and a documentation of the
    options we are using see, e.g.,
    https://pytorch.org/docs/1.7.1/notes/randomness.html
    https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility

    As of today (July 2021), even after seeding and setting some directives,
    there remain unfortunate contradictions:
    1. CUDNN
    - having CUDNN enabled leads to
      - non-determinism in Pytorch when using the GPU, cf. MORPH-10999.
    - having CUDNN disabled leads to
      - most regression tests in Qrunchy failing, cf. MORPH-11103
      - significantly increased execution time in some cases
      - performance degradation in some cases
    2. torch.set_deterministic(d)
    - setting d = True leads to errors for Pytorch algorithms that do not (yet) have a deterministic
      counterpart, e.g., the layer `adaptive_avg_pool2d_backward_cuda` in vgg16__torchvision.

    Thus, we leave the choice of enforcing determinism by disabling CUDNN and non-deterministic
    algorithms to the user. To keep it simple, we only have one switch for both.
    This situation could be re-evaluated upon updates of Pytorch, CUDA, CUDNN.
    """

    assert isinstance(seed, int), f"RNG seed must be an integer ({seed})"
    assert seed >= 0, f"RNG seed must be a positive integer ({seed})"

    # Builtin RNGs
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Numpy RNG
    np.random.seed(seed)

    # CUDNN determinism (setting those has not lead to errors so far)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Torch RNGs
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Problematic settings, see docstring. Precaution: We do not mutate unless asked to do so
    if deterministic is True:
        torch.backends.cudnn.enabled = False

        torch.set_deterministic(True)  # Use torch.use_deterministic_algorithms(True) in torch 1.8.1
        # When using torch.set_deterministic(True), it is advised by Pytorch to set the
        # CUBLAS_WORKSPACE_CONFIG variable as follows, see
        # https://pytorch.org/docs/1.7.1/notes/randomness.html#avoiding-nondeterministic-algorithms
        # and the link to the CUDA homepage on that website.
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def assert_allclose(actual, desired, *args, **kwargs):
    """A more beautiful version of torch.all_close."""
    np.testing.assert_allclose(to_numpy(actual), to_numpy(desired), *args, **kwargs)


def count_params(module):
    return len(nn.utils.parameters_to_vector(module.parameters()))


class StopForwardException(Exception):
    """Used to throw and catch an exception to stop traversing the graph."""

    pass


class StopForwardHook:
    def __call__(self, module, *args):
        raise StopForwardException


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class CosineTempDecay:
    def __init__(self, t_max, temp_range=(20.0, 2.0), rel_decay_start=0):
        self.t_max = t_max
        self.start_temp, self.end_temp = temp_range
        self.decay_start = rel_decay_start * t_max

    def __call__(self, t):
        if t < self.decay_start:
            return self.start_temp

        rel_t = (t - self.decay_start) / (self.t_max - self.decay_start)
        return self.end_temp + 0.5 * (self.start_temp - self.end_temp) * (1 + np.cos(rel_t * np.pi))


class BaseEnumOptions(Flag):
    def __str__(self):
        return self.name

    @classmethod
    def list_names(cls):
        return [m.name for m in cls]


class ClassEnumOptions(BaseEnumOptions):
    @property
    def cls(self):
        return self.value.cls

    def __call__(self, *args, **kwargs):
        return self.value.cls(*args, **kwargs)


MethodMap = partial(namedtuple("MethodMap", ["value", "cls"]), auto())


def split_dict(src: dict, include=(), remove_prefix: str = ""):
    """
    Splits dictionary into a DotDict and a remainder.
    The arguments to be placed in the first DotDict are those listed in `include`.
    Parameters
    ----------
    src: dict
        The source dictionary.
    include:
        List of keys to be returned in the first DotDict.
    remove_suffix:
        remove prefix from key
    """
    result = DotDict()

    for arg in include:
        if remove_prefix:
            key = arg.replace(f"{remove_prefix}_", "", 1)
        else:
            key = arg
        result[key] = src[arg]
    remainder = {key: val for key, val in src.items() if key not in include}
    return result, remainder




class QuantizerNotInitializedError(Exception):
    """Raised when a quantizer has not been initialized"""

    def __init__(self):
        super(QuantizerNotInitializedError, self).__init__(
            "Quantizer has  not been initialized yet"
        )
