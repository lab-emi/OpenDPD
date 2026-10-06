"""Arena-only CUDA replay for reviewed native, stateless float32 models.

The optimizer stays eager: its updates, weight decay and changing learning rate
are identical to the ordinary path. Return ``None`` from this factory or from
the callable means the caller must perform the ordinary forward/backward and
clipping, clearing gradients with ``set_to_none=False`` while a helper exists.
The supported set does not extend the training path outside Arena.
"""

from __future__ import annotations

import math
import os

import torch
from torch import nn

from modules.cuda_fast_training import FastCudaStep

ARENA_REPLAY_BACKBONES = frozenset({
    "gru", "dgru", "lstm", "tres_gru", "rvtdcnn", "vdlstm", "tcn",
    "mcldnn", "pgjanet", "dvrjanet", "bojanet", "apnrru", "gmp",
    "qgru", "qgru_amp1", "deltagru", "deltajanet",
    "tres_deltagru", "user_template",
})


def _supported_model(net):
    from models import CascadedModel, CoreModel
    if type(net) is CoreModel:
        return net.backbone_type in ARENA_REPLAY_BACKBONES
    arena_cascade = (isinstance(net, CascadedModel)
                     and type(net).__name__ == "ArenaCascade"
                     and type(net).__module__ == "opendpd.core.arena_engine"
                     and getattr(type(net), "_arena_static_cascade", False) is True)
    if type(net) is CascadedModel or arena_cascade:
        return (_supported_model(net.dpd_model) and _supported_model(net.pa_model)
                and not any(parameter.requires_grad for parameter in net.pa_model.parameters()))
    return False


def make_arena_step(net, criterion, parameters, grad_clip):
    """Return an optional replay callable with the existing ``FastCudaStep`` API."""
    from opendpd.core.arena_training_backends import dense_training_module
    parameters = tuple(parameters)
    if (os.getenv("OPENDPD_DISABLE_CUDA_FAST_PATH", "0") == "1"
            or not _supported_model(net)
            or type(criterion) not in (nn.MSELoss, nn.L1Loss)
            or criterion.reduction != "mean"
            or torch.are_deterministic_algorithms_enabled()
            or not parameters
            or not math.isfinite(grad_clip) or grad_clip < 0
            or tuple(map(id, parameters)) != tuple(id(p) for p in net.parameters() if p.requires_grad)
            or any(not p.is_cuda or p.dtype != torch.float32 for p in net.parameters())):
        return None
    devices = {parameter.device for parameter in net.parameters()}
    if len(devices) != 1:
        return None
    # None of the reviewed native architectures has persistent buffers. A
    # buffer, observer, hook or stochastic layer therefore signals an altered
    # model whose capture semantics need a separate review (including QAT).
    for module in net.modules():
        if (tuple(module.buffers(recurse=False))
                or hasattr(module, "statistics") and not dense_training_module(module)
                or hasattr(module, "activation_post_process")
                or hasattr(module, "weight_fake_quant")
                or isinstance(module, nn.modules.dropout._DropoutNd) and module.p > 0
                or isinstance(module, nn.RNNBase) and module.dropout > 0
                or module._forward_hooks or module._forward_pre_hooks
                or module._backward_hooks or module._backward_pre_hooks):
            return None
    return FastCudaStep(net, criterion, parameters, grad_clip)
