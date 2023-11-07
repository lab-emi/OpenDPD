import torch
import torch.nn.functional as F
from torch import nn
import collections
from typing import DefaultDict, Tuple, List, Dict
from functools import partial

def save_activations(
        activations: DefaultDict,
        name: str,
        module: nn.Module,
        inp: Tuple,
        out: torch.Tensor
) -> None:
    """PyTorch Forward hook to save outputs at each forward
    pass. Mutates specified dict objects with each fwd pass.
    """
    activations[name].append(out.detach().cpu())

def register_activation_hooks(
        model: nn.Module,
        layers_to_save: List[str]
) -> DefaultDict[List, torch.Tensor]:
    """Registers forward hooks in specified layers.
    Parameters
    ----------
    model:
        PyTorch model
    layers_to_save:
        Module names within ``model`` whose activations we want to save.

    Returns
    -------
    activations_dict:
        dict of lists containing activations of specified layers in
        ``layers_to_save``.
    """
    activations_dict = collections.defaultdict(list)

    for name, module in model.named_modules():
        if name in layers_to_save:
            module.register_forward_hook(
                partial(save_activations, activations_dict, name)
            )
    return activations_dict



# to_save = ["conv1", "conv2"]

# # register fwd hooks in specified layers
# saved_activations = register_activation_hooks(net, layers_to_save=to_save)