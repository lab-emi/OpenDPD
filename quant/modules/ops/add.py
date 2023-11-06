import torch
import torch.nn as nn
class Add(nn.Module):
    r"""Applies the element-wise function:
    .. math::
        \text{Add}(x) = x + y
    
    Examples::
        >>> m = nn.Add()
        >>> in1, in2 = torch.randn(2), torch.randn(2)
        >>> output = m(in1, in2)
    """
    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        return torch.add(input1, input2)