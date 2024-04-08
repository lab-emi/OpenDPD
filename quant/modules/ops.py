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


class Mul(nn.Module):
    r"""Applies the element-wise function:
    .. math::
        \text{Mul}(x) = x * y
    
    Examples::
        >>> m = nn.Mul()
        >>> in1, in2 = torch.randn(2), torch.randn(2)
        >>> output = m(in1, in2)
    """
    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        return torch.mul(input1, input2)

class Sqrt(nn.Module):
    r"""Applies the element-wise function:
    .. math::
        \text{Sqrt}(x) = \sqrt{x}
    
    Examples::
        >>> m = nn.Sqrt()
        >>> in1 = torch.randn(2)
        >>> output = m(in1)
    """
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(input)
    
class Pow(nn.Module):
    r"""Applies the element-wise function:
    .. math::
        \text{Pow}(x) = x^power
    
    Examples::
        >>> m = nn.Pow(2)
        >>> in1 = torch.randn(2)
        >>> output = m(in1)
    """
        
    def __init__(self, power):
        super(Pow, self).__init__()
        self.power = power
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.pow(input, self.power)