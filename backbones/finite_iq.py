"""Finite amplitude and phase features for exact zero IQ samples.

Nonzero squared magnitudes use the original square root without an epsilon.
At zero, amplitude has the finite zero subgradient and phase normalization uses
a unit denominator, so a padded or muted sample maps to zero phase features.
"""

import torch


def amplitude(squared_magnitude):
    zero = squared_magnitude == 0
    safe_squared = torch.where(zero, torch.ones_like(squared_magnitude), squared_magnitude)
    return torch.where(zero, torch.zeros_like(squared_magnitude), torch.sqrt(safe_squared))


def phase_denominator(magnitude):
    return torch.where(magnitude == 0, torch.ones_like(magnitude), magnitude)


def phase(i, q):
    """Keep atan2's zero phase convention, with a finite zero subgradient."""
    zero = (i == 0) & (q == 0)
    return torch.atan2(torch.where(zero, torch.zeros_like(q), q),
                       torch.where(zero, torch.ones_like(i), i))
