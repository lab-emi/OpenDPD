"""Utility entry points for quantization helpers."""
from __future__ import annotations

from typing import Any

from .modules import Add, Mul, Pow, Sqrt
from .quant_envs import AttrDict, Base_GRUQuantEnv

__all__ = ["get_quant_model", "AttrDict", "Base_GRUQuantEnv", "Add", "Mul", "Pow", "Sqrt"]


def _build_quant_args(proj: Any) -> AttrDict:
    return AttrDict({
        "n_bits_w": getattr(proj, "n_bits_w", 8),
        "n_bits_a": getattr(proj, "n_bits_a", 8),
        "pretrained_model": getattr(proj, "pretrained_model", ""),
        "quant_dir_label": getattr(proj, "quant_dir_label", ""),
    })


def get_quant_model(proj: Any, model) -> Any:
    """Return a (possibly) quantized version of ``model``.

    If ``proj.quant`` is truthy we attempt to construct the quantization
    environment defined in ``quant.quant_envs``. Studio fails on setup errors;
    the historical warning/fallback is retained for legacy CLI callers only.
    """
    if not getattr(proj, "quant", False):
        return model

    try:
        quant_args = _build_quant_args(proj)
        env = Base_GRUQuantEnv(model, args=quant_args)
        setattr(proj, "quant_env", env)
        return env.q_model
    except Exception as exc:
        if getattr(proj, "studio_strict_quantization", False):
            raise RuntimeError(f"requested quantization setup failed: {exc}; Studio will not substitute a float model") from exc
        print(f"[WARN] Quantization setup failed: {exc}. Using float model instead.")
        return model
