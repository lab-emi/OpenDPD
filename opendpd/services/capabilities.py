"""What this machine can run: detected accelerators (never "tested", that is the registry's word)."""

from __future__ import annotations

from typing import Any, Dict

_device_cache: Dict[str, Any] = {}


def detect_devices() -> Dict[str, Any]:
    """Detected accelerators, cached per process: ``{"cuda": {detected, count, name}, "mps": {detected}}``."""
    if _device_cache:
        return _device_cache
    info: Dict[str, Any] = {"cuda": {"detected": False, "count": 0, "name": None}, "mps": {"detected": False}}
    try:
        import torch
        if torch.cuda.is_available():
            info["cuda"] = {"detected": True, "count": torch.cuda.device_count(),
                            "name": torch.cuda.get_device_name(0)}
        mps = getattr(torch.backends, "mps", None)
        info["mps"] = {"detected": bool(mps and mps.is_available())}
    except Exception as err:  # noqa: BLE001 - torch missing or broken driver
        info["error"] = str(err)
    _device_cache.update(info)
    return _device_cache


def device_available(device: str) -> bool:
    return device == "cpu" or bool(detect_devices().get(device, {}).get("detected"))
