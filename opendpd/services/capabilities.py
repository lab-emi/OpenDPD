"""What this machine can run: detected accelerators (never "tested", that is the registry's word)."""

from __future__ import annotations

from typing import Any, Dict, Optional

_device_cache: Dict[str, Any] = {}


def detect_devices() -> Dict[str, Any]:
    """Detected accelerators, including each visible CUDA device's logical index."""
    if _device_cache:
        return _device_cache
    info: Dict[str, Any] = {"cuda": {"detected": False, "count": 0, "name": None, "instances": []},
                            "mps": {"detected": False}}
    try:
        import torch
        if torch.cuda.is_available():
            instances = [{"index": index, "name": torch.cuda.get_device_name(index)}
                         for index in range(torch.cuda.device_count())]
            info["cuda"] = {"detected": bool(instances), "count": len(instances),
                            "name": instances[0]["name"] if instances else None, "instances": instances}
        mps = getattr(torch.backends, "mps", None)
        info["mps"] = {"detected": bool(mps and mps.is_available())}
    except Exception as err:  # noqa: BLE001 - torch missing or broken driver
        info["error"] = str(err)
    _device_cache.update(info)
    return _device_cache


def device_available(device: str, *, detected: Optional[Dict[str, Any]] = None) -> bool:
    """Check a concrete device against local or hosted discovery without changing it."""
    if device == "cpu":
        return True
    info = detect_devices() if detected is None else detected
    if device == "cuda" or device.startswith("cuda:"):
        index = device.partition(":")[2] if ":" in device else "0"
        cuda = info.get("cuda", {})
        return (index.isascii() and index.isdecimal() and bool(cuda.get("detected"))
                and int(index) < cuda.get("count", 0))
    return device == "mps" and bool(info.get("mps", {}).get("detected"))
