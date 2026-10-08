"""Environment diagnostics returned as ordinary Python/MATLAB data."""

from __future__ import annotations

import importlib
import platform
import sys


def doctor() -> dict:
    from opendpd import __version__

    modules, errors = {}, []
    for name in ("numpy", "scipy", "torch", "pydantic", "fastapi", "uvicorn", "psutil"):
        try:
            module = importlib.import_module(name)
            modules[name] = str(getattr(module, "__version__", "installed"))
        except Exception as err:
            errors.append(f"{name}: {err}")
    return {"api_version": 1, "opendpd_version": __version__, "python": platform.python_version(),
            "executable": sys.executable, "platform": platform.platform(), "modules": modules,
            "ok": not errors, "errors": errors,
            "apply_models": ["gru"], "apply_execution": ["offline_segmented"],
            "note": "MATLAB release compatibility and MATLAB execution must be checked in MATLAB."}
