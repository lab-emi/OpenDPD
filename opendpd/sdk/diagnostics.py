"""Environment diagnostics returned as ordinary Python/MATLAB data."""

from __future__ import annotations

import importlib
import platform
import sys


def doctor() -> dict:
    from opendpd import __version__
    from opendpd.services.inference import APPLY_MODELS, EXECUTIONS, streaming_variants

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
            "apply_models": list(APPLY_MODELS), "apply_execution": list(EXECUTIONS),
            "apply_streaming_models": sorted(m.weights_from for m in streaming_variants()),
            "note": "MATLAB release compatibility and MATLAB execution must be checked in MATLAB."}
