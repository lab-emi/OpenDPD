"""Process transport for the MATLAB toolbox: ``python -m opendpd.sdk._fit --job DIR``.

MATLAB writes ``request.json``, ``x.npy`` and ``y.npy`` into DIR and starts this module with any Python that has OpenDPD
installed; no Python-in-MATLAB (``pyenv``) is involved. It appends one JSON line per event to ``progress.jsonl``, stops
when a file named ``cancel`` appears, and writes ``result.json`` or ``error.json``. Exit status: 0 done, 130
cancelled, 1 failed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    job = parser.parse_args().job
    progress = job / "progress.jsonl"

    def emit(event):
        with progress.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, allow_nan=False) + "\n")

    try:
        from .client import SDKError
        from .workflow import fit

        request = json.loads((job / "request.json").read_text(encoding="utf-8"))
        x = np.load(job / "x.npy", allow_pickle=False)
        y = np.load(job / "y.npy", allow_pickle=False)
        result = fit(request.pop("workspace"), x, y, on_event=emit, cancelled=lambda: (job / "cancel").exists(), **request)
        (job / "result.json").write_text(json.dumps(result, allow_nan=False), encoding="utf-8")
        return 0
    except Exception as err:
        code = getattr(err, "code", "fit_failed")
        (job / "error.json").write_text(json.dumps({"code": code, "message": str(err)}), encoding="utf-8")
        return 130 if code == "cancelled" else 1


if __name__ == "__main__":
    raise SystemExit(main())
