"""Isolated CPU inference for the SDK: rebuilds the run's model in its own process.

PyTorch is imported here, never in the Python host that MATLAB attaches to.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--execution", default="auto")
    parser.add_argument("--chunk-samples", type=int, default=0)
    args = parser.parse_args()
    try:
        import torch

        torch.set_num_threads(1)
        from opendpd.services.inference import apply_waveform
        from opendpd.services.workspace import Workspace

        x = np.load(args.directory / "input.npy", allow_pickle=False)
        y, metadata = apply_waveform(Workspace.open(Path(args.workspace)), args.run_id, x, execution=args.execution,
                                     chunk_samples=args.chunk_samples or None)
        np.save(args.directory / "output.npy", y, allow_pickle=False)
        (args.directory / "metadata.json").write_text(json.dumps(metadata, allow_nan=False), encoding="utf-8")
        return 0
    except Exception as err:
        (args.directory / "error.json").write_text(json.dumps({
            "code": getattr(err, "code", "inference_failed"), "message": str(err)}), encoding="utf-8")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
