"""Isolated model export for the SDK: rebuilds the run's model in its own process and writes the package.

PyTorch is imported here, never in the Python host that MATLAB attaches to.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    try:
        import torch

        torch.set_num_threads(1)         # the same run must give the same bytes
        from opendpd.services.model_export import export_model
        from opendpd.services.workspace import Workspace

        summary = export_model(Workspace.open(Path(args.workspace)), args.run_id, args.destination)
        (args.directory / "summary.json").write_text(json.dumps(summary, allow_nan=False), encoding="utf-8")
        return 0
    except Exception as err:
        (args.directory / "error.json").write_text(json.dumps({
            "code": getattr(err, "code", "export_failed"), "message": str(err)}), encoding="utf-8")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
