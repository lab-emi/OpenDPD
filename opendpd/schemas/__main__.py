"""Export helpers: ``python -m opendpd.schemas export-mocks --out DIR``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pydantic import BaseModel

from .examples import all_examples


def dump(obj) -> object:
    if isinstance(obj, list):
        return [dump(o) for o in obj]
    assert isinstance(obj, BaseModel)
    return obj.model_dump(mode="json")


def export_mocks(out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for name, example in all_examples().items():
        path = out / f"{name}.json"
        payload = {"_mock": True, "_note": "Generated from opendpd.schemas.examples; do not edit by hand.",
                   "data": dump(example)}
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        written.append(path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m opendpd.schemas")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("export-mocks", help="write mock fixtures generated from the contract examples")
    p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "export-mocks":
        for path in export_mocks(args.out):
            print(path)


if __name__ == "__main__":
    main()
