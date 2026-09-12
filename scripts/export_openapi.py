"""Export the OpenAPI document (source of the frontend types).

    python scripts/export_openapi.py            # writes docs/contracts/openapi.json
    python scripts/export_openapi.py --check    # exit 1 when the committed file is stale
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "docs" / "contracts" / "openapi.json"


def current_schema() -> dict:
    from opendpd.server.app import create_app

    with tempfile.TemporaryDirectory() as tmp:
        app = create_app(Path(tmp), bootstrap_token="export")
        return app.openapi()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    text = json.dumps(current_schema(), indent=2, sort_keys=True) + "\n"
    if args.check:
        if not TARGET.exists() or TARGET.read_text() != text:
            print(f"{TARGET} is stale; run python scripts/export_openapi.py", file=sys.stderr)
            return 1
        print("openapi.json is up to date")
        return 0
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(text)
    print(f"wrote {TARGET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
