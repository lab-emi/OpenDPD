"""VM process supervised exactly like a local worker; CUDA execution is host-pulled."""
import argparse
import json
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    result = Path(args.workspace) / "runs" / args.run_id / ".gpu-result"
    while not result.exists():
        time.sleep(0.2)
    return int(json.loads(result.read_text())["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
