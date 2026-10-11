"""Spike helper: start a throwaway Studio service in a scratch workspace and keep it up until a stop file appears.

usage: python real_service.py WORKSPACE_DIR
Writes WORKSPACE_DIR/../spike-ready once the service is up and stops it when WORKSPACE_DIR/../spike-stop appears.
Nothing is read from or written to any other workspace.
"""
import sys
import time
from pathlib import Path

from opendpd.sdk import open_project

workspace = Path(sys.argv[1]).resolve()
control = workspace.parent
project = open_project(workspace)
(control / "spike-ready").write_text(str(workspace))
try:
    while not (control / "spike-stop").exists():
        time.sleep(0.2)
finally:
    project.close(stop_service=True)
