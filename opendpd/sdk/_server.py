"""Studio service with a cross-platform graceful stop signal for SDK clients."""

from __future__ import annotations

import argparse
import json
import os
import socket
import threading
import uuid
from pathlib import Path

SERVICE_FILE = ".sdk-service.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    workspace = args.workspace.expanduser().resolve()

    from opendpd.runtime.procs import process_identity
    from opendpd.services.workspace import write_json_atomic
    from opendpd.studio.launcher import HOST, launch

    def serve(app, host, port):
        import uvicorn

        pid, created = process_identity(os.getpid())
        stop_name = f".sdk-stop-{uuid.uuid4().hex}"
        stop_file = workspace / stop_name
        meta_file = workspace / SERVICE_FILE
        write_json_atomic(meta_file, {"pid": pid, "create_time": created, "stop_file": stop_name})
        server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level="warning",
                                             timeout_graceful_shutdown=1))
        done = threading.Event()

        def watch():
            while not done.wait(0.1):
                if stop_file.exists():
                    server.should_exit = True
                    return

        watcher = threading.Thread(target=watch, name="sdk-stop", daemon=True)
        watcher.start()
        try:
            server.run(sockets=[listener])
        finally:
            done.set()
            watcher.join()
            stop_file.unlink(missing_ok=True)
            try:
                if json.loads(meta_file.read_text())["pid"] == pid:
                    meta_file.unlink()
            except (OSError, ValueError, KeyError):
                pass

    # Keep the OS-selected port reserved until uvicorn takes ownership. Separate
    # workspaces starting together must never advertise the same unbound port.
    with socket.create_server((HOST, 0)) as listener:
        return launch(workspace, mode="none", serve=serve, reserved_listener=listener)


if __name__ == "__main__":
    raise SystemExit(main())
