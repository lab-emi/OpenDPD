"""Linux VM entrypoint. Configuration and secrets stay out of the browser build."""

import argparse
import os
import resource
from pathlib import Path

from opendpd.web.policy import WebConfig
from opendpd.web.runtime import clear_sessions, prepare_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cleanup", action="store_true", help="purge disposable workspaces after stopping the service")
    parser.add_argument("--host", choices=["127.0.0.1", "0.0.0.0"], default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    root = Path(os.environ.get("OPENDPD_WEB_ROOT", "/run/opendpd-web"))
    if args.cleanup:
        lock = prepare_root(root)
        try:
            clear_sessions(root)
        finally:
            lock.close()
        return
    from opendpd.web.app import create_web_app
    import uvicorn
    os.umask(0o077)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (64 * 1024 * 1024, 64 * 1024 * 1024))
    config = WebConfig(root=root, origin=os.environ["OPENDPD_WEB_ORIGIN"], api_host=os.environ["OPENDPD_WEB_API_HOST"],
                       tunnel_host=os.environ["OPENDPD_WEB_TUNNEL_HOST"])
    uvicorn.run(create_web_app(config), host=args.host, port=args.port, proxy_headers=False,
                access_log=False, limit_concurrency=32, timeout_keep_alive=5)


if __name__ == "__main__":
    main()
