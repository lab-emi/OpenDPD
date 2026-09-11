"""``opendpd gui``: start the local service, wait until it answers, open the workbench.

Sequence (plan §3): bind loopback → serve → probe ``/healthz`` → print the
bootstrap URL → open a native window (desktop extra) or the default browser →
block until the window closes or Ctrl+C → stop the supervisor (workers
terminated) → release the workspace lock.

Instance reuse: a running Studio writes ``<workspace>/.studio.lock`` with its
pid, port and bootstrap URL. Starting again on the same workspace opens the
existing instance instead of a second server on the same SQLite store.
"""

from __future__ import annotations

import json
import errno
import locale
import os
import secrets
import socket
import sys
import threading
import time
import urllib.request
import webbrowser
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Optional

DEFAULT_PORT = 8765
PORT_SEARCH = 20
LOCK_FILE = ".studio.lock"
GUARD_FILE = ".studio.guard"
HOST = "127.0.0.1"
Mode = Literal["auto", "window", "browser", "none"]
SHUTDOWN_GRACE_S = 15.0     # supervisor stop timeout (10 s) plus a margin for the server thread to exit


class LaunchError(RuntimeError):
    pass


class WorkspaceBusy(LaunchError):
    pass


@contextmanager
def workspace_guard(workspace: Path):
    """Hold an OS lock for the server lifetime, including startup and shutdown.

    Keep the file in place: unlinking it would let another process lock a new
    inode while an existing waiter still holds the old one. The OS releases
    the lock even when a server crashes before it can remove its metadata.
    """
    fd = os.open(workspace / GUARD_FILE, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            if os.name == "nt":
                import msvcrt
                if os.fstat(fd).st_size == 0:
                    os.write(fd, b"\0")
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as err:
            if err.errno in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                raise WorkspaceBusy(f"workspace is already in use: {workspace}") from err
            raise
        yield
    finally:
        os.close(fd)


@dataclass
class Lock:
    pid: int
    create_time: float
    port: int
    url: str

    @classmethod
    def read(cls, path: Path) -> Optional["Lock"]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return cls(int(data["pid"]), float(data["create_time"]), int(data["port"]), str(data["url"]))
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def alive(self) -> bool:
        from opendpd.runtime.procs import is_same_process
        return is_same_process(self.pid, self.create_time)


def default_workspace() -> Path:
    env = os.environ.get("OPENDPD_WORKSPACE")
    return Path(env).expanduser() if env else Path.home() / "opendpd-workspace"


def port_is_free(port: int, host: str = HOST) -> bool:
    """Mirrors uvicorn's bind (SO_REUSEADDR) so a port left in TIME_WAIT by the
    previous instance is not reported as busy."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


def choose_port(requested: Optional[int]) -> int:
    """An explicit port must be free; otherwise search upwards from the default."""
    if requested is not None:
        if not port_is_free(requested):
            raise LaunchError(f"port {requested} is already in use; pick another with --port or omit it to auto-select")
        return requested
    for port in range(DEFAULT_PORT, DEFAULT_PORT + PORT_SEARCH):
        if port_is_free(port):
            return port
    raise LaunchError(f"no free port in {DEFAULT_PORT}-{DEFAULT_PORT + PORT_SEARCH - 1}; use --port")


def probe(url: str, timeout: float = 1.0) -> Optional[dict]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:   # noqa: S310 - loopback only
            return json.loads(resp.read().decode("utf-8") or "{}")
    except Exception:  # noqa: BLE001 - not up yet / connection refused
        return None


def wait_until_healthy(port: int, deadline_s: float = 30.0, sleep_s: float = 0.2) -> bool:
    end = time.monotonic() + deadline_s
    while time.monotonic() < end:
        if probe(f"http://{HOST}:{port}/healthz") is not None:
            return True
        time.sleep(sleep_s)
    return False


def open_browser(url: str, opener: Callable[[str], bool] = webbrowser.open) -> bool:
    """Returns False (and never raises) when no browser is available, e.g. headless servers."""
    try:
        return bool(opener(url))
    except Exception:  # noqa: BLE001
        return False


def write_lock(workspace: Path, port: int, url: str) -> Path:
    from opendpd.runtime.procs import process_identity
    pid, create_time = process_identity(os.getpid())
    path = workspace / LOCK_FILE
    payload = json.dumps({"pid": pid, "create_time": create_time, "port": port, "url": url})
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(payload)
    return path


def existing_instance(workspace: Path) -> Optional[Lock]:
    lock = Lock.read(workspace / LOCK_FILE)
    if lock is None:
        return None
    if lock.alive():
        # A live server may still be starting or temporarily unresponsive.
        # Its metadata must not be removed by another launcher or `doctor`.
        return lock if probe(f"http://{HOST}:{lock.port}/healthz") is not None else None
    try:
        (workspace / LOCK_FILE).unlink()   # stale lock from a crashed/killed instance
    except OSError:
        pass
    return None


# -- surfaces: native window or browser ----------------------------------------------

class BackgroundServer:
    """uvicorn in a daemon thread: the GUI toolkit needs the main thread."""

    def __init__(self, app, host: str, port: int):
        import uvicorn
        self.server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level="warning"))
        self.thread = threading.Thread(target=self.server.run, name="opendpd-server", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self, timeout: float) -> None:
        self.server.should_exit = True
        self.thread.join(timeout)


def _active_run_count(app) -> int:
    """Runs the supervisor would have to stop if the server went away now."""
    store = getattr(app.state, "store", None)
    if store is None:
        return 0
    from opendpd.schemas import RunStatus
    return sum(store.count_runs(status=s) for s in (RunStatus.queued, RunStatus.running, RunStatus.cancel_requested))


def preferred_language(workspace: Path) -> str:
    """The workspace's stored UI language, else the OS locale when supported, else English."""
    from opendpd.schemas import UI_LANGUAGES
    try:
        from opendpd.services.workspace import Workspace
        stored = Workspace.open(workspace).settings().language
        if stored:
            return stored
    except Exception:  # noqa: BLE001 - no workspace yet or a broken file: never blocks the window
        pass
    try:
        code = (locale.getlocale()[0] or "").split("_")[0].lower()
    except ValueError:
        code = ""
    return code if code in UI_LANGUAGES else "en"


def _default_window_runner(url: str, active_runs: Callable[[], int], workspace: Path) -> None:
    from opendpd.studio import window as window_shell
    from opendpd.studio.strings import shell_strings
    window_shell.run_window(url, active_runs=active_runs, strings=lambda: shell_strings(preferred_language(workspace)),
                            icon=window_shell.icon_path())


def _choose_surface(mode: Mode, availability: Callable[[], "object"]) -> tuple:
    """(surface, note): the surface actually used and, for a fallback, why."""
    if mode in ("browser", "none"):
        return mode, ""
    avail = availability()
    if avail.ok:
        return "window", ""
    if mode == "window":
        return "unavailable", avail.reason
    return "browser", f"native window not available ({avail.reason}); opening the browser instead"


def _block_until_interrupted() -> None:
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


def launch(workspace: Path, *, port: Optional[int] = None, mode: Mode = "auto", out=None, serve=None,
           opener: Callable[[str], bool] = webbrowser.open, window_runner=None, background_server=None,
           availability=None) -> int:
    """Run the Studio server for ``workspace``; returns a process exit code.

    ``mode``: ``auto`` opens a native window when the desktop extra can show one
    and the browser otherwise; ``window`` insists on the window; ``browser``
    always uses the browser; ``none`` only prints the URL.
    """
    out = out or sys.stdout
    workspace = workspace.expanduser().resolve()
    if availability is None:
        from opendpd.studio.window import availability as _availability
        availability = _availability
    surface, note = _choose_surface(mode, availability)
    if surface == "unavailable":
        print(f"error: native window not available: {note}", file=sys.stderr)
        return 2
    if note:
        print(note, file=out)
        _flush(out)
    window_runner = window_runner or (lambda url, active: _default_window_runner(url, active, workspace))
    try:
        workspace.mkdir(parents=True, exist_ok=True)
        with workspace_guard(workspace):
            return _launch_locked(workspace, port=port, surface=surface, out=out, serve=serve, opener=opener,
                                  window_runner=window_runner, background_server=background_server or BackgroundServer)
    except WorkspaceBusy:
        running = existing_instance(workspace)
        if running is not None:
            return _reuse_instance(workspace, running, surface, out, opener, window_runner)
        print(f"error: workspace {workspace} is already starting, running, or shutting down; "
              "wait for that instance or use a different --workspace", file=sys.stderr)
        return 2
    except OSError as err:
        print(f"error: could not launch Studio in {workspace}: {err}; "
              "check workspace permissions and available disk space", file=sys.stderr)
        return 2


def _reuse_instance(workspace: Path, running: Lock, surface: str, out, opener, window_runner) -> int:
    print(f"OpenDPD Studio is already running for {workspace} (pid {running.pid}) at {running.url}", file=out)
    if surface == "window":
        try:
            window_runner(running.url, lambda: 0)     # a window on someone else's server: never asks, never stops it
            return 0
        except Exception as err:  # noqa: BLE001 - toolkit failure: fall back like the launch path
            print(f"warning: the native window failed ({err}); opening the browser instead", file=out)
            surface = "browser"
    if surface == "browser" and not open_browser(running.url, opener):
        print("could not open a browser; open the URL above yourself", file=out)
    return 0


def _launch_locked(workspace: Path, *, port: Optional[int], surface: str, out, serve,
                   opener: Callable[[str], bool], window_runner, background_server) -> int:
    running = existing_instance(workspace)
    if running is not None:
        return _reuse_instance(workspace, running, surface, out, opener, window_runner)

    previous = Lock.read(workspace / LOCK_FILE)
    if previous is not None and previous.alive():
        # Older Studio versions have no OS guard. Never replace a live legacy
        # instance just because its health endpoint has not answered yet.
        print(f"error: a live Studio instance (pid {previous.pid}) owns {workspace} "
              "but is not responding; wait for it or stop that instance first", file=sys.stderr)
        return 2

    try:
        chosen = choose_port(port)
    except LaunchError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2

    token = secrets.token_urlsafe(32)
    url = f"http://{HOST}:{chosen}/bootstrap?token={token}"

    from opendpd.server.app import create_app, static_status
    status = static_status()
    if status["problem"]:
        print(f"warning: {status['problem']} - the page will show a diagnostic instead of the workbench", file=out)

    app = create_app(workspace, bootstrap_token=token)
    lock_path = write_lock(workspace, chosen, url)

    if surface == "window":
        return _serve_with_window(app, chosen, url, lock_path, out=out, opener=opener,
                                  window_runner=window_runner, background_server=background_server)

    def after_ready() -> None:
        if not wait_until_healthy(chosen):
            print("error: the service did not become healthy within 30 s", file=sys.stderr)
            return
        _print_ready(chosen, url, out, stop_hint="Press Ctrl+C to stop.")
        if surface == "browser" and not open_browser(url, opener):
            print("could not open a browser (no desktop session?); open the URL above yourself", file=out)

    threading.Thread(target=after_ready, name="opendpd-launcher", daemon=True).start()
    _sigterm_as_keyboard_interrupt()
    try:
        if serve is None:
            import uvicorn
            uvicorn.run(app, host=HOST, port=chosen, log_level="warning")
        else:
            serve(app, HOST, chosen)
        return 0
    except KeyboardInterrupt:
        return 0
    finally:
        try:
            lock_path.unlink()
        except OSError:
            pass


def _print_ready(port: int, url: str, out, stop_hint: str) -> None:
    ready = probe(f"http://{HOST}:{port}/readyz") or {}
    for problem in ready.get("problems", []):
        print(f"warning: {problem}", file=out)
    print(f"OpenDPD Studio: {url}", file=out)
    print(stop_hint, file=out)
    _flush(out)


def _flush(out) -> None:
    """The URL must reach a piped or redirected stdout while the server is still running."""
    try:
        out.flush()
    except (AttributeError, OSError):
        pass


def _serve_with_window(app, port: int, url: str, lock_path: Path, *, out, opener, window_runner, background_server) -> int:
    """Window mode: the server in a thread, the toolkit on the main thread; closing the window stops the server."""
    server = background_server(app, HOST, port)
    server.start()
    try:
        if not wait_until_healthy(port):
            print("error: the service did not become healthy within 30 s", file=sys.stderr)
            return 2
        _print_ready(port, url, out, stop_hint="Close the window or press Ctrl+C to stop.")
        try:
            window_runner(url, lambda: _active_run_count(app))
        except Exception as err:  # noqa: BLE001 - toolkit failure after the probe said it would work
            print(f"warning: the native window failed ({err}); opening the browser instead", file=out)
            if not open_browser(url, opener):
                print("could not open a browser; open the URL above yourself", file=out)
            _block_until_interrupted()
        return 0
    finally:
        server.stop(timeout=SHUTDOWN_GRACE_S)
        try:
            lock_path.unlink()
        except OSError:
            pass


def _sigterm_as_keyboard_interrupt() -> None:
    """Uvicorn re-raises a captured SIGTERM after its graceful shutdown; turning it
    into KeyboardInterrupt lets the ``finally`` above release the lock file."""
    import signal

    def handler(signum, frame):
        raise KeyboardInterrupt

    try:
        signal.signal(signal.SIGTERM, handler)
    except (ValueError, AttributeError, OSError):
        pass    # not the main thread / platform without SIGTERM handling


def doctor(workspace: Optional[Path] = None, out=None) -> int:
    """Print what `opendpd gui` will rely on; exit 1 when something blocks a launch."""
    import platform

    out = out or sys.stdout

    from opendpd import __version__
    from opendpd.server.app import static_status

    problems = []
    print(f"opendpd {__version__}  python {platform.python_version()}  {platform.platform()}", file=out)
    for name in ("torch", "fastapi", "uvicorn", "psutil", "pydantic"):
        try:
            mod = __import__(name)
            extra = ""
            if name == "torch":
                extra = f"  cuda={'yes' if mod.cuda.is_available() else 'no'}"
                mps = getattr(mod.backends, "mps", None)
                extra += f"  mps={'yes' if mps and mps.is_available() else 'no'}"
            print(f"  {name:<9} {getattr(mod, '__version__', '?')}{extra}", file=out)
        except Exception as err:  # noqa: BLE001
            problems.append(f"{name} not importable: {err}")
            print(f"  {name:<9} MISSING", file=out)
    status = static_status()
    print(f"  frontend  {'present' if status['present'] else 'MISSING'}"
          f"{'  version ' + str(status['version']) if status['version'] else ''}", file=out)
    if not status["present"]:
        problems.append("frontend assets missing: install a release wheel or run `npm run build` in frontend/")
    elif status["problem"]:
        problems.append(status["problem"])
    from opendpd.studio import window as window_shell
    avail = window_shell.availability()
    print(f"  window    {avail.backend if avail.ok else 'not available: ' + avail.reason}", file=out)
    ws = (workspace or default_workspace()).expanduser()
    print(f"  workspace {ws}{'' if ws.exists() else '  (will be created)'}", file=out)
    if ws.exists():
        from opendpd.services.workspace import Workspace
        try:
            problems.extend(Workspace.open_or_create(ws).preflight())
        except Exception as err:  # noqa: BLE001
            problems.append(f"workspace cannot be opened: {err}")
        running = existing_instance(ws)
        if running:
            print(f"  running   pid {running.pid} on port {running.port}", file=out)
    try:
        print(f"  port      {choose_port(None)} free", file=out)
    except LaunchError as err:
        problems.append(str(err))
    for p in problems:
        print(f"problem: {p}", file=out)
    print("ok: ready to launch" if not problems else f"{len(problems)} problem(s)", file=out)
    return 0 if not problems else 1
