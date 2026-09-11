"""Native application window for the Studio (pywebview): a display surface only.

The window loads the same loopback URL a browser would. It has no JavaScript
bridge: the page's Content-Security-Policy refuses injected scripts, and the
window must never be able to script the page. Nothing here is imported by the
browser path; ``availability()`` is the only function the launcher and the
doctor call before deciding to open a window.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from opendpd.studio.strings import ShellStrings

INSTALL_HINT = 'install the desktop extra: pip install "opendpd[desktop]"'
LINUX_HINT = ("install the WebKit2GTK bindings (Debian/Ubuntu: python3-gi gir1.2-webkit2-4.1) "
              'or pip install "pywebview[qt]"')
ICON_DIR = Path(__file__).resolve().parent
DEFAULT_SIZE = (1366, 860)
MIN_SIZE = (960, 600)
TITLE = "OpenDPD Studio"


@dataclass(frozen=True)
class Availability:
    backend: Optional[str]
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.backend is not None


def pywebview_version() -> Optional[str]:
    try:
        from importlib.metadata import version
        return version("pywebview")
    except Exception:  # noqa: BLE001 - not installed
        return None


def desktop_session(platform: str = sys.platform, environ=os.environ) -> Optional[str]:
    """None when a window can be shown; otherwise the reason it cannot."""
    if platform == "darwin":
        try:
            import Quartz
            if Quartz.CGSessionCopyCurrentDictionary() is None:
                return "no graphical login session (SSH or a headless Mac)"
        except Exception as err:  # noqa: BLE001 - pyobjc missing or the window server refused
            return f"cannot query the window server ({err})"
        return None
    if platform.startswith("linux") or platform.endswith("bsd"):
        if not (environ.get("DISPLAY") or environ.get("WAYLAND_DISPLAY")):
            return "no display (DISPLAY and WAYLAND_DISPLAY are unset)"
    return None


def gui_backend() -> str:
    """Name of the pywebview backend usable here ("cocoa", "edgechromium", "gtk", "qt"); raises otherwise."""
    import importlib
    # The package attribute ``webview.guilib`` is a placeholder pywebview fills at start();
    # the probe needs the submodule of the same name.
    guilib = importlib.import_module("webview.guilib")
    guilib.initialize()
    return guilib.guilib.__name__.rsplit(".", 1)[-1]


def availability(platform: str = sys.platform, environ=os.environ) -> Availability:
    version = pywebview_version()
    if version is None:
        return Availability(None, f"pywebview is not installed; {INSTALL_HINT}")
    reason = desktop_session(platform, environ)
    if reason:
        return Availability(None, reason)
    try:
        backend = gui_backend()
    except Exception as err:  # noqa: BLE001 - WebViewException or a backend import error
        hint = LINUX_HINT if platform.startswith("linux") else INSTALL_HINT
        return Availability(None, f"{err}; {hint}")
    return Availability(f"{backend} (pywebview {version})")


def icon_path() -> Optional[Path]:
    path = ICON_DIR / ("icon.ico" if os.name == "nt" else "icon.png")
    return path if path.is_file() else None


def should_close(active: int, ask: Callable[[str, str], bool], strings: ShellStrings, out=None) -> bool:
    """Close policy: never ask when nothing runs; ask with the count otherwise; a broken dialog never blocks the close."""
    if active <= 0:
        return True
    try:
        return bool(ask(strings.quit_title, strings.quit_body.format(count=active)))
    except Exception as err:  # noqa: BLE001 - backend without dialogs
        print(f"warning: could not ask before closing ({err}); stopping the running experiments",
              file=out or sys.stderr)
        return True


def _name_the_application(title: str) -> None:
    """macOS shows the process name ("Python") in the application menu unless the bundle says otherwise."""
    if sys.platform != "darwin":
        return
    try:
        from Foundation import NSBundle
        NSBundle.mainBundle().infoDictionary()["CFBundleName"] = title
    except Exception:  # noqa: BLE001 - cosmetic only
        pass


# -- macOS: Ctrl+C and Cmd+Q must end in a window close, never in a process exit -----------

_mac_quit_target = None          # the pywebview window the quit delegate closes
_mac_quit_delegate = None        # kept alive for the lifetime of the process
_mac_quit_delegate_class = None  # an Objective-C class may be defined only once per process


def _macos_quit_delegate_class():
    global _mac_quit_delegate_class
    if _mac_quit_delegate_class is None:
        import AppKit
        import Foundation

        class OpenDPDQuitDelegate(AppKit.NSObject):
            def applicationShouldTerminate_(self, app):
                # Quit from the menu behaves like the close button: it may ask, and the window
                # close is what ends the run loop, so webview.start() returns and the launcher cleans up.
                target = _mac_quit_target
                try:
                    target.native.performClose_(None)
                except Exception:  # noqa: BLE001 - no native handle: close without asking
                    target.destroy()
                return Foundation.NO     # NSTerminateCancel: AppKit must never exit the process itself

            def applicationSupportsSecureRestorableState_(self, app):
                return Foundation.YES

        _mac_quit_delegate_class = OpenDPDQuitDelegate
    return _mac_quit_delegate_class


def _install_macos_hooks(window, finished) -> None:
    """Runs in a helper thread once the Cocoa loop is up.

    pywebview installs pyobjc's Mach interrupt handler for Ctrl+C and an
    application delegate for Cmd+Q; both end in ``NSApplication.terminate``,
    which exits the process without returning from ``webview.start()``. The
    launcher's cleanup (supervisor stop, workers terminated, lock released)
    runs after that return, so the delegate is replaced by one that answers
    ``terminate`` with a window close: Ctrl+C and Cmd+Q then behave exactly
    like the close button, including the question while runs are active.
    """
    import time

    import AppKit
    from PyObjCTools import AppHelper

    global _mac_quit_target, _mac_quit_delegate
    window.events.shown.wait()
    app = AppKit.NSApplication.sharedApplication()
    deadline = time.monotonic() + 10.0
    while not app.isRunning() and time.monotonic() < deadline and not finished.is_set():
        time.sleep(0.05)
    if finished.is_set():
        return
    _mac_quit_target = window
    _mac_quit_delegate = _macos_quit_delegate_class().alloc().init()
    AppHelper.callAfter(app.setDelegate_, _mac_quit_delegate)     # on the main thread, like every AppKit call


def run_window(url: str, *, active_runs: Callable[[], int], strings: Callable[[], ShellStrings],
               title: str = TITLE, icon: Optional[Path] = None, out=None) -> None:
    """Show the workbench in a native window and return when it is closed.

    ``strings()`` is read when the window opens (menus, dialogs of the toolkit)
    and again at every close request (the quit question), so a language change
    made in the page is honoured without restarting.

    Ctrl+C / SIGTERM: the main thread sits in the toolkit's native run loop and
    executes no Python there, so a Python-level signal handler would never run.
    The interpreter's C-level handler writes to a wake-up socket instead; a
    watcher thread reads it and destroys the window, which ends the run loop.
    """
    import select
    import signal
    import socket
    import threading

    import webview

    webview.settings["ALLOW_DOWNLOADS"] = True
    webview.settings["OPEN_EXTERNAL_LINKS_IN_BROWSER"] = True
    _name_the_application(title)
    window = webview.create_window(title, url, width=DEFAULT_SIZE[0], height=DEFAULT_SIZE[1],
                                   min_size=MIN_SIZE, text_select=True)

    def on_closing():
        return should_close(active_runs(), window.create_confirmation_dialog, strings(), out)

    window.events.closing += on_closing

    finished = threading.Event()
    wake_r, wake_w = socket.socketpair()
    wake_r.setblocking(False)
    wake_w.setblocking(False)

    def watcher() -> None:
        while not finished.is_set():
            try:
                ready, _, _ = select.select([wake_r], [], [], 0.5)
            except (OSError, ValueError):     # sockets closed underneath: the window is gone
                return
            if ready and not finished.is_set():
                try:
                    window.destroy()
                except Exception:  # noqa: BLE001 - already closing
                    pass
                return

    def request_stop(signum, frame):
        pass    # the wake-up socket carries the request; nothing to do when Python runs again

    previous_handlers = {}
    for sig in (signal.SIGINT, getattr(signal, "SIGTERM", None)):
        if sig is None:
            continue
        try:
            previous_handlers[sig] = signal.signal(sig, request_stop)
        except (ValueError, OSError):
            pass    # not the main thread
    try:
        previous_fd = signal.set_wakeup_fd(wake_w.fileno(), warn_on_full_buffer=False)
    except (ValueError, OSError):
        previous_fd = None
    watcher_thread = threading.Thread(target=watcher, name="opendpd-window-signals", daemon=True)
    watcher_thread.start()
    if sys.platform == "darwin":
        threading.Thread(target=_install_macos_hooks, args=(window, finished), name="opendpd-window-macos",
                         daemon=True).start()
    try:
        webview.start(localization=strings().localization, private_mode=True,
                      icon=str(icon) if icon else None)
    finally:
        finished.set()
        watcher_thread.join(1.0)
        if previous_fd is not None:
            try:
                signal.set_wakeup_fd(previous_fd)
            except (ValueError, OSError):
                pass
        for sig, handler in previous_handlers.items():
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                pass
        wake_r.close()
        wake_w.close()
