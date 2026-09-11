# Native Window Shell Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `opendpd gui` opens the Studio workbench in a native application window (pywebview) when the `desktop` extra is installed, with the browser as the unchanged fallback.

**Architecture:** The launcher decides a *surface* (window, browser or none) before starting anything. In window mode uvicorn runs in a daemon thread and the GUI toolkit owns the main thread; closing the window (or Ctrl+C) stops the server through the existing supervisor shutdown, after a native confirmation when runs are active. The window is a display surface only: no JavaScript bridge, same loopback URL, same session/CSRF/CSP boundary.

**Tech Stack:** Python 3.10+, pywebview 6.2 (WKWebView on macOS, WebView2 on Windows, WebKit2GTK/Qt on Linux), uvicorn `Server` in a thread, Pillow for the icon, pytest.

Spec: `docs/superpowers/specs/2026-09-11-native-window-shell-design.md`.

## Global Constraints

- Code comments, docstrings, commit messages and documentation are in English.
- `import opendpd`, `opendpd.commands` and the browser path never import pywebview (`tests/unit/test_lazy_imports.py` must stay green).
- Dependency pin: `pywebview>=6.2,<7`, only in the new `desktop` extra.
- No JavaScript bridge (`js_api`) and no change to `opendpd/server/security.py`.
- Flags `--browser`, `--window`, `--no-browser` are mutually exclusive; `--no-browser` keeps its exact behaviour.
- Windows and Linux stay *unverified* in the support matrix; only what runs here (macOS Apple Silicon) may be recorded as verified.
- Every commit ends with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

---

## File structure

| File | Responsibility |
|---|---|
| `opendpd/studio/strings.py` (new) | `ShellStrings` dataclass and the English strings the host shows outside the page |
| `opendpd/studio/window.py` (new) | availability probe, close policy, `run_window` (the only module that imports pywebview) |
| `opendpd/studio/launcher.py` (modify) | surface decision, background uvicorn thread, window lifecycle, doctor line |
| `opendpd/commands.py` (modify) | `gui` flags → `mode` |
| `scripts/make_app_icon.py` (new), `opendpd/studio/icon.png`, `opendpd/studio/icon.ico` (generated) | application icon from the favicon geometry |
| `pyproject.toml`, `MANIFEST.in`, `.github/workflows/ci.yml` (modify) | `desktop` extra, package data, CI installs the extra |
| `tests/unit/test_window.py` (new), `tests/unit/test_launcher.py`, `tests/integration/test_launcher_ownership.py` (modify) | L0 coverage without pywebview |
| `docs/architecture/adr/0002-native-window-shell.md` (new), `docs/tutorials/gui-quickstart.md`, `docs/releases/support-matrix.md`, `docs/releases/release-notes-2.2.0.md`, `docs/releases/backlog.md`, `README.md`, `OpenDPD_Studio_Development_Plan.md` (modify) | decision record and user documentation |

---

### Task 1: Shell strings and the window module (no launcher changes yet)

**Files:**
- Create: `opendpd/studio/strings.py`
- Create: `opendpd/studio/window.py`
- Test: `tests/unit/test_window.py`

**Interfaces:**
- Produces: `ShellStrings(quit_title: str, quit_body: str, localization: Dict[str, str])`, `ENGLISH: ShellStrings`, `shell_strings(language: Optional[str] = None) -> ShellStrings`.
- Produces: `Availability(backend: Optional[str], reason: str = "")` with `.ok`; `availability(platform=sys.platform, environ=os.environ) -> Availability`; `should_close(active: int, ask: Callable[[str, str], bool], strings: ShellStrings, out=None) -> bool`; `run_window(url: str, *, active_runs: Callable[[], int], strings: Callable[[], ShellStrings], title: str = "OpenDPD Studio", icon: Optional[Path] = None, out=None) -> None`; `icon_path() -> Optional[Path]`.
- Test seams: module functions `pywebview_version()`, `desktop_session(platform, environ)`, `gui_backend()` are monkeypatched by tests.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_window.py
"""Native window shell: availability probe and close policy, without pywebview (L0)."""

from opendpd.studio import window
from opendpd.studio.strings import ENGLISH, shell_strings


def test_should_close_without_active_runs_never_asks():
    asked = []
    assert window.should_close(0, lambda *a: asked.append(a) or False, ENGLISH) is True
    assert asked == []


def test_should_close_asks_with_the_count_and_respects_the_answer():
    asked = []

    def ask(title, body):
        asked.append((title, body))
        return False

    assert window.should_close(2, ask, ENGLISH) is False
    assert asked == [("Quit OpenDPD Studio?", "2 experiment(s) are running. Quit OpenDPD Studio and stop them?")]
    assert window.should_close(1, lambda title, body: True, ENGLISH) is True


def test_should_close_allows_the_close_when_the_dialog_fails(capsys):
    def broken(title, body):
        raise RuntimeError("no dialog")

    assert window.should_close(3, broken, ENGLISH) is True
    assert "could not ask before closing" in capsys.readouterr().err


def test_availability_names_the_extra_when_pywebview_is_missing(monkeypatch):
    monkeypatch.setattr(window, "pywebview_version", lambda: None)
    a = window.availability()
    assert not a.ok and 'pip install "opendpd[desktop]"' in a.reason


def test_availability_needs_a_display_on_linux(monkeypatch):
    monkeypatch.setattr(window, "pywebview_version", lambda: "6.2.1")
    a = window.availability(platform="linux", environ={})
    assert not a.ok and "DISPLAY" in a.reason


def test_availability_reports_the_backend(monkeypatch):
    monkeypatch.setattr(window, "pywebview_version", lambda: "6.2.1")
    monkeypatch.setattr(window, "desktop_session", lambda platform, environ: None)
    monkeypatch.setattr(window, "gui_backend", lambda: "cocoa")
    a = window.availability()
    assert a.ok and a.backend == "cocoa (pywebview 6.2.1)" and a.reason == ""


def test_availability_reports_a_missing_backend_with_the_linux_hint(monkeypatch):
    monkeypatch.setattr(window, "pywebview_version", lambda: "6.2.1")
    monkeypatch.setattr(window, "desktop_session", lambda platform, environ: None)

    def boom():
        raise RuntimeError("You must have either QT or GTK with Python extensions installed")

    monkeypatch.setattr(window, "gui_backend", boom)
    a = window.availability(platform="linux", environ={"DISPLAY": ":0"})
    assert not a.ok and "QT or GTK" in a.reason and "gir1.2-webkit2" in a.reason


def test_english_strings_are_complete():
    s = shell_strings()
    assert "{count}" in s.quit_body
    assert {"global.quit", "global.cancel", "global.ok", "global.saveFile", "global.quitConfirmation"} <= set(s.localization)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_window.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'opendpd.studio.window'`.

- [ ] **Step 3: Write `opendpd/studio/strings.py`**

```python
"""Strings the desktop window shows outside the page: the quit dialog and pywebview's own dialogs/menus.

The page translates itself (frontend/src/i18n). These few host-side strings
start in English; the languages plan adds the other six.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass(frozen=True)
class ShellStrings:
    quit_title: str
    quit_body: str                                   # carries a {count} placeholder
    localization: Dict[str, str] = field(default_factory=dict)   # pywebview localization keys


ENGLISH = ShellStrings(
    quit_title="Quit OpenDPD Studio?",
    quit_body="{count} experiment(s) are running. Quit OpenDPD Studio and stop them?",
    localization={
        "global.quit": "Quit",
        "global.cancel": "Cancel",
        "global.ok": "OK",
        "global.saveFile": "Save file",
        "global.quitConfirmation": "Do you really want to quit?",
    },
)


def shell_strings(language: Optional[str] = None) -> ShellStrings:
    """Strings for a UI language code; English until the other languages exist."""
    return ENGLISH
```

- [ ] **Step 4: Write `opendpd/studio/window.py`**

```python
"""Native application window for the Studio (pywebview): a display surface only.

The window loads the same loopback URL a browser would. It has no JavaScript
bridge: the page's Content-Security-Policy refuses injected scripts, and the
window must never be able to script the page. Nothing here is imported by the
browser path; `availability()` is the only function the launcher and the
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
    from webview import guilib
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


def run_window(url: str, *, active_runs: Callable[[], int], strings: Callable[[], ShellStrings],
               title: str = TITLE, icon: Optional[Path] = None, out=None) -> None:
    """Show the workbench in a native window and return when it is closed.

    ``strings()`` is read when the window opens (menus, dialogs of the toolkit)
    and again at every close request (the quit question), so a language change
    made in the page is honoured without restarting.
    """
    import signal

    import webview

    webview.settings["ALLOW_DOWNLOADS"] = True
    webview.settings["OPEN_EXTERNAL_LINKS_IN_BROWSER"] = True
    _name_the_application(title)
    window = webview.create_window(title, url, width=DEFAULT_SIZE[0], height=DEFAULT_SIZE[1],
                                   min_size=MIN_SIZE, text_select=True)

    def on_closing():
        return should_close(active_runs(), window.create_confirmation_dialog, strings(), out)

    window.events.closing += on_closing

    def on_signal(signum, frame):
        window.destroy()

    previous = {}
    for sig in (signal.SIGINT, getattr(signal, "SIGTERM", None)):
        if sig is None:
            continue
        try:
            previous[sig] = signal.signal(sig, on_signal)
        except (ValueError, OSError):
            pass    # not the main thread
    try:
        webview.start(localization=strings().localization, private_mode=True,
                      icon=str(icon) if icon else None)
    finally:
        for sig, handler in previous.items():
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                pass
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/test_window.py tests/unit/test_lazy_imports.py -q`
Expected: all PASS (the lazy-import test proves `opendpd.commands` still loads without pywebview).

- [ ] **Step 6: Commit**

```bash
git add opendpd/studio/strings.py opendpd/studio/window.py tests/unit/test_window.py
git commit -m "feat(studio): native window module with availability probe and close policy

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: Launcher surface decision, window lifecycle, CLI flags, doctor line

**Files:**
- Modify: `opendpd/studio/launcher.py` (signature of `launch`, `_reuse_instance`, `_launch_locked`, `doctor`)
- Modify: `opendpd/commands.py:591-595` (`cmd_gui`) and `:1045-1048` (the `gui` parser)
- Modify: `tests/unit/test_launcher.py` (call sites `open_in_browser=` → `mode=`), `tests/integration/test_launcher_ownership.py:46`
- Test: `tests/unit/test_launcher.py`

**Interfaces:**
- Consumes: `opendpd.studio.window.availability`, `run_window`, `icon_path`; `opendpd.studio.strings.shell_strings`.
- Produces: `Mode = Literal["auto", "window", "browser", "none"]`; `launch(workspace, *, port=None, mode: Mode = "auto", out=None, serve=None, opener=webbrowser.open, window_runner=None, background_server=None, availability=None) -> int`; `BackgroundServer(app, host, port)` with `start()` and `stop(timeout)`; `window_runner(url: str, active_runs: Callable[[], int]) -> None` (default: `_default_window_runner`); `_active_run_count(app) -> int`.
- The old keyword `open_in_browser` is removed (`True` → `mode="browser"`, `False` → `mode="none"`).

- [ ] **Step 1: Update the existing call sites and write the failing tests**

In `tests/unit/test_launcher.py` replace `open_in_browser=True` with `mode="browser"` (line 53) and `open_in_browser=False` with `mode="none"` (lines 102, 121). In `tests/integration/test_launcher_ownership.py:46` replace `open_in_browser=False` with `mode="none"`. Then append:

```python
from opendpd.studio import window


class FakeServer:
    """Stands in for the uvicorn thread: 'healthy' as soon as started, records stop()."""

    def __init__(self, healthy):
        self.healthy = healthy
        self.started = False
        self.stopped = None

    def start(self):
        self.started = True
        self.healthy.set()

    def stop(self, timeout):
        self.stopped = timeout


def _fake_probe(healthy):
    return lambda url, timeout=1.0: {"status": "ok"} if healthy.is_set() else None


def test_window_mode_serves_in_the_background_and_stops_when_the_window_closes(tmp_path, monkeypatch, capsys):
    healthy = threading.Event()
    servers = []

    def factory(app, host, port):
        assert host == "127.0.0.1"
        servers.append(FakeServer(healthy))
        return servers[-1]

    monkeypatch.setattr(launcher, "probe", _fake_probe(healthy))
    seen = {}

    def runner(url, active_runs):
        assert servers[0].started and servers[0].stopped is None, "the window opens while the server runs"
        seen["url"], seen["active"] = url, active_runs()

    ws = tmp_path / "ws"
    rc = launcher.launch(ws, mode="window", availability=lambda: window.Availability("fake"),
                         window_runner=runner, background_server=factory)
    assert rc == 0
    assert seen["url"].startswith("http://127.0.0.1:") and "/bootstrap?token=" in seen["url"]
    assert seen["active"] == 0
    assert servers[0].stopped is not None, "closing the window stops the server"
    assert not (ws / launcher.LOCK_FILE).exists(), "lock released on exit"
    out = capsys.readouterr().out
    assert "Close the window or press Ctrl+C to stop." in out


def test_window_mode_gives_up_when_the_service_never_becomes_healthy(tmp_path, monkeypatch, capsys):
    servers = []

    def factory(app, host, port):
        servers.append(FakeServer(threading.Event()))
        return servers[-1]

    monkeypatch.setattr(launcher, "probe", lambda url, timeout=1.0: None)
    monkeypatch.setattr(launcher, "wait_until_healthy", lambda port, deadline_s=30.0, sleep_s=0.2: False)
    opened = []
    rc = launcher.launch(tmp_path / "ws", mode="window", availability=lambda: window.Availability("fake"),
                         window_runner=lambda url, active_runs: opened.append(url), background_server=factory)
    assert rc == 2 and opened == [] and servers[0].stopped is not None
    assert "did not become healthy" in capsys.readouterr().err


def test_auto_mode_falls_back_to_the_browser_with_a_reason(tmp_path, monkeypatch, capsys):
    opened = []
    healthy = threading.Event()

    def fake_serve(app, host, port):
        time.sleep(0.3)
        healthy.set()
        time.sleep(0.5)

    monkeypatch.setattr(launcher, "probe", _fake_probe(healthy))
    rc = launcher.launch(tmp_path / "ws", mode="auto", serve=fake_serve,
                         availability=lambda: window.Availability(None, "pywebview is not installed; install it"),
                         opener=lambda url: opened.append(url) or True)
    assert rc == 0 and len(opened) == 1
    out = capsys.readouterr().out
    assert "native window not available (pywebview is not installed" in out and "opening the browser instead" in out


def test_window_mode_refuses_without_a_backend(tmp_path, capsys):
    served = []
    rc = launcher.launch(tmp_path / "ws", mode="window", serve=lambda *a: served.append(a),
                         availability=lambda: window.Availability(None, "no display"))
    assert rc == 2 and served == []
    assert "native window not available: no display" in capsys.readouterr().err


def test_a_failing_window_falls_back_to_the_browser_and_keeps_serving(tmp_path, monkeypatch, capsys):
    healthy = threading.Event()
    servers = []

    def factory(app, host, port):
        servers.append(FakeServer(healthy))
        return servers[-1]

    monkeypatch.setattr(launcher, "probe", _fake_probe(healthy))
    monkeypatch.setattr(launcher, "_block_until_interrupted", lambda: None)
    opened = []

    def broken(url, active_runs):
        raise RuntimeError("no window server")

    rc = launcher.launch(tmp_path / "ws", mode="window", availability=lambda: window.Availability("fake"),
                         window_runner=broken, background_server=factory, opener=lambda url: opened.append(url) or True)
    assert rc == 0 and len(opened) == 1 and servers[0].stopped is not None
    assert "the native window failed (no window server); opening the browser instead" in capsys.readouterr().out


def test_running_instance_gets_a_window_in_window_mode(tmp_path, monkeypatch):
    ws = tmp_path / "ws"
    ws.mkdir()
    from opendpd.runtime.procs import process_identity
    pid, ctime = process_identity(os.getpid())
    (ws / launcher.LOCK_FILE).write_text(json.dumps({"pid": pid, "create_time": ctime, "port": 8791,
                                                      "url": "http://127.0.0.1:8791/bootstrap?token=abc"}))
    monkeypatch.setattr(launcher, "probe", lambda url, timeout=1.0: {"status": "ok"})
    served, shown, opened = [], [], []
    rc = launcher.launch(ws, mode="auto", serve=lambda *a: served.append(a), availability=lambda: window.Availability("fake"),
                         window_runner=lambda url, active_runs: shown.append((url, active_runs())),
                         opener=lambda url: opened.append(url) or True)
    assert rc == 0 and served == [] and opened == []
    assert shown == [("http://127.0.0.1:8791/bootstrap?token=abc", 0)]


def test_gui_flags_map_to_modes(monkeypatch):
    from opendpd import commands
    calls = []
    monkeypatch.setattr("opendpd.studio.launcher.launch", lambda ws, **kw: calls.append(kw) or 0)
    parser = commands.build_parser()
    for argv in (["gui"], ["gui", "--window"], ["gui", "--browser"], ["gui", "--no-browser"]):
        args = parser.parse_args(argv)
        assert args.func(args) == 0
    assert [c["mode"] for c in calls] == ["auto", "window", "browser", "none"]
    with pytest.raises(SystemExit):
        parser.parse_args(["gui", "--window", "--browser"])


def test_doctor_reports_the_window_backend(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(window, "availability", lambda: window.Availability(None, "no display"))
    launcher.doctor(tmp_path / "ws")
    out = capsys.readouterr().out
    assert "window    not available: no display" in out
    assert "problem: no display" not in out, "a missing window is never a blocking problem"
```

Also extend `test_doctor_reports_frontend_and_port`: add `assert "window" in out`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_launcher.py -q`
Expected: the new tests FAIL with `TypeError: launch() got an unexpected keyword argument 'mode'` (and the CLI test with `argparse` errors); the updated old tests also fail until `launch` accepts `mode`.

- [ ] **Step 3: Implement the launcher changes**

In `opendpd/studio/launcher.py`:

1. Extend the imports: `from typing import Callable, Literal, Optional` and add after the constants:

```python
Mode = Literal["auto", "window", "browser", "none"]
SHUTDOWN_GRACE_S = 15.0     # supervisor stop timeout (10 s) plus a margin for the server thread to exit


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


def _default_window_runner(url: str, active_runs: Callable[[], int]) -> None:
    from opendpd.studio import window as window_shell
    from opendpd.studio.strings import shell_strings
    window_shell.run_window(url, active_runs=active_runs, strings=shell_strings, icon=window_shell.icon_path())


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
```

2. Replace `launch` and `_reuse_instance`:

```python
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
    window_runner = window_runner or _default_window_runner
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
```

3. In `_launch_locked`, change the signature to `(workspace, *, port, surface, out, serve, opener, window_runner, background_server)`, replace the `running is not None` line with `return _reuse_instance(workspace, running, surface, out, opener, window_runner)`, and replace everything from `def after_ready()` to the end of the function with:

```python
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
```

4. In `doctor`, after the `frontend` print block add:

```python
    from opendpd.studio import window as window_shell
    avail = window_shell.availability()
    print(f"  window    {avail.backend if avail.ok else 'not available: ' + avail.reason}", file=out)
```

5. Update the module docstring's first paragraph: "bind loopback → serve → probe /healthz → print the bootstrap URL → open a native window (desktop extra) or the default browser → block until the window closes or Ctrl+C → stop the supervisor (workers terminated) → release the workspace lock."

In `opendpd/commands.py`:

```python
def cmd_gui(args) -> int:
    from opendpd.studio.launcher import default_workspace, launch

    workspace = Path(args.workspace) if args.workspace else default_workspace()
    mode = "none" if args.no_browser else "browser" if args.browser else "window" if args.window else "auto"
    return launch(workspace, port=args.port, mode=mode)
```

and the parser:

```python
    p = sub.add_parser("gui", help="start the local Studio service and open the workbench "
                                   "(a native window when opendpd[desktop] is installed, else your browser)")
    p.add_argument("--workspace", default=None, help="workspace directory (default: $OPENDPD_WORKSPACE or ~/opendpd-workspace)")
    p.add_argument("--port", type=int, default=None, help="loopback port (default: first free from 8765)")
    surface = p.add_mutually_exclusive_group()
    surface.add_argument("--no-browser", dest="no_browser", action="store_true",
                         help="print the URL only (SSH sessions, servers without a desktop)")
    surface.add_argument("--browser", action="store_true", help="open the system browser even when the native window is available")
    surface.add_argument("--window", action="store_true", help="require the native window; fail when it is not available")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/test_launcher.py tests/unit/test_window.py tests/unit/test_lazy_imports.py tests/integration/test_launcher_ownership.py -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add opendpd/studio/launcher.py opendpd/commands.py tests/unit/test_launcher.py tests/integration/test_launcher_ownership.py
git commit -m "feat(studio): opendpd gui opens a native window when available, browser otherwise

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: Application icon

**Files:**
- Create: `scripts/make_app_icon.py`
- Create (generated, committed): `opendpd/studio/icon.png`, `opendpd/studio/icon.ico`
- Modify: `pyproject.toml` (`[tool.setuptools.package-data]`), `MANIFEST.in`
- Test: `tests/unit/test_window.py`

**Interfaces:**
- Consumes: `window.icon_path()` from Task 1.
- Produces: the two icon files as package data.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_window.py`:

```python
def test_icon_files_ship_with_the_package(monkeypatch):
    for name in ("icon.png", "icon.ico"):
        assert (window.ICON_DIR / name).is_file(), name
    monkeypatch.setattr(window.os, "name", "posix")
    assert window.icon_path().name == "icon.png"
    monkeypatch.setattr(window.os, "name", "nt")
    assert window.icon_path().name == "icon.ico"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_window.py::test_icon_files_ship_with_the_package -q`
Expected: FAIL on `icon.png` missing.

- [ ] **Step 3: Write the generator and run it**

```python
# scripts/make_app_icon.py
"""Render the Studio application icon (Dock / taskbar) from the favicon geometry.

    python scripts/make_app_icon.py     # writes opendpd/studio/icon.png and icon.ico

Same shapes as frontend/public/favicon.svg: a rounded #0B5FA5 square and a
white five-point polyline. Drawn 8x oversampled and downscaled for smooth edges.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "opendpd" / "studio"
SIZE, SCALE = 256, 8
BLUE, WHITE = (11, 95, 165, 255), (255, 255, 255, 255)
POINTS = [(6, 22), (11, 12), (16, 20), (21, 8), (26, 22)]     # favicon viewBox 0..32


def render() -> Image.Image:
    big = SIZE * SCALE
    unit = big / 32
    img = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((0, 0, big - 1, big - 1), radius=int(6 * unit), fill=BLUE)
    pts = [(x * unit, y * unit) for x, y in POINTS]
    width = int(3 * unit)
    draw.line(pts, fill=WHITE, width=width, joint="curve")
    for x, y in (pts[0], pts[-1]):          # round caps
        r = width / 2
        draw.ellipse((x - r, y - r, x + r, y + r), fill=WHITE)
    return img.resize((SIZE, SIZE), Image.LANCZOS)


def main() -> None:
    img = render()
    img.save(OUT / "icon.png")
    img.save(OUT / "icon.ico", sizes=[(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)])
    print(f"wrote {OUT / 'icon.png'} and {OUT / 'icon.ico'}")


if __name__ == "__main__":
    main()
```

Run: `.venv/bin/python scripts/make_app_icon.py`

- [ ] **Step 4: Ship the files**

In `pyproject.toml` change the package-data line to `"opendpd.studio" = ["static/*", "static/assets/*", "icon.png", "icon.ico"]`. In `MANIFEST.in` add `include opendpd/studio/icon.png` and `include opendpd/studio/icon.ico` under the Studio block.

- [ ] **Step 5: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/test_window.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/make_app_icon.py opendpd/studio/icon.png opendpd/studio/icon.ico pyproject.toml MANIFEST.in tests/unit/test_window.py
git commit -m "feat(studio): application icon for the native window

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: Packaging extra, CI, documentation and ADR

**Files:**
- Modify: `pyproject.toml` (`[project.optional-dependencies]`), `.github/workflows/ci.yml` (Python test job install line)
- Create: `docs/architecture/adr/0002-native-window-shell.md`
- Modify: `docs/tutorials/gui-quickstart.md`, `docs/releases/support-matrix.md`, `docs/releases/release-notes-2.2.0.md`, `docs/releases/backlog.md`, `README.md` (Studio paragraph), `OpenDPD_Studio_Development_Plan.md` (§3.1 amendment)

- [ ] **Step 1: Add the extra and the CI install**

`pyproject.toml`, after the `gui` extra:

```toml
# Native application window for `opendpd gui` (pywebview: WKWebView on macOS, WebView2 on Windows,
# WebKit2GTK or Qt on Linux). Pure Python; the browser stays the fallback.
desktop = [
    "opendpd[gui]",
    "pywebview>=6.2,<7",
]
```

`.github/workflows/ci.yml`: in the `test` job change `pip install -e ".[dev,gui]"` to `pip install -e ".[dev,gui,desktop]"` with the comment `# desktop: the window module's import path and availability probe run headless here`.

Verify locally: `.venv/bin/python -c "import tomllib; d=tomllib.load(open('pyproject.toml','rb')); print(d['project']['optional-dependencies']['desktop'])"`.

- [ ] **Step 2: Write ADR-0002**

```markdown
# ADR-0002: Native window shell for `opendpd gui`

- **Status:** accepted (maintainer decision, 2026-09-11)
- **Date:** 2026-09-11
- **Plan step:** post-S20 (Studio launch surface)
- **Deciders:** OpenDPD maintainers

## Context

The workbench was a browser tab opened by `opendpd gui`. The maintainer asked
for an application that runs without a browser, like draw.io's desktop build,
while keeping the browser path for SSH tunnels and headless machines. The plan's
principles apply: one compute core, no Node.js for users, loopback-only
security, no silent scientific change.

## Decision

| Concern | Decision | Notes |
|---|---|---|
| Shell | **pywebview** (`desktop` extra: `pywebview>=6.2,<7`) hosting the same loopback URL | WKWebView / WebView2 / WebKit2GTK / Qt; no bundled browser engine, no Rust or Node toolchain |
| Default surface | window when the extra is installed and a desktop session exists; otherwise the browser with a printed reason; `--browser`, `--window`, `--no-browser` override | `opendpd doctor` reports the backend |
| Process model | uvicorn in a daemon thread, the toolkit on the main thread; closing the window (or Ctrl+C) stops the supervisor and releases the lock | a native confirmation when runs are queued, running or stopping |
| Boundary | no JavaScript bridge, no CSP change, private WebKit storage | the page's CSP refuses injected scripts anyway |
| Preferences | none in the window; the server holds them per workspace | see ADR-0003 |

## Alternatives considered

| Option | Why not |
|---|---|
| Tauri or Electron application | a second toolchain in CI and releases, signing and notarisation, and the binary still needs the user's Python and torch: two installs instead of one |
| Bundling Python and torch into an installer | multi-gigabyte artefacts, no sane CUDA story; can be built later on top of this shell |
| Rewriting the UI in a native toolkit | duplicates 66 components, their tests and accessibility evidence; violates "one compute core, three entry points" |

## Consequences

- `opendpd[desktop]` adds pyobjc (macOS, about 35 MB) or pythonnet (Windows); Linux needs the distribution's WebKit2GTK bindings, which pip cannot install, so the browser fallback stays first class there.
- Verified on macOS Apple Silicon only; Windows and Linux rows of the support matrix stay unverified until a person runs them.
- Agents may not add a JavaScript bridge, relax the CSP or persist data in the webview without a new ADR.

## Verification

`tests/unit/test_window.py`, `tests/unit/test_launcher.py` (window mode, fallback, refusal, reuse, CLI flags, doctor); the macOS evidence in `docs/releases/support-matrix.md`.
```

- [ ] **Step 3: Update the user documentation**

`docs/tutorials/gui-quickstart.md`: change the opening block to

````markdown
```bash
pip install "opendpd[desktop]"   # or "opendpd[gui]" for the browser-only variant
opendpd gui
```

`opendpd gui` starts a local service on `127.0.0.1`, waits until it answers
and opens the workbench in a native application window (`opendpd[desktop]`)
or, without that extra, in your default browser after printing a one-time
URL such as `http://127.0.0.1:8765/bootstrap?token=…`. No Node.js, no second
terminal, no copying of addresses. Close the window or press Ctrl+C to stop;
running workers are terminated and the workspace lock is released. When runs
are still active, closing the window asks first.
````

and extend the option table with `| --browser | open the system browser even when the native window is available |` and `| --window | require the native window; fail with the reason when it is not available |`. Add a section:

```markdown
## Native window

The window is the same page a browser would show, hosted by the operating
system's web view (WKWebView on macOS, WebView2 on Windows, WebKit2GTK or Qt
on Linux). Downloads go through the platform's save dialog. The window cannot
script the page and keeps no data of its own: your language choice and every
other preference live in the workspace. `opendpd doctor` prints the backend
in use or the reason none is available (on Linux install `python3-gi
gir1.2-webkit2-4.1` or `pip install "pywebview[qt]"`).
```

`docs/releases/support-matrix.md`: add after the GUI table

```markdown
## Native window (`opendpd[desktop]`)

| OS | Backend | Status | Evidence |
|---|---|---|---|
| macOS Apple Silicon (macOS 26.6, Python 3.13) | cocoa / WKWebView, pywebview 6.2.1 | verified (manual, 2026-09-11) | `opendpd gui` opened the window on the bootstrap URL; session, datasets, a smoke run and a download through the save dialog worked; closing with the run active asked and *Cancel* kept the window; *Quit* terminated the worker and released the lock; Ctrl+C in the terminal closed the window and stopped the server |
| Windows x86-64 | edgechromium / WebView2 | **unverified** | needs a person: WebView2 runtime, downloads, Ctrl+C in a console |
| Linux x86-64 | gtk / WebKit2GTK or qt | **unverified** | needs a person with the distribution packages; CI only proves the headless fallback reason |
```

(Fill the macOS row from the real run in Task 5; leave "verified" out until that run happened.)

`docs/releases/release-notes-2.2.0.md`: add under "What is new" the bullet `- **Native window**: \`pip install "opendpd[desktop]"\` makes \`opendpd gui\` open the workbench in an application window (WKWebView / WebView2 / WebKit2GTK) instead of a browser tab; the browser stays the fallback and \`--browser\` / \`--window\` / \`--no-browser\` choose explicitly.`

`docs/releases/backlog.md`: add row `| 2a | **human** Native window checks on Windows and Linux (`opendpd gui` with the desktop extra: window, downloads, close-with-confirmation, Ctrl+C) | the support matrix rows are "unverified" | G2 |` after row 2.

`README.md`: in the Studio section change the install block to `pip install "opendpd[desktop]"` and the sentence to "`opendpd gui` starts a loopback-only service and opens the workbench in an application window (or, with the plain `gui` extra, in your browser after printing a one-time URL)".

`OpenDPD_Studio_Development_Plan.md` §3.1, after the "成功行为必须是" paragraph, add:

```markdown
> **修订（2026-09-11，维护者决定）**：安装 `opendpd[desktop]` 后，`opendpd gui` 默认在原生应用窗口（pywebview）中打开工作台；未安装该 extra、无桌面会话或使用 `--browser` 时仍按上文打开默认浏览器。`--window` 强制窗口，`--no-browser` 行为不变。详见 ADR-0002。
```

- [ ] **Step 4: Run the docs-command test and the full unit layer**

Run: `.venv/bin/python -m pytest tests/unit -q -x` and `.venv/bin/python -m pytest tests/integration/test_docs_commands.py -q -k quickstart`
Expected: PASS (the quickstart's commands are executed by that test; `opendpd gui` lines are not executed there — confirm the test only runs the headless commands it lists).

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml .github/workflows/ci.yml docs/architecture/adr/0002-native-window-shell.md docs/tutorials/gui-quickstart.md docs/releases/support-matrix.md docs/releases/release-notes-2.2.0.md docs/releases/backlog.md README.md OpenDPD_Studio_Development_Plan.md
git commit -m "docs(studio): desktop extra, ADR-0002 and user documentation for the native window

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: Real-desktop verification on macOS (evidence for the support matrix)

**Files:**
- Modify: `docs/releases/support-matrix.md` (fill the macOS row with what actually happened)

- [ ] **Step 1: Install the extra into the repo environment**

Run: `uv pip install --python .venv/bin/python "pywebview>=6.2,<7"` then `.venv/bin/opendpd doctor`
Expected: the `window` line reads `cocoa (pywebview 6.2.1)`; `ok: ready to launch`.

- [ ] **Step 2: Launch and exercise the window**

Run in the background with a scratch workspace: `.venv/bin/opendpd gui --workspace <scratch>/ws-window --port 8797`
Expected: the console prints the URL and "Close the window or press Ctrl+C to stop."; a window titled "OpenDPD Studio" shows the Home page with the workspace path (no "Session required" page).

Then, in the window: register the example dataset, start `pa-gru-smoke-v1`, download the resolved configuration (a save dialog appears and writes the file), close the window while the run is active (the confirmation names the count; *Cancel* keeps the window), wait for the run to finish, close the window (the process exits with code 0, `.studio.lock` is gone). Start again and press Ctrl+C in the terminal: the window closes and the process exits 0.

- [ ] **Step 3: Record the evidence**

Fill the macOS row of the "Native window" table with the observed results, the macOS build, Python and pywebview versions and the date. Anything that did not work is written as observed, never as passed.

- [ ] **Step 4: Commit**

```bash
git add docs/releases/support-matrix.md
git commit -m "docs(studio): record the macOS native window evidence

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```
