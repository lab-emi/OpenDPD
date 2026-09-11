"""`opendpd gui` launch logic without a real browser or uvicorn (L0)."""

import json
import os
import socket
import threading
import time
from pathlib import Path

import pytest

from opendpd.studio import launcher


def test_explicit_port_must_be_free_and_auto_port_skips_busy_ones(monkeypatch):
    with socket.socket() as busy:
        busy.bind((launcher.HOST, 0))
        busy.listen(1)
        port = busy.getsockname()[1]
        with pytest.raises(launcher.LaunchError, match="already in use"):
            launcher.choose_port(port)
        monkeypatch.setattr(launcher, "DEFAULT_PORT", port)
        assert launcher.choose_port(None) == port + 1 or launcher.port_is_free(launcher.choose_port(None))


def test_open_browser_never_raises():
    def broken(_url):
        raise RuntimeError("no display")
    assert launcher.open_browser("http://127.0.0.1:1/x", opener=broken) is False
    assert launcher.open_browser("http://127.0.0.1:1/x", opener=lambda u: True) is True


def test_browser_opens_only_after_health_check_with_bootstrap_url(tmp_path, monkeypatch, capsys):
    opened, order = [], []
    healthy = threading.Event()

    def fake_probe(url, timeout=1.0):
        if url.endswith("/healthz"):
            return {"status": "ok"} if healthy.is_set() else None
        if url.endswith("/readyz"):
            return {"ready": False, "problems": ["frontend assets are not built/installed"]}
        return None

    def fake_serve(app, host, port):
        order.append("serve")
        assert host == "127.0.0.1"
        time.sleep(0.3)
        healthy.set()
        time.sleep(0.5)     # let the launcher thread observe health and open the browser

    monkeypatch.setattr(launcher, "probe", fake_probe)
    ws = tmp_path / "空格 workspace with spaces"
    rc = launcher.launch(ws, port=None, mode="browser", serve=fake_serve,
                         opener=lambda url: (opened.append(url), order.append("browser"))[1] is None)
    assert rc == 0
    assert order == ["serve", "browser"], order
    assert opened and opened[0].startswith("http://127.0.0.1:") and "/bootstrap?token=" in opened[0]
    out = capsys.readouterr().out
    assert "warning: frontend assets are not built/installed" in out
    assert not (ws / launcher.LOCK_FILE).exists(), "lock released on exit"


def test_running_instance_is_reused_not_duplicated(tmp_path, monkeypatch):
    ws = tmp_path / "ws"
    ws.mkdir()
    from opendpd.runtime.procs import process_identity
    pid, ctime = process_identity(os.getpid())
    (ws / launcher.LOCK_FILE).write_text(json.dumps({"pid": pid, "create_time": ctime, "port": 8790,
                                                      "url": "http://127.0.0.1:8790/bootstrap?token=abc"}))
    monkeypatch.setattr(launcher, "probe", lambda url, timeout=1.0: {"status": "ok"})
    opened = []
    served = []
    # browser mode explicitly: on a desktop with the extra installed, 'auto' would open a real window
    rc = launcher.launch(ws, mode="browser", serve=lambda *a: served.append(a), opener=lambda u: opened.append(u) or True)
    assert rc == 0 and served == [] and opened == ["http://127.0.0.1:8790/bootstrap?token=abc"]


def test_stale_lock_is_removed(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / launcher.LOCK_FILE).write_text(json.dumps({"pid": 999999, "create_time": 1.0, "port": 1, "url": "u"}))
    assert launcher.existing_instance(ws) is None
    assert not (ws / launcher.LOCK_FILE).exists()
    (ws / launcher.LOCK_FILE).write_text("not json")
    assert launcher.existing_instance(ws) is None


def test_doctor_reports_frontend_and_port(tmp_path, capsys):
    rc = launcher.doctor(tmp_path / "ws")
    out = capsys.readouterr().out
    assert "frontend" in out and "port" in out and "window" in out
    static = Path(launcher.__file__).parent / "static" / "index.html"
    assert rc == (0 if static.exists() else 1)


def test_live_starting_instance_keeps_its_metadata(tmp_path, monkeypatch):
    lock = launcher.write_lock(tmp_path, 8790, "http://127.0.0.1:8790/bootstrap?token=test")
    before = lock.read_text()
    monkeypatch.setattr(launcher, "probe", lambda *a, **kw: None)
    assert launcher.existing_instance(tmp_path) is None
    assert lock.read_text() == before
    served = []
    assert launcher.launch(tmp_path, serve=lambda *a: served.append(a), mode="none") == 2
    assert served == []
    assert lock.read_text() == before


def test_guard_is_released_after_a_failed_launch(tmp_path):
    with pytest.raises(RuntimeError, match="startup failed"):
        with launcher.workspace_guard(tmp_path):
            with pytest.raises(launcher.WorkspaceBusy):
                with launcher.workspace_guard(tmp_path):
                    pytest.fail("a second writer acquired the workspace")
            raise RuntimeError("startup failed")
    with launcher.workspace_guard(tmp_path):
        assert (tmp_path / launcher.GUARD_FILE).exists()


def test_workspace_creation_error_is_actionable(tmp_path, capsys):
    workspace = tmp_path / "not-a-directory"
    workspace.write_text("keep this file")
    assert launcher.launch(workspace, mode="none") == 2
    assert "check workspace permissions" in capsys.readouterr().err
    assert workspace.read_text() == "keep this file"


# --- native window surface (L0: fake window runner, fake background server) ---------

from opendpd.studio import window  # noqa: E402


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


def test_preferred_language_reads_the_workspace_then_the_os_locale(tmp_path, monkeypatch):
    from opendpd.schemas import WorkspaceSettings
    from opendpd.services.workspace import Workspace
    ws = Workspace.create(tmp_path / "ws")
    monkeypatch.setattr(launcher.locale, "getlocale", lambda: ("fr_FR", "UTF-8"))
    assert launcher.preferred_language(ws.root) == "fr"
    ws.save_settings(WorkspaceSettings(language="ja"))
    assert launcher.preferred_language(ws.root) == "ja"
    monkeypatch.setattr(launcher.locale, "getlocale", lambda: ("pt_BR", "UTF-8"))
    assert launcher.preferred_language(tmp_path / "missing") == "en"
    ws.settings_path.write_text("{broken")
    assert launcher.preferred_language(ws.root) == "en", "a broken file never blocks the window"
