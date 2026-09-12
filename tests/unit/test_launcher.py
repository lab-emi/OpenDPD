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
    rc = launcher.launch(ws, port=None, open_in_browser=True, serve=fake_serve,
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
    rc = launcher.launch(ws, serve=lambda *a: served.append(a), opener=lambda u: opened.append(u) or True)
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
    assert "frontend" in out and "port" in out
    static = Path(launcher.__file__).parent / "static" / "index.html"
    assert rc == (0 if static.exists() else 1)


def test_live_starting_instance_keeps_its_metadata(tmp_path, monkeypatch):
    lock = launcher.write_lock(tmp_path, 8790, "http://127.0.0.1:8790/bootstrap?token=test")
    before = lock.read_text()
    monkeypatch.setattr(launcher, "probe", lambda *a, **kw: None)
    assert launcher.existing_instance(tmp_path) is None
    assert lock.read_text() == before
    served = []
    assert launcher.launch(tmp_path, serve=lambda *a: served.append(a), open_in_browser=False) == 2
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
    assert launcher.launch(workspace, open_in_browser=False) == 2
    assert "check workspace permissions" in capsys.readouterr().err
    assert workspace.read_text() == "keep this file"
