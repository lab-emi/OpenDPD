"""Native window shell: availability probe and close policy, without pywebview (L0)."""

import sys
import threading
from types import SimpleNamespace

import pytest

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


def test_icon_files_ship_with_the_package(monkeypatch):
    for name in ("icon.png", "icon.ico"):
        assert (window.ICON_DIR / name).is_file(), name
    monkeypatch.setattr(window.os, "name", "posix")
    assert window.icon_path().name == "icon.png"
    monkeypatch.setattr(window.os, "name", "nt")
    assert window.icon_path().name == "icon.ico"


def test_every_language_has_every_native_string():
    from opendpd.schemas import UI_LANGUAGES
    from opendpd.studio.strings import STRINGS
    assert set(STRINGS) == set(UI_LANGUAGES)
    for code, s in STRINGS.items():
        assert "{count}" in s.quit_body, code
        assert set(s.localization) == set(ENGLISH.localization), code
        assert all(v.strip() for v in s.localization.values()), code
    assert shell_strings("xx") is ENGLISH and shell_strings(None) is ENGLISH
    assert shell_strings("ja").quit_title != ENGLISH.quit_title


@pytest.mark.parametrize("platform", ["darwin", "linux", "win32"])
def test_window_close_dispatch_respects_cancel_and_reads_current_language(monkeypatch, platform):
    """Cocoa closing handlers run on the UI thread: never enter the toolkit's
    queue-and-wait confirmation API there. Other backends keep their own API.
    """
    callbacks = []
    asked = []
    state = {"active": 2, "language": "en", "answer": False}

    class ClosingEvent:
        def __iadd__(self, callback):
            callbacks.append(callback)
            return self

    def toolkit_ask(title, message):
        if platform == "darwin":
            pytest.fail("Cocoa would deadlock waiting for work queued to this same thread")
        asked.append((title, message))
        return state["answer"]

    def native_ask(title, message, strings):
        assert threading.current_thread() is threading.main_thread()
        assert strings is shell_strings(state["language"])
        asked.append((title, message))
        return state["answer"]

    def start(**kwargs):
        close, = callbacks
        assert close() is False                  # Cancel must veto the close.
        state.update(language="zh", answer=True)
        assert close() is True                   # A subsequent Quit is allowed.
        state.update(active=0, answer=False)
        assert close() is True                   # An idle close needs no dialog.

    fake_window = SimpleNamespace(events=SimpleNamespace(closing=ClosingEvent()),
                                  create_confirmation_dialog=toolkit_ask)
    fake_webview = SimpleNamespace(settings={}, create_window=lambda *a, **k: fake_window, start=start)
    monkeypatch.setitem(sys.modules, "webview", fake_webview)
    monkeypatch.setattr(window.sys, "platform", platform)
    monkeypatch.setattr(window, "_name_the_application", lambda title: None)
    monkeypatch.setattr(window, "_install_macos_hooks", lambda *a: None)
    monkeypatch.setattr(window, "_macos_confirm_quit", native_ask)

    window.run_window("http://127.0.0.1:8797/", active_runs=lambda: state["active"],
                      strings=lambda: shell_strings(state["language"]))

    assert asked == [(shell_strings(code).quit_title, shell_strings(code).quit_body.format(count=2))
                     for code in ("en", "zh")]


@pytest.mark.parametrize('width,height,scale', [(1024, 768, 1), (1280, 800, 2), (1920, 1080, 1), (1920, 1080, 2), (3840, 2160, 1), (600, 480, 1)])
def test_window_geometry_fits_logical_display_without_double_retina_scaling(width, height, scale):
    from types import SimpleNamespace
    screen = SimpleNamespace(width=width, height=height, scale=scale)
    size, minimum = window.window_geometry(screen)
    assert size[0] <= width - 80 and size[1] <= height - 100
    assert minimum[0] <= size[0] and minimum[1] <= size[1]
    assert size[0] <= window.DEFAULT_SIZE[0] and size[1] <= window.DEFAULT_SIZE[1]
    assert window.window_geometry(SimpleNamespace(width=width, height=height, scale=1))[0] == size
