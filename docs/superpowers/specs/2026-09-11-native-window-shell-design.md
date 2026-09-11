# Native window shell for OpenDPD Studio — design

- **Date:** 2026-09-11
- **Status:** approved by the maintainer (option A: pywebview shell, window by default)
- **Records:** ADR-0002 (written with the implementation)
- **Companion spec:** `2026-09-11-ui-languages-design.md` (the shell's native strings
  follow the UI language chosen there)

## 1. Goal

`opendpd gui` opens the workbench in a native application window instead of a
browser tab whenever the optional `desktop` extra is installed and a desktop
session exists. Nothing else changes: the same FastAPI service on `127.0.0.1`,
the same built React bundle, the same session / CSRF / CSP boundary, the same
workspace lock and the same lifecycle rules. The browser mode stays available
and unchanged for SSH tunnels, headless servers and users without the extra.

The model is draw.io's: one web application, two shells. The window is a
*display surface only*. It cannot script the page and the page cannot reach
the host process.

### Non-goals

Installers (dmg / msi / AppImage), bundling Python or torch, a custom title
bar or custom menus, a JavaScript bridge between the window and the page, any
visual redesign (the light theme stays), Windows and Linux verification (no
such machine is available here; those support-matrix rows stay *unverified*).

## 2. User-facing contract

Install: `pip install "opendpd[desktop]"` (equals `opendpd[gui]` plus
pywebview). Start: `opendpd gui`.

| Situation | Behaviour |
|---|---|
| `opendpd gui`, window backend available, desktop session | a native window titled "OpenDPD Studio" opens on the bootstrap URL once `/healthz` answers; the console prints the URL and "Close the window or press Ctrl+C to stop." |
| `opendpd gui`, backend missing (extra not installed, Linux without WebKit2GTK, SSH session, no display) | the browser exactly as today, plus one line saying why the window was not used and how to get it |
| `opendpd gui --browser` | browser, never a window |
| `opendpd gui --window` | window; exit code 2 with the reason and the fix when it is not available; no server is started |
| `opendpd gui --no-browser` | unchanged: print the URL only |
| second `opendpd gui` for a workspace that is already served | a window (or the browser) on the existing instance's URL; closing that window never stops a server it does not own |
| window closed, no run active | the server stops through the Ctrl+C path: supervisor stop, workers terminated, lock released |
| window closed while runs are queued, running or stopping | a native confirmation: "N experiments are running. Quit OpenDPD Studio and stop them?" — *Quit* stops as above, *Cancel* keeps the window |
| Ctrl+C in the terminal (window mode) | the window closes and the server stops through the same path (verified on macOS) |
| downloads (artifacts, configuration JSON, CSV, packages, reports) | the platform's save dialog (pywebview `ALLOW_DOWNLOADS`) |
| links to other sites | none exist today; if one is added it opens in the system browser (`OPEN_EXTERNAL_LINKS_IN_BROWSER`) |
| `opendpd doctor` | a new line `window    cocoa (pywebview 6.2.1)` or `window    not available: <reason>`; never counted as a blocking problem |

`--browser`, `--window` and `--no-browser` are mutually exclusive.

## 3. Architecture

### 3.1 New module `opendpd/studio/window.py`

Imported only when a window may be used; `import opendpd`, the CLI and the
browser path never import pywebview.

- `availability() -> Availability` with `backend: Optional[str]` and
  `reason: str`. Checks, in order: pywebview importable (reason names
  `pip install "opendpd[desktop]"`); a desktop session exists (macOS:
  `Quartz.CGSessionCopyCurrentDictionary()` is not `None`; Linux: `DISPLAY` or
  `WAYLAND_DISPLAY` set; Windows: assumed); pywebview finds a GUI backend
  (`webview.guilib.initialize()`, whose `WebViewException` message becomes the
  reason, extended with the Linux fix `python3-gi gir1.2-webkit2-4.1` or
  `pip install "pywebview[qt]"`).
- `run_window(url, *, title, on_close_request, strings, icon) -> None`:
  creates the window, registers the closing handler, calls `webview.start()`
  and returns when the window is gone. `on_close_request()` returns `True` to
  allow the close. The module knows nothing about runs or the server.
- `strings` is a small dataclass (dialog title and body template, plus the
  pywebview `localization` dictionary) defined in `opendpd/studio/strings.py`,
  which starts with English; the languages spec adds the other six.

Window parameters: 1366×860, minimum 960×600, resizable, `private_mode=True`
(nothing persists in the WebKit store; the server holds preferences),
`text_select=True`, no `js_api`. The page's CSP (`script-src 'self'`) blocks
pywebview's injected bridge anyway, as the spike showed (`EvalError` from the
CSP). Application icon through `webview.start(icon=…)`: `icon.png` (macOS
Dock, GTK) or `icon.ico` (Windows), both generated from the geometry of
`frontend/public/favicon.svg` by `scripts/make_app_icon.py` (Pillow) and
committed as package data. On macOS the application menu name is set to
"OpenDPD Studio" through the main bundle's info dictionary, best effort: a
failure leaves "Python" and is not an error.

### 3.2 Launcher changes (`opendpd/studio/launcher.py`)

`launch(workspace, *, port=None, mode="auto", out=None, serve=None,
opener=webbrowser.open, window_runner=None)`; `mode` is one of `auto`
(window if available, else browser with a printed reason), `window` (or exit
2), `browser` (today's behaviour) and `none` (today's `--no-browser`).
`window_runner(url, on_close_request)` is injectable for tests, like `serve`
and `opener` today.

Process model in window mode: the GUI toolkit needs the main thread (Cocoa
refuses otherwise), so uvicorn runs in a daemon thread through
`uvicorn.Server(uvicorn.Config(app, host, port, log_level="warning"))`. The
main thread waits for `/healthz` (30 s as today), prints the URL and the
readiness warnings, then runs the window. When the window returns, the
launcher sets `server.should_exit = True` and joins the thread, bounded by
the application's shutdown timeout plus five seconds; the lock file is removed
in `finally` as today. Browser mode keeps uvicorn on the main thread,
unchanged. If the server does not become healthy, no window is opened: the
error is printed, the server thread is stopped and the exit code is 2 (in the
browser mode the launcher keeps serving after that message; a window would
have nothing to show).

Close request: `on_close_request` counts runs in `queued`, `running` and
`cancel_requested` through the in-process store (`app.state.store.count_runs`)
and, when the count is positive, asks through pywebview's
`create_confirmation_dialog`. If the dialog cannot be shown the close is
allowed and a warning is printed: the supervisor's stop path marks and
terminates the workers, nothing is left unmarked.

Reuse path: `mode` `auto` or `window` with an available backend opens a plain
window (no close hook, no server) on the running instance's URL; otherwise the
browser as today.

Signals: browser mode keeps `_sigterm_as_keyboard_interrupt`. In window mode
SIGINT and SIGTERM handlers call `window.destroy()` so the normal shutdown path
runs after `webview.start()` returns.

### 3.3 CLI and packaging

`opendpd gui` gains `--window` and `--browser` in a mutually exclusive group
with `--no-browser`; `cmd_gui` maps the flags to `mode`. `opendpd doctor`
prints the window line. `pyproject.toml`: `desktop = ["opendpd[gui]",
"pywebview>=6.2,<7"]`; package data adds the two icon files. CI installs
`.[dev,gui,desktop]` on the Python test job so the import path and
`availability()` (which reports "no display" there) are exercised; the
packaging test (wheel without `gui`) is unchanged.

## 4. Error handling

| Failure | Behaviour |
|---|---|
| extra or backend missing, `mode=auto` | browser plus one explanatory line on stdout |
| `--window` with the backend missing | `error: native window not available: <reason>; <fix>` on stderr, exit 2, nothing started |
| server not healthy within 30 s (window mode) | error on stderr, server thread stopped, lock released, exit 2 |
| pywebview raises while creating or running the window | caught; message printed; fall back to the browser opener; the server keeps running; exit code 0 |
| readiness problems (frontend assets missing) | the window opens on the diagnostic page the server renders, as the browser would |
| confirmation dialog unavailable | close allowed, warning printed |

## 5. Testing

Unit (L0, no pywebview, fake `window_runner` / `serve` / `opener`):

- mode table: `auto` + unavailable → opener called and the reason printed; `auto` + available → window runner receives the bootstrap URL only after `/healthz`; `window` + unavailable → exit 2, nothing served; `browser` and `none` unchanged.
- window lifecycle: serve runs off the main thread, the window opens after health, returning from the window stops the server, the lock is released.
- close request: no active run → allowed without asking; active runs → asks; *Cancel* keeps the window (the fake runner loops), *Quit* stops.
- reuse: a running instance and an available backend → a window on its URL, no server started.
- `availability()` reasons for a missing package and a missing display (monkeypatched checks).
- CLI: the three flags are mutually exclusive; `--window` maps to `mode="window"`.
- doctor prints the window line and never counts it as a problem.

Real evidence on this Mac, recorded in the support matrix with the commands:
`opendpd gui` opens the window; a smoke run trains; a download goes through
the save dialog; closing with the run active asks and *Cancel* keeps it;
*Quit* terminates the worker and releases the lock; Ctrl+C closes the window
and stops the server. Windows and Linux rows stay *unverified*.

## 6. Documentation

ADR-0002 (launch surface: window by default when installed, browser fallback,
no JavaScript bridge, pywebview pinned below 7); `docs/tutorials/gui-quickstart.md`
(install line, flag table, "Native window" section); `docs/releases/support-matrix.md`
(new table "Native window (`opendpd[desktop]`)"); README Studio paragraph;
`docs/releases/release-notes-2.2.0.md`; `docs/releases/backlog.md` (human
item: Windows and Linux window checks); a short amendment in
`OpenDPD_Studio_Development_Plan.md` §3.1 recording the maintainer's decision
of 2026-09-11 that the default launch surface is the native window when the
desktop extra is installed.
