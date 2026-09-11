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
| Process model | uvicorn in a daemon thread, the toolkit on the main thread; closing the window, quitting the application or Ctrl+C stops the supervisor and releases the lock | a native confirmation when runs are queued, running or stopping; on macOS a replacement application delegate turns `terminate` into a window close because pywebview's own answer to Ctrl+C and Cmd+Q would exit the process before the cleanup |
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
