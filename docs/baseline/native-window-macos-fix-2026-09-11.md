# macOS active-run exit fix — 2026-09-11

**Outcome: all six native Cancel/Quit paths passed with real CPU workers.**
This resolves the blocking failure in the
[initial desktop verification](native-window-macos-2026-09-11.md).

## Changes

The Cocoa close callback now runs an `NSAlert` directly on the main thread.
It no longer calls pywebview's confirmation API, which queued work to that
same thread and then waited on a semaphore. The alert uses the current
workspace language, explicit Quit/Cancel buttons and the Studio icon.

Further testing exposed a second failure in the same exit flow: an open SSE
response prevented Uvicorn from reaching application lifespan shutdown.
The launcher abandoned its daemon server thread after 15 seconds, released
the workspace lock and exited 0 while the worker remained alive. That failure
was reproduced through an actual Quit button and is retained in
[before-http-shutdown-fix.json](native-window-macos-fix-2026-09-11/before-http-shutdown-fix.json).

The background server now allows one second for HTTP connections to drain,
then waits for lifespan shutdown to complete. The existing supervisor retains
its own bounded worker shutdown budget. Workspace ownership is released only
after server cleanup finishes. No training, metric, split or checkpoint
selection semantics changed.

## Native verification

Environment: macOS 26.6.2 (25G83), arm64, Python 3.13.12, pywebview 6.2.1,
pyobjc 12.2.2, Torch 2.14.0 and uvicorn 0.52.4. Source checkout with the existing
built frontend, based on commit `f43cd32`; exact tested source SHA-256 values
are retained in [results.json](native-window-macos-fix-2026-09-11/results.json).

Launched with `.venv/bin/opendpd gui --workspace "$task_root/workspace" --port 8797`
in a PTY. Real API submissions used the built-in dataset and the PA GRU smoke
configuration with 3,000 epochs solely to keep workers active during interaction.
Each final run was cancelled well before that limit. These are lifecycle checks,
not benchmark runs. The native window displayed each run and its live events.

With the user's authorization, System Events operated the real red close
button, Cmd+Q keystroke and native dialog buttons. Ctrl+C was byte `0x03` sent
to the launcher's foreground PTY, twice in the same session. Window-specific
captures isolate the actual alert from other desktop applications.

| Trigger | Cancel | Quit |
|---|---|---|
| Red close button | Pass: same worker, epoch 210 → 224; window usable | Pass: approximately 2.95 s, launcher exit 0, run cancelled |
| Cmd+Q | Pass: same worker, epoch 290 → 303; window usable | Pass: approximately 2.60 s, launcher exit 0, run cancelled |
| Terminal Ctrl+C | Pass: same worker, epoch 155 → 172; window usable | Pass: approximately 2.70 s, launcher exit 0, run cancelled |

Every Quit check confirmed the launcher and all captured worker descendants
were gone, port 8797 was free, `.studio.lock` was absent, and the OS workspace
guard could be reacquired. Timing includes desktop automation overhead.
Worker cancellation exit codes are recorded separately from launcher exit 0.

The red-button session began in English. Switching to 中文 in the page without
restarting also changed the subsequent native confirmation title, body and
buttons to Chinese. Cmd+Q and Ctrl+C were checked in Chinese.

Screenshots: [Cmd+Q confirmation](native-window-macos-fix-2026-09-11/cmdq-cancel.png),
[Ctrl+C confirmation](native-window-macos-fix-2026-09-11/ctrlc-quit.png).

## Automated checks and cleanup

```sh
.venv/bin/python -m pytest tests/unit/test_window.py tests/unit/test_launcher.py tests/integration/test_launcher_ownership.py tests/integration/test_launcher_shutdown.py tests/integration/test_studio_api.py -q
```

**57 passed in 51.78 s**, with two dependency deprecation warnings.
The new unit regression guards Cocoa dispatch, Cancel handling and reading the
current language; other platforms' dispatch is tested with doubles only.
The new integration regression keeps a real SSE reader open during launcher
shutdown and verifies real CPU worker cleanup before ownership is released.
It uses a headless window callback, so native evidence comes from the separate
desktop checks above.

An initial version of the new integration fixture mutated a shared recipe's
epoch field. That test setup was corrected to copy the training configuration;
the interrupted first suite run is not counted as passing evidence. Existing
expectations and scientific defaults were not edited. The reported run above
used a fresh process and the corrected fixture.

Critical flake8 checks (`E9,F63,F7,F82`), compilation and `git diff --check`
passed. No frontend source changed. Windows/Linux native behavior and GPU
execution remain unverified. Temporary workspaces, checkpoints and automation
helpers were removed after retaining evidence; no test server or worker remains.
