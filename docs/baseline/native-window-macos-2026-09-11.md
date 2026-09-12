# Native window verification on macOS — 2026-09-11

**Historical result on `f43cd32`: the active-run close confirmation fails.**

**Update:** the defect and a subsequently reproduced HTTP shutdown blocker were
fixed and all six native Cancel/Quit paths passed. See the
[fix verification](native-window-macos-fix-2026-09-11.md).

The remainder of this report preserves the original observations.

**Original outcome:** Downloads, application
menu naming, the Dock icon, and all three idle exit paths passed on this Mac.
The native window must not be described as fully verified until the close
deadlock is fixed and the Cancel/Quit branches are exercised with real workers.

## Environment and method

- Checkout: `OpenDPD-Studio`, commit `f43cd32d8385efc3bf44980e19689caa1ae66e64`;
  working tree clean before verification. No application code was changed.
- macOS 26.6.2 (25G83), arm64; Python 3.13.12, pywebview 6.2.1,
  pyobjc-core / pyobjc-framework-Cocoa 12.2.2, Torch 2.14.0, uvicorn 0.52.4.
- Cocoa/WKWebView, source checkout with its existing built frontend and `.venv`.
  This is not a fresh-wheel installation test.
- Separate temporary workspace; loopback port 8797; built-in `dpa-200mhz` data;
  CPU only. No instrumentation or mocked API/computation.
- The desktop control interface could not resolve the unbundled Python process.
  With the user's explicit permission, AppleScript/System Events drove the
  actual application by PID; `screencapture` recorded its UI. Files, process
  identities, HTTP health and OS locks were checked independently from Python.

Commands used (the workspace path was generated with `tempfile.mkdtemp`):

```sh
sw_vers
uname -m
.venv/bin/python -V
.venv/bin/opendpd doctor
.venv/bin/opendpd gui --workspace "$task_root/workspace" --port 8797
sample 58426 2 -file "$task_root/evidence/active-close-sample.txt"
```

The fourth launch ran in a PTY; sending byte `0x03` tested terminal Ctrl+C.
AppleScript used `first application process whose unix id is <launcher PID>`,
the real `AXCloseButton`, native Save/Cancel buttons and Cmd+Q keystrokes.

## Results

| Check | Result | Observed evidence |
|---|---|---|
| Default native launch and session | Pass | Actual `OpenDPD Studio` window, authenticated Home page, `/healthz` returns `{"status":"ok"}`. |
| Real training through the window | Pass | Registered the example dataset in the UI and ran `pa-gru-smoke-v1`, named `desktop-smoke`, CPU, 3 epochs. Run `run-20260911-203615-1e5644` succeeded in approximately 3.62 s according to run timestamps, with real results (`is_mock=false`). This only proves the workflow. |
| Download configuration: Save | Pass | Clicking Configuration → Download configuration opened the native Save dialog with `config.resolved.json`. Used Cmd+Shift+G to select the temporary downloads directory, then Save. The resulting 1,590 bytes match the run's resolved configuration byte for byte. SHA-256: `0049c3a0c73a94ac89d2a45cfd0cd7ed1d6057f318ee9a40a68c52f7dbc87953`. |
| Download configuration: Cancel | Pass | Opened the dialog a second time, clicked Cancel, returned to the functioning configuration page. |
| Application menu name | Pass | Menu bar says `OpenDPD Studio`; English About/Hide/Quit items include that name. |
| Chinese setting and native menus after restart | Pass | Selected 中文 in the actual window; `settings.json` contains `{"language":"zh"}`. Relaunch restored the Chinese page and native 编辑 / 显示 / 关于 OpenDPD Studio / 退出 OpenDPD Studio menu labels. |
| Dock icon | Pass, with naming caveat | Actual Dock item shows the project's blue square with white waveform. Its accessibility item name remains `python3.13`; this check does not claim an installed `.app` bundle identity. |
| Idle red close button | Pass | Exit 0, approximately 0.75 s including automation overhead; launcher PID gone, `.studio.lock` removed, port free, OS workspace guard reacquired. |
| Idle Cmd+Q | Pass | Real keyboard action, exit 0, approximately 1.15 s including automation overhead; same cleanup checks passed. |
| Idle Ctrl+C | Pass | Real PTY interrupt, exit 0; launcher gone, `.studio.lock` removed, port free, guard reacquired, no remaining test process. |
| Red close button while training | **Fail — blocking** | No confirmation appeared. Window froze while real training continued. The screenshot remained at epoch 33/300 while the run file subsequently reported 80, 185 and finally 300/300. Native accessibility requests stopped returning a usable window. The UI remained stuck after the run completed. |
| Active-run Cancel and Quit branches | **Blocked** | Neither button could be exercised because the confirmation never appeared. |
| Active-run Cmd+Q and Ctrl+C | **Not verified** | Not exercised independently after the close-button failure. Shared code suggests exposure to the same issue; that is an inference, not a passed or failed desktop test for these two triggers. |
| Windows / Linux native windows | **Not verified** | These platforms were not available in this session. |

Download coverage is the resolved JSON configuration; ZIP exports, other file
types, overwrite prompts and Unicode destination names were not tested here.
Native menu translation was checked after relaunch, not for live menu updates.

## Blocking defect: confirmation waits on its own UI thread

Reproduction on the unmodified commit:

1. Launch the native window and register the example dataset.
2. Select the PA GRU smoke recipe. For this lifecycle test only, set Advanced
   settings → Epochs to 300 to leave time to interact; keep CPU and the other
   fields unchanged. Name the run `desktop-close-active`.
3. Start the run, then press the red window close button while it is running.
4. Observe that no confirmation appears and the window stops responding, while
   the supervisor/worker and `/healthz` continue functioning.

Run `run-20260911-203933-827f4b` ran from 20:39:33.630308Z to
20:40:30.532731Z and succeeded in the background. These 300 smoke-layout epochs
were a lifecycle probe, not a research recipe or benchmark acceptance run.

The local code explains the observed deadlock:

- `opendpd/studio/window.py:202–205` registers a closing callback that calls
  `window.create_confirmation_dialog` synchronously.
- pywebview 6.2.1's `Window.events.closing = Event(self, True)` runs that handler
  synchronously inside Cocoa's `windowShouldClose_` on the main thread.
- `webview/platforms/cocoa.py:1437–1452` schedules `_confirm` with
  `AppHelper.callAfter`, then calls `semaphore.acquire()`. Since the caller is
  already the main thread, it blocks the thread needed to show the alert and
  release the semaphore.
- The process sample has the main thread inside `NSWindow __close`, a Python
  callback and `_PySemaphore_Wait` for every sample of the two-second capture.

Required follow-up: change the native close/dialog scheduling so the main thread
does not wait for work queued to itself. Re-test both Cancel (same worker keeps
progressing) and Quit (worker terminated, exit 0, lock/port released), through
the red button, Cmd+Q and terminal Ctrl+C. No mock-only check can close this item.

## Cleanup and retained evidence

The stuck launcher (PID 58426) did not exit after SIGTERM and a three-second
wait; SIGKILL was required, with exit 137. The real training worker had already
completed and exited. Forced termination left `.studio.lock` metadata, while
the OS guard was released. A subsequent normal launch recovered that workspace;
its idle Cmd+Q cleanup removed the metadata. The final Ctrl+C launch also exited
cleanly. No test worker or server remained.

Evidence is retained in [native-window-macos-2026-09-11/](native-window-macos-2026-09-11/):

- [results.json](native-window-macos-2026-09-11/results.json): environment, run
  metadata, download comparison, exit and cleanup observations.
- [main-thread-sample.txt](native-window-macos-2026-09-11/main-thread-sample.txt):
  sample metadata and blocked main-thread stack.
- [save-dialog.png](native-window-macos-2026-09-11/save-dialog.png),
  [menu-en.png](native-window-macos-2026-09-11/menu-en.png),
  [menu-zh.png](native-window-macos-2026-09-11/menu-zh.png), and
  [dock-icon.png](native-window-macos-2026-09-11/dock-icon.png).
- [active-close-hang.png](native-window-macos-2026-09-11/active-close-hang.png):
  frozen native window with active training and no confirmation dialog.

The temporary workspace, datasets, checkpoints, UI helper and redundant
screenshots were removed after retaining the evidence. No application code,
scientific protocol, numerical expectation, dependency, release note or remote
branch was changed. Source tests were not rerun for this documentation-only
verification task.
