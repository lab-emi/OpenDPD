# OpenDPD Studio quality audit — 2026-09-06

## Assessment

The branch is a working alpha with substantial shared-service and numerical
regression coverage. It is not ready to treat all implementation stages as
accepted release gates. The original backend suite passed, but independent
failure-path checks still found a reproducible double-writer startup race,
a validation-order defect blocking DPD creation in the GUI, and four
experiment-form defects. The completed export journey also exposed an unquoted
workspace path in the generated reproduction command. Final input checks found
an additional polynomial-parameter validation failure. All eight findings are
repaired without changing metrics, splits, recipes, seeds, checkpoint selection
or numerical tolerances.

This audit compares the installed application and source with
`OpenDPD_Studio_Development_Plan.md`, particularly sections 3, 4, 6 and 8.
It does not substitute an agent-operated browser for an external human trial.

The strongest work is the common execution service, versioned evidence and
numerical regression coverage. The weakest part is acceptance discipline:
advanced features have broad implementation coverage while an ordinary GUI
DPD workflow could not pass its own preflight. Direct service tests and mock
browser tests covered different halves of that failure. This is a finding
about this implementation, not a general ranking of the model that authored it.

## Baseline and environment

- Branch: `OpenDPD-Studio`; initial HEAD: `13b3cc0`.
- Final code revision: `7bfce76`; code fixes are `9ae15cb` (workspace ownership),
  `34fae73` (experiment forms and shared validation) and `7bfce76` (package commands).
- OS: macOS 26.6.2, arm64; Apple M2, 24 GiB RAM.
- Isolated editable environment: `.venv`, Python 3.13.12.
- PyTorch 2.14.0, NumPy 2.5.3, SciPy 1.18.1, Pydantic 2.13.5,
  FastAPI 0.141.1, Uvicorn 0.52.4.
- Node.js 26.8.1; dependencies installed with `npm ci` from the committed lockfile.
- Playwright 1.63.0: Chromium 153, Firefox 155, WebKit 26.6 on macOS.
- All computation in this audit used CPU. CUDA, MPS execution, Windows,
  Linux and RF hardware were **not verified** in this environment.
- Browser workspace: `/tmp/OpenDPD Studio 审查` (resolved by macOS under
  `/private/tmp`); no existing user workspace was modified.

## Defects repaired

| Priority | Finding and reproduction | Change | Plan |
|---|---|---|---|
| P1 | Two launchers started together against one new workspace on separate loopback ports both became healthy. The JSON `.studio.lock` did not exclude a second supervisor. | Hold an OS file lock before server creation through completed shutdown; reuse a healthy owner or refuse while it starts/stops. Preserve a live owner's metadata when health probing fails. Crash recovery releases the OS lock automatically. Workspace creation errors now include a usable diagnostic. | S03, S06; section 3.2 |
| P1 | Choosing a succeeded PA run in the real DPD form still disabled Start: the route ran pure resolution before binding `pa_reference.run_id`. Direct submission bound first, so existing pipeline tests missed the GUI blocker. | Share workspace binding and resolution between preview and submission. The API and optional CLI `validate --workspace` use the same validation service. Unknown model references produce field errors instead of server errors. | S02, S04, S09; section 4.1 |
| P1 | Re-running a missing/unavailable source run could fall back to the default recipe and validate an unrelated experiment. | Block the recipe fallback while a source run is requested; display loading/error/retry until its configuration is available. | S02, S09 |
| P1 | An imported PA configuration referenced `dpa-200mhz`, while the editable dataset selector displayed `audit-capture`; edits to those selectors were ignored on submission. | Import mode shows the submitted configuration and optional name only. Discarding the import explicitly restores the recipe form. | S02, S05, S09 |
| P2 | Entering `num_layers=1.5` silently submitted `1` because the frontend truncated integers before validation. | Pass the actual numeric edit to the shared server validator. The real API now returns `num_layers must be an integer` and the GUI blocks submission. | S02, S04, S09 |
| P2 | A failed validation request left a disabled Start button without a reason or recovery action. | Preserve and show the error; retry the same configuration explicitly. Also show model-list loading errors. | S04, S05 |
| P2 | The imported package's generated evaluation command split a workspace containing spaces into several arguments; copying it into zsh exited with `unrecognized arguments`. | Quote the generated POSIX-shell command with `shlex.join`. Extend the package round-trip test to a workspace with spaces and Unicode and verify the command's argument boundaries. | S06, S11 |
| P2 | A minimal `mp_ls` configuration raised `KeyError('Q')`; an invalid `Q` raised `ValueError` during workspace checks, yielding an API 500 instead of a validation result. | Bind and normalize through the common resolver before workspace checks read model-context parameters, then freeze the workspace warnings. Omitted parameters use registry defaults; invalid parameters produce field errors and submission returns 422 without creating a run. | S02, S04, S09 |

The four new experiment-form regressions failed before their fixes and passed
afterward. Existing numerical reference files and expected values were not edited.

The new real-API regression first reproduced the DPD validation failure after
training a real PA. It now validates, trains and applies a real DPD, checks that
preview and submitted hashes match, compares the CLI and API reports with a
controlled resolution timestamp, and rejects missing references, incompatible
training settings, unknown models and fractional integer parameters without
creating runs. No metric expectations are replaced or relaxed.

The launcher regression delays a real Uvicorn server before listening, attempts
a competing process, verifies the original metadata, starts the server, reuses
it over real HTTP, kills it without cleanup, restarts it, and checks graceful
shutdown. Unit checks additionally cover legacy live metadata and lock release
on exceptions. The guard file intentionally remains on disk; deleting it while
a server owns the lock would undermine exclusion.

## Executed checks

Commands are run at the repository root unless `frontend/` is stated.

| Check | Command | Result |
|---|---|---|
| Original CPU backend, compatibility, goldens and integration baseline | `MPLBACKEND=Agg OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest tests -m "not extended and not packaging" -q --maxfail=3 --durations=15` | 536 passed, 17 skipped, 22 deselected; 496.77 s. Includes the real PA/DPD pipeline, GUI/CLI deterministic consistency, package re-evaluation and documented command workflows. |
| Complete CPU regression after the main fixes | Same backend command | 541 passed, 17 skipped, 22 deselected; 513.06 s. The final polynomial-input normalization was added afterward and checked by the focused suites recorded below. |
| Final validation and real computation regression | `MPLBACKEND=Agg OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest tests/integration/test_studio_api.py tests/integration/test_cli_run.py tests/integration/test_entry_consistency.py tests/integration/test_g1_flow.py tests/unit/test_config_resolve.py -q --maxfail=2` | 59 passed; 53.70 s. Includes the final normalization fix, real API/CLI training, exact configuration hashes, CPU numerical consistency and cross-workspace reproduction. |
| Final polynomial baseline regression | `MPLBACKEND=Agg OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest tests/integration/test_baselines.py -q` | 7 passed; 5.83 s. |
| Original frontend without competing test jobs | `npm test -- --maxWorkers=1` in `frontend/` | 53 passed; 60.45 s. |
| New form regressions | `npm test -- --maxWorkers=1 src/pages/NewExperimentPage.test.tsx` in `frontend/` | Before: 4 failed, 5 passed. After: 9 passed; 17.55 s. |
| Final frontend suite | `npm test -- --maxWorkers=1` in `frontend/` | 57 passed; 55.85 s. |
| Real reference-binding regression | `MPLBACKEND=Agg OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest tests/integration/test_studio_api.py -k reference_validation -q` | Before: validation rejected a succeeded PA reference. After: 1 passed; 53.17 s, including actual PA training, DPD training and application. |
| Package path regression | `MPLBACKEND=Agg OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest tests/integration/test_packages.py -k full_package_round_trips -q` | 1 passed; 28.61 s. The generated command also succeeded when executed verbatim by zsh against `/tmp/OpenDPD Studio 引号复算`. Windows shell syntax is not verified. |
| Final runtime and recovery regression | `MPLBACKEND=Agg OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest tests/integration/test_runtime.py tests/integration/test_workspace_recovery.py tests/integration/test_launcher_ownership.py tests/unit/test_launcher.py -q` | 21 passed; 70.51 s. |
| Wheel installed outside the repository | `MPLBACKEND=Agg OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest tests/packaging -q` after the frontend build | 3 passed; 55.54 s. Fresh environment borrows installed third-party dependencies by path, not the editable OpenDPD package. Covers real packaged CPU training, data files and GUI serving. |
| Final wheel after all code fixes | Same packaging command | 3 passed; 16.40 s. |
| No Node.js at runtime, automatic browser opening | `env PATH=/usr/bin:/bin:/usr/sbin:/sbin <installed-venv>/bin/opendpd gui --workspace '/tmp/OpenDPD packaged 无Node' --port 8877` | Passed on macOS. `shutil.which('node')` and `shutil.which('npm')` both returned `None`; the installed wheel opened the default Chrome browser automatically, authenticated, removed the bootstrap token from the URL and displayed the correct empty workspace. The test instance was shut down afterward. |
| Source distribution assets | `.venv/bin/python -m build --sdist --outdir /tmp/opendpd-studio-audit-dist` | Built successfully; archive inspection found the frontend index and assets, with no audit scratch directories or session metadata. |
| Production frontend | `npm run build`, `npm run typecheck`, `npm run lint`, `npm run types:check` in `frontend/` | Passed. |
| Browser contracts, keyboard navigation, accessibility and layout | `npx playwright test --ignore-snapshots --workers=1` in `frontend/` | 40 passed, 12 skipped; 3.8 min. Chromium at 1366×768 and 1920×1080, Firefox and WebKit at 1366×768. These use the repository's mock API; they are not computation evidence. The live/performance probes are the 12 skipped cases. Linux image baselines were not regenerated. |

### Test failures and their treatment

- The first unrestricted frontend run, concurrent with backend work, had 12
  timing/lookup failures. A serialized run of the unchanged suite passed 53/53.
- A later measured-capture interaction test repeatedly hit its unchanged 5 s
  timeout, including in a serialized run. It simulates several long operator
  descriptions character by character. That test now pastes the same text as
  complete user edits. Its upload, submitted payload, numeric values, timestamp,
  navigation assertions and timeout remain unchanged. Keyboard-specific tests
  still exercise keyboard navigation and Enter submission.
- One packaging attempt overlapped the audit's own frontend rebuild and observed
  the transiently empty asset directory. This was an audit sequencing error.
  Re-running after the build completed passed all three packaging tests.
- FastAPI/Starlette emitted deprecation warnings about their HTTPX test adapter
  and AnyIO alias. No dependency upgrades were made to suppress them.

## Real browser journey

The headed browser connects to `opendpd gui --no-browser --port 8876 --workspace
"/tmp/OpenDPD Studio 审查"`, serving the production bundle directly from FastAPI.
No API mocking is used in this journey.

1. Registered the measured built-in DPA_200MHz data and trained the unmodified
   PA GRU smoke recipe through the GUI. Run `run-20260906-173611-43303c` succeeded;
   its logs, checkpoint, metric profiles and plots were inspected.
2. Created a four-column CSV from the 23,040 paired samples in the bundled
   measured training capture (`tx_i`, `tx_q`, `rx_i`, `rx_q`). This is a format
   import test using measured data, not a new independent measurement or an
   external user's dataset.
3. Uploaded that CSV through the GUI, inspected its first rows and column
   mapping, confirmed 800 MHz sample rate, 200 MHz bandwidth, 20 MHz subchannels,
   10 subchannels, nperseg 2560 and normalized amplitude units. Kept the existing
   256-sample guard protocol unchanged. Dataset id: `audit-capture`.
4. Ran Dataset Doctor through the GUI: no blocking problems; the report
   retained a gain/phase warning and showed alignment, amplitude and spectrum
   evidence. No corrective preprocessing was applied to suppress the warning.
5. Trained PA GRU on the imported dataset through the GUI with the smoke defaults.
   Run `run-20260906-174609-22c34f` succeeded. Its raw file hash is
   `dd1abaaa458c1d844835ed8c17178ab8691bb381002a5ee538c71416724b0aa6`.
6. Selected the imported dataset's PA in the DPD GRU smoke form. The original
   code blocked submission. After the shared-service fix, validation succeeded
   and GUI run `run-20260906-175720-bf7808` completed all three epochs.
7. Applied the DPD through the GUI to the test split using its training PA
   surrogate. Run `run-20260906-175808-25df2c` succeeded with the expected
   `dpd_surrogate` evidence type.
8. Exported a full private package and downloaded it through the browser.
   The downloaded bytes matched the server export, SHA-256
   `80a9c03613cedafd792b61fddf2010c1cfdbce2a8b143c34e556894e25c8981a`.
   CLI import into `/tmp/OpenDPD Studio 复算` restored the measured data, the
   PA reference and the DPD run with no missing artifacts.
9. Ran `opendpd evaluate run-20260906-175720-bf7808 --workspace
   '/tmp/OpenDPD Studio 复算' --profile legacy-opendpd-v1 --json`. All 15 primary
   and baseline metric values matched with maximum absolute difference **0**,
   within the existing relative `1e-4` / absolute `1e-5` tolerance. Model weight
   hashes matched. The CLI result is recomputation of saved checkpoints, not
   a claim that independent retraining must produce identical weights.

Selected screenshots: [DPD run](studio-audit-2026-09-06/dpd-run.png) and
[completed GUI export](studio-audit-2026-09-06/export-ready.png).

## Remaining gate limitations and blocking notes

The 1366×768 run screenshot also shows a chart legend close enough to the x-axis
title to overlap. This remains a visual-polish item for G2; the Linux screenshot
baseline was not verified or rewritten in this macOS audit.

| Gate | Independent evidence from this audit | Remaining boundary |
|---|---|---|
| G0 | Real macOS browser, production assets, installed wheel, automatic default-browser opening without Node.js on PATH, startup ownership and crash/exit checks | Other operating systems and the complete platform support matrix remain unverified here. |
| G1 | Real GUI measured-data import, diagnosis, PA/DPD training, DPD application and export; downloaded package verified and re-evaluated by CLI in another workspace | This is an agent-operated review, not the required independent human acceptance trial. |
| G2 | Three browser engines, keyboard/axe checks, bounded regression tests and packaged computation | No full 20-start/100-sample performance protocol, 30-minute memory soak, 100-million-sample stress import or external user trial was performed. Do not promote G2 based on the pass counts above. |
| G3 | CPU tests of the extended services pass as software regression checks | Independent scientific validation, accelerator evidence, measured instruments and deployment hardware remain separately gated. |
| G4 | Existing tooling is covered by repository tests | External submissions, independent recomputations and community review were not established by this audit. |

Two security/lifecycle protocol gaps need a separately reviewed follow-up:

- **Bootstrap credentials are reusable.** Two real `POST
  /api/v1/session/bootstrap` requests using the launch credential both returned
  HTTP 200. `SessionStore.exchange` retains the bootstrap token; the session map
  has no server-side expiry. This differs from the one-time credential language
  in section 7.5 and the current threat-model document. Simply consuming the
  token would break the current launcher, which reopens the same bootstrap URL.
  Define credential rotation, existing-session reopening and expiry together,
  then add replay/expiry/reopen tests before calling this requirement accepted.
- **Instance reuse does not verify service identity/version.** The OS guard
  repairs concurrent normal launcher ownership. Reuse still accepts a live PID
  and any JSON response at `/healthz`; it does not bind that response to the
  expected workspace and package version. A future instance handshake should
  verify those fields and reject incompatible or copied instance metadata.

These are recorded rather than guessed into a new authorization protocol.
The development plan section 9.3 lists authorization-boundary changes as requiring
separate review; `AGENTS.md` also requires a blocking note when a protocol is
ambiguous. Current loopback binding, Host/Origin checks, CSRF, file boundaries,
and RF interlocks were preserved. No protected scientific paths, release notes,
release publication, external uploads, telemetry or CI permissions were changed.

## Scope and cleanup

The final ablation review kept only production changes exercised by the stated
regressions or the real browser journey. No runtime dependency was added.
Temporary browser snapshots, session files and scratch screenshots were removed
from the repository; only the two linked evidence screenshots remain. Audit
servers and browser sessions were closed. The disposable workspaces and logs
remain under `/tmp` for local inspection. Changes are local commits on
`OpenDPD-Studio`; no release or remote publication was performed.
