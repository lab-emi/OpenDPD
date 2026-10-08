# MATLAB toolbox 0.1.0 development preview

Validated on 2026-09-29 in the `codex/studio-2.4` worktree, based on commit
`7418ba3`. This is an unpublished development preview for OpenDPD 2.4.
The backend package metadata still uses the 2.3 baseline; toolbox version is
0.1.0 and SDK API version is 1.

## Implemented

The [toolbox guide](../tutorials/matlab-toolbox.md) covers installation and use.
MATLAB can select a Python environment or development checkout, import I/Q and
MAT v7 captures, submit PA/DPD experiments, query or cancel jobs, reconnect by
run ID, read stored results and export a DPD waveform. These jobs share Studio's
workspace and supervisor. The standalone `apply` function supports ordinary
GRU models on CPU with frozen offline segment settings and checked weight hashes.

The new Python SDK reuses existing services and schemas. Scientific scoring,
model implementations and Arena protocols are unchanged.

## Validation

Environment: Linux, MATLAB R2026a, Python 3.13.14, NumPy 2.4.4, SciPy 1.18.0
and PyTorch 2.13.0+cu132. All new training and inference checks used CPU.

| Check | Result |
| --- | --- |
| SDK numeric/MAT-file tests, real service/worker integration, launcher unit/ownership regressions | **62 passed** |
| MATLAB `buildtool package` | **6 tests passed**; test and package tasks succeeded |
| Installed toolbox in a fresh MATLAB process | Installed from `.mltbx`; entry point resolved inside the temporary add-on folder |
| Installed example | PA and DPD trained; standard export completed; saved **718 complex single samples**, matching input length and all finite |
| Uninstall | Succeeded; isolated installed-toolbox catalog empty afterwards |
| Python wheel and sdist | Built; wheel contains the SDK, sdist contains toolbox source and Apache-2.0 license |
| Installed Python wheel outside the checkout | Real PA training and 129-sample inference passed; packaged Studio returned `ready: true` |
| Python critical lint and OpenAPI consistency | Passed |
| Studio frontend build | Passed |
| MATLAB static check | Parsed successfully with MISS_HIT 0.9.44; name-value syntax style advisories are informational |

The numeric comparison uses fixed checkpoints and compares `apply` with the
existing Python evaluator for PA output and DPD input, with `rtol=1e-5` and
`atol=1e-6`. Checks also cover segment resets, partial tails, changed checkpoint
hashes, edits to dataset metadata, simultaneous connection attempts, cancellation
and reconnecting to stop a service.

MATLAB packaging and installation used separate temporary preference folders.
A fresh profile initially had an empty Add-Ons installation setting; selecting
a writable temporary folder for that session allowed installation. User
preferences were not changed. The Python dependencies came from the existing
environment, and `SourceDirectory` selected the 2.4 worktree for the installed
example.

## Package record

The local build is `Matlab/toolbox/dist/OpenDPD-0.1.0.mltbx` (41,349 bytes).
SHA-256: `ec00ba60b663db39e7c1b86593268da0d83af13dfc532a6fc7fd57b9ca912105`.
The final package includes the updated verification guide. Its 28 MATLAB
source files and license are byte-identical to the installed-and-tested build.

## Current limits

- R2024b, R2025b, R2026b, Windows, macOS and CUDA bridge execution need their own
  runtime checks. A manual GitHub workflow provides the release checks and
  packages an artifact; no workflow was dispatched or release published here.
- `apply` resets at each stored segment. The existing `runDPD` exporter carries
  state across the whole test waveform. Their metadata states the execution
  rules; the example saves the segmented training result with its `apply` output.
- MAT v7.3, streaming `apply`, other inference backbones, Simulink and physical
  RF measurements are future work.
- The short synthetic runs validate the workflow and carry no hardware or
  benchmark performance claim.
- Full strict documentation build remains blocked by **11 existing warnings**
  in the 2.3 Arena/navigation documents. The new toolbox pages add no warnings.
