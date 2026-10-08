# OpenDPD Toolbox for MATLAB — development preview

Run PA and DPD experiments from MATLAB using the same workspace, queue and
results as OpenDPD Studio. Training runs in Python/PyTorch. MATLAB receives
ordinary structs and complex signal vectors.

This is the toolbox for the unreleased OpenDPD **2.4.0** (SDK protocol **1**). The
toolbox version follows the OpenDPD release it ships with. Install the Python code from
this checkout: the released 2.3 package does not contain this SDK. The Python package
metadata is still the 2.3 baseline until the release is cut.

## What works in this preview

- Open OpenDPD Studio as the toolbox GUI from `opendpd.studio()` or Apps.
- Use Studio's **MATLINK** tab to select MATLAB signals and import a paired capture.
- Select experiments by name and send saved reports to MATLAB, immediately or
  when training finishes. Destination variable names are generated automatically.
- Read six bundled offline guides with `opendpd.help()`.
- Start a local service or connect to a workspace already open in Studio.
- Import paired complex I/Q vectors or named numeric variables from MAT files of any version (v7.3 included).
- Submit PA/DPD training, query progress, cancel, and reconnect by run ID.
- Read stored evaluation results and queue the standard DPD waveform export.
- Apply a trained PA or DPD (`gru`, `tres_gru`, `gmp`, `mp_ls`, `gmp_ls`) to a supplied
  waveform on CPU, with the **offline segment boundaries** of the Python evaluator, or as
  a **stream** for the models that have a registered streaming variant (`gru`, `gmp`).
- Score a capture with the metrics Studio uses, without a project or server: `opendpd.waveform`,
  `opendpd.metrics.evm`, `aclr` and `evaluate`.
- Export a trained PA or DPD as an `opendpd-model-v1` package (`opendpd.export`) and run it in plain
  MATLAB with no Python (`opendpd.load`, `opendpd.verify`, `opendpd.apply(model, x)`, streaming as a
  System object); for `mp_ls`, `model.commCoefficients()` gives the `comm.DPD` coefficients.
- Build a source-only `.mltbx` using MATLAB, after running the MATLAB tests.

Training uses the existing model registry. `apply` supports the five models above
and refuses any other model, quantization-aware runs and streaming for models without
a registered streaming variant, with the reason. A model joins the list together with
a test that compares its output with the Python evaluator. `runDPD` follows the
existing supported export workflow. The Python integration tests exercise real
training workers and compare inference with the existing evaluator. The MATLAB workflow has also been
run locally with **MATLAB R2026a, Python 3.13.14 and CPU execution on Linux**.

## Studio GUI and MATLINK

```matlab
link = opendpd.studio("/path/to/experiment-workspace", ...
    PythonExecutable="/path/to/python");
opendpd.help();        % bundled getting-started guide
opendpd.help("gui");   % MATLINK walkthrough
```

Studio opens at **MATLINK**. Choose **OpenDPD Signal Generator** to configure
waveforms and a virtual PA; generated input/output captures are saved to MATLAB
before experiment setup opens with the dataset selected. Alternatively select
existing input/output variables from **MATLAB workspace**. After training,
MATLINK shows experiment metrics and spectra. **Save to MATLAB** returns the
report, resolved settings and available plot arrays to a suggested, editable
variable name. **Send when ready** queues delivery during training. Existing
variables are preserved; **Open in MATLAB** opens the delivered struct.

`opendpd.studio()` uses remembered setup settings. `opendpd.studio(p)` attaches
an existing Project, leaving it available to scripts. The MATLAB session retains
the bridge even if you do not keep its returned handle. Closing the browser
leaves MATLAB connected. `opendpd.disconnect()` detaches all bridges in this
MATLAB process; `opendpd.disconnect(link)` or `opendpd.disconnect(workspace)`
detaches one. Training and the service continue. Disconnecting or closing MATLAB
ends pending transfers; reconnect and send the saved report again when needed.

MATLAB executes named requests through a timer. A blocking MATLAB computation
can delay interaction and make presence appear offline until callbacks resume.
The browser cannot run arbitrary MATLAB expressions. Reports arrive as structs;
waveform inference and export remain available through the script API.

MATLAB's public `web` API launches the existing Studio application. MathWorks
documents external web application embedding as unsupported in `uihtml`, so
Studio runs in the system browser and communicates through the shared local
service. This requires neither MATLAB Compiler nor Web App Server. See the
[uihtml reference](https://www.mathworks.com/help/matlab/ref/uihtml.html),
[web reference](https://www.mathworks.com/help/matlab/ref/web.html), and
`opendpd.help("architecture")`.

Offline guide topics are `index`, `gui`, `workflow`, `reference`, `architecture`
and `troubleshooting`. Sources live in `docs/`, packaged HTML in `resources/docs/`.
Regenerate after editing with `python scripts/build_matlab_docs.py` from the
repository root (requires the `Markdown` Python package). `--check` verifies
that generated pages are current and their local links resolve.

## Install from this checkout

Start with desktop MATLAB R2024b or later and a compatible CPython environment.
Python 3.11 is a candidate common to the planned MATLAB release matrix. Check
the [MathWorks Python compatibility table](https://www.mathworks.com/support/requirements/python-compatibility.html)
for your release. Base MATLAB is sufficient for the synthetic example;
the Python environment supplies PyTorch and OpenDPD's dependencies.

In a terminal, from the repository root:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
```

On Windows, create the environment with `py -3.11 -m venv .venv`, then use
`.venv\Scripts\python.exe -m pip install -e ".[dev]"`.

In MATLAB, using the absolute paths on your machine:

```matlab
repo = "/absolute/path/to/OpenDPD";
addpath(fullfile(repo, "Matlab", "toolbox"));
info = opendpd.setup(PythonExecutable="/absolute/path/to/OpenDPD/.venv/bin/python");
```

For Windows, point `PythonExecutable` at `.venv\Scripts\python.exe`.
`setup` selects an existing environment; it does not install packages or stop
a loaded Python session. The default is `OutOfProcess`. If MATLAB has already
loaded an incompatible Python environment, follow the `pyenv` error guidance
before switching. `opendpd.doctor()` reports interpreter, dependency and SDK
versions.

For development alongside another OpenDPD checkout, reuse a Python environment
with the dependencies already installed and select this source tree explicitly:

```matlab
info = opendpd.setup(PythonExecutable="/path/to/existing/python", SourceDirectory=repo);
```

This changes only the selected MATLAB Python session's import path. If another
OpenDPD checkout is already loaded, restart that Python session before switching.

To use Studio and MATLINK from a source checkout, also build the web interface once
from the repository root:

```bash
npm --prefix frontend ci
npm --prefix frontend run build
```

The SDK itself works without built frontend assets. A workspace service stays
on loopback; this preview does not connect to a remote server or MATLAB Online.

## Run the synthetic example

```matlab
addpath(fullfile(repo, "Matlab", "toolbox", "examples"));
summary = opendpdQuickstart();
```

The example creates a fresh workspace, generates a synthetic nonlinear PA
capture, trains small GRUs, and writes `matlab-dpd-output.mat` containing the
held-out input, predistorted waveform, inference metadata and stored report.
It stops its service when finished. Two training epochs check the workflow;
the result is not a performance benchmark.

Reopen the experiment in Studio:

```matlab
p = opendpd.openProject(summary.workspace);
opendpd.openStudio(p);
% When finished with an SDK-started service:
opendpd.closeProject(p, StopService=true);
```

## Use your own paired I/Q data

`x` is the PA input and `y` is the corresponding PA output. Supply aligned
vectors, the sample rate, the occupied signal bandwidth and the segment length
explicitly.

```matlab
p = opendpd.openProject("my-pa-workspace");
ds = opendpd.importIQ(p, x, y, ...
    Name="capture-001", SampleRate=491.52e6, Bandwidth=200e6, ...
    Origin="measured", SegmentSamples=2048);   % PSD segment length, see below

training = struct('epochs', 150, 'frame_length', 200, 'seed', 0);
paJob = opendpd.trainPA(p, ds, Model="gru", Training=training);
opendpd.status(paJob)
pa = opendpd.wait(paJob, Timeout=3600);
dpd = opendpd.wait(opendpd.trainDPD(p, ds, PA=pa, ...
    Model="gru", Training=training), Timeout=3600);

[u, info] = opendpd.apply(dpd, xTest);
save("predistorted-input.mat", "u", "info", "xTest", "-v7");
report = opendpd.result(dpd);
```

Use an independent test waveform for `xTest`, with the training sample rate
and preprocessing. `u` is the **predistorted PA input**, before the physical
PA. The stored DPD report evaluates a learned PA surrogate. Hardware evidence
requires a separate capture and measured-evaluation workflow.

`ModelParameters` and `Training` are structs using the existing OpenDPD schema,
such as `struct('hidden_size', 8)` and `struct('epochs', 5, 'frame_stride', 16)`.
`opendpd.submit(p, config)` accepts a complete `ExperimentConfig` struct.
`Device="auto"` (the default) takes Studio's own default: the first detected of
`cuda`, `mps`, `cpu`, and `cpu` for least-squares models. Pass `Device="cpu"` for a
reproducible small run, or `Device="cuda"` with `DeviceIndex=1` to pick a GPU.
`NumThreads=0` (the default) lets the service decide.

To register the standard test-split waveform and surrogate evaluation in the
workspace, use `exported = opendpd.wait(opendpd.runDPD(dpd))`. Its artifacts
are available through Studio and `opendpd.result(exported)` reads its report.
The baseline `runDPD` exporter carries state over the whole test waveform;
`apply` resets at segment boundaries unless you ask for `Execution="streaming"`.
Read the export sidecar for its execution semantics. The example saves the segmented training-run report with `apply`'s
waveform and records the separate export run ID.

### MAT files

```matlab
ds = opendpd.importMAT(p, "capture.mat", ...
    InputVariable="tx", OutputVariable="rx", Name="capture-002", ...
    SampleRate=491.52e6, Bandwidth=200e6, SegmentSamples=2048, Origin="measured");
```

MATLAB reads the file, so any MAT-file version works, v7.3 included. Variables are dense
numeric `single`/`double`. Complex row and column vectors are accepted. Real MAT vectors
are I-only signals; real matrices with two columns represent I/Q. Cells, structs, sparse
arrays, integer classes and missing variables produce an actionable error.

`SegmentSamples` has no default. It is the PSD (Welch) segment length of every spectral metric and the
interval at which evaluation restarts a model's state, so choose it for your signal; Studio's generated
signals use 512-4096.

### Data and execution rules

- MATLAB vectors may be rows or columns; `apply` returns a complex `single`
  column vector of the same length. Complex samples use `I + 1j*Q`.
- The backend stores float32 I/Q. It records source dtype and shape, and
  performs no amplitude normalization, delay correction or gain fitting at
  the bridge boundary. Nonfinite values and float32 overflow are rejected.
- Import uses the existing contiguous split with a default 256-sample guard.
  The guard must cover the training frame context. Dataset names must be
  unique IDs, for example `capture-001`.
- `apply(..., Execution="offline_segmented")` (the default) is how the run was scored. It
  restarts the model's state at the run's frozen `SegmentSamples` boundary, pads the final
  segment with zeros and trims the padding from its result. A model that reads future
  samples (`tres_gru` reads 16) sees zero padding within that distance of a segment end.
  `info.limitations` says so. It does not carry state between calls.
- `apply(..., Execution="streaming")` (alias of `"streaming_stateful"`) carries one state across
  the waveform in chunks of `ChunkSamples`. It exists only for `gru` and `gmp`, whose
  registered streaming variants are `gru_stream` and `gmp_stream`; other models are refused
  rather than approximated. The output is a different signal from the one the stored report
  scored, and `info.streaming` carries the measured warm-up, look-ahead and chunk
  consistency. Choose it deliberately for a waveform that will run continuously.
- Each call loads the recorded model in an isolated CPU process; it is intended for whole
  waveforms.
- Inference metadata includes checkpoint and sample hashes, output role, sample count,
  execution semantics, segment length, sample rate and preprocessing version. Later edits
  to dataset metadata do not change these inference settings. A checkpoint whose hash
  changed is refused.
- MAT import records the file name, its SHA-256 and the variable names (class, complexity,
  size), and retains the converted I/Q source in the workspace. Keep the original MAT file
  if its other variables are part of your experiment record.

## Jobs and service lifetime

```matlab
record = opendpd.status(paJob);
opendpd.cancel(paJob);                   % explicit cancellation request
job = opendpd.getRun(p, savedRunID);      % reconnect after reopening a project
opendpd.closeProject(p);                 % disconnect; service/jobs continue
opendpd.closeProject(p, StopService=true); % stops an SDK service and its jobs
```

Save `job.ID` and `p.Workspace` as strings to reconnect later. Python objects
inside the MATLAB handles are not portable MAT-file model artifacts.
Timeout or Ctrl+C while waiting leaves a submitted job running. A job being
cancelled stays `cancel_requested` until its worker stops.

`StopService=true` also works after reconnecting to an SDK-started service.
It affects all jobs in that workspace. A service launched by Studio itself
must be stopped through its original launcher. Service logs live in the
workspace as `.sdk-service-*.log`; they contain the private local bootstrap
URL, so keep them local.
If a data import loses its connection, inspect the dataset list in Studio
before retrying the same name. The server may still finish; the SDK retains
its staged input under `imports/` and reports that file's location.

## Test and package

Python, frontend and guide checks, from the repository root:

```bash
python -m pytest tests/unit/test_sdk_iq.py tests/unit/test_studio_navigation.py \
  tests/unit/test_matlink.py tests/integration/test_sdk_matlab.py \
  tests/integration/test_matlink_api.py -q
python scripts/build_matlab_docs.py --check
npm --prefix frontend run typecheck
npm --prefix frontend test
```

In MATLAB, set the interpreter for the test suite, then build from this folder:

```matlab
setenv("OPENDPD_MATLAB_PYTHON", "/absolute/path/to/.venv/bin/python");
cd(fullfile(repo, "Matlab", "toolbox"));
buildtool test
buildtool package
```

The package task runs tests first and writes `dist/OpenDPD-2.4.0.mltbx`.
The archive contains MATLAB source, tests, examples, help and the Apache-2.0
license; Python and model weights remain in the selected environment/workspace.
Install it with `matlab.addons.toolbox.installToolbox`, then select Python
with `opendpd.setup`. Uninstalling it leaves Python environments and experiment
workspaces intact.

The **MATLAB toolbox** GitHub workflow runs on pull requests that touch the toolbox,
SDK, MATLINK or `apply`, and can be started by hand for another MATLAB release. It runs
the Python SDK tests (including the `apply` parity tests), the MATLAB tests, builds the
package, then installs it and runs the example in a fresh MATLAB session, on Linux and
on Windows (Windows reports without blocking until it has been seen to pass). It
uploads a workflow artifact; it does not publish a release or submit anything to File
Exchange.

## Verification status

Current changes should pass their MATLAB, server and frontend checks before
packaging. Validation records live in `docs/performance/` and `docs/releases/`.

MATLAB R2024b, R2025b, R2026b, Windows, macOS and CUDA bridge workflows still need
release/platform-specific execution. The package declares R2024b as a candidate
minimum. If a fresh batch MATLAB profile reports that its installation path is
not accessible, set a writable Add-Ons folder in MATLAB Settings. The installation
check uses a temporary folder for its own session.

MATLAB's [Python interface](https://www.mathworks.com/help/matlab/matlab_external/ways-to-call-python-from-matlab.html)
and [ToolboxOptions](https://www.mathworks.com/help/matlab/ref/matlab.addons.toolbox.toolboxoptions.html)
are the underlying integration and packaging APIs.
