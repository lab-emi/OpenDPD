# Function reference

All functions live in the `opendpd` namespace. MATLAB's `help opendpd.function`
also shows each function's short reference. Numeric sample rate and bandwidth
arguments use **Hz**; MATLINK displays **MHz**.

## Environment and MATLINK

| Call | Purpose and options |
| --- | --- |
| `info = opendpd.setup(...)` | Select Python. `PythonExecutable=""`, `SourceDirectory=""`, `ExecutionMode="OutOfProcess"`. SourceDirectory is an optional development checkout. Does not install packages or terminate Python. |
| `info = opendpd.doctor()` | Dependency checks, Python/OpenDPD versions, SDK protocol and supported inference models. |
| `p = opendpd.openProject(workspace, ...)` | Open or attach a local workspace. `StartService=true`, `Timeout=30` seconds. `p.Workspace` is the resolved path. |
| `opendpd.closeProject(p, ...)` | Disconnect a Project. `StopService=false`; true explicitly stops an SDK-started service and its jobs. Use `disconnect` to detach MATLINK. |
| `link = opendpd.studio(...)` | Connect MATLAB and open Studio at MATLINK. Uses remembered workspace, Python and source settings when omitted. Options: `PythonExecutable=""`, `SourceDirectory=""`, `OpenBrowser=true`, `Label` defaulting to the MATLAB release. |
| `link = opendpd.studio(workspace, ...)` | Connect a specified workspace; reuse its active bridge in this MATLAB process. |
| `link = opendpd.studio(p, ...)` | Connect MATLINK through an existing Project. That Project remains caller-owned and usable after MATLINK disconnects. |
| `opendpd.disconnect()` | Disconnect every MATLINK bridge in the current MATLAB process; leave services and jobs running. |
| `opendpd.disconnect(link)` | Disconnect one returned MATLABBridge. |
| `opendpd.disconnect(workspace)` | Disconnect the bridge for one workspace. |
| `opendpd.openStudio(p, ...)` | Open a Studio page in the system browser. `Page="home"`, `RunID=""`, `OpenBrowser=true`. This alone does not create a MATLAB bridge. |
| `opendpd.help(topic)` | Open a bundled guide. Topics: `index` (default), `gui`, `workflow`, `reference`, `architecture`, `troubleshooting`. `OpenBrowser=false` returns the local filename without opening it. |

`studio` returns an `opendpd.MATLABBridge` and retains it for the MATLAB session,
so calling without an output keeps MATLINK active. Use `disconnect` for lifecycle
control. Internal request messages and timer state are implementation details,
not a stable public automation API. When MATLAB is blocked, requests and the
heartbeat wait until callbacks can run again.

```matlab
link = opendpd.studio("my-pa-workspace", Label="PA capture session");
opendpd.help("gui");
opendpd.disconnect(link);
```

`OpenBrowser=false` connects the bridge without launching a browser. Studio pages
supported by `openStudio` are `home`, `matlink`, `datasets`, `experiments`,
`new-experiment`, `results`, `run` and `result`. The last two require `RunID`,
which is useful in scripts; the MATLINK GUI uses named experiment selection.
Only supported local routes are accepted. A returned URL from `openStudio` is a
**private bootstrap link**; keep it local. Normal use prints no token in the
Command Window.

```matlab
opendpd.openStudio(p, Page="datasets");
opendpd.openStudio(p, Page="result", RunID=job.ID);
```

## Import

```matlab
ds = opendpd.importIQ(p, x, y, SampleRate=80e6, Bandwidth=20e6, SegmentSamples=2048);
ds = opendpd.importMAT(p, "capture.mat", SampleRate=80e6, Bandwidth=20e6, SegmentSamples=2048);
```

`ds` is a dataset struct including `dataset_id`, sample count and provenance.
`importIQ` accepts dense MATLAB single/double vectors; rows and columns are
equivalent. `importMAT` reads a MAT file of any version in MATLAB and additionally accepts
real N×2 I/Q matrices.

| Option | Default / requirement |
| --- | --- |
| `SampleRate`, `Bandwidth` | Required, finite and positive, in Hz; backend validates the signal specification. |
| `SegmentSamples` | Required, integer at least 2, no default. PSD segment length of every spectral metric and the interval at which evaluation restarts a model's state. Studio's generated signals use 512-4096. |
| `Name` | `""` for an automatic ID; otherwise a unique dataset slug such as `capture-001`. |
| `GuardSamples` | `256`, nonnegative integer. Must cover the training frame context. |
| `Subchannels` | `1`, positive integer. |
| `Origin` | `"unknown"`; also `"measured"`, `"synthetic"`. |
| `AmplitudeUnits` | `"unknown"`; also `"normalized"`, `"volts"`. Describes input units, does not transform amplitudes. |
| `InputVariable`, `OutputVariable` | MAT adapter only; defaults `"x"`, `"y"`. |
| `Source` | `importIQ` only; a struct recorded as the dataset's provenance (`importMAT` fills it). |

## Jobs

| Call | Returns / behavior |
| --- | --- |
| `job = opendpd.trainPA(p, ds, ...)` | Queue PA training. `ds` may be a dataset struct or ID. |
| `job = opendpd.trainDPD(p, ds, PA=pa, ...)` | Queue DPD training using a succeeded PA Job from the same workspace. |
| `job = opendpd.submit(p, config)` | Submit a complete ExperimentConfig struct; server validates it. |
| `job = opendpd.getRun(p, runID)` | Create a handle for a saved run. |
| `record = opendpd.status(job)` | Read status and error information as a struct. |
| `job = opendpd.wait(job, ...)` | Wait for success; `Timeout=600`, `PollInterval=0.5`, in seconds. Errors on failure or timeout; timeout does not cancel. |
| `record = opendpd.cancel(job)` | Request cancellation through the shared service. |
| `report = opendpd.result(job)` | Read the stored result struct. Requires a saved result. |

Training options shared by `trainPA` and `trainDPD`:

| Option | Default |
| --- | --- |
| `Model` | `"gru"` |
| `ModelParameters`, `Training` | Empty structs; use existing OpenDPD schema fields. |
| `Device` | `"auto"`: Studio's default (first detected of `cuda`, `mps`, `cpu`; `cpu` for least-squares models). Or `"cpu"`, `"cuda"`, `"mps"`, which require locally supported hardware and model. |
| `DeviceIndex` | `0`; the logical CUDA device when `Device` resolves to `"cuda"`. |
| `NumThreads` | `0`: the service decides. A positive integer sets the CPU thread count. |
| `Profile` | `"opendpd-spectral-v2"` |

`job.ID` and `job.Project.Workspace` identify a run for reconnection. Complete
settings, artifacts and logs remain in the workspace and Studio.

## One call: fit

`opendpd.fit` does what the project API does in order - import the capture, train a PA model, train a DPD through it,
export both - in one call, and returns the two models as `opendpd.Model` objects that run in plain MATLAB.

| Call | Returns / behavior |
| --- | --- |
| `[dpd, pa, report] = opendpd.fit(x, y, Workspace=w, SampleRate=fs, Bandwidth=bw, SegmentSamples=n, ...)` | `x` is the PA input and `y` the PA output (complex vectors of one length; nothing is normalised). `dpd` and `pa` are loaded model packages (`opendpd.apply(dpd, xNew)`, streaming for `gru` and `gmp`). `report` has `Workspace`, `Dataset`, `PA` and `DPD` (`RunID`, `Result`, `Package` file, `Verify`), `Seconds`, `Python` and `Job`. |

| Option | Default / requirement |
| --- | --- |
| `Workspace` | Required. The folder for datasets, runs and packages; created if missing. No default, so results never land somewhere you did not choose. Every step is an ordinary run you can open in Studio. |
| `SampleRate`, `Bandwidth`, `SegmentSamples` | Required, as for `importIQ`. |
| `DPDModel`, `PAModel` | `"gru"` for both. Both must be exportable (`gru`, `tres_gru`, `gmp`, `mp_ls`, `gmp_ls`), and `PAModel` must be gradient-trained (`gru`, `tres_gru`, `gmp`) because the DPD is trained through it. Checked before anything trains. |
| `DPDParameters`, `PAParameters`, `Training` | Empty structs: OpenDPD defaults. `Training` is shared by both runs (a DPD must use its surrogate's seed and frame length). |
| `Device`, `DeviceIndex`, `NumThreads`, `Profile` | As for `trainPA`. |
| `Name`, `Subchannels`, `GuardSamples`, `Origin`, `AmplitudeUnits` | As for `importIQ`. |
| `OutputFolder` | `<Workspace>/exports`; packages are `<run id>.opendpd.zip`. |
| `Timeout` | Seconds; none by default. When it passes the running job is cancelled and the call fails with `opendpd:Timeout`. |
| `PythonExecutable`, `SourceDirectory` | See below. `SourceDirectory` is a development checkout put on `PYTHONPATH`. |
| `Verbose` | `true`: print the import, the epochs (about ten lines per model) and the exports. |

**Python is a separate process.** `fit` starts `python -m opendpd.sdk._fit` as a child of MATLAB and exchanges files with
it; MATLAB's Python integration (`pyenv`, the `py.` namespace) is not used or loaded. The Python version therefore need
not be one your MATLAB release supports, and a crash in Python cannot take MATLAB down. The interpreter is the first
that exists of: `PythonExecutable`, the `OPENDPD_PYTHON` environment variable, the one `opendpd.studio` or
`opendpd.setup` remembered, MATLAB's configured `pyenv` (reading the setting does not load Python). MATLAB's own library
folders are removed from the child's library path (`LD_LIBRARY_PATH`, `DYLD_*`, and `PATH` on Windows) so that system
libraries are not replaced by MATLAB's; other entries stay. Ctrl+C asks the running job to cancel and returns once it has
stopped (up to a minute, then the process is ended). Verified on R2026a with Python 3.13 on Linux, in a fresh session
that left `pyenv` unloaded; Windows and macOS are not verified.

**The workspace service.** `fit` works through the workspace's Studio service. If none is running it starts one and stops it
when the call ends. A service that was already running - a Studio you have open on that workspace, or another session's -
is used and left running, and so is one that still has other runs; `fit` only stops what it started. If the service it
started does not stop promptly, `fit` still returns its result and warns (`opendpd:fit:ServiceNotStopped`).

The exported models are verified against their golden vectors on this MATLAB release before `fit` returns; if one does
not reproduce OpenDPD's outputs, `fit` fails with `opendpd:Verification` and names the package. Errors from Python come
back with its message: `opendpd:FitFailed`, `opendpd:Cancelled`, `opendpd:Timeout`.

## Metrics without a run

These need no project, workspace or server: they call the code that Studio and the run service use, so a number
computed here equals the one in a Studio result for the same signal and settings. Metadata that changes a number
has no default (`SegmentSamples`, `SampleRate`); a metric that cannot be computed is returned with a `Status` and a
`Reason`, never a guess.

| Call | Returns / behavior |
| --- | --- |
| `w = opendpd.waveform(Seed=1, Subframes=10)` | The `ofdm-lte20-v1` test waveform with known symbols, regenerated from its seed: `w.x` (complex column, 30.72 MS/s, unit power), `w.Symbols` (symbols × 1200), `w.SampleRate`, `w.Seed`, `w.Subframes`, `w.SHA256`. A test signal, not a conformance signal. |
| `m = opendpd.metrics.evm(y, w, SampleRate=fs)` | Data-aided EVM of a capture of `w`: `m.EVM_RMS` (percent), `m.EVM_dB`, `m.Status`, `m.Reason`. `fs` must convert to 30.72 MS/s with a small exact ratio. `m.Status` is `"missing_reference"` when `y` does not correlate with `w` (wrong seed or length, or a carrier offset above about 75 Hz for 10 subframes). |
| `m = opendpd.metrics.aclr(y, SampleRate=fs, SegmentSamples=n, Waveform=w)` | Adjacent-channel leakage in negative dBc: `m.ACLR_L`, `m.ACLR_R`, `m.Status`, `m.Reason`. `Profile="ofdm-lte20-evm-v1"` (default; needs `Waveform` and `fs` of at least 58 MS/s) or `"opendpd-spectral-v2"` (needs `Bandwidth`, `Subchannels`). `SegmentSamples` is the Welch segment length and moves the ratio. |
| `t = opendpd.metrics.evaluate(y, SampleRate=fs, SegmentSamples=n, Waveform=w, ...)` | Every metric of a profile as a table (`Name`, `Value`, `Unit`, `Status`, `Reason`). Also `Profile="general-spectral-v1"`, `Reference=` (target signal for reference-based metrics), `Bandwidth`, `Subchannels`. |

How these compare with MathWorks functions is measured, not assumed: EVM agrees with an independent
`lteOFDMDemodulate`/`lteEVM` chain to better than 1e-8 percentage points on the registered signals, and the
profile's ACLR differs from `comm.ACPR` by up to 0.2 dB because `comm.ACPR` integrates the bins that enclose a band
edge while OpenDPD sums the bins whose centre lies inside it. The numbers, budgets and the cases that do not agree
are in `docs/performance/matlab-parity.md` of the OpenDPD repository.

## Inference and export

| Call | Returns / behavior |
| --- | --- |
| `[y, info] = opendpd.apply(job, x, ...)` | Complex single column vector plus metadata. `Execution="offline_segmented"` (default, how the run was scored) or `"streaming_stateful"` / `"streaming"` (one state across chunks; `gru` and `gmp` only), `ChunkSamples=0` (the default chunk), `Timeout=120` seconds. CPU, unquantized `gru`, `tres_gru`, `gmp`, `mp_ls`, `gmp_ls`. `info.execution`, `info.limitations` and `info.streaming` state what produced `y`. |
| `exported = opendpd.runDPD(dpd)` | Job handle for standard test-split waveform export and surrogate evaluation. Wait for completion before reading its report/artifacts. |

See [execution semantics](workflow.html#apply-the-dpd) before comparing exported
and applied waveforms. `apply` uses stored sample-rate metadata and assumes the
supplied samples have the corresponding physical rate; vectors do not carry
their own time base.

## Model packages: run a trained model without Python

`opendpd.export` writes a trained PA or DPD as an `opendpd-model-v1` package: a zip of data (`manifest.json`,
`weights.mat` and `weights.npz` with the same arrays, `golden/` with a test input and the outputs OpenDPD produced for it,
a README). It holds no code. `opendpd.load` reads it back as an `opendpd.Model` that runs in plain MATLAB: no Python,
no project, no server, no other toolbox. A run trained in Studio can be exported from a terminal with
`opendpd export-model RUN_ID --workspace WORKSPACE --out file.opendpd.zip`; it writes the same bytes.

| Call | Returns / behavior |
| --- | --- |
| `s = opendpd.export(job, file, ...)` | Write a succeeded PA or DPD run to `file` (`*.opendpd.zip`). Needs the Python SDK. `Timeout=300` seconds. Models: `gru`, `tres_gru`, `gmp`, `mp_ls`, `gmp_ls` (those `apply` supports), unquantized. The same run always gives the same bytes. `s` has `path`, `sha256`, `model`, `role`, `run_id`, `files`, `golden_samples`, `execution`. |
| `model = opendpd.load(file)` | Read a package as data and return an `opendpd.Model`. Properties: `Manifest` (model, signal, scaling, execution semantics, evidence, provenance), `Source`, `SHA256` of the file. |
| `report = opendpd.verify(model)` | Run the package's golden test vector on this MATLAB release. `report.passed`, `report.offline_max_abs_error`, `report.streaming_max_abs_error` (`NaN` for a model without a streaming variant), `report.tolerance_abs` (at most `1e-5`; a package can ask for a stricter test, never a looser one). A non-finite output never passes. |
| `[y, info] = opendpd.apply(model, x, ...)` | Same options and result as for a job (`Execution`, `ChunkSamples`), computed by `opendpd.runtime` in MATLAB. `Timeout` does not apply. Input is rounded to single first, as the Python evaluator does. |
| `y = model(chunk)`, `reset(model)` | The model as a System object for streaming (`gru`, `gmp`): one state across calls, any chunk size. `reset` returns to the start of a stream. |
| `[C, info] = model.commCoefficients()` | `mp_ls` only: `C = reshape(w, Q, K)`, the `Coefficients` of `comm.DPD('PolynomialType', 'Memory polynomial')`. Other models: error `opendpd:NoMathWorksEquivalent`. |

```matlab
opendpd.export(dpd, "apa-dpd.opendpd.zip");           % where the model was trained (Python SDK)
model = opendpd.load("apa-dpd.opendpd.zip");          % anywhere: plain MATLAB
assert(opendpd.verify(model).passed)                  % this release computes the model as OpenDPD did
u = opendpd.apply(model, xTest);                      % the predistorted PA input, like opendpd.apply(dpd, xTest)
```

What `verify` shows and what it does not: a pass means this MATLAB release computes the package's model on the
golden input to within `1e-5` of what OpenDPD's `apply` produced. It does not show that the model is good for your
amplifier, and it does not make a package from an unknown source trustworthy: the golden vector comes from the
same file. Compare `model.SHA256` with the value published by whoever gave you the package.

Reading a package is deliberately narrow. Only the six known file names are accepted (nothing else is extracted, no
path from the archive is used), every file must match the SHA-256 in the manifest, entries are copied out with a hard
size limit, and the arrays are read from the `.npz` files by a strict parser that only ever interprets bytes as float32,
float64, complex64 or complex128 numbers. The `.mat` files are for your own code and are never opened by the toolbox:
`load` and `whos -file` both call `loadobj` for classes on the MATLAB path, so opening a MAT file from an untrusted
source can itself run code. The golden test input is synthetic noise with the training input's amplitude statistics,
never a slice of your data, so a package can be shared without sharing a measurement.

Execution and numerics: `opendpd.apply(model, x)` uses the same two semantics as for a job (`offline_segmented`, the
default, restarts state every `Manifest.signal.nperseg` samples and zero pads the last segment; `tres_gru` reads 16
future samples inside a segment). MATLAB computes in double precision; PyTorch uses float32, so outputs agree to
about `1e-7`, which is what the golden test measures. Speed on a development machine (R2026a, Linux, no GPU, one MATLAB
process) is roughly 0.3 us per sample for `mp_ls`, 1 us for `gmp_ls`, 2-5 us for a two-layer `gru` or `tres_gru` of
hidden size 6-64, and 13 us for a `gmp` of 495 terms; it is for evaluating waveforms, not a real-time implementation.
Not verified: MATLAB Coder, a Simulink MATLAB System block, HDL generation, other MATLAB releases or operating systems.

## Measured captures: `opendpd.lab`

`opendpd.lab.Session` supervises a measurement made by **your** instrument code. RF stays off until a named person arms
the session, every limit is checked before anything is sent, and every abnormal path - an error in your functions, a
timeout, a lost link, a capture that is empty or not finite, Ctrl+C, a failed RF-off - switches RF off and leaves the
session *tripped*, which cannot be armed again. It follows OpenDPD's Python interlock (`opendpd/instruments/safety.py`,
`docs/architecture/instruments.md`); the instrument adapter is a pair of function handles, so the code your laboratory
already has (VISA, a vendor driver, a remote laboratory) is used as it is. No driver ships with the toolbox.

| Call | Returns / behavior |
| --- | --- |
| `lab = opendpd.lab.Session(MeasureFcn=@f, RFOffFcn=@g, Operator="Name", ...)` | A disarmed session. `f(u, fs)` plays the complex baseband signal `u` (digital full scale 1.0) at sample rate `fs` and returns the captured complex vector or an N-by-2 `[I Q]`; it is called as `f(u)` when `measure` gets no `SampleRate`. `g()` switches the output off: required, idempotent, no default. Optional: `HeartbeatFcn` (returns when the link is alive, errors otherwise), `PowerCalibration(u)` (the dBm that playing `u` produces, one finite number; for a safety ceiling give peak power), `MaxPeak=1`, `MaxOutputPower_dBm` (none), `Timeout=30` seconds, `RFOffAfterMeasure=true`, `Name`, `Description`. |
| `lab = opendpd.lab.Session(Instrument=opendpd.lab.MockInstrument())` | A dry-run session around a fixed synthetic PA. It emits nothing, arms with a name alone, and records `mock: true` everywhere. For learning the procedure and testing your scripts; what it returns is never evidence about an amplifier. The class is sealed, so a subclass that drives hardware cannot pass as the mock. |
| `arm(lab)`, `arm(lab, "Name")` | A named person arms the session (an empty name is refused). A session built from functions also needs the environment variable `OPENDPD_ALLOW_RF_OUTPUT=1`, which only an approved laboratory session sets: this toolbox, its tests and automated tooling never set it, and its tests do not run when it is set. |
| `y = measure(lab, u, SampleRate=fs, RequestedPower_dBm=p)` | Refuses unless armed, and before anything is sent refuses a signal that is empty, not finite, not a vector or real N-by-2, whose peak exceeds `MaxPeak`, or whose power exceeds `MaxOutputPower_dBm` (the larger of the calibration and `RequestedPower_dBm`). Then calls your `MeasureFcn` and returns the capture as a complex column. Stricter than the Python interlock: with a power ceiling set and no way to know the power, it refuses. RF is switched off after the measurement unless `RFOffAfterMeasure=false`. |
| `disarm(lab)`, `abort(lab, reason)` | RF off. `abort` also trips the session; `disarm` does not clear a trip. Deleting a session that is still armed switches RF off. |
| `r = record(lab)`, `record(lab, Compact=true)`, `saveRecord(lab, file)` | The record: operator, limits, the SHA-256 of every played and captured signal (float32 interleaved I/Q, the hash Python session records use), the log of every state change, whether the instrument was a mock. `Compact=true` fits a dataset's 2000-character notes (free text is cut: operator at 100 characters, trip reason at 200) and carries the SHA-256 of the complete record, which `saveRecord` writes as JSON and returns `file` and `sha256` for. |

```matlab
lab = opendpd.lab.Session(MeasureFcn=@myMeasure, RFOffFcn=@myRFOff, Operator="Your Name", ...
    MaxPeak=0.9, MaxOutputPower_dBm=30, PowerCalibration=@myPeakPower_dBm, Timeout=60);
% The next line needs OPENDPD_ALLOW_RF_OUTPUT=1, which you set only in an approved laboratory session.
arm(lab);
y = measure(lab, u, SampleRate=fs);                % limits first, then myMeasure(u, fs); RF off afterwards
disarm(lab);
ds = opendpd.importIQ(p, u, y, SampleRate=fs, Bandwidth=bw, SegmentSamples=2048, Origin="measured", ...
    Source=record(lab, Compact=true));              % the dataset keeps who armed the session and what was played
saveRecord(lab, "session-1.json");                 % the complete record, next to your data
```

**What MATLAB cannot do.** It cannot interrupt a running function, so `Timeout` is checked when `MeasureFcn` returns,
and the heartbeat is checked before and after a measurement, not during it. Give your instrument calls their own time
limits (for example the `Timeout` of a `visadev`) so that a hung instrument returns by itself. Ctrl+C trips the session
through an `onCleanup` guard that MATLAB documents; that path was not automated in tests (`matlab -batch` did not react
to SIGINT, so the cleanup could not be exercised). If `RFOffFcn` itself fails the session warns (`opendpd:lab:RFOffFailed`), is tripped and says that the output may
still be on: switch it off at the instrument.

**What this is not.** A session is a safety wrapper and a record, not a calibration: it does not align delay or gain,
does not know your attenuators, and does not establish the amplifier's real output power unless your `PowerCalibration`
does. It has been exercised only on the mock, with injected faults; no instrument chain has been tested with it, so the
supervised trial that `docs/protocols/measured-dpd.md` requires is still to be done on a real bench.
