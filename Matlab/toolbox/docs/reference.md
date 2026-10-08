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
