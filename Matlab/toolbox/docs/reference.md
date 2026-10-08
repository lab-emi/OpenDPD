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
ds = opendpd.importIQ(p, x, y, SampleRate=80e6, Bandwidth=20e6);
ds = opendpd.importMAT(p, "capture.mat", SampleRate=80e6, Bandwidth=20e6);
```

`ds` is a dataset struct including `dataset_id`, sample count and provenance.
`importIQ` accepts dense MATLAB single/double vectors; rows and columns are
equivalent. `importMAT` additionally accepts real N×2 I/Q matrices.

| Option | Default / requirement |
| --- | --- |
| `SampleRate`, `Bandwidth` | Required, finite and positive, in Hz; backend validates the signal specification. |
| `Name` | `""` for an automatic ID; otherwise a unique dataset slug such as `capture-001`. |
| `SegmentSamples` | `256`, integer at least 2. Frozen offline evaluation segment length. |
| `GuardSamples` | `256`, nonnegative integer. Must cover the training frame context. |
| `Subchannels` | `1`, positive integer. |
| `Origin` | `"unknown"`; also `"measured"`, `"synthetic"`. |
| `AmplitudeUnits` | `"unknown"`; also `"normalized"`, `"volts"`. Describes input units, does not transform amplitudes. |
| `InputVariable`, `OutputVariable` | MAT adapter only; defaults `"x"`, `"y"`. |

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
| `Device` | `"cpu"`; `"cuda"` and `"mps"` require locally supported hardware and model. |
| `NumThreads` | `1` |
| `Profile` | `"opendpd-spectral-v2"` |

`job.ID` and `job.Project.Workspace` identify a run for reconnection. Complete
settings, artifacts and logs remain in the workspace and Studio.

## Inference and export

| Call | Returns / behavior |
| --- | --- |
| `[y, info] = opendpd.apply(job, x, ...)` | Complex single column vector plus metadata. `Execution="offline_segmented"`, `Timeout=120` seconds. CPU, ordinary unquantized GRU only. |
| `exported = opendpd.runDPD(dpd)` | Job handle for standard test-split waveform export and surrogate evaluation. Wait for completion before reading its report/artifacts. |

See [execution semantics](workflow.html#apply-the-dpd) before comparing exported
and applied waveforms. `apply` uses stored sample-rate metadata and assumes the
supplied samples have the corresponding physical rate; vectors do not carry
their own time base.
