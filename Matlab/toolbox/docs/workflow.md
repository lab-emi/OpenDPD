# Script workflow

Scripts and MATLINK use the same workspace service as Studio. Script workflows
can save workspace paths and run IDs for reproducible reconnection; the MATLINK
GUI selects experiments by name.

## Run the complete synthetic example

After [setup](index.html#install-and-select-python):

```matlab
toolboxRoot = fileparts(fileparts(which('opendpd.setup')));
addpath(fullfile(toolboxRoot, "examples"));
summary = opendpdQuickstart();
```

The example creates a fresh workspace and a synthetic nonlinear PA capture,
trains small PA and DPD GRUs for two epochs on CPU, checks inference, exports a
waveform and writes `matlab-dpd-output.mat`. It closes its service when finished.
The MAT file includes `xTest`, `u`, `info`, `fs` and `report`. This is a workflow
check, not a model benchmark. Reopen it with:

```matlab
p = opendpd.openProject(summary.workspace);
link = opendpd.studio(p);
```

## Import your capture

Let `x` be the PA input and `y` its aligned output, recorded at the same sample
rate. Provide occupied signal bandwidth explicitly, in Hz.

```matlab
p = opendpd.openProject("my-pa-workspace");
ds = opendpd.importIQ(p, x, y, Name="capture-001", ...
    SampleRate=491.52e6, Bandwidth=200e6, Origin="measured", ...
    SegmentSamples=256, GuardSamples=256);
```

Rows and columns are accepted. Backend storage uses float32; source precision
and shape are recorded. No delay correction, normalization or gain fitting is
performed at the bridge boundary. Nonfinite values and float32 overflow are
rejected. Import uses the existing contiguous train/validation/test split; set
the guard to cover your training frame context.

For a MAT v7 file with numeric variables:

```matlab
ds = opendpd.importMAT(p, "capture.mat", ...
    InputVariable="tx", OutputVariable="rx", Name="capture-002", ...
    SampleRate=491.52e6, Bandwidth=200e6, Origin="measured");
```

The MAT adapter accepts dense single/double complex vectors and real N×2 I/Q
matrices. Real vectors are I-only. Save selected variables with `save(...,
'-v7')` for v7.3 sources. Source file hash and variable names are recorded;
keep the original MAT file if its other variables matter to your experiment.

## Train the PA and DPD

```matlab
training = struct('epochs', 150, 'frame_length', 200, ...
    'frame_stride', 100, 'seed', 0);
paJob = opendpd.trainPA(p, ds, Model="gru", Training=training);
opendpd.status(paJob)
pa = opendpd.wait(paJob, Timeout=3600);
dpdJob = opendpd.trainDPD(p, ds, PA=pa, Model="gru", Training=training);
dpd = opendpd.wait(dpdJob, Timeout=3600);
report = opendpd.result(dpd);
```

`ModelParameters` and `Training` use the existing OpenDPD schema, for example
`struct('hidden_size', 8)` and `struct('epochs', 5)`. CPU is the default. A
complete experiment configuration can be sent with `opendpd.submit(p, config)`.
Training remains in Python/PyTorch and uses the model registry and validation
already used by Studio.

## Apply the DPD

Use an independent `xTest` waveform with the training sample rate and
preprocessing:

```matlab
[u, info] = opendpd.apply(dpd, xTest);
save("predistorted-input.mat", "u", "info", "xTest", "-v7");
```

`u` is a complex single column vector with the same sample count as `xTest`.
For a DPD run it is the **predistorted PA input**, before a physical or simulated
PA. Applying a PA run instead produces the modeled PA output.

The preview supports ordinary, unquantized GRU models on CPU. It resets state
at the trained run's frozen segment boundaries, zero-pads the final segment and
trims padding from the returned vector. Calls do not share hidden state. Each
call loads the recorded model in an isolated process, so pass a whole waveform
rather than repeatedly calling it for individual samples.

Inference metadata includes checkpoint and sample hashes, execution mode,
sample count, segment length, sample rate, preprocessing version and output
role. Later edits to dataset metadata do not alter frozen inference settings.
A checkpoint whose hash changed is refused.

To queue the standard test-split waveform export:

```matlab
exported = opendpd.wait(opendpd.runDPD(dpd));
exportReport = opendpd.result(exported);
opendpd.openStudio(p, Page="run", RunID=exported.ID);
```

The existing `runDPD` exporter carries state across the whole test waveform;
`apply` resets at segment boundaries. These outputs can differ. Read the export
sidecar for its execution semantics. The quickstart saves the segmented
training-run report alongside `apply`'s waveform and separately records the
export run ID. Stored DPD reports evaluate a PA surrogate. A new waveform passed
to `apply` does not update that stored report or establish hardware performance.

## Reconnect, cancel and stop

MATLINK report delivery does not require script IDs. Use its **Send to MATLAB**
or **Send when ready** actions, then select **Open in MATLAB** to inspect the
saved report. For scripts, keep Job handles during a session and save IDs for
later reconnection:


```matlab
savedRunID = dpd.ID;
savedWorkspace = p.Workspace;
opendpd.disconnect(p.Workspace); % if MATLINK is attached
opendpd.closeProject(p);  % disconnect; service and jobs continue
p = opendpd.openProject(savedWorkspace);
job = opendpd.getRun(p, savedRunID);
opendpd.status(job)
opendpd.cancel(job);      % explicit cancellation request, when applicable
```

If MATLINK is attached, `opendpd.disconnect(p.Workspace)` detaches that bridge
without stopping training or closing a caller-owned Project. Calling
`opendpd.disconnect()` detaches all MATLINK bridges in this MATLAB process.
Disconnecting ends pending report delivery; reconnect and request a saved report
again when needed.

Save paths and run IDs as strings; MATLAB handles containing Python objects
are not portable model artifacts. A timeout or Ctrl+C during `wait` leaves the
job running. Cancellation remains `cancel_requested` until the worker stops.

`opendpd.closeProject(p, StopService=true)` stops an SDK-started service and all
jobs in that workspace, including after reconnecting. Stop externally launched
Studio through its original launcher. Uninstalling the toolbox leaves Python
environments and experiment workspaces intact.
