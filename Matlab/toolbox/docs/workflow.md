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
rate. Provide occupied signal bandwidth explicitly, in Hz, and the segment length.

```matlab
p = opendpd.openProject("my-pa-workspace");
ds = opendpd.importIQ(p, x, y, Name="capture-001", ...
    SampleRate=491.52e6, Bandwidth=200e6, Origin="measured", ...
    SegmentSamples=2048, GuardSamples=256);
```

`SegmentSamples` has no default. It is the PSD (Welch) segment length of every spectral metric and the
interval at which evaluation restarts a model's state, so choose it for your signal; Studio's generated
signals use 512-4096.

Rows and columns are accepted. Backend storage uses float32; source precision
and shape are recorded. No delay correction, normalization or gain fitting is
performed at the bridge boundary. Nonfinite values and float32 overflow are
rejected. Import uses the existing contiguous train/validation/test split; set
the guard to cover your training frame context.

For a MAT file with numeric variables:

```matlab
ds = opendpd.importMAT(p, "capture.mat", ...
    InputVariable="tx", OutputVariable="rx", Name="capture-002", ...
    SampleRate=491.52e6, Bandwidth=200e6, SegmentSamples=2048, Origin="measured");
```

MATLAB reads the file, so every MAT-file version works, v7.3 included. The MAT adapter
accepts dense single/double complex vectors and real N×2 I/Q matrices. Real vectors are
I-only. The source file name, SHA-256 and variable names are recorded; keep the original
MAT file if its other variables matter to your experiment.

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
`struct('hidden_size', 8)` and `struct('epochs', 5)`. `Device="auto"` is the default: the
first detected of `cuda`, `mps`, `cpu` (least-squares models stay on `cpu`), which is also
Studio's default. Pass `Device="cpu"` for a small reproducible run. A complete experiment
configuration can be sent with `opendpd.submit(p, config)`.
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

`apply` supports `gru`, `tres_gru`, `gmp`, `mp_ls` and `gmp_ls` on CPU, unquantized. Each
has a test that compares its output with the Python evaluator. By default
(`Execution="offline_segmented"`) it is how the run was scored: state resets at the trained
run's frozen segment boundaries, the final segment is zero padded and the padding is trimmed
from the returned vector. `tres_gru` reads 16 future samples, so within 16 samples of a
segment end it sees zero padding rather than the waveform; `info.limitations` says so.
Calls do not share hidden state. Each call loads the recorded model in an isolated process,
so pass a whole waveform rather than repeatedly calling it for individual samples.

For a waveform that will run continuously, `Execution="streaming"` carries one state across
chunks (`ChunkSamples`) for `gru` and `gmp`, whose registered streaming variants are
`gru_stream` and `gmp_stream`. Other models are refused rather than approximated. Streaming
output is a different signal from the one the stored report scored; `info.streaming` records
the measured warm-up, look-ahead and chunk consistency of that execution.

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
`apply` resets at segment boundaries by default. These outputs can differ. Read the export
sidecar for its execution semantics. The quickstart saves the segmented
training-run report alongside `apply`'s waveform and separately records the
export run ID. Stored DPD reports evaluate a PA surrogate. A new waveform passed
to `apply` does not update that stored report or establish hardware performance.

## One call from a capture to models: `fit`

When you have a paired capture and want models that run in MATLAB, `fit` does the whole sequence and starts Python as a
separate process (no `pyenv`):

```matlab
[dpd, pa, report] = opendpd.fit(x, y, Workspace="my-pa-workspace", SampleRate=fs, Bandwidth=bw, ...
    SegmentSamples=2048, PAModel="gru", DPDModel="gru");
u = opendpd.apply(dpd, xNew);        % the predistorted PA input, plain MATLAB
yHat = opendpd.apply(pa, u);         % the PA model's output for it
report.DPD.Result                    % the stored evaluation result of the DPD run
```

The runs are ordinary runs in `Workspace`: open it in Studio (`opendpd.studio("my-pa-workspace")`) to see the curves
and spectra. Use the project API (`openProject`, `importIQ`, `trainPA`, `trainDPD`) when you need finer control. See the
[function reference](reference.html#one-call-fit) for the options, how Python is found and what is checked first.

## Score a capture without training

Metrics need no project or server. Play `opendpd.waveform` through your amplifier, capture its output and score it
with the code Studio uses:

```matlab
w = opendpd.waveform(Seed=1, Subframes=10);       % the waveform to play (loop it); w.x is 30.72 MS/s
% ... play w (resampled to your generator's rate), capture the PA output as y at SampleRate fs ...
evm  = opendpd.metrics.evm(y, w, SampleRate=fs);                                  % evm.EVM_RMS in percent
aclr = opendpd.metrics.aclr(y, SampleRate=fs, SegmentSamples=2048, Waveform=w);   % aclr.ACLR_L, aclr.ACLR_R in dBc
```

`SegmentSamples` has no default because the ratio depends on it: it is the Welch segment length, and a longer
segment resolves the band edges better. Report it with the number. If `Status` is not `"ok"`, read `Reason`; for
example `"missing_reference"` means the capture does not correlate with `w`. See the
[function reference](reference.html#metrics-without-a-run) for all options.

## Take a trained model out of OpenDPD

A trained PA or DPD can leave the Python environment as a package of data and run in plain MATLAB, on a machine
that has no Python:

```matlab
opendpd.export(dpd, "apa-dpd.opendpd.zip");       % needs the Python SDK; the same run always gives the same bytes
model = opendpd.load("apa-dpd.opendpd.zip");      % plain MATLAB from here on
report = opendpd.verify(model)                    % golden test: OpenDPD's outputs vs this MATLAB release, within 1e-5
u = opendpd.apply(model, xTest);                  % offline_segmented, like opendpd.apply(dpd, xTest)
y = model(chunk); reset(model);                   % streaming, for gru and gmp
```

Use the training dataset's sample rate and amplitude units: nothing is normalised or aligned
(`model.Manifest.signal`, `model.Manifest.scaling`). For an `mp_ls` model, `model.commCoefficients()` returns the matrix
for `comm.DPD` and `rf.PAmemory`, so a memory polynomial fitted by OpenDPD can be run by MathWorks code; one `nperseg` segment
of OpenDPD and one `comm.DPD` stream from a zero state agree (`rf.PAmemory` starts its delay line with the first sample: see the
reference for the zero pad). See the [function
reference](reference.html#model-packages-run-a-trained-model-without-python) for what a package contains, what `verify`
shows and what it does not, and how a package from elsewhere is read.

To use the model in Simulink or with MATLAB Coder, write it as a standalone class:

```matlab
r = opendpd.generateCode(model, "apa-dpd-class", Name="ApaDpd", Execution="streaming");
addpath(r.Folder); ApaDpdCheck()                  % the golden test again, on the generated class
```

Add a *MATLAB System* block with the System object name `ApaDpd`, or run `codegen ApaDpdStep ...`. The example
`opendpdSimulink` builds a DPD-then-PA transmit chain. Choose `streaming` when samples arrive one at a time or in
frames of any size; the default restarts the state every `nperseg` samples, as in training. See the [function
reference](reference.html#standalone-classes-for-matlab-coder-and-simulink).

A fixed-point deployment package (`opendpd deploy`, or the Studio's Deployment panel) is loaded the same way and checked bit
for bit against its golden vectors; the same model gives you the expected integers for your own implementation:

```matlab
fixed = opendpd.load("deploy.zip");               % an opendpd.FixedModel: integers held exactly in double precision
opendpd.verify(fixed).status                      % "bit_exact", or the case, sample and signal where it first differs
[yq, state, trace] = fixed.runInteger(xq);        % integers in, integers out, the state after every sample
```

See the [function reference](reference.html#fixed-point-deployment-packages-check-an-implementation-bit-for-bit).

## Measure on your bench

`opendpd.lab.Session` wraps the instrument code you already have so that a measurement is supervised and recorded
(see the [function reference](reference.html#measured-captures-opendpdlab) for the rules). Learn the procedure on the
dry-run mock first; it emits nothing and needs no Python:

```matlab
rng(1); u = 0.1 * complex(randn(8192, 1), randn(8192, 1));       % any complex baseband signal below MaxPeak
lab = opendpd.lab.Session(Instrument=opendpd.lab.MockInstrument(), Operator="Your Name");
arm(lab);
y = measure(lab, u, SampleRate=30.72e6);                         % the mock's synthetic PA, not an amplifier
disarm(lab);
record(lab)                                                       % operator, limits, hashes of played and captured, log
```

For your own chain, replace `Instrument=` by `MeasureFcn=` and `RFOffFcn=` (and, if you have them, `HeartbeatFcn=` and
`PowerCalibration=`). Arming such a session needs the person's name and the environment variable
`OPENDPD_ALLOW_RF_OUTPUT=1`; nothing in this toolbox sets that variable.

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
