# MATLINK walkthrough

The toolbox opens **OpenDPD Studio itself** as its GUI. The **MATLINK** tab
contains MATLAB signal exchange and report delivery. Use Studio's existing
Datasets, Experiments, Results and plots throughout the rest of the workflow.

## 1. Connect from MATLAB

```matlab
link = opendpd.studio("/path/to/my-pa-workspace", ...
    PythonExecutable="/path/to/python");
```

`opendpd.studio()` also works after Python has been configured. You can pass an
existing Project:

```matlab
p = opendpd.openProject("my-pa-workspace");
link = opendpd.studio(p);
```

Studio opens at **MATLINK** in the system browser. Selecting the same workspace
as an existing Studio session attaches to the same service. Otherwise the SDK
starts a local service. The connection remains active without keeping `link`
in the workspace. Supplying `p` leaves that Project available to your scripts.

MATLINK displays session presence and the variables MATLAB has made available.
If multiple MATLAB sessions are connected, choose the session that holds your
signals before requesting a transfer. Its connection updates depend on MATLAB processing timer callbacks. A long
blocking MATLAB operation can temporarily make the session appear offline;
the heartbeat resumes when MATLAB can process callbacks again.

## 2. Choose a signal source

MATLINK offers two source cards. Offline connection history is hidden. With one
online MATLAB session, that session is selected automatically. With several,
choose the intended session. A selected session becoming unavailable never
silently redirects a transfer to another MATLAB process.

### OpenDPD Signal Generator

1. Select **OpenDPD Signal Generator**, then **Open Signal Generator**.
2. Use Studio's existing waveform presets, bandwidth, sample rate, modulation,
   sample count and advanced settings. Select one or several captures.
3. Choose the **Virtual PA model** and its parameters. The generator produces
   PA input `x`; the virtual PA creates output `y`, which PA training requires.
   The default is the library's solid-state AM/AM + AM/PM model (`rapp-am-pm`).
   Both signals are explicitly synthetic.
4. Click **Generate & prepare experiment**. Studio generates the inputs,
   creates the paired dataset, and asks the connected MATLAB session to save
   every capture. It waits for MATLAB's acknowledgment before navigating.
5. The experiment page opens with the generated dataset and `raw-v1` selected.
   Sample rate, bandwidth and spectral segment length are inherited from the
   dataset. Training defaults come from the existing recipe and detected
   device. Review the normal experiment steps and start training.

Generated data is saved as an ordinary base-workspace struct named
`opendpdSignals`, or a name with a suffix if that variable already exists:

```matlab
x = opendpdSignals.captures(1).x;       % complex single column vector
y = opendpdSignals.captures(1).y;      % simulated PA output
fs = opendpdSignals.captures(1).signal.sample_rate_hz;
plot(abs(x), abs(y), '.');
```

Each capture keeps its own sampling metadata, split boundaries and simulation
provenance. Full continuous arrays are preserved, including split guards.
MATLINK verifies the stored I/Q hashes before loading. For a batch, every capture
is saved; the experiment starts with the first capture. Choose a different
dataset in experiment setup to train another capture. Nothing starts training
automatically. The normal Signal Generator outside MATLINK retains its existing
preview and Virtual PA Library workflow.

If MATLAB is busy, the page waits. If delivery fails, **Retry transfer** retries
saving the existing paired dataset. It does not regenerate its signals. The
paired dataset remains registered in Studio even if MATLAB disconnects.

### MATLAB workspace

Select dense, nonempty `single` or `double` PA input and output vectors from the
base workspace. They must be different variables with equal sample counts.
Complex samples represent `I + 1j*Q`; real vectors are I-only. Enter sample rate
and occupied bandwidth in **MHz**, plus measured/synthetic/unknown provenance.
The script API uses **Hz**. An optional unique dataset name can be supplied.

Click **Use these variables in an experiment**. After MATLAB validates and
imports the pair, Studio opens experiment setup with the imported data selected.
Import preserves sample order and amplitude, stores float32 I/Q, and performs
no normalization or alignment. This path uses 256-sample evaluation segments,
256-sample split guards and one subchannel; use `opendpd.importIQ` for other
settings. Invalid values, unequal lengths and duplicate IDs are reported.

## 3. Train in Studio

Use Studio's normal experiment flow. PA training creates a surrogate; DPD
training references a successful PA experiment. Model choices, configuration,
progress and comparisons remain in the existing Studio pages. MATLAB and the
browser share one queue. Defaults are starting points, not tuned hyperparameters
or a guarantee of model quality.

## 4. Review and save results

Return to **MATLINK** and choose an experiment from the searchable list. The
preview shows the stored metrics, their evidence type, metric profile and
available spectrum. Unavailable metrics retain their status instead of showing
zero. Use **Open experiment** for the full Studio analysis.

**Save to MATLAB workspace** provides a suggested variable name, which you can
edit. Click **Save to MATLAB** to deliver a struct containing:

- The saved primary evaluation report, including metrics and evidence.
- `configuration`: the resolved experiment settings.
- `plots`: available spectrum, time-window and AM–AM/AM–PM plot arrays.
- Dataset provenance and software metadata from the report.

Plot arrays retain the stored preview resolution. They are not full capture
exports or native MATLAB model weights. For full predistorted waveforms use
`opendpd.apply` or `opendpd.runDPD` in the [script workflow](workflow.html#apply-the-dpd).

```matlab
report = opendpd_train_pa_gru;  % use the variable name shown by MATLINK
struct2table(report.metrics)
report.configuration.training
report.plots.spectrum
```

The actual saved variable name is shown after MATLAB confirms delivery.
**Open in MATLAB** opens it in the variable editor. Existing variables are
preserved by adding a suffix. Repeating the same completed transfer reuses its
variable when it still holds the same report; if it has been cleared or replaced,
**Send again** creates a safe new variable.

For a running experiment, **Send when ready** queues delivery after successful
completion. MATLAB must stay connected and process callbacks. Disconnecting
ends pending delivery; the experiment can still finish and be saved after
reconnecting. Failed and cancelled experiments do not produce a successful
result delivery. Run IDs never need to be copied or typed in this workflow.

## Documentation and lifecycle

```matlab
opendpd.help();         % bundled guide, available offline
opendpd.help("gui");    % this walkthrough
opendpd.disconnect();   % detach MATLAB; training continues
```

Closing the browser does not disconnect MATLAB. Reopen Studio to return to the
same workspace and MATLINK session. Closing MATLAB or calling `disconnect`
ends the session and its pending exchanges. Neither action cancels training.
If you passed an existing Project to `studio`, disconnecting MATLINK leaves
that Project usable.

To stop a service started by the SDK explicitly:

```matlab
p = opendpd.openProject("my-pa-workspace");
opendpd.closeProject(p, StopService=true);
```

This stops the workspace's service and jobs. Stop a service launched through
Studio's own launcher using that launcher.
