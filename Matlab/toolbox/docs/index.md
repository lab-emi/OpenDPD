# OpenDPD for MATLAB

**Toolbox 2.4.0 (unreleased) · OpenDPD 2.4 development · SDK protocol 1**

Use OpenDPD Studio as the toolbox GUI. Prepare signals in MATLAB, import them
from Studio's **MATLINK** tab, train with the existing Studio workflow, and send
saved reports back to MATLAB by selecting an experiment. Both applications
share one local workspace, dataset collection and job queue.

## Start here

```matlab
opendpd.studio()
```

This connects MATLAB and opens Studio's **MATLINK** tab in your system browser.
The MATLAB session keeps the connection alive even when you do not save the
returned handle. After installing the package, **Apps → OpenDPDStudio** opens
the same workflow. `opendpd.help()` opens this bundled offline guide.

- [MATLINK walkthrough](gui.html): import signals, train, and send reports back.
- [Script workflow](workflow.html): reproducible PA and DPD experiments.
- [Function reference](reference.html): arguments, outputs and defaults.
- [Integration architecture](architecture.html): how Studio and MATLAB connect.
- [Troubleshooting](troubleshooting.html): environment, connection and data issues.

## Install and select Python

Use desktop MATLAB and a compatible CPython environment containing OpenDPD and
PyTorch. Base MATLAB is sufficient for the synthetic example. The earlier 0.2
integration was exercised on Linux with MATLAB R2026a and Python 3.13.14 using
CPU. MATLINK adds a new browser-to-MATLAB path; see the development validation
record for its current checks. R2024b is the candidate minimum. Other releases,
operating systems and CUDA bridge workflows need their own execution checks.
Consult the [MathWorks Python compatibility table](https://www.mathworks.com/support/requirements/python-compatibility.html)
for your MATLAB release.

Install the `.mltbx` by opening it in MATLAB, or:

```matlab
matlab.addons.toolbox.installToolbox("/path/to/OpenDPD-2.4.0.mltbx");
opendpd.studio("/path/to/experiment-workspace", ...
    PythonExecutable="/path/to/python");
```

The package includes MATLAB source, this offline
guide, tests, examples and the Apache-2.0 license. Install Python dependencies
separately. The released OpenDPD 2.3 package does not contain this SDK: install
Python code from the **2.4 development checkout**. Its package version still
reflects the 2.3 baseline until the coordinated prerelease.

From the checkout root, create an environment, install dependencies and build
Studio. These are terminal commands; choose a Python version supported by your
MATLAB release (3.11 is a candidate for the planned release matrix).

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
npm --prefix frontend ci
npm --prefix frontend run build
```

On Windows, use `py -3.11 -m venv .venv` and
`.venv\Scripts\python.exe -m pip install -e ".[dev]"`.

For development without installing the toolbox package:

```matlab
repo = "/absolute/path/to/OpenDPD";
addpath(fullfile(repo, "Matlab", "toolbox"));
opendpd.studio("/path/to/experiment-workspace", ...
    PythonExecutable="/path/to/python", SourceDirectory=repo);
```

`SourceDirectory` selects development source in this MATLAB Python session.
The workspace is where experiment data is stored; it is a separate choice from
the source checkout. Setup uses an existing Python environment and never
installs packages. Finish work in an already loaded incompatible Python session
before restarting or switching it.

## Try MATLINK

1. Confirm MATLINK shows your connected MATLAB session.
2. Choose **OpenDPD Signal Generator** or **MATLAB workspace**.
3. For generated signals, configure the waveform and virtual PA, then click
   **Generate & prepare experiment**. The complete paired signals are saved to
   MATLAB before Studio opens experiment setup with the dataset selected.
4. Review the model and training settings. Start a small PA experiment; two
   epochs suffice for a workflow check, without establishing model performance.
5. Return to MATLINK, select the experiment, inspect its metrics and spectrum,
   then **Save to MATLAB**. The suggested destination is editable and existing
   variables are preserved. **Send when ready** also works during training.

See the [MATLINK walkthrough](gui.html) for both signal sources, automatic
parameter selection, batch captures and the saved struct fields. For a complete
PA → DPD → inference script example, follow the [script workflow](workflow.html).

## Closing and supported scope

Closing the browser leaves MATLAB connected. To detach MATLAB:

```matlab
opendpd.disconnect()
```

Disconnecting or closing MATLAB ends MATLINK interaction and pending report
delivery. The workspace service and training jobs continue. Reconnect and send
a completed report again when ready.

Training uses the existing OpenDPD model registry. MATLAB `apply` supports
unquantized **`gru`, `tres_gru`, `gmp`, `mp_ls` and `gmp_ls` on CPU**, using offline
segments by default or one continuous state for `gru` and `gmp`. MATLINK
returns saved report structs; use the script API for waveform inference and
export. DPD reports evaluate a learned PA surrogate. Hardware performance
requires a separate measured capture and evaluation. This preview runs on the
local desktop; MATLAB Online and remote services are outside its current scope. Simulink
is supported only through the standalone classes that `opendpd.generateCode` writes from a
model package.
