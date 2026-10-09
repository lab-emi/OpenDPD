# Troubleshooting

## MATLAB cannot find opendpd.studio

Install the toolbox (`OpenDPD-2.4.0.mltbx`), or add this checkout's `Matlab/toolbox` folder to the
MATLAB path. Use `which opendpd.studio -all` and `which opendpd.setup -all` to
check for older installations. Only the toolbox root needs to be on the path.

## Python or OpenDPD cannot load

Run `pyenv` and `opendpd.doctor()`. Select a Python executable supported by your
MATLAB release and install dependencies into that exact environment. A regular
OpenDPD 2.3 release does not contain the preview SDK; select the 2.4 checkout via
`SourceDirectory` or install it into the environment:

```matlab
opendpd.studio("/path/to/workspace", ...
    PythonExecutable="/path/to/python", SourceDirectory="/path/to/OpenDPD");
```

If MATLAB has already loaded a different interpreter or checkout, finish work
using it before restarting Python or MATLAB. The toolbox does not terminate that
session automatically. Consult [pyenv](https://www.mathworks.com/help/matlab/ref/pyenv.html)
for execution-mode and release-specific behavior.

## Studio does not open or MATLINK is missing

Build the frontend from the selected development checkout:

```bash
npm --prefix frontend ci
npm --prefix frontend run build
```

Then run `opendpd.studio(workspace)` again. If a workspace service was started
before the update, finish its jobs and restart it with the current checkout.
The toolbox does not silently restart an existing service. Inspect the service
log and `opendpd.doctor()` if readiness fails. Keep `.sdk-service-*.log` files
local because they may contain a private bootstrap URL.

A Studio tab alone does not attach MATLAB. Start the connection from the MATLAB
session you want to use and select the same workspace folder.

## MATLINK shows no MATLAB session or says offline

Run `opendpd.studio(workspace)` in MATLAB. Check that the browser and MATLAB use
the same workspace. MATLAB must remain running and able to process callbacks.
A long blocking computation, modal interaction or breakpoint can delay the
heartbeat and pending actions. Once MATLAB resumes callback processing, its
presence updates again.

Closing the browser does not disconnect the bridge. Calling
`opendpd.disconnect()` or closing MATLAB does. After an intentional disconnect,
connect again before requesting imports or report delivery.

## No variables appear, or import fails

- Load signals into MATLAB's **base workspace** and wait for MATLINK to refresh.
  Function-local variables can be imported directly through the script API.
- Use dense nonempty single/double vectors of equal length. Correct NaN and Inf
  in the capture; float32 overflow is refused.
- Confirm units: **MHz in MATLINK, Hz in scripts**.
- Use enough samples for train/validation/test splits and their guards.
- Choose a unique dataset ID. Imports never replace an existing dataset.
- `importMAT` reads any MAT-file version, v7.3 included. It needs full single/double
  variables; cells, structs, sparse arrays and integer classes are refused.

If an import loses its connection, inspect Studio's dataset list before retrying.
The server may still finish. The SDK retains staged input under `imports/` and
reports its location when the outcome is uncertain.

## A report has not arrived

Check MATLINK's session and delivery status. **Send to MATLAB** needs a completed
experiment with a saved result. **Send when ready** waits for successful
completion and for MATLAB to process callbacks. Failed and cancelled runs
cannot complete that delivery.

Disconnecting or closing MATLAB ends pending delivery. Reconnect, select the
completed experiment, and send its report again. There is no need to copy a
run ID or type a variable name. MATLINK shows the generated variable name after
success; **Open in MATLAB** opens it in MATLAB. Existing variables are preserved.

A delivered report describes the stored evaluation. It does not contain a newly
predistorted waveform or recalculate metrics for a waveform passed to `apply`.
Use the [script workflow](workflow.html#apply-the-dpd) for inference and export.

## Training keeps running after closing MATLAB

This is the shared-service lifecycle. Reconnect to the workspace and use Studio
or `opendpd.cancel(job)` to cancel a particular job. To stop an SDK-started service
and its jobs, use `opendpd.closeProject(p, StopService=true)`. Stop a service
launched by Studio through its original launcher.

## apply rejects a model or differs from runDPD

`apply` supports unquantized `gru`, `tres_gru`, `gmp`, `mp_ls` and `gmp_ls` on CPU. Any
other model, a quantization-aware run, or `Execution="streaming"` for a model without a
registered streaming variant (only `gru` and `gmp` have one) is refused with the reason;
the output is never approximated. Offline segments use frozen run settings. The baseline
test-split exporter carries recurrent state across the waveform, so it can differ from
segmented `apply`; ask `apply` for `Execution="streaming"` to compare like with like. Check
`info.execution`, `info.limitations` and `info.streaming`. A changed checkpoint hash is
rejected; restore the original artifact.

A missing `SegmentSamples` error from `importIQ` or `importMAT` is deliberate: see the
import options in the reference.

## Documentation is blank

`opendpd.help()` opens the bundled guide without Python or an internet connection.
Check that `resources/docs` was included in the installation, and reinstall the
package if resources are missing. External links need internet access. This
preview requires desktop MATLAB.

## Package installation path is not accessible

Choose a writable Add-Ons installation folder in MATLAB Settings. A fresh batch
profile may have no installation folder set. The installation smoke test uses
a temporary folder for its own session.
