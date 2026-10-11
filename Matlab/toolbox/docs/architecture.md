# MATLAB and the Studio web application

## Chosen design: Studio with MATLINK

**OpenDPD Studio is the toolbox GUI.** `opendpd.studio()` connects the active
MATLAB session and opens Studio's **MATLINK** tab in the system browser.
MATLINK provides signal import and report delivery. Existing Studio pages
provide datasets, model configuration, training, plots and comparisons.

```text
MATLAB desktop                         OpenDPD Studio in browser
  base workspace                         MATLINK + existing workflow
       |                                             |
  MATLABBridge timer                                 |
       |                                             |
  Python SDK -------- authenticated local Studio service
                              |
                workspace / datasets / queue / results
                              |
                       Python training workers
```

The browser sends a small set of named requests to the workspace service.
A MATLAB timer publishes session presence and numeric variable metadata,
collects requests, executes permitted operations, and reports their outcomes.
The browser cannot evaluate arbitrary MATLAB code. Variable access accepts
validated identifiers; report delivery generates unused names before writing
to the base workspace.

This design lets users select visible variables and named experiments. It
removes copying run IDs between applications and keeps the existing Studio
training flow. Script users can still use Job handles and saved IDs for
reproducible automation and reconnection.

## Can a web app be a MATLAB GUI?

MATLAB supports local HTML/CSS/JavaScript controls through `uihtml`, including
MATLAB-to-JavaScript data and events. MathWorks documents that `uihtml` cannot
embed third-party web applications using external URLs. See the
[uihtml reference](https://www.mathworks.com/help/matlab/ref/uihtml.html).

MATLAB's public `web(url, '-browser')` API opens Studio in the system browser.
Using this supported launch path lets the toolbox reuse the complete existing
web application. See the [web reference](https://www.mathworks.com/help/matlab/ref/web.html).
The preview does not try to place a remote URL inside a MATLAB iframe.

| Option | Fit for OpenDPD |
| --- | --- |
| Studio in the browser + MATLINK | Current default. Reuses Studio's screens and flow; a MATLAB session performs signal exchange. |
| Local `uifigure` + `uihtml` window | Not shipped. It would duplicate Studio's screens, and a second GUI would have to be kept in step with the first. |
| Entire Studio in `uihtml` via external URL/iframe | Outside the documented uihtml support contract and incompatible with Studio's anti-frame protection. |
| Rewrite Studio in native App Designer controls | Would duplicate model configuration, plots and experiment workflows. |
| MATLAB Web App Server | Deploys compiled App Designer applications; it is not a host for the existing React Studio application. |

MathWorks describes its Compiler/App Designer deployment workflow in
[Create and Deploy a Web App](https://www.mathworks.com/help/webappserver/ug/create-and-deploy-a-web-app.html)
and [Web App Server requirements](https://www.mathworks.com/support/requirements/matlab-web-app-server.html).
The MATLINK design requires neither MATLAB Compiler nor Web App Server.

## Connection and request lifetime

`opendpd.studio()` retains a timer-backed bridge in the MATLAB session, including
when called without an output. The bridge owns a Project connection when it
creates one. A Project passed by the caller remains caller-owned and is usable
after MATLINK disconnects.

- Closing a browser tab leaves the MATLAB bridge active.
- MATLAB executes requests when it can process callbacks. Long blocking MATLAB
  work can delay requests and make presence appear offline.
- Returning to callback processing renews the heartbeat.
- Disconnecting or closing MATLAB ends pending exchanges and automatic delivery.
- The workspace service and training continue independently. Reconnect and send
  a saved report again when needed.

A completed experiment's primary report, resolved configuration and registered
plot arrays are transferred as a MATLAB struct. Generated paired datasets are
loaded as full continuous I/Q arrays after verifying their recorded hashes.
All batch captures are included, with per-capture metadata.
**Send when ready** waits for successful completion while the session is
connected. This preview keeps waveform inference and export in the script API;
it does not transfer model weights into native MATLAB execution.

## Service and data boundaries

The SDK reuses Studio's launcher, workspace lock, sessions, CSRF protection,
queue and workers. Both clients attach to one local workspace service.
Opening Studio exchanges a private bootstrap token for a session cookie and
redirects to a supported local route. Same-origin, CSRF and anti-frame
protections remain in place. This preview serves loopback only.

The allowed browser requests cover refreshing variable metadata, generating
paired-dataset delivery, importing a selected pair, receiving a saved result bundle, and opening
a delivered variable. MATLAB executes these fixed operations through toolbox
code. It never accepts a general expression or script from MATLINK. Dataset
storage and signal execution rules match the [script workflow](workflow.html).

## Packaging and future scope

The `.mltbx` contains MATLAB source, examples,
tests and offline help. Python dependencies and built Studio assets belong to
the selected OpenDPD environment/checkout; experiment artifacts belong to the
workspace. The Apps gallery entry launches `opendpd.studio()` through the public
[ToolboxOptions API](https://www.mathworks.com/help/matlab/ref/matlab.addons.toolbox.toolboxoptions.html).

Further work includes waveform selection and transfer in MATLINK, more validated
inference adapters, MATLAB release/platform checks and measured hardware
workflows. Simulink, stateful streaming and remote deployment require their own
execution and lifecycle designs.

## Toolbox capabilities

The MATLAB class advertises `dataset_export` and `result_bundle` capabilities.
Older toolbox connections remain usable for their original operations; the
server rejects new operations before an older class could ignore their options.
The generator uses the same batch and Virtual PA dataset APIs as Studio.
Transfer acknowledgments gate navigation to the experiment page. A retry reuses
the generated dataset, and queued requests use stable idempotency keys.
