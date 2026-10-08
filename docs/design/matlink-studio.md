# MATLINK: use Studio as the MATLAB toolbox GUI

## User flow

OpenDPD Studio becomes the toolbox's primary GUI. `opendpd.studio()` connects
the current MATLAB session and opens Studio at `/matlink`. Users import MATLAB
signals, train through Studio's existing experiment flow, and return saved
reports to MATLAB by selecting named experiments.

MATLINK contains three parts:

1. **Connection** — online sessions only; a single connection is automatic.
   A disconnected selected session stays pinned until explicitly changed.
2. **Signal source** — Studio Signal Generator or local MATLAB workspace vectors.
   The generator path includes an explicit virtual PA, creates paired data,
   saves all captures to MATLAB and opens the selected experiment after acknowledgment.
3. **Review and save** — searchable experiments, metric/evidence cards, spectrum,
   saved-content preview and an editable suggested MATLAB destination.

The GUI requires no run-ID copying. Experiment setup inherits dataset signal
metadata and normal recipe/device defaults. It never starts training itself.
MATLAB naming preserves existing variables; delivered reports include resolved
settings and registered plot arrays. All batch captures are saved, while an
individual experiment starts with the first selected capture.

## Integration

Studio keeps its existing routes, React components, model configuration, plots,
queue and workspace service. A new MATLINK route presents MATLAB exchange
actions. The toolbox launches it with MATLAB's public
[`web`](https://www.mathworks.com/help/matlab/ref/web.html) API.

A persistent MATLAB bridge publishes a heartbeat and numeric variable metadata,
then receives a bounded set of named requests through the authenticated local
service. Its timer performs the MATLAB operations and reports results. The
browser does not send expressions for general evaluation. Import accepts
validated variable identifiers; report delivery uses collision-safe names.
Studio's same-origin, CSRF and anti-frame protections remain enabled.

MathWorks documents external application embedding as unsupported in
[`uihtml`](https://www.mathworks.com/help/matlab/ref/uihtml.html). Using the system
browser reuses the full Studio without relying on MATLAB browser internals.
MATLAB Compiler and Web App Server are unnecessary for this local design.

## Lifetime and feedback

| Event | Outcome |
| --- | --- |
| Open Studio from MATLAB | Register/reuse this process's bridge for the workspace and open MATLINK. |
| Close browser tab | Bridge, service and jobs continue. |
| MATLAB executes blocking work | Heartbeat and requests can pause; the UI may show the session as offline. |
| MATLAB processes callbacks again | Heartbeat resumes and available requests can proceed. |
| Disconnect MATLINK or close MATLAB | Pending interaction and delivery end; service and training jobs continue. |
| Reconnect after training | Select the completed experiment and send its saved report. |

`opendpd.disconnect()` detaches all bridges in the current MATLAB process.
Passing a bridge or workspace detaches one. A Project passed into `studio`
remains caller-owned; disconnecting the bridge leaves that Project usable.

## Scope

MATLINK exchanges dense single/double base-workspace vectors, generated capture
collections and saved result bundles. Waveform inference/export stays in `opendpd.apply` and
`opendpd.runDPD`. Existing I/Q provenance, float32 conversion, split guards and
frozen inference semantics are unchanged. The integration is local desktop only.

The six bundled offline guides describe setup, MATLINK, scripts, APIs,
architecture and troubleshooting. `opendpd.help()` remains usable without a
Python service or internet connection. Guide generation checks local links and
anchors to keep the packaged documentation usable offline.
