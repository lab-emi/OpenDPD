# MATLINK toolbox preview 0.3.0

Validated on 2026-09-29 in `codex/studio-2.4`. The toolbox now uses OpenDPD
Studio as its primary GUI. `opendpd.studio()` connects MATLAB and opens Studio's
**MATLINK** page; model training, plots and experiment management keep their
existing Studio workflows.

## User workflow

- Select MATLAB PA input/output variables in MATLINK and import the pair into
  Studio, or create demo variables with one click.
- Find an experiment by its name, model or dataset and select **Send to MATLAB**.
  The bridge chooses a unique variable name and keeps existing variables.
- Select **Send when ready** during training for automatic report delivery.
- Select **Open in MATLAB** to inspect the report. **Send again** also handles
  a report variable that has been cleared or reassigned.
- Use `opendpd.disconnect()` to detach MATLAB. The service and training jobs
  continue; requests addressed to that explicitly disconnected bridge end.

The browser submits validated named requests to the local Studio service.
MATLAB executes those requests through a timer and returns the outcome.
Heartbeat metadata contains names, types and dimensions, without signal values.
Signal import uses the SDK's existing I/Q path. Report delivery uses the saved
evaluation result.

## Validation

Environment: Linux, MATLAB R2026a, Python 3.13.14, CPU training/inference.

| Check | Result |
| --- | --- |
| MATLAB package build | **15 tests passed**, including four real MATLINK integration tests |
| Native MATLINK round trip | Demo creation, variable metadata, I/Q import, queued training/report delivery, unique names, duplicate delivery and reassigned-variable preservation passed |
| Lost acknowledgment | A real MATLAB operation was completed while its response was deliberately lost; retry did not create duplicate variables |
| Broker/API/navigation/SDK | **63 tests passed**; three additional stale-session reclamation tests added afterward, with **31 broker/API tests** passing in the focused rerun |
| Existing SDK regression tests | **44 passed** |
| Frontend | Related initial suite **63 passed**; latest MATLINK-specific suite **13 passed**, including resend recovery and immediate run-status refresh |
| Frontend static checks | Full lint, typecheck, generated API types and production build passed |
| Responsive UI | 1440, 768 and 390 px fixture checks had no page overflow or page errors; accessibility scan reported no WCAG violations in the main content |
| Live browser + real MATLAB | Clicked Create demo signals and Import to Studio, submitted a PA run through the shared service, then clicked Send when ready while running. Report arrived automatically; MATLAB verified a struct with the correct run identity. No browser page errors |
| Fresh `.mltbx` installation | Installed entry resolved under isolated Add-Ons; connected MATLINK, executed a demo request creating 4,096 samples, loaded the offline guide and uninstalled successfully |
| Interactive desktop session | Launched through the public MATLAB Engine API, observed 14 timer callbacks and continuing heartbeats while idle, then created demo variables from the actual MATLINK page |
| Documentation | Six offline HTML pages rebuilt; source consistency and all bundled local links/anchors passed |
| Python wheel and sdist | Built with MATLINK service, SDK, frontend assets and toolbox sources |

The local package is `Matlab/toolbox/dist/OpenDPD-0.3.0.mltbx`, **130,173 bytes**.
SHA-256: `56af58bfd4de1cdfac5655595c29ca3d08f7270758c661387fe3e17c6279996e`.

## Recovery and current limits

Commands and acknowledgments are idempotent. MATLAB caches completed operations
before acknowledging them, so a lost response does not rerun an operation.
Studio restarts invalidate the previous bridge identity. Repeating
`opendpd.studio()` recreates a stale connection. At connection capacity, the
broker can retire the oldest offline session without pending work, while keeping
live sessions and queued transfers.

MATLAB must remain open and process callbacks. A long MATLAB command can pause
transfers and make presence appear offline; requests resume when callbacks run.
On this host, invoking the bridge through an external `matlab -desktop -r`
startup command produced an initial heartbeat but no subsequent timer callbacks.
The interactive MATLAB Engine desktop session worked normally. For normal use,
run `opendpd.studio()` from the MATLAB Command Window or Apps gallery; the
external `-r` startup path needs separate investigation before being supported.
Transfer history is in memory and resets when Studio restarts. Imported datasets,
training jobs and reports remain in the workspace. Explicit disconnect does not
delete variables already imported into MATLAB.

MATLINK currently transfers paired dense single/double vectors and saved report
structs. Waveform generation remains available through `apply` and `runDPD`.
MATLAB Online, remote bridges, Simulink and streaming execution remain outside
this preview. Other MATLAB releases/platforms require their own execution checks.

The repository-wide strict MkDocs build still has the same **11 pre-existing
Arena/navigation warnings**. No new MATLINK guide warnings were introduced.
No release, File Exchange submission or CI workflow was published/dispatched.
