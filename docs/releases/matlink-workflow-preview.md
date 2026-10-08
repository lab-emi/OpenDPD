# MATLINK workflow preview 0.4.0

Validated on 2026-09-29 in the OpenDPD 2.4 worktree, on Linux with MATLAB
R2026a, Python 3.13.14 and the local Studio service. This is toolbox preview
0.4.0; the repository's coordinated core version remains the 2.3 baseline.

## Delivered workflow

- Only online MATLAB connections appear in the normal session control. One
  connection is automatic; an unavailable selected connection stays pinned
  until the user explicitly selects a replacement.
- MATLINK offers Studio Signal Generator and MATLAB workspace source cards.
- Generated inputs pass through an explicitly selected virtual PA. The existing
  batch generator and PA Library APIs create paired data. All captures, their
  full complex I/Q vectors and per-capture metadata are saved to MATLAB before
  experiment setup opens with the first capture and `raw-v1` selected.
- Dataset signal settings and normal recipe/device defaults populate experiment
  setup. The user reviews and starts training through the existing Studio flow.
- MATLINK presents searchable experiments, stored metric/evidence cards, spectrum
  previews, export-content details and a suggested editable MATLAB variable.
  Saved bundles contain the primary report, resolved configuration and available
  spectrum, time, AM–AM and AM–PM plot arrays. Plot arrays retain preview resolution.
- Existing variables are preserved. Retry-safe delivery and capability checks
  cover old toolbox sessions. Python member lookup supports upgrades in an
  already running MATLAB desktop, without clearing its signal variables.

## Validation

| Check | Result |
| --- | --- |
| Python broker, HTTP, SDK and navigation tests | 69 passed |
| MATLAB toolbox tests, including generated collections and result bundles | 16 passed |
| MATLINK, generator and experiment frontend tests | 38 passed; 17 affected tests rerun after layout refinement |
| Frontend typecheck, lint and production build | Passed; existing Plotly chunk-size advisory remains |
| Generated offline documentation and local links | Six pages checked |
| Package build and installation in the existing MATLAB desktop | Passed |
| Browser → generator → MATLAB → experiment → MATLINK → MATLAB | Passed |

The live browser check used a 30,720-sample NR-numerology stimulus and the
solid-state AM/AM + AM/PM virtual PA. MATLAB's stored input and output complex
vectors matched the registered continuous arrays exactly. A two-epoch CPU GRU
workflow check completed. Saving its results from MATLINK preserved all metric
names, values, units and statuses, its dataset selection and three plot groups.
This short run checks the workflow; it does not establish model performance.

The trial leaves `opendpdSignals` and `opendpd_MATLINK_workflow_check` in the
connected MATLAB workspace. MATLAB's JSON conversion represents JSON nulls as
empty arrays; numeric scores and waveform values were checked separately.

## Package

- File: `Matlab/toolbox/dist/OpenDPD-0.4.0.mltbx`
- SHA-256: `c8f47fa3227db24aa8a8125b4f715dac472b8ae4c238619f6bb02315e71dde92`
- Installed toolbox source matches this package and the 2.4 development backend.
- Other MATLAB releases and operating systems still require their own execution
  checks; no release or external publication was performed.

See the [MATLINK walkthrough](../tutorials/matlab-toolbox.md) for the user flow.
