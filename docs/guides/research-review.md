# Reviewing RF results in Studio

Use Results to select runs, then Compare. Choose a reference and inspect the
condition differences before interpreting a gain. Same-condition comparison
marks best values only when the existing protocol checks pass. Cross-condition
mode shows the operating points without a best badge. Deltas are candidate
minus reference and are unavailable across incompatible protocols or units.

The RF facts panel distinguishes evaluated signal metadata from later physical
declarations. Missing power stays missing. Record RF conditions accepts a
source/change note, reference plane and optional power rails; every revision is
retained. It never rewrites the original result or recalculates its metrics.
Efficiency uses declared linear powers and an explicit DC boundary. These
values are declarations, not an independent calibration of normalized IQ.

Spectrum integration bands follow the selected result's metric profile.
Legacy adjacent bands cover a subchannel, and its denominator is the strongest
subchannel. General spectral ACPR uses the whole main-channel power. The
display spectrum remains a stored Welch density estimate; drawing legacy
boundaries does not turn it into the legacy metric estimator. Bands outside
the capture range are identified. The cursor reports each trace's nearest
saved frequency bin without interpolation.

Save view preserves the reference, selected profiles, visible traces, axis
ranges, bands and cursor. Reopen it from Saved views. Export figure + data
downloads the figure in PNG/SVG/PDF, chart-data CSV, formal metric CSV,
profiles, conditions, full provenance, a manifest and a standalone replay
script. Unzip and run `python replay.py .` with matplotlib installed. Replay
checks hashes and rebuilds the figure from the saved arrays. To re-evaluate
the raw IQ, use the result's full run package, or the combined private reproduction download in the Publication figure editor described below.
If a source result, declaration or plot changed, the old view refuses to
silently render the new evidence; save a new view.

## Measurement sessions

Measured-result pages have a Measurement sessions section. Create a session
from existing evaluated captures, enter the actual physical acquisition IDs,
and choose the profile and power tolerance declared by the experiment protocol.
Several files or evaluations of the same acquisition retain the same ID; an
extra evaluation must carry an exclusion reason. Reusing identical raw data
under different acquisition IDs is refused. Mock captures can only join a mock
session. The original raw files are checked against their recorded hashes.

The session holds optional structured instrument and calibration records:
reference plane, calibration version/time/method, fixture and de-embedding,
receiver bandwidth/noise/compression limits and an uncertainty-budget note.
Each capture retains its raw hash, playback hash, alignment and processing
record. Failed and explicitly excluded captures remain visible. Export session
record downloads the complete snapshot.

Repeat statistics require at least two included independent acquisitions,
matching methods/models/calibration and declared output powers within the
session tolerance. They report arithmetic statistics of the stored metric
values (including dB), sample standard deviation with `ddof=1`, and first-to-last
drift. An optional Student t 95% confidence interval assumes independent,
stationary captures. It is a repeatability interval, not total measurement
uncertainty. Training seeds are counted separately from acquisitions; missing
seed provenance remains unavailable. A single capture produces no repeatability
number. Changing the source result does not rewrite a saved session: a warning
identifies the changed source while retaining the recorded snapshot.

## Four five-minute demonstration scripts

Generate the local demonstration with:

```bash
python scripts/studio_research_demo.py --workspace /tmp/studio-review-demo
opendpd gui --workspace /tmp/studio-review-demo
```

The generator prints run IDs and the comparison URL and writes
`review-demo/index.json` and `review-demo/figure.zip` inside that workspace.
It uses registered measured no-DPD input/output data and actual CPU smoke
training. Its DPD outputs are **surrogate simulations**. It does not create
fake measured captures, power readings or target hardware evidence.

| Task | Five-minute sequence | Evidence to record |
| --- | --- | --- |
| System trade-offs (de Vreede) | 1 min identify surrogate evidence and bandwidth; 2 min fix MP as reference and inspect GRU absolute/delta values; 1 min inspect missing calibrated power/DC rails; 1 min save the candidate view. | Can the user explain why no system-efficiency improvement is established? |
| Architecture-related distortion (Alavi) | 1 min inspect x/u/y and no-DPD baselines; 2 min inspect AM/AM and AM/PM; 1 min compare MP and GRU with the same surrogate; 1 min inspect training paths and missing independent I/Q scan. | Does the user distinguish a simple baseline from a structural scan/LUT claim? |
| Timing and implementation (Babaie) | 1 min fix reference; 1 min inspect preprocessing and gain rules; 2 min inspect execution semantics/lookahead and available deployment evidence; 1 min identify missing calibration ablation and hardware timing. | Does the user distinguish algorithmic lookahead, host timing and hardware evidence? |
| Measurement credibility (Spirito) | 1 min identify the measured no-DPD source and simulated DPD output; 1 min inspect full data/model hashes; 1 min read a frequency cursor and profile bands; 2 min export and replay the saved view. | Can another person reproduce the figure without relabelling simulation as measurement? |

These are task hypotheses, not claims about the named researchers' personal
habits. Real session/repeat/condition validation requires independent capture
files. Do not substitute chunks of one capture. A usability study needs real
participants and a record of task success, active operation time and replay
success; the suggested 80% completion and 20% time reduction are unverified
targets until that study is run.

## Versioned measured timing diagnostics

The capture-import dialog defaults to `measurement-integer-v1`. Choose
`measurement-fractional-v2` for a new, separately recorded evaluation. Existing
results are never rewritten to use the new protocol. Re-evaluation reads the
stored protocol, raw capture and played-signal hashes.

The fractional protocol first performs the existing integer correlation search,
then fits a residual delay within ±0.5 sample against the waveform actually
played. For declared loop playback it reuses the preprocessing FFT phase ramp
on one full period. Single-shot playback uses a 129-tap Kaiser-windowed sinc
(beta 8.6) and excludes 64 samples on both edges; retained samples do not use
circular wrapping. The result records the effective-rate delay in samples and
ns, boundary method and half-open valid interval. This is offline processing;
it does not establish a causal implementation or target hardware latency.

The alignment table shows the original timing, integer timing and fractional
timing on the **same retained interval**. Amplitude-only and complex-gain
reference fits expose the residual phase contribution. These NMSE diagnostics
are distinct from the formal metric profile; captured amplitudes are not
normalised to match output power. Sparse periodic tones may have ambiguous
coarse delays, and the finite interpolation filter requires validation for
near-Nyquist signals. Analytic and MOCK tests are not real capture validation.

## Create and execute a matrix

Open **Experiments → Sweep Board**, or use the matrix entry on Robustness.
Select same-condition method comparison or cross-condition adaptation. In the
same-condition mode all DPD methods and seeds use one explicitly selected PA
checkpoint; its `fixed_surrogate` seed policy is recorded. Frame compatibility
is still checked. Ordinary runs keep the legacy seed rule by default.

For cross-condition work, declare the DUT, one varied dimension, each dataset's
acquisition batch and the source condition. Identical raw captures cannot become
different conditions by renaming datasets. VSWR requires reflection phase.
Choose PA/DPD methods, associate DPD methods with a PA method ID, and select
zero-update, few-shot sample budgets and/or full retraining. The existing
conditions service determines dependencies and which target PA models are needed.

Preview displays all cells, training versus evaluation counts, sample×epoch work
and a declared wall-clock limit. Sample×epochs does not estimate fitting FLOPs,
energy or duration. Validation requiring a future model checkpoint is identified
as deferred until that prerequisite succeeds. Registering a plan saves it;
**Start matrix** submits ordinary workers through the existing queue. The time
budget includes queue waiting and cancellation uses the worker's normal grace
period. Configuration/seed selection remains a pre-registration responsibility;
target test results must not guide hyperparameter selection.

Cancellation, interruption and failure preserve each attempt. **Resume unfinished
cells** retains successful cells and creates new runs for unfinished attempts.
Changes to registered source metadata or the fixed PA require a new preview.
The seed summary copies stored metrics and labels sample SD; it is not hardware
repeatability. Download the plan/attempts and seed summary alongside relevant
result packages for a review record.

## Synthetic data, uploads and public review

In **Datasets**, choose **Generate synthetic datasets** to create a suite locally.
The default is three normalized drive settings and two separate synthetic
realizations each. Every entry is marked SYNTHETIC and stores the generator
version, RNG seeds, polynomial and memory coefficients. Use the generated
conditions button to open a populated matrix draft. These fixtures do not
replace real independent captures, calibrated power, hardware cost reports or
external EVM validation.

**Create Your Own Dataset** accepts paired finite CSV data through the existing
whole-file validation and split-review flow. Uploads default to private. Choose
public contribution in the last step, or open **Dataset sharing** later. Enter a
public description, attribution and license, then inspect/download the exact
package. Only that package is submitted after the rights/disclosure confirmation.

Public submission creates a fresh branch from the latest upstream default
branch, pushes the data-only folder under `dataset/community/`, and opens a PR
for human review. Data are publicly accessible on GitHub while review is pending.
The current code checkout is not staged or pushed. Failed submission can resume
the same branch and recover an existing PR; no automatic merge is performed.
Questions about data or review: [emi.lab@outlook.com](mailto:emi.lab@outlook.com).

Local Studio uses Git and an authenticated GitHub CLI account on the host.
Administrators of hosted Studio can opt in with
`OPENDPD_WEB_DATASET_PUBLICATIONS=1` and a dedicated authenticated GitHub CLI
identity. The feature is otherwise unavailable for submission, while private
preview/download still work. Hosted defaults allow two new submissions per
network and twenty per day; retries keep their existing reservation. Credentials
are never sent to the browser. Configuring this feature does not change the
requirement for human review.


## Hardware costs and precision

Open **Hardware costs and RF trade-offs** from a result or comparison. Select up
 to eight runs under one released profile. Storage counts refer to serialized
 tensor payload, not effective sparse parameters. Only reviewed dense GRU
 affine operations are counted automatically; additional operators and unknown
 buffers remain explicit. Existing GRU fixed-point exports add separate resource
 and C99 reference timing records. CPU timing is not power measurement.

**Add cost report** uploads a PDF/JSON/TXT/RPT/CSV (up to 5 MiB) and records
 user-transcribed values, precision by module, process, clock, batch, activity
 and implementation boundary. Energy axes require a sourced energy value;
 operation counts and CPU timing alone cannot establish one. Synthetic and
 changed-source records are labelled; stale entries do not supply plot points.
 Linked RF metrics retain their measured/surrogate evidence and execution
 semantics. A hardware cost report does not independently verify the linked
 model's RF performance on that implementation.

In a same-condition matrix, expand **Model parameter overrides** to choose a
 model and precision. QGRU QAT supports separate weight/activation bits and an
 optional compatible float DPD pretraining run. Its exact checkpoint is bound
 and included in full exports. Float pretraining consumes additional budget;
 features remain FP32 and the hardware accumulator is unspecified. Unsupported
 PA/cross-condition QAT is refused rather than silently run as float.

## Multi-panel publication and full reproduction

Choose **Publication figure** in the spectrum review. Add up to four PSD,
 AM/AM, AM/PM, residual-CDF or declared-output-power panels. Trace controls
 retain visibility, colour and line style. **Preview panels** validates all
 sources; zooming then **Save view** preserves axes. Saved figures retain their
 per-run profiles and reference. Single/double-column exports include PNG,
 SVG, PDF, CSV data, facts, profiles, caption, hashes and a rendering script.
 The R1/R2 legend aliases map to full run IDs in the caption.

Residual CDF uses every valid evaluated sample, with 64 right-closed bins for
 |output − saved evaluation reference| / RMS(reference). It adds no timing,
 phase or gain fit and is **not demodulated EVM**. Power scans need explicitly
 declared average output power in dBm; normalized IQ does not provide it.
 Older runs without a stored residual artifact show that source as unavailable.

**Download full metric + figure reproduction** is private: it includes raw
 datasets, weights, run metadata and the exact numerical Python source. Extract
 the bundle, install its numerical dependencies in a separate environment if
 needed, then run one of the documented commands in `FULL-REPRODUCTION.md`.
 `--use-bundled-source` explicitly executes the included source after checking
 all hashes. Use a new workspace path; existing workspaces are never replaced.

The script independently reconstructs predictions, compares primary/baseline
 metrics and display coordinates, and renders the saved layout with rebuilt
 arrays. Read `reproduction-check.json`, `reproduced-figures/` and `replayed/`.
 CPU checks use the repository's **proposed**, not independently approved,
 relative 1e-4 / absolute 1e-5 checkpoint tolerance; other devices need their
 recorded tolerances. Re-training is a separate experiment. Real RF repeats,
 hardware measurements, independent EVM validation and participant testing
 remain external acceptance items.
