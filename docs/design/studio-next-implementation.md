# Studio Next implementation and acceptance ledger

Source: [research-driven plan](opendpd-studio-research-driven-plan-2026-09-13.md).
Branch: `studio-next`, based on `OpenDPD-Studio` / `origin/main` at `d09ebb6`.
P0 and P1 software development is delivered, with the independent EVM and physical
acceptance gates explicitly retained. P2 remains conditional on the hardware and
data prerequisites in the plan. No physical measurement validation
or user study is claimed from software tests or mock captures.

## Delivery ledger

| Requirement | Implementation / evidence | State |
| --- | --- | --- |
| W1 RF facts, definitions, missing fields, declared conditions | Review context snapshots, RF facts panel, immutable declaration revisions, power/efficiency boundaries, full provenance | Implemented; tested |
| W2 fixed reference, deltas, condition differences, ranking guards | Reference and review mode in URL; exact full hashes and conditions; per-run profiles; incompatible deltas withheld | Implemented; tested |
| W3 trace provenance, cursor, protocol bands, metric invariance | Actual frequency coordinates per trace; nearest-bin cursor; profile-specific integration boundaries; custom plot gestures retained | Implemented; tested |
| W4 saved views | Versioned figure spec binds runs, profiles, traces/visibility, reference, axes, bands and cursor; stale sources refused | Implemented; tested |
| W5 figure/data export | PNG/SVG/PDF, numerical CSV, metrics, facts, profiles, rendering environment, hashes and standalone replay | Implemented; tested. Figure replay alone does not re-evaluate metrics |
| W6 four research tasks | Four approximately five-minute scripts in the research-review guide; measured no-DPD source with surrogate smoke runs; additional explicitly MOCK timing/session demo | Implemented demo material; actual participant validation pending |
| P1 sessions/calibration/repeats | Structured instrument/calibration/reference-plane records, raw hashes, explicit acquisition IDs, exclusion history, power tolerance, SD/optional Student t CI, separate capture and seed counts | Implemented; unit and mock-pipeline integration tests. Real repeat captures required |
| P1 fractional processing | Opt-in `measurement-fractional-v2`, integer default unchanged, sample/ns delays, documented interpolation/boundaries, valid intervals, common-interval timing/phase-fit diagnostic stages | Implemented; analytic waveform and mock import/replay tests. Real capture validation pending |
| P1 condition cards and matrices | Sweep Board page; condition creation/audit, copied configs, method parameters, seeds and sample budgets, count/work preview, ordinary Supervisor workers, cancel/retry/restart, preserved attempts, seed summaries | Implemented first version; real browser creation/start/completion passed at both sizes. Advanced model/precision controls, strict QAT setup and bound float pretraining tested; GUI precision preview passed at both desktop sizes |
| P1 fixed surrogate with multiple DPD seeds | Explicit PA `seed_policy=fixed_surrogate`, exact checkpoint materialisation; legacy default still enforces seed compatibility | Actual training test passes; same PA hash across distinct PA/DPD seeds |
| P1 hardware costs and trade-offs | Full checkpoint/report hashes; tensor storage, qualified operation counts, state/lookahead, existing fixed-point resources and CPU reference timing; user report upload; sourced scatter axes | Implemented; actual deployment/API tests and browser workflow passed. No physical report supplied |
| P1 independent EVM validation | Existing pending profile remains hidden in GUI; no independent reference data supplied | External cross-validation/review pending |
| P1 multi-panel publication authoring | Up to four PSD/AM-AM/AM-PM/declared-power/residual-CDF panels, saved styles/axes/profiles, SVG/PDF/PNG and numerical data | Implemented; four-panel export and browser save/reopen passed. Missing dBm remains unavailable |
| Combined metric and view reproduction | Private full run/data packages, numerical dependency versions, exact runtime source snapshot, metric/display checks and regenerated figure | Implemented; fresh-workspace replay through bundled source reproduced metrics, plot coordinates and PNG exactly |
| Desktop layout/performance | Real browser review/save/reopen/download at 1366×768 and 1920×1080; newest screenshots inspected | P0 and session/matrix workflows passed at both sizes |
| Physical evidence | Need actual with/without-DPD, independent repeats, ≥3 independent working conditions, hardware cost report | User confirmed real data are not yet available; synthetic fixtures and user uploads are the authorized development path |
| Actual user study | Need 6–8 participants, task completion/time, metric/view reconstruction | External; participants not contacted |

## Validation recorded so far

- `python -m pytest -q tests/unit tests/golden tests/integration/test_research_review_flow.py tests/integration/test_sweep_board.py tests/integration/test_cli_run.py`: **367 passed** (52.20 s). Later small source-integrity/metadata changes require focused rechecks.
- `npm --prefix frontend test`: **45 files, 207 tests passed** (18.97 s).
- `npm --prefix frontend run build`, type generation/typecheck and lint succeeded; the existing large Plotly chunk warning remains.
- Fractional alignment: **13 analytic-waveform tests** cover positive/negative residuals, known phase/gain, power preservation, loop/single boundary rules and unchanged integer output. The generator evaluates delayed sinusoids independently of the correction implementation.
- Actual measurement import/re-evaluation checks retain raw/played hashes, effective sample rate and processing version; integer and fractional results cannot be ranked as identical protocols. Capture tampering is refused.
- A real subprocess matrix over **three synthetic conditions / eight cells** completed, including source DPD transfer through each target condition's own PA model. This checks dependency execution, not physical cross-condition evidence.
- Browser review benchmark (`scripts/verify_research_review.mjs`): two runs, **7,680 evaluated samples/run**, reference switch **372 / 393 ms** at 1366×768 / 1920×1080. Save/reload restored the exact new figure ID, axes, bands, cursor and reference; downloaded filename matched that ID; result payload unchanged; no page errors or horizontal overflow. Measurements describe this local demo, not arbitrary dataset sizes.
- Selected browser artifacts are retained in `docs/performance/studio-next`; complete raw captures are under `/tmp/studio-next-browser-qa-latest` and `/tmp/studio-next-workflows-qa`.

## Reproducible local demonstrations

```bash
python scripts/studio_research_demo.py --workspace /tmp/studio-review-demo
python scripts/studio_measurement_demo.py --workspace /tmp/studio-review-demo
```

The first creates CPU PA/GRU/MP smoke runs on the registered DPA no-DPD source.
The second adds MOCK captures with a declared synthetic delay, phase and gain
through the real measurement import pipeline. It never accesses an instrument.
The generated index files identify evidence types and missing physical data.
See [research-review guide](../guides/research-review.md) for review/export tasks.

## Invariants and remaining work

- Historical metric algorithms and integer-alignment numbers remain unchanged.
- Display interactions never recalculate formal metrics.
- Training seeds, acquisition IDs, capture slices and hardware repeats remain distinct.
- Missing RF/DC power, calibration or cost evidence is not inferred from normalized IQ.
- Full hashes identify data/models/configuration; display prefixes are not identity keys.
- Fractional timing fitting assumes a uniquely correlating waveform. Sparse periodic tones can have ambiguous coarse delays; near-Nyquist single-shot interpolation and real receiver impairments require separate validation.
- EVM is not released based on internal self-comparison or mock data.
- Remaining acceptance gates require real repeat captures, independent operating conditions, hardware reports, independent EVM references/review and actual participants. Synthetic/software validation does not discharge those gates.

## Follow-up integrity checks and implementation findings

- Focused session/sweep recheck after raw-source validation and dispatch checks: **15 passed** (25.85 s).
- Real browser session/alignment plus matrix preview/register/start/CPU completion passed at **1366×768 and 1920×1080** with no page errors or horizontal overflow. Both matrices finished through real Supervisor workers. Measurement evidence is explicitly MOCK.
- The current legacy QAT helper catches setup failures and returns a float model; its `pretrained_run_id` also needs verified binding. Resolved: Studio setup now fails explicitly, PA QAT/unsupported models are rejected, and float QAT pretraining binds full checkpoint hashes. Actual training, metric replay and full-package import passed. Existing fixed-point deployment reports have a separate explicit specification and bit-exact verification path.

## User-directed dataset creation and public contributions

The user confirmed that real captures, hardware reports and independent EVM
reference data are not ready. They requested synthetic data with explicit GUI
labels, user uploads, a private/public choice, and an automatic branch → push →
PR workflow with human review and contact address `emi.lab@outlook.com`.

- Added six generated datasets under `dataset/synthetic/studio-research-v1/`: three
  normalized drive settings × two independent random realizations, 16,384 samples
  each, 80 MS/s, 64QAM OFDM-like input, three-tap causal polynomial memory PA.
  No calibrated dBm, DC power, efficiency or physical-repeat claims are provided.
- `synthetic-memory-pa-v1` saves all generator inputs and RNG seeds. Synthetic
  origin cannot be changed to measured through the manifest editor. The GUI can
  regenerate suites, inspect truth metadata and prefill a three-condition matrix.
- Existing CSV validation/upload is reused. New uploads remain private by default.
  Public selection leads to a concrete package preview, license and attribution
  choice, and explicit confirmation of public disclosure/rights.
- Only canonical finite IQ CSV and selected metadata are copied. No private notes,
  local paths, scripts, credentials or unrelated checkout changes enter the PR.
- `PublicationController` retains exact package/source hashes and submission
  stages. An isolated sparse Git checkout starts at the central repository's
  current default branch, pushes one `codex/dataset-*` branch (using a fork when
  required), and opens a PR. Retries recover an existing branch/PR. No merge API
  exists. The GUI states that data are public while the PR awaits human review.
- The data-only catalog is discovered by updated Studio installations. Existing
  `datasets/` layouts remain compatible. Synthetic fixtures are packaged in wheels.
- Hosted submission is explicitly configurable by the operator and defaults off
  until a dedicated GitHub CLI account is available. Hosted preview/generation are
  tenant-scoped; publication has per-IP/daily quotas and requires exact consent.
  No deployed service configuration was changed.

Validation: six dataset/publication integration tests exercise real local bare
Git repositories, direct and fork branches, real pushes, post-push failure/retry,
package hashing, synthetic regeneration, origin locking, API authentication and
consent. GitHub PR calls are controlled test replacements: no real PR was sent.
The packaged-dataset HTTP test was extended to read the six new entries and to
confirm that they do not borrow legacy demodulators. The seven focused checks
passed (34.75 s); later sparse-checkout changes passed all six contribution tests
(7.49 s). Hosted boundary checks: 32 passed (7.67 s).

Real GUI tests at 1366×768 and 1920×1080 generated suites, previewed their
three-condition matrices, uploaded a CSV, confirmed the private default, prepared
a public package and downloaded the exact ZIP. No page errors, horizontal
overflow or public writes. Artifacts: `/tmp/studio-next-datasets-qa`. Selected
1366 screenshots were visually inspected.

QAT: float DPD → bound 8-bit-weight/12-bit-activation QAT → re-evaluation → full
package export/import → re-evaluation passed through the actual CPU trainer
(11.37 s). This remains software fake quantization with FP32 feature extraction.
Float pretraining contributes an additional training budget and is retained in
lineage and full packages. Cross-condition QAT is explicitly rejected by the
current conditions-v1 protocol because it does not carry precision consistently.


## Completed hardware, publication and reproduction workflows

- The hardware ledger separates exact checkpoint tensor storage, limited reviewed
  GRU affine operation counts, fixed-point specification resources, CPU reference
  timing and user-reported implementation evidence. Unknown buffers, skipping,
  throughput, power and energy are not inferred. Reports are immutable, fully
  hashed and marked stale if the source/result/checkpoint changes. Pending EVM
  profiles cannot supply a trade-off axis. Different RF/execution protocols remain
  visible as incompatibilities, with no automatic ranking.
- Figure v2 authors up to four PSD, AM/AM, AM/PM, declared average output-power
  and residual-CDF panels. The CDF is a separately versioned display diagnostic
  over all valid samples: no additional fit and no demodulated-EVM claim. Power
  points copy stored metrics and explicit dBm declarations; normalized drive has
  no invented dBm conversion. Figure v1 remains readable. Export legends use R1,
  R2, etc., with full run identities in the caption and bindings.
- Full reproduction includes each selected run's raw dataset (including built-in
  data), checkpoints, private metadata and a hash-verified runtime source
  snapshot. The recipient can explicitly use that source without needing an
  uncommitted checkout. The script imports into new workspaces, reconstructs the
  predictions, checks metrics and plot coordinates, and draws the saved view.
  It reports the proposed CPU tolerances and exact-PNG status separately; it does
  not retrain or claim external scientific validation.
- Result links now preserve the selected released profile. Pending profiles are
  refused in the GUI, including through direct URLs. Synthetic dataset origin
  appears in RF facts, publication output and cost evidence.

Final checks:

- Backend regression: **442 passed**, with one old test expecting only three plot
  artifacts. Both PA and DPD expectations were updated for the new residual plot;
  the affected real pipeline test then **passed** (8.25 s). The full selection
  covered **443** tests. A later focused figure/reproduction run passed its other
  **18** checks; the remaining failure was that same second artifact expectation,
  now fixed. No metric-algorithm failure was observed.
- Frontend: **49 files / 213 tests passed** with four workers (26.07 s). Default
  high-core concurrency twice exceeded existing one-second async UI waits;
  the affected tests passed in isolation, and all files passed with bounded
  concurrency. The runner now uses four workers without changing timeouts.
- Type generation consistency, TypeScript build, production build, lint and
  `git diff --check` passed. The existing large Plotly chunk warning remains.
- Wheel verification found all **six** synthetic catalogs, matching full CSV
  hashes and **98,304** total samples, and the reproduction runtime.
- Actual GUI at **1366×768 and 1920×1080**: four-panel preview, zoom, save,
  reopen and both downloads; synthetic cost-report upload and energy-axis
  selection; QGRU W8/A12 matrix preview. No page errors, horizontal overflow,
  result changes or public writes. The downloaded bundles were each replayed
  using their included source, preserving the released profile and producing
  matching metrics and byte-identical PNGs.

Artifacts: `docs/performance/studio-next/publication-hardware.json` and selected
screenshots; full private bundles and replay workspaces remain under
`/tmp/studio-next-publication-qa-final-styles`. Those are explicitly synthetic GUI
fixtures, not user or hardware evidence. No real PR, email, merge or deployment
was performed. Hosted public dataset submission still requires operator GitHub
configuration and the explicit enable flag described above.

## Signal Generator and model onboarding (2026-09-13)

- PA / DPD task cards are merged into two model workspaces, each with Training
  and Testing tabs. Original run task types and deep links remain compatible.
  Testing reads exact split counts for the selected preprocessing version.
- Get Started prioritizes Signal Generator as its only highlighted action;
  existing datasets and CSV upload follow in that order. The generator also has
  a dedicated sidebar entry, with simple family/preset selection and advanced
  parameters for geometry, OFDMA allocation, pilots, CP and impairments.
- Twenty presets cover NR FR1/FR2 and Wi-Fi 6/7 numerology, experimental Wi-Fi 8,
  and custom OFDM, RRC QAM/PSK, tone, multitone and chirp. These are continuous
  uncoded engineering stimuli with generic pilots, not complete standards
  implementations or conformance test models. This distinction is shown in
  the GUI and all exports; Wi-Fi 8 has no draft-specific UHR implementation.
- Full-waveform PAPR/CCDF, Welch PSD, RMS/peak, 99% occupied bandwidth, time-domain
  I/Q, FFT-recovered constellation, diagnostic reference EVM and resource maps
  are integrated. RF carrier remains metadata. Saved JSON, exact I/Q ZIP export
  and configurable synthetic PA datasets connect generation to model training.
- Validation: 41 generator numerical/API checks and 33 hosted-boundary checks
  pass. Frontend regression passes all 217 tests across 51 files; build, lint,
  TypeScript and generated API schema checks pass. Two real browser sizes
  (1366×768 / 1920×1080) complete generation, exact waveform export, dataset
  creation and eight actual CPU PA/DPD training/testing runs. Both exported
  CSVs reproduce NPY float32 values exactly, with verified SHA-256 and PAPR.
  Test set count is 6,452 for each 32,768-sample browser fixture. No public
  dataset submission, email, merge or deployment was performed.

See `docs/guides/signal-generator.md` for parameters, coverage and measurement
definitions; `docs/performance/studio-next/signal-generator/` contains selected
browser evidence. Full private QA exports remain in `/tmp/studio-next-generator-qa`.
