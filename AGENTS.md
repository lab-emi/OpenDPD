# Agent and Contributor Operating Rules

This file defines what automated coding agents (and humans acting in the same
role) may change on their own, what needs a human decision, and when to stop.
It is the executable companion of `OpenDPD_Studio_Development_Plan.md`.

## 1. One compute core, three entry points

GUI (`opendpd gui`), CLI (`opendpd run` / legacy `opendpd-cli`) and the Python
API must call the *same* application services for training, preprocessing,
metrics and export. Never re-implement a metric, a data split or a default
value in the frontend, in a route handler or in a shell script.

## 2. Writable scope for ordinary agent PRs

Agents may freely edit, within an approved task:

- `opendpd/` (schemas, services, runtime, server, studio, cli)
- `frontend/`
- `tests/` **except** the protected paths below
- `docs/` **except** `docs/protocols/` and release notes
- CI workflow files, provided permissions are not widened

## 3. Protected paths (human science review required)

Changes to any of the following must be a separate PR, must carry the
`science-review-approved` label, and must explain *why* the scientific
semantics change. CI (`.github/workflows/protected-paths.yml`) fails otherwise.

| Path | Why it is protected |
|---|---|
| `tests/golden/**` | Frozen numerical references (metric goldens, checkpoint fixtures). |
| `opendpd/core/metrics/**` | Metric definitions and versioned profiles. |
| `opendpd/core/splits.py` | Data-split and boundary-isolation protocol. |
| `benchmark/**` | Published benchmark results, protocol and provenance. |
| `docs/protocols/**` | Acceptance thresholds, evaluation and comparison protocols. |
| `.github/CODEOWNERS`, `.github/workflows/protected-paths.yml` | The guard itself. |

Agents must never relax a tolerance, change a seed set, change a checkpoint
selection metric, or edit an expected value to make a test pass. If a test
disagrees with the code, stop and report; do not "fix" the expectation.

## 4. Resource budget and stop conditions

- At most **2 agent PRs awaiting human review** at any time.
- One core objective per agent at a time; no scope growth inside a PR.
- Ordinary PR CI must stay CPU-only and finish in under 30 minutes.
  GPU / multi-seed validation is a separate, human-approved job bound to an
  exact commit.
- Stop and write a blocking note (instead of guessing) when:
  - acceptance evidence for the task cannot be produced in this environment,
  - a protocol is ambiguous,
  - a change would touch a protected path,
  - a platform (Windows/macOS/GPU) claim cannot be verified here.

## 5. Evidence rules

- "Mock succeeded" is never end-to-end evidence. After G0, feature completion
  requires real API + real computation.
- Missing hardware is reported as **not verified**, never as passed.
- Every stage ends with an ablation pass: remove code, dependencies or
  abstractions that no acceptance criterion exercises, and delete scratch files.
- Reports record commands, environment, results and failure causes
  (`docs/baseline/`, `docs/releases/`).

## 6. Things agents never do on their own

Publish a release, upload data or models to external platforms, enable
telemetry, bind the server to non-loopback addresses by default, unlock RF
output on instruments (never set `OPENDPD_ALLOW_RF_OUTPUT`, register a real
instrument adapter or relax a safety limit; only the mock adapter runs in CI),
or widen CI permissions.
