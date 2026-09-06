# OpenDPD Studio — stage progress

Per-stage status against the acceptance criteria of
`OpenDPD_Studio_Development_Plan.md`. "done" means evidence exists in this
repository; "pending human" means a maintainer decision is required;
"not verified" means the environment could not produce the evidence.

| Stage | Status | Commit | Notes |
|---|---|---|---|
| S00 baseline & governance | done except maintainer approvals | `6f026b1` | `docs/baseline/baseline-report.md` |
| S01 contracts & UX | done except prototype (delivered with S05) | — | schemas, ADR-0001, UX spec, tokens, mock examples |

## S00 acceptance items

| Item | Status |
|---|---|
| `baseline-report.md` with commands, env, results, failure causes | done |
| CLI/API compatibility cases, metric goldens, legacy checkpoint load case registered | done (`tests/golden`) |
| Maintainer approval of G0–G2 scope, support matrix, risks, thresholds | **pending human** |
| Protected paths cannot be changed by ordinary PRs | done (`protected-paths.yml`, CODEOWNERS) |
| Agent scope, budget, max pending PRs, stop conditions configured | done (`AGENTS.md`) |

## S01 acceptance items

| Item | Status |
|---|---|
| Three journeys with success / failure / exit paths | done (`docs/architecture/ux-spec.md`) |
| Page structure reviewed by maintainer; advanced params collapsed | spec written; **pending human review** |
| Every result expresses `pa_modeling` / `dpd_surrogate` / `dpd_measured` with source and profile version | done (`EvaluationResult` validators) |
| Schema examples: missing metadata, N/A metric, interrupted run, log disconnect, legacy import | done (`opendpd/schemas/examples.py`, 49 tests) |
| Frontend prototype on fixed, labelled mock | deferred to S05 (mock fixtures are generated from the same examples) |
