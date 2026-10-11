## Task
<!-- Sxx.y — one independently reviewable objective -->

## User problem
<!-- Who hits what problem in which flow? What is the evidence? -->

## Scope / Non-goals
<!-- What changes, which modules; what is explicitly not done here -->

## Acceptance evidence
<!-- Commands, environment (OS / Python / device), logs, artifact paths.
     Screenshots are for UI checks only and never replace backend results.
     Missing hardware is reported as "not verified", not as passed. -->

## Unfinished / follow-ups

## Risk / approvals
- [ ] Touches a protected path (`tests/golden`, metrics, splits, benchmark, protocols)? If yes: separate PR + `science-review-approved` label.
- [ ] Changes permissions, release, telemetry, network binding or hardware control? If yes: independent authorized human or agent review required; operational permissions still apply.

## Contributors and independent review
<!-- List actual people/agents and development session IDs. Required for shared-account
     agent review. Copy the opendpd-contributors JSON block from
     docs/architecture/review-policy.md. Prefer a different agent for review/merge.
     The reviewer records the current full head SHA, findings and a separate session.
     Final merge commit: Contributed-By, Reviewed-By, Reviewed-Commit, Review-URL.
     Do not claim a review that has not happened. -->

## Rollback
<!-- How to revert code or restore data. Deleting evidence is not a rollback. -->
