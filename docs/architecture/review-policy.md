# Review and merge policy

OpenDPD accepts contributions, reviews and merges by people and authorized agents.
A human is **not required in the PR review loop**. An agent may implement a change,
open its PR, and hand it to an independent reviewer agent that can approve and
merge it. Prefer different systems: Codex develops → Claude Code reviews and
merges, or Claude Code develops → Codex reviews and merges. A separate session of
the same system is acceptable when another system is unavailable; record why the
review is independent. The implementing session must not approve its own work.

This policy supersedes older **PR-review** wording that required a human. It does
not grant permission to publish private data, operate physical instruments, spend
money, deploy, or publish a release. Existing consent, hardware interlocks and
release authorization still apply. Scientific evidence and acceptance thresholds
are not relaxed by allowing agents to review.

## What must pass

- Work on a branch and open a PR. Keep unfinished work as a draft.
- Include the problem, scope, verification, limitations and contributor identities.
- Review the actual diff and relevant tests, including scientific and security
  effects. An agent reviewer records its identity, a separate session and findings.
- Approval must refer to the **current full head SHA**. Any push, including a merge
  from main, requires renewed approval. Resolve requested changes before merging.
- All required CI, documentation, dependency, packaging and review checks must
  pass; the branch must be current with main. An agent must not bypass a check,
  dismiss another review without resolution, or weaken protections to merge.
- The reviewer should perform the merge. Use the merge helper below so the final
  squash commit records contributors, reviewers, the reviewed SHA and evidence.

Scientific changes (metrics, splits, golden references, acceptance thresholds,
protocols and published benchmarks) remain separately reviewable PRs with the
`science-review-approved` label after scientific review. The reviewer may be an
agent competent to assess that evidence. A label by itself never grants approval.
Workflow, deployment, packaging, uploaded-backbone and permissions changes also
need an independent authorized reviewer; CODEOWNERS identifies responsible teams.

## Authenticated review records

A native GitHub `APPROVED` review from an independent account with `admin` or
`maintain` permission counts, whether a person or an agent operates the account.
Bot accounts are allowed. Author self-approvals, stale and dismissed approvals do
not count. Outstanding authorized `CHANGES_REQUESTED` reviews block merging,
including requests on an earlier head, until superseded or explicitly dismissed.

Agents sharing one authorized GitHub account cannot use native self-approval.
Instead, declare every contributor and development session in the PR description:

```opendpd-contributors
{"contributors":[{"name":"Codex <noreply@openai.com>","session":"codex:development-session-id"}]}
```

The independent reviewer then posts its findings and exactly one structured block
in a PR comment, using an account with `admin` or `maintain` permission:

```opendpd-review
{"head":"0123456789abcdef0123456789abcdef01234567","decision":"approve","reviewer":"Claude Code <noreply@anthropic.com>","session":"claude:review-session-id","independent":true,"summary":"Reviewed the diff and relevant tests; describe commands, results, findings and evidence here."}
```

Use real session IDs and the actual reviewed SHA. `request_changes` blocks merge;
`dismiss` withdraws that review. Post a replacement with the same reviewer/session
or edit the original record to change the decision. A newer decision supersedes
that session's old decision; an unresolved request from another reviewer/session
still blocks. If the reviewer name matches a contributor, add an
`independence_reason` explaining the separate session and fresh review.

This is an **authenticated maintainer attestation**, not cryptographic proof of
agent separation. GitHub authenticates the account; operators are responsible for
truthful identities, separate sessions and protecting credentials. Prefer separate
GitHub identities when available. Never invent reviewer credit. Ordinary comments,
labels, an author's claim that tests passed, or an unprivileged account's block do
not grant approval. A reviewer making implementation changes becomes a contributor
and must hand that revision to another independent reviewer.

## Enforcement and rechecks

`Protected paths` runs trusted target-branch code and publishes `independent-review`
on the PR head for **every PR**, including documentation. `User backbone review`
uses the same review policy and additionally reads uploaded templates as bounded
blobs, checks hashes and scans them without executing PR code. Its status is
`user-backbone-review`. Both gates recheck review/comment creation, edits and
deletion/dismissal, PR updates and PR-description edits. The gate code never trusts
the executable contents of the PR it is checking.

For fork review events with read-only tokens, an authorized operator can trigger
the trusted workflows manually (`pr_number`) or post a PR comment to recheck. An
ordinary trigger comment is not approval. The merge helper rereads live review
records to avoid relying solely on an earlier green status.

Apply `deployment/user-backbone-ruleset.json` as an additional active, no-bypass
ruleset alongside `deployment/dependency-status-ruleset.json`. Require PRs,
conversation resolution, no force pushes/deletion, strict up-to-date branches and
both review statuses from GitHub Actions (app 15368), preserving all existing CI
checks. Native approval count is zero because independence is enforced by the
head-bound gate and shared-account agents cannot approve themselves in GitHub.
CODEOWNERS remains a routing mechanism, not a human-only approval requirement.
Merge queues are not enabled by this policy; do not enable one until every required
check, including `independent-review`, reports on its synthetic merge commits.

Deploy the trusted implementation through a reviewed, fully tested PR **before**
enabling the new required statuses. Do not temporarily remove CI protections to
bootstrap it. Existing PRs must update from main and rerun the review workflows.
Repository administrators remain the root of trust. Settings are external to Git;
verify the live rulesets rather than assuming a checked-in template is active.

Routine version-only Dependabot updates retain the narrowly validated automated
review in `scripts/dependency_policy.py`. It is a deterministic reviewer separate
from Dependabot, not a fictitious human or LLM review. Non-routine or major updates
need independent human/agent review. Full CI remains mandatory. See
[dependency maintenance](../guides/dependency-maintenance.md).

## Commit attribution and merge

Development commits identify actual contributors with `Contributed-By:`; use
`Co-Authored-By: Name <email>` as well when applicable. A reviewer is credited only
after completing the review. Do not amend the reviewed head merely to add reviewer
credit: that would invalidate the approval. Put it in the final squash/merge
commit, preserving all contributors (including people) and the evidence link:

```text
Contributed-By: Codex <noreply@openai.com>
Reviewed-By: Claude Code <noreply@anthropic.com>
Reviewed-Commit: <full PR head SHA>
Review-URL: <GitHub review or attestation URL>
Review-Session: claude:<review-session-id>
```

From a trusted checkout, the **reviewer** runs:

```bash
python -I scripts/merge_reviewed_pr.py PR_NUMBER --head FULL_REVIEWED_SHA
python -I scripts/merge_reviewed_pr.py PR_NUMBER --head FULL_REVIEWED_SHA --merge
```

The first command validates and prints the commit body. The second rereads the
PR/reviews, checks full CI and the current base, and submits a squash merge with an
expected head SHA and attribution trailers. GitHub still enforces its rulesets.
Equivalent native/API merges must preserve the same trailers; avoid a rebase merge
that loses reviewer attribution. Automated dependency merges write Dependabot and
the deterministic policy reviewer into their final commits too. Trailer creation
is enforced by these merge tools and reviewer procedure, not by GitHub's native
approval counter.

## Releases

Only publishing a GitHub release starts production PyPI publication. Its tag must
match the package version; the `pypi` environment owns the OIDC write permission.
A PR approval does not itself authorize release publication or deployment.
Run dependency preparation until the known dependency backlog is empty, verify
regression and packaging evidence, review the exact release source, and merge.
Then follow the authorized release/deployment process with rollback and hosted
workflow verification. Never change thresholds or branch protection to make a
release pass.

The [2026-10-11 development audit](../releases/2.4-development-audit-2026-10-11.md) records the 2.4 PR backlog and the previous enforcement gap.
