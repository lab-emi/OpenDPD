# Review and release controls

Protected paths include scientific references and protocols, workflow definitions, deployment units, the public boundary and packaging metadata. `.github/CODEOWNERS` identifies their maintainers.

The protected-path check runs trusted base-branch code and reads changed files (including previous filenames) from the GitHub API. It requires an independent admin/maintainer approval attached to the current PR head; comments and labels do not substitute for approval. After a review, rerun the check or add a PR comment to trigger a fresh check. Repository rulesets that require Code Owner approval and dismiss stale approvals are the primary controls; their settings must be verified separately from repository code.

Routine, version-only Dependabot minor/patch updates have a narrowly validated
automatic-review exception. It covers existing frontend dependencies,
documentation requirements and verified immutable Action pins; it excludes
major/core updates and any source, script, workflow-structure or permission
change. This is an explicit automation policy, not a human approval. Full CI,
documentation, dependency audits and strict up-to-date branch checks are still
required. See [Dependencies before a release](../guides/dependency-maintenance.md).

User backbone contributions have a stricter, separate gate: only an independent
human maintainer's `APPROVED` review of the exact head counts; the comment mechanism
above is not sufficient. Studio refuses contributions until the required no-bypass
repository rules are enabled. See [user backbone templates](user-backbones.md).

Only publishing a GitHub release starts the production PyPI workflow. Its tag must match the package version. The publisher has its own `pypi` environment and the only OIDC write permission. Maintainers should configure environment reviewers when more than one independent reviewer is available; the workflow does not assume such an approval exists. Build and publish actions are commit pinned and Dependabot proposes updates.

For a release: run dependency preparation until the known dependency PR backlog is empty, run the regression suites and packaging checks, review the exact head, merge under the repository's configured permissions, build the pinned runtime, wait for active jobs to drain, deploy with rollback copies, then verify the real hosted workflow and installed wheel. The publishing workflow checks the dependency backlog and exact release source again before building and uploading. Never alter branch protection to make a release pass.
