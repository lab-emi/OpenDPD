# Review and release controls

Protected paths include scientific references and protocols, workflow definitions, deployment units, the public boundary and packaging metadata. `.github/CODEOWNERS` identifies their maintainers.

The protected-path check requires an admin or maintainer review attached to the current PR head. An approval counts; a maintainer comment review can record `Reviewed head: <full commit SHA>` when the reviewer and PR author use the same account. This records the review identity and exact revision without pretending it is an independent human approval. A subsequent commit requires a fresh review. Labels never grant approval. Repository rules and code-owner review requirements remain the primary merge controls.

Only publishing a GitHub release starts the production PyPI workflow. Its tag must match the package version. The publisher has its own `pypi` environment and the only OIDC write permission. Maintainers should configure environment reviewers when more than one independent reviewer is available; the workflow does not assume such an approval exists. Build and publish actions are commit pinned and Dependabot proposes updates.

For a release: run the regression suites and packaging checks, review the exact head, merge under the repository's configured permissions, build the pinned runtime, wait for active jobs to drain, deploy with rollback copies, then verify the real hosted workflow and installed wheel. Never alter branch protection to make a release pass.
