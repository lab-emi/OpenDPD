# Dependencies before a release

Dependabot checks dependencies daily. Routine npm, Python and GitHub Actions
minor/patch updates are grouped by ecosystem, so a release does not leave a
separate PR for every small upgrade.

## Automatic processing

The **Dependency maintenance** workflow starts automatically when a PR changes the
version-bearing manifests, after CI/documentation/dependency checks finish, on
weekdays, and on manual dispatch. It reads PR metadata
and files from GitHub using trusted code checked out from `main`. It never checks
out PR code or installs dependencies while holding its write token.

Automatic handling is restricted to same-repository Dependabot PRs that contain:

- Existing frontend dependency version updates and their npm registry lock entries.
- Minor/patch lower-bound updates for the two documentation requirements.
- Minor/patch updates of known GitHub Actions, pinned to immutable commits whose
  upstream version tags match.

New packages, major upgrades, breaking 0.x updates, core Python dependencies,
scripts, source-code changes, workflow structure, permissions, new registries and
symlinks require human review. Labels alone never authorize an automatic merge.
The scoped review checks report routine dependency eligibility explicitly; they
do not pretend that a human reviewed the PR.

The workflow refreshes an eligible branch against current `main`, then requires
the full **CI**, **Docs** and **Deployment dependency audit** workflows to pass
on that exact commit. Required jobs must run successfully, not merely be skipped.
An active GitHub ruleset must require up-to-date branches and all of those
checks; otherwise automatic merging refuses to run. This also closes the race
between checking the base revision and merging. Merges use an expected head SHA.
Additional repository review rules remain effective; the automation never
bypasses them.
After a token-created merge, the workflow explicitly dispatches main's validation
workflows. Empty refreshed dependency PRs are closed as already included.
Concurrent closures, stale-head responses and merge conflicts are re-read and
reported per PR. A conflicting branch waits for Dependabot's normal refresh;
other PRs continue to be checked. Preparation stays blocked if a conflict or
repository review rule is still unresolved when its wait limit expires.

These dispatches account for [GitHub's token-triggering rules](https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/trigger-a-workflow);
they require no personal access token, stored maintainer credential or automatic
human-review impersonation.

## Prepare each version

Version PRs start reconciliation automatically. To clear the queue before creating
the version PR, or to wait for a single explicit readiness result, run:

```bash
gh workflow run dependency-maintenance.yml -f prepare_release=true
```

This waits up to two hours for eligible dependency updates to be refreshed,
tested and merged. Failures, major upgrades and other manual-review cases fail
the preparation with the affected PR numbers and check links. Resolve those
items and rerun preparation before continuing.

Then prepare the version/release-notes PR, wait for its checks, and merge it.
The required **Release dependency readiness** CI job detects changes to the
Python package and frontend version fields and blocks that version PR while
other dependency PRs remain open. Ordinary dependency PRs can still merge to
clear the queue, so the gate cannot deadlock its own preparation.
Publish the tag from that resulting `main` commit. Both the package build and the
final PyPI upload independently check that:

1. No Dependabot or `dependencies`-labelled PR remains open.
2. The release tree matches current `main`, including maintenance commits.
3. Required validation passed on the release commit, or on a merged PR with the
   **identical source tree**. A failed release-commit run cannot be replaced by an
   older successful PR run.

Thus a release cannot silently proceed with a known unresolved dependency PR.
New upstream versions can still appear after publication; the same workflow
handles those updates for the next release. Dependabot security updates remain
enabled.

## Operator checks

For a read-only gate, use a GitHub token with repository contents, Actions and PR
read access:

```bash
GITHUB_REPOSITORY=lab-emi/OpenDPD python -I scripts/maintain_dependencies.py --check --sha COMMIT_SHA
```

`GH_TOKEN` supplies the token; do not print it or store it in the repository.
The automatic workflow additionally needs contents, PR and Actions write
permissions for merging, refreshing branches and dispatching checks. It does not
change branch protection or create releases.

Apply the reviewed `deployment/dependency-status-ruleset.json` template as an active
repository ruleset. The automation reads effective rules using metadata access;
it does not need an administrator token.

The required branch checks are the eleven CI jobs declared in
`scripts/maintain_dependencies.py`, plus `Build site` and `audit`, with strict
up-to-date checking. Keep that list aligned when changing the CI matrix.
