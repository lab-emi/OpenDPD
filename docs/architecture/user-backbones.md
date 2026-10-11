# User backbone templates (v1)

PA and DPD training both offer **Download template (.py)** and **Upload
backbone** beside architecture selection. The uploaded template is immediately
available in **My uploaded backbones**, in that workspace. The usual training
presets, frozen PA reference, checkpoint selection, validation/test metrics and
plain tensor checkpoint loader still apply. The resolved configuration embeds
the complete canonical network graph, so testing, retry and export do not depend
on a mutable filename or a future catalog update.

## Template contract

The download contains one `BACKBONE = {...}` literal assignment and an optional
module docstring. Edit this description using the examples in the template.
It is Python syntax for **data**, not a general Python plugin interface.

Required fields: `schema_version` (1), `name`, `description`, `author`, `license`
(`Apache-2.0`), `nodes`, `output`. Each node has a unique lowercase `id`, an `op`
and `inputs` referring to `input` or earlier node IDs. No cycles or unused nodes.
The final node must produce exactly two real IQ features. Input and output shapes
are `[batch, time, 2]`, and every operation preserves the time dimension.

| Operation | Options | Behavior |
| --- | --- | --- |
| `linear` | `features`, optional `bias=True` | Per-sample linear projection |
| `gru`, `lstm` | `features`, optional `layers=1` | Unidirectional, batch-first; state resets per frame |
| `conv1d` | `features`, `kernel_size`, optional `dilation=1`, `bias=True` | Causal left padding |
| `relu`, `tanh`, `gelu`, `silu` | none | Fixed PyTorch activation |
| `layer_norm` | none | Normalize features, never the time axis |
| `dropout` | `p` from 0 to 0.8 | Disabled during evaluation |
| `identity` | none | Pass through |
| `iq_features` | none | 2 → 6: I, Q, envelope, power, envelope × power, power²; envelope includes a 1e-12 stabilizer |
| `add`, `concat` | 2–4 inputs | Residual sum or concatenate on the feature axis |

Limits: UTF-8 `.py` up to 32 KiB; 4,096 AST nodes; 16 literal nesting levels;
24 graph nodes; 128 features per node; 1,024 total node features; 1–250,000
trainable parameters. Recurrent layers are limited to two per node. Convolution
kernel size is at most 33, dilation at most eight, history at most 128 samples
per convolution. These are per-template bounds, not a guarantee of training
accuracy or low latency. Chained convolutions can have a longer total history.
All operations are causal; evaluation uses existing offline-segmented semantics.

Imports, functions, classes, custom `forward`, decorators, expressions, attribute
access, filesystem/network/process operations, dependencies, binary files,
pickle and archives are rejected. There is no `exec`, dynamic import or Python
bytecode execution of uploaded content. Static scans cannot prove arbitrary
Python safe; this design removes that execution path entirely. Only trusted
PyTorch implementations of the bounded graph execute. It does not claim to
eliminate vulnerabilities in Python, PyTorch, drivers or a compromised host.

## Private upload and optional contribution

The contribution checkbox is **unchecked by default**. Selecting it confirms
rights to publish the *entire source file*, including comments and the `author`
field, under Apache-2.0. Changing the file clears this consent. A blank public
author is accepted for a private upload but refused for a contribution.

Uploads are stored outside the import path under
`<workspace>/user-backbones/bbpr-<full source SHA256>/`. Submission rechecks the
original bytes, graph, manifest, file list, destination and both consent hashes.
One isolated contribution branch contains only:

```
backbones/user_uploaded/<name>_<source-hash-prefix>/
  backbone.py
  manifest.json
```

The branch is `codex/backbone-<full-source-SHA256>`. The host's authenticated
GitHub CLI creates a branch in `lab-emi/OpenDPD`, or in the contributor's verified
fork when upstream write access is unavailable, and opens a PR targeting `main`.
Credentials never come from the browser or uploaded file. No token is stored in
the upload record. Git hooks are disabled, source hashes are checked after copying
and staging, and existing branches are never force-pushed. Interrupted submissions
recover the same branch and PR. If an existing branch includes unrelated changes,
or main advanced in a way that prevents the isolated diff from being verified,
automatic retry stops for manual review on GitHub.

Studio never merges PRs or enables auto-merge. A PR remains outside the community
catalog until merged. **Refresh community backbones** reads only GitHub `main`,
pinned to an immutable commit; it checks regular-file Git modes, source SHA256,
the schema and resource limits again. Successful entries appear in **User Uploaded
Backbones** immediately, without executing downloaded Python or installing packages.
Refresh is atomic: invalid upstream content or a GitHub outage leaves the previous
catalog available. Anonymous GitHub API rate limits can delay refreshing.

Desktop uploads persist with the workspace. Hosted uploads have the same temporary
lifetime as the hosted session. The graph is included in exported run configurations;
private raw source is not automatically attached to a run export or sent to GitHub.

## One-time repository and hosting setup

The feature deliberately refuses public submission unless it can verify an active
repository ruleset for `main` with no bypass actors, current independent approvals,
an independent review gate and the required `independent-review` and
`user-backbone-review` statuses from the GitHub
Actions app (15368). A warning explains why the contribution checkbox is disabled;
private upload and training continue to work.

An account that cannot read the ruleset's bypass metadata cannot confirm this
boundary. In that case standalone contribution stays disabled; use a configured
hosted Studio or ask the host operator. Missing metadata is never treated as an
empty bypass list.

Deployment order:

1. Independently review and merge the trusted scanner and workflows.
2. Apply `deployment/user-backbone-ruleset.json` alongside the existing CI rules.
   It requires PRs, resolved conversations, current branches, no bypass actors,
   and the `independent-review` and `user-backbone-review` statuses from GitHub
   Actions (15368). Native approval counts need not require a person.
3. Give intended human or agent reviewers an authorized admin/maintain identity.
   Prefer separate GitHub identities; shared-account agents use the explicit
   session-bound attestation in the [review policy](review-policy.md).
4. On a public host, opt in with `OPENDPD_WEB_BACKBONE_PUBLICATIONS=1` after
   configuring its dedicated contribution identity. Publication stays disabled
   by default. Private uploads and training do not require it.

The guard executes only trusted base code and reads bounded blobs at the PR head.
It accepts independent authorized human or agent approval under the shared policy;
self-review, stale/dismissed approvals and unresolved changes requests fail.
For fork events with read-only tokens, post a PR comment or manually dispatch
with `pr_number` to trigger a trusted recheck. Ordinary comments are not approvals.
Studio only reads the rules; it never changes protections or merges contributions.
Repository administrators remain the root of trust.

## Verification

`tests/unit/test_backbone_template.py` exercises malicious syntax, size/shape
bounds, every operation's gradients and causal behavior, and independent-review rules.
`tests/integration/test_user_backbones.py` covers API authentication/CSRF,
workspace isolation, consent/tampering, real local bare Git branches with mocked
GitHub PR transport, fork recovery, main-only catalog refresh and real PA/DPD
training plus checkpoint evaluation on CPU and CUDA when available. The UI tests
cover default-private upload, revoked consent on file changes, disabled publishing
without enforcement, and selection of merged community entries.

These are execution and security-boundary tests, not ACPR/ACLR performance claims
for arbitrary submitted architectures. No fixture PR is sent to the real repository.
