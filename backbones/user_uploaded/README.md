# User uploaded backbone templates

Studio opens an opt-in contribution PR in a separate branch. It never merges
that PR. An independent human maintainer must approve its current head before
it can merge. See [the contribution guide](../../docs/architecture/user-backbones.md).

Each contribution directory contains exactly `backbone.py` and `manifest.json`.
`backbone.py` is a template v1 **literal network description**, not an executable
plugin. Studio parses it as data, revalidates its shape and resource limits, and
builds the network using trusted PyTorch layers. Do not import these files.

Only files merged into `main` are included in the community catalog. In Studio,
click **Refresh community backbones** to populate **User Uploaded Backbones**.
Open PRs and private workspace uploads never enter that catalog.
