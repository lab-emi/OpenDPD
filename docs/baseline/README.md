# Validation reports and local evidence

Commit reviewed Markdown reports that record commands, environment, results and
failure causes. Screenshots and reusable verification scripts may be included
after checking them for private data and session URLs.

Raw experiment and validation captures stay local: logs, JSON/JSONL responses,
CSV data, terminal dumps, traces and source snapshots are ignored here. Existing
reports may refer to these local evidence paths; those files are intentionally
absent from a fresh checkout. Reproduce them using the commands in each report.
Removing a capture from Git tracking does not delete its local copy.

Keep user captures, exports, checkpoints and Studio workspaces outside the
repository (the default is `~/opendpd-workspace`). The repository's published
PA datasets, synthetic tutorial, contract examples and frozen test fixtures
remain versioned. New dataset captures are ignored by default.

Environment examples must contain placeholders only. Never force-add credentials,
private keys, authenticated browser state or `.studio.lock` (which contains the
bootstrap URL). Ignore rules prevent accidental staging of untracked files;
they do not remove content from earlier Git commits or detect every possible
secret. Review the staged diff before pushing.
