# OpenDPD Studio 2.2.10

This maintenance release addresses the release blockers from the repository code review and improves hosted resource use, uploads and Python API behavior.

- A worker that cannot start fails once, releases its queue slot and records an actionable error.
- All local upload consumers run outside the event loop. Cancellation waits for file consumers to finish, and imported package archives are removed.
- Built-in catalog reads reuse a fingerprinted cache; expensive public reads share the existing compute budget. Generated paired datasets no longer write and reparse a temporary CSV.
- Plotting uses deterministic local styles without executing personal plugin code or changing the global plotting backend.
- Python training calls validate explicit arguments without changing `sys.argv` or `sys.path`. CLI model choices match available constructors. QGRU and APNRRU initialization errors are fixed; checkpoint parameter layouts are retained.
- API request/response contracts use strict validation. Shared error, identifier and content-store helpers reduce inconsistent checks. Unknown request fields are rejected.
- Malformed run links keep navigation usable. Download and contribution links are validated, query reads propagate cancellation, HTML sinks are checked, and production builds omit the fixture gallery.
- Hosted cleanup can recover after a successful sweep. Retiring a workspace is synchronized with GPU result delivery. Worker heartbeat threads finish before closing their event files.
- Publishing runs only for a matching release tag. Actions are pinned to commits, protected changes require a current-head maintainer review, and distribution checks exclude deployment files, papers and planning documents from the source archive.
- GPU/VM runtime dependencies are hash locked. The GPU agent has explicit capability, filesystem and network limits; the tunnel identity is restricted to the API loopback port. The VM bridge token is read from a protected file.

Install with `uv pip install "opendpd==2.2.10" --torch-backend=auto`, or use [Studio on the web](https://opendpd.com/studio/).

The default installation still includes PyTorch, the desktop window and built-in datasets. Existing models and scientific metric definitions remain readable. Larger compatibility-sensitive refactors from the review are tracked in the [review disposition](review-2.2.10.md); this patch does not claim that every architectural recommendation is complete.
