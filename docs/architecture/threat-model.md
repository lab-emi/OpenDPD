# Local threat model (S04)

Scope: a single user running `opendpd gui` on their own machine. The server
binds `127.0.0.1` only. Remote use is limited to an SSH tunnel the user
creates; binding `0.0.0.0`, shared lab accounts and multi-user service are
out of scope and refused by design.

## Assets

Raw PA measurements and their manifests, trained checkpoints, experiment
results, the workspace database, and the ability to start compute jobs.

## Adversaries and mitigations

| Threat | Vector | Mitigation | Test |
|---|---|---|---|
| Malicious web page drives the local API | `fetch("http://127.0.0.1:port/api/v1/runs", {method:"POST"})` | session cookie required (SameSite=Strict, HttpOnly); writes need `X-OpenDPD-CSRF` that only same-origin scripts can read; no CORS headers, preflight refused; Origin/Referer must match Host | `tests/integration/test_studio_api.py::test_cross_origin_and_csrf` |
| DNS rebinding | attacker host resolves to 127.0.0.1 | Host header must be a loopback name | `test_host_header_enforced` |
| Guessing the session | brute force | 256-bit random ids; bootstrap token printed only to the local console/URL; sessions die with the process | — |
| Path traversal / symlink escape | `GET /artifacts/{run}/{id}` | downloads by registered artifact id; the resolved path must stay inside the run directory | `test_artifact_download_by_id_only` |
| Arbitrary code via uploads/models | pickle, scripts | checkpoints loaded with `weights_only=True`; user data imports (S07) accept CSV/NumPy numeric arrays only, object arrays refused | S00 golden test; S07 tests |
| Shell injection | API → worker | workers are started with an argument list from validated structured input; no shell, no user strings in commands | code review; `Supervisor._spawn` |
| Resource exhaustion | huge bodies, unbounded logs | 2 MB JSON cap (413); logs paged by byte offset; one worker per device; workspace preflight | `test_payload_too_large`, `test_logs_are_paged` |
| Leaking secrets in URLs/logs | bootstrap token | token exchanged once via `GET /bootstrap` then removed by redirect; never logged by the app | `test_bootstrap_redirect_strips_token` |

## Accepted residual risks (first version)

- Any local process running as the same OS user can read the workspace and
  the console output; OS user isolation is the boundary.
- HTTP without TLS on loopback: traffic never leaves the host.
- Sessions live in memory; a server restart requires re-opening the URL
  printed by the launcher.
