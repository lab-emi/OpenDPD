# Local threat model (S04, hardened in S13)

Scope: a single user running `opendpd gui` on their own machine. The server
binds `127.0.0.1` only. Remote use is limited to an SSH tunnel the user
creates; binding `0.0.0.0`, shared lab accounts and multi-user service are
out of scope and refused by design.

## Assets

Raw PA measurements and their manifests, trained checkpoints, experiment
results, the workspace database, and the ability to start compute jobs.

## Adversaries and mitigations

Every row has a negative test that exercises the refusal through the real
ASGI application (`tests/integration/test_hardening.py` unless stated).

| Threat | Vector | Mitigation | Test |
|---|---|---|---|
| Malicious web page drives the local API | `fetch("http://127.0.0.1:port/api/v1/runs", {method:"POST"})` | session cookie required (SameSite=Strict, HttpOnly); writes need `X-OpenDPD-CSRF` that only same-origin scripts can read; no CORS headers, preflight refused; Origin/Referer must match Host | `tests/integration/test_studio_api.py::test_cross_origin_and_csrf` |
| DNS rebinding | attacker host resolves to 127.0.0.1 | Host header must be a loopback name | `test_studio_api.py::test_host_header_enforced` |
| Script injection into the page (XSS) | run names, notes, dataset ids, log lines, package contents rendered in the UI | React escapes every string; no `dangerouslySetInnerHTML`/`innerHTML`/`eval` in the frontend sources; server-side diagnostic pages HTML-escape their text; a Content-Security-Policy on every response allows scripts only from this origin (no inline scripts, no `eval`, no CDN), plus `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `frame-ancestors 'none'`, `Referrer-Policy: same-origin` | `test_every_response_carries_the_security_headers`, `test_diagnostic_pages_escape_their_text`, `test_frontend_sources_have_no_html_sinks`, `test_user_text_is_stored_and_returned_as_data`; the real-server journey (`frontend/e2e/live.spec.ts`) fails on any console error, which is where CSP violations surface |
| Guessing the session | brute force | 256-bit random ids; bootstrap token printed only to the local console/URL; sessions die with the process | `tests/unit/test_security.py` |
| Path traversal / symlink escape on download | `GET /artifacts/{run}/{id}` | downloads by registered artifact id; the resolved path (symlinks followed) must stay inside the run directory | `test_studio_api.py::test_artifact_download_by_id_only`, `test_artifact_symlink_outside_the_run_is_refused` |
| Reading files outside the import roots | `../` or a symlink inside a root pointing elsewhere | every source path is resolved and must stay inside its root; a symlink that leaves the root is neither listed nor readable | `tests/unit/test_datasets_service.py::test_import_roots_refuse_traversal_and_unknown_roots`, `test_import_root_symlink_escapes_are_invisible_and_unreadable` |
| Malicious experiment package | traversal member names, symlink/device members, decompression bombs, member floods, damaged files | member names are checked lexically and after resolution; entries typed as anything but a regular file or directory are refused; every member is read at most one byte past its manifest-recorded size (hash and size verified before any write, and again bounded during extraction); at most 10 000 members; the manifest is capped at 8 MB; the declared total must fit the free space of the workspace volume; nothing is written before every check passed | `test_symlink_members_are_refused`, `test_members_larger_than_recorded_are_refused_early`, `test_member_count_is_bounded`, `test_traversal_members_are_refused_before_any_write`, `tests/integration/test_packages.py::test_damaged_packages_are_refused_with_a_specific_diagnostic` |
| Arbitrary code via checkpoints or data files | pickled objects in `.pt`/`.npy` | every checkpoint goes through one loader, `opendpd.services.legacy_adapter.load_checkpoint`, which uses `torch.load(weights_only=True)` and turns a refusal into a workspace error; there is **no** trusted/unrestricted loading path anywhere in the tree (legacy `main.py` checkpoints are plain state_dicts and load under the same restriction); NumPy sources are opened with `allow_pickle=False` and object arrays are refused | `test_malicious_checkpoints_are_refused_not_executed`, `test_legacy_state_dict_checkpoints_still_load_under_the_restriction`, `test_no_unrestricted_pickle_loading_in_the_tree`, `test_datasets_service.py::test_numpy_imports_and_object_arrays_refused`, `tests/golden/test_legacy_checkpoint_loads.py` |
| Shell injection | API → worker | workers are started with an argument list from validated structured input; no shell, no user strings in commands | code review; `Supervisor._spawn` |
| Resource exhaustion by request bodies | huge JSON, chunked bodies without `Content-Length`, huge uploads | 2 MB cap on API bodies enforced by header *and* by counting the bytes as they arrive (a 413 either way); uploads stream to disk under a 2 GB cap and the partial file is removed on overflow (413) | `test_studio_api.py::test_payload_too_large`, `test_chunked_bodies_without_content_length_are_capped`, `test_oversized_uploads_are_413_and_leave_no_file` |
| Resource exhaustion by data | unbounded logs, event replays, huge captures | logs are paged by byte offset and the viewer keeps a bounded window in the DOM; run listings are paged and searched on the server; binary captures are memory-mapped and converted in chunks; the dataset doctor analyses a bounded central window and says so | `test_studio_api.py::test_logs_are_paged`, `test_runs_are_paged_and_searchable_on_the_server`, `test_datasets_service.py::test_numpy_imports_stream_from_the_source_without_copies`, `test_doctor_analyses_a_bounded_central_window_and_says_so`; measured in `docs/releases/performance-report.md` |
| Accelerator held by another process | CUDA/MPS allocation failures | the failure is classified as `device_busy_or_out_of_memory` with a hint; the run is never retried or moved to another device silently | `test_accelerator_refusals_are_classified_with_a_hint` |
| Leaking secrets in URLs/logs | bootstrap token | token exchanged once via `GET /bootstrap` then removed by redirect; never logged by the app; API responses are `Cache-Control: no-store` | `test_studio_api.py::test_bootstrap_redirect_strips_token`, `test_every_response_carries_the_security_headers` |
| Leaking private data in outward packages | share packages | share packages leave user data, worker logs and machine paths out and list what was removed under `redaction` | `tests/integration/test_packages.py::test_share_package_is_redacted_and_says_what_is_missing` |

## Dependency and code scanning

`pip-audit` and `npm audit` are run for every release candidate; results and
dates are recorded in `docs/releases/hardening-report.md`. A finding blocks the
release unless it is recorded there as an accepted residual risk with a reason.

## Accepted residual risks (first version)

- Any local process running as the same OS user can read the workspace and
  the console output; OS user isolation is the boundary.
- HTTP without TLS on loopback: traffic never leaves the host.
- Sessions live in memory; a server restart requires re-opening the URL
  printed by the launcher.
- The CSP allows inline *styles* (`style-src 'unsafe-inline'`): Plotly and
  MUI set element styles at runtime. Scripts remain same-origin only, so
  a style injection cannot execute code.
- Plotly renders a small HTML subset (`<b>`, `<i>`, `<br>`, links) in legend
  and hover text. Trace names come from the server's plot artifacts (roles
  such as "PA output"), never from user text, so this is not reachable today;
  keep it that way when adding user-named traces.
- Provenance files keep the exact command that was executed, including the
  absolute dataset path of that machine. They are historical records shipped
  only in *full* packages, which are private by definition.
