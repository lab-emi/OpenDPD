# Studio 2.2.6 security review

Review date: 2026-09-14. Scope: local/public HTTP boundaries, workspace isolation, uploads and filesystem writes, browser rendering, shared resource admission, cleanup, the private GPU bridge and its deployed runtime dependencies. This is a code review with adversarial regression tests and dated dependency scans, not independent penetration-test certification.

## Findings addressed

| Area | Change and verification |
| --- | --- |
| Filesystem traversal | Validate request identifiers before creating dataset/version directories and training output paths; re-check at service entry points. Tests include absolute paths, separators and `..`, with no outside writes. Includes the contribution in PR #38. |
| Public route disclosure | Explicitly deny `/datasets/import-roots`; a route-surface test detects static local endpoints accidentally matched by identifier patterns. |
| Request ambiguity | Reject duplicate security headers, malformed loopback hosts, ports and origins. Error responses retain security headers and a single `no-store` policy. CSRF secret comparisons use constant-time byte comparisons. |
| Local sessions | Enforce seven-day expiry server-side and cap the session store at 128. The launcher's bootstrap secret remains valid for that process so reopening its URL can renew a session; it is not described as single-use. |
| Resource exhaustion | Serialize workspace mutations, cap heavy API operations at two globally and reserve storage atomically before receiving heavy writes. Reads and cancellation remain available. GPU containers retain CPU/memory/PID/time limits; the host agent additionally caps CPU and tasks. |
| Cleanup availability | Run storage scans off the event loop, tolerate atomic file replacement and serialize sweeps. GPU-finalization errors now mark health unavailable and stop new-session admission while tenant expiry continues. Unexpected maintenance failures no longer silently terminate the periodic task. |
| Telemetry privacy | Expose only bounded numeric aggregates to authenticated sessions. No process names, IPs, identifiers, paths or dataset details. The private resource endpoint requires a separate bridge secret, forbids browser origins and caps payloads at 4 KiB. Missing or stale samples are not reported as zero. |
| Browser/outbound requests | Strengthen the web bundle's CSP base-URI policy and HTTPS upgrade behavior. Public About makes no GitHub activity request. Existing React escaping and restricted KaTeX rendering remain covered by adversarial tests. |
| Dependencies | Raise affected minimum versions, update the pinned PyTorch/CUDA image and Ubuntu packages, remove unused compiler/header packages and package installers, and add frontend/Python advisory checks to CI. |

The existing public route allowlist continues to deny arbitrary filesystem imports, uploaded checkpoints/code, shell commands, RF control and executable exports. CSV admission scans every row in quarantine with byte/sample limits. NumPy loads forbid pickled arrays; checkpoint loads use the restricted weights loader. Cross-session reads, cancellation and private GPU calls remain denied in integration tests.

## Dependency results

The pre-upgrade GPU image audit reported 89 advisory matches across 16 Python packages, including aliases for some vulnerabilities. The updated image's installed Python distributions and frontend lockfile have no known advisory matches in the dated scans. The public API environment was also audited separately.

The final container OS scan reports **no critical/high findings and no vendor-provided fixes left unapplied**. Remaining medium/low findings are retained in the [machine-readable scan summary](../performance/studio-2.2.6/security-scan.json), rather than suppressed. They affect Ubuntu runtime libraries/utilities, including Expat, SQLite, glibc, zlib and system utilities. OpenDPD's public upload path accepts numeric CSV, not XML, databases, archives or native executables. Those restrictions and the offline, read-only, non-root worker reduce reachable attack surface; they do not make a vulnerable system library universally safe.

The scan summary records the image identity, severity totals, package versions, advisory identifiers and available fixes. PyTorch vendor suffixes such as `+cu132` are mapped to their upstream release for advisory lookup. Actual import precedence is respected when both system and environment metadata exist. Only OpenDPD itself is omitted from package-index lookup because its source is reviewed here.

## Deployment boundaries and residual limits

The API and VM administration forwards bind host loopback. The guest cannot initiate Internet/LAN connections; the public API accepts the configured HTTPS tunnel and exact frontend origin. API responses are non-cacheable, and the static frontend uses a restrictive CSP. The API process runs as an unprivileged user with read-only source and bounded tmpfs. GPU jobs have no network, host sockets or capabilities, and run as UID 65532.

The host GPU agent needs privilege to create Podman namespaces and expose the GPU device. It listens on no socket and pulls only through the authenticated private bridge; the experiment container drops those privileges. The GPU is shared rather than hardware-partitioned. Host kernel, GPU driver, hypervisor, DNS/CDN account security and volumetric denial of service remain operator responsibilities and are not certified by an application/image scan. The review did not rotate unrelated credentials or modify the host operating system.

See [validation](../performance/studio-2.2.6.md) and [hosting architecture](../architecture/public-studio.md). Review criteria follow the [OWASP REST security guidance](https://cheatsheetseries.owasp.org/cheatsheets/REST_Security_Cheat_Sheet.html) and [file-upload guidance](https://cheatsheetseries.owasp.org/cheatsheets/File_Upload_Cheat_Sheet.html); resource readings follow [psutil semantics](https://psutil.readthedocs.io/en/latest/) and [NVIDIA's utilization definitions](https://docs.nvidia.com/deploy/nvidia-smi/).
