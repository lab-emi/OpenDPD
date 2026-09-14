# Studio 2.2.6 validation

Validation date: 2026-09-14. Resource values are real host measurements; synthetic PA/DPD runs establish software behavior, not physical RF performance.

## Application and UI

- Python regression: 951 tests passed in the complete run; its API snapshot comparison was rerun successfully after the new nullable job-count fields were exported. Additional cleanup-failure and request-boundary tests were then run against the final implementation. The release PR runs the complete configured CI matrix again.
- Frontend: 234 tests in 55 files passed, with strict TypeScript and lint checks. OpenAPI generation and the committed TypeScript contract agree.
- Real Chromium over the installed shared CDP browser: local telemetry; web layout at 1366, 390 and 360 pixels; one top-bar UTC timestamp; no old per-page alert; no horizontal overflow or toolbar collisions. Stale samples hide utilization, and navigating away stops resource polling. The web layout check used a local API adapter; production verification is recorded separately with the release.
- Current Home and local load-page [screenshots](studio-2.2.6/screenshots.json) were captured from a clean workspace. Earlier unchanged signal-chain and ILC images retain their 2.2.5 provenance.
- Security regression covers directory traversal, local-only route denial, cross-session access, private telemetry authentication/schema/body limits, expired sessions, duplicate/malformed headers, concurrent writes, heavy-work limits and low temporary storage. GPU-cleanup failure must leave health unavailable while expired tenants are still removed.

## Resource overhead and GPU execution

The [loopback HTTP check](studio-2.2.6/status-http.json) made 100 authenticated status reads while other Python tests were running: median **1.01 ms**, P95 **1.56 ms**, maximum **1.87 ms**, and at most **557 bytes** per local response. This measures cached endpoint latency on this host, not public Internet latency or a capacity benchmark. Tests verify that reading the cache never invokes `nvidia-smi`; one sampler per machine runs every five seconds with a two-second GPU-command timeout. Public job-count collection is shared across viewers for five seconds.

The [GPU smoke record](studio-2.2.6/gpu.json) uses the same non-root, network-free, read-only container command and resource limits as production. It runs one-epoch GRU PA training, GRU DPD training and default ILC/ILA against 32,768 synthetic pairs, with 6,452 held-out test samples. The Ideal test-feedback waveform remains separate from fitted DPD training. These are execution checks, not convergence or quality guarantees.

## Advisory and release checks

[Security review](../releases/security-review-2.2.6.md) and [scan summary](studio-2.2.6/security-scan.json) document remaining findings without suppressions. Frontend and actual Python environment audits accompany the full OS/container scan.

The GitHub release carries merged source, deployed API/agent/image, Pages build and published-wheel identities, followed by a public session and CUDA workflow check. Linux/macOS/Windows default-install CI verifies dependency resolution and headless browser fallback; native-window and MPS hardware coverage remain limited to the [support matrix](../releases/support-matrix.md).
