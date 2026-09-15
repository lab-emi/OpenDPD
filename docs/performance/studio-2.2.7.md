# Studio 2.2.7 capacity validation

Date: 2026-09-15. Measurements use the candidate source over real loopback HTTP
on the development host. The benchmark uses a fixed daytime clock and private
temporary storage; it does not consume production workspace slots.

| Measurement | Observed |
|---|---:|
| Open empty workspaces | 256 |
| Live waiting-room tickets | 1,024 |
| Process RSS with both | 225.97 MiB |
| Public background threads | 2 (sampler and shared dispatcher) |
| Idle CPU, percent of one core | 0.50% |
| Workspace creation median / p95 | 2.45 / 6.00 ms |
| Workspace creation maximum, including cold initialization | 790.26 ms |
| Status reads, 100 requests at concurrency 8, median / p95 | 11.21 / 20.19 ms |
| Status reads maximum | 22.76 ms |
| Admission beyond the waiting-room limit | HTTP 429, bounded queue |

The full Python regression was running separately during the benchmark.
These are empty-workspace admission and status measurements, not a guarantee
of 256 simultaneous training jobs or an Internet latency SLA. The production
deployment retains one compute slot, two heavy API operations, a 2 GiB shared
tmpfs, 256 MiB per workspace and its existing CPU/RAM/container limits.

[Machine-readable measurement](studio-2.2.7/admission.json).
Reproduce from the repository with:

```sh
PYTHONPATH=. python scripts/benchmark_web_admission.py --output /tmp/admission.json
```

## Regression coverage

The 59-test focused public boundary/GPU/telemetry/admission run passed, including
actual ASGI requests for FIFO admission, lost-response replay, network/global
limits, queue credential isolation, cancellation racing with admission,
abandoned-ticket expiry, exact noon cleanup and reopening, low-storage waiting,
and 32 idle workspaces without per-visitor threads. Existing real CPU training
and private GPU lease/cancellation regressions also passed.

The 238 frontend tests passed, including waiting-position rendering, automatic
admission, cancellation and rejection of a late session response after leaving
the queue. Type checking, lint and regenerated API types passed.

Chromium connected to the real candidate HTTP API under a controlled two-slot
policy. Five independent browser contexts verified FIFO positions, automatic
entry after another user confirmed End workspace, cancellation, preservation of
the ticket across refresh, and recovery after a dropped network request. Desktop,
390 px and 360 px screenshots had no document overflow or JavaScript errors.
See [browser evidence](studio-2.2.7/browser.json).

The new pinned GPU image reports OpenDPD 2.2.7 and PyTorch 2.14.0+cu132 and passed
a real CUDA tensor operation on the host GPU. Full-regression and production
workflow evidence is added to the release record after verification. Synthetic
PA/DPD runs validate the software workflow; they are not measured RF hardware results.
