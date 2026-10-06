# Hosted Arena evaluation

Hosted Arena submissions use the existing private GPU broker and its pinned,
isolated worker image. They share FIFO scheduling, the global compute semaphore,
session/IP/service run budgets, and workspace expiry with ordinary Studio jobs.
An Arena evaluation trains a backbone at up to four parameter budgets, so it has
a **7-day wall-clock guard**; ordinary Studio jobs retain their 30-minute limit.
This operational allowance does not change the frozen 240 complete epochs and
94,320 optimizer updates per seed and configuration on APA_200MHz_b, the data,
seeds, validation policy, or scoring gates. The API VM does not run Arena training.

Enable submissions with `OPENDPD_WEB_ARENA_SUBMISSIONS=1` on the web service, or
`WebConfig(arena_submissions=True)` when embedding it. The default is disabled.
The service must also have its existing private GPU token configured. Build and
pin a worker image containing the same OpenDPD version, Arena runner, protocol,
and packaged calibration assets as the API service. Publishing an image or
restarting a service is an operator action; setting the flag alone does not
perform either action.

The Arena allowance is bounded by `WebConfig.arena_max_runtime_seconds` (604,800
seconds by default and at most); reducing it does not change the ordinary-job
setting. Local Studio and the official benchmark launcher use the same 7-day
Arena guard. A running worker's original start remains the deadline origin when
an operator hands it to a replacement supervisor; supervision does not restart
training or add epochs.

At startup the host agent probes Arena support inside that pinned image and
verifies the packaged data and PA weight hashes. It advertises the protocol hash
over the existing private broker connection. The submission API and Studio's
Submit view become available only while the live agent advertises the exact
protocol accepted by the API. An older image, missing assets, a stale heartbeat,
or a different protocol keeps submission unavailable while rankings remain
readable.

Each required condition and seed consumes one unit of the existing run quota,
and that unit trains the case at every budget of the parameter sweep. Thus a
neural entry uses three units on the APA_200MHz_b board,
and a deterministic baseline uses one unit. Each entry
trains up to four models per unit and can occupy the shared worker for up to
the 7-day guard. Operators who enable hosted submissions should size
`runs_per_session`, `runs_per_ip` and `runs_per_day` with that in mind. A
request for a stale protocol or an unknown backbone is refused before any
budget is charged. Both ordinary runs and
Arena entries count toward the same pending-job limits; opening another session
does not reset the IP or service budget.

The broker sends only the server-validated request and optional bounded template
definition. Calibration data and evaluators come from the read-only image.
Container output is limited to the result, progress, and worker log; it cannot
overwrite the authoritative request or submission. The API checks the returned
model identity, re-derives the registered sweep and recomputes the operation
counts, scores, required case coverage, frozen PA hashes, and the output-power gate
before displaying a rank. No host is timed, so a hosted result is directly comparable
with the shipped rows. Closing or expiring the workspace
cancels queued and active evaluations, and late results cannot restore them.

Results remain private to the temporary workspace and disappear with its normal
cleanup. A hosted submission does not publish a global leaderboard record.

## Isolated container smoke test

Hosted submissions are off by default. Before enabling them on a deployment, run one
real four-configuration MP submission through the pinned GPU image and confirm that
it completes: the host agent unpacks the broker's input, the unprivileged,
network-isolated container writes its result, and only the selected output files
are transferred back. At startup the agent already runs the pinned image's
exact-protocol capability check and a cold CUDA probe. Automated tests cover the
agent and broker transfer contract with the container stubbed
(`tests/unit/test_arena_gpu_adapter.py`); they do not start a container or use a GPU.

The v6 local and hosted runners prefer CUDA when available, including
zero-threshold Delta cells with reviewed dense training equivalents. The
distributed reference launcher may assign selected fits to CPU and records the
actual host/device. TF32 is disabled. These execution paths have dedicated
forward, gradient, optimizer and inference checks. The container smoke checks the
execution boundary; the complete 218-fit reference has a separate independent audit.
