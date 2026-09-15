# OpenDPD Studio 2.2.7

Hosted Studio previously exhausted its 16 workspace slots even when no jobs
were running. Version 2.2.7 supports **256 open workspaces**, an automatic
waiting room, and cleanup every **12 hours**.

- **Waiting room:** up to 1,024 lightweight tickets, a visible position,
  automatic admission, page-refresh recovery and explicit cancellation.
  Disconnected tickets expire after three minutes. Shared-storage pressure and
  scheduled cleanup also use the waiting room.
- **Lower idle cost:** one shared dispatcher services submitted work; idle
  workspaces no longer create supervisor and sweep threads. Heavy analysis and
  GPU concurrency stay bounded independently of workspace capacity.
- **Cleanup:** access expires at the next 11:55 or 23:55 UTC deadline shown in
  the top bar. Independent resets at 11:59 and 23:59 remove remaining data.
  Late arrivals get less than 12 hours. The Server load page offers an explicit
  **End workspace** confirmation to stop jobs, delete data and release a slot.
- **Shared compute:** FIFO job scheduling remains global, with 128 pending
  jobs maximum and two per workspace. More workspace slots do not imply more
  simultaneous GPU jobs.
- **Admission protection:** queue capabilities never enter URLs or query
  caches, cannot access datasets or GPU endpoints, and support safe retries
  after a lost admission response. Network, request and storage budgets remain
  enforced before materialization.

Install with `uv pip install "opendpd==2.2.7" --torch-backend=auto`, or use
[Studio on the web](https://opendpd.com/studio/).

See [waiting-room behavior](../guides/server-load.md),
[capacity validation](../performance/studio-2.2.7.md), and the
[deployment policy](../architecture/public-studio.md). Local Studio retains
persistent workspaces and has no hosted waiting room or scheduled deletion.
