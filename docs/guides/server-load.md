# Server load and workspace cleanup

Open **Server load** in the sidebar to see current resource usage and shared compute availability. The page refreshes every five seconds while visible and stops polling when you leave it. Use **Refresh** for an immediate reading.

| Reading | Meaning |
| --- | --- |
| Active users | An estimate: authenticated sessions with API activity in the last five minutes. One person can have multiple sessions; this is not a count of distinct people. |
| Running jobs | Experiments running or stopping across all active workspaces. |
| Queued jobs | Experiments waiting for the shared training slot. |
| Temporary workspaces | Unexpired sessions, including currently inactive ones, versus the configured capacity. |
| API server | CPU and memory inside the public service's virtual machine. |
| Compute host | Total host CPU, RAM, GPU 0 utilization and GPU memory, including workloads outside OpenDPD. |

Local Studio shows **Local machine** with its CPU, RAM and NVIDIA GPU 0 when available. It shows a dash for active users because local browser sessions are not a reliable user count. GPU utilization telemetry requires `nvidia-smi`; a dash on a CPU/MPS machine does not mean that its accelerator is disabled.

Samples older than 20 seconds show **Stale** and hide numerical utilization until collection resumes. Missing values and failed job-count collection show a dash, never a fabricated zero. After an API error the page backs off to 30 seconds. Aggregate telemetry contains no usernames, process names, IP addresses, dataset details or workspace paths.

![Local Studio resource status](../../pics/studio-server-load.png)

## Cleanup timestamp

Public Studio shows one persistent cleanup timestamp in the top bar, for example **2026-09-14 23:55:00 UTC**. Its tooltip explains the corresponding local time. Mobile screens place it on a separate compact top-bar row.

Access expires and cleanup starts at the displayed time. Download your datasets, checkpoints and results beforehand. Normal sweeps run every 15 seconds and wait for active requests before removing their files; an independent reset at 23:59 UTC stops remaining workers and clears temporary storage. New sessions are available after 00:00 UTC. Visiting again does not extend the deadline. The local app uses persistent workspaces and has no daily cleanup.

See [hosting architecture](../architecture/public-studio.md) for the operator limits and isolation model.
