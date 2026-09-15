# Server load and workspace cleanup

Open **Server load** in the sidebar to see current resource usage and shared compute availability. The page refreshes every five seconds while visible and stops polling when you leave it. Use **Refresh** for an immediate reading.

| Reading | Meaning |
| --- | --- |
| Active users | An estimate: authenticated sessions with API activity in the last five minutes. One person can have multiple sessions; this is not a count of distinct people. |
| Running jobs | Experiments running or stopping across all active workspaces. |
| Queued jobs | Experiments waiting for the shared training slot. |
| Temporary workspaces | Unexpired sessions, including currently inactive ones, versus the configured capacity (256 on the hosted service). |
| Waiting visitors | Live waiting-room tickets; these do not yet own a workspace or a compute slot. |
| API server | CPU and memory inside the public service's virtual machine. |
| Compute host | Total host CPU, RAM, GPU 0 utilization and GPU memory, including workloads outside OpenDPD. |

Local Studio shows **Local machine** with its CPU, RAM and NVIDIA GPU 0 when available. It shows a dash for active users because local browser sessions are not a reliable user count. GPU utilization telemetry requires `nvidia-smi`; a dash on a CPU/MPS machine does not mean that its accelerator is disabled.

Samples older than 20 seconds show **Stale** and hide numerical utilization until collection resumes. Missing values and failed job-count collection show a dash, never a fabricated zero. After an API error the page backs off to 30 seconds. Aggregate telemetry contains no usernames, process names, IP addresses, dataset details or workspace paths.

![Local Studio resource status](../../pics/studio-server-load.png)

## Cleanup timestamp

Public Studio shows one persistent cleanup timestamp in the top bar, for example **2026-09-15 11:55:00 UTC**. Its tooltip explains the corresponding local time. Mobile screens place it on a separate compact top-bar row.

Access expires and cleanup starts at the displayed time. Download your datasets, checkpoints and results beforehand. All hosted sessions expire at the next **11:55 or 23:55 UTC** cutoff. Sweeps check expiry every five seconds and wait for active requests before removing their files; independent resets at **11:59 and 23:59 UTC** stop remaining workers and clear temporary storage. New sessions resume after **12:00 or 00:00 UTC**. This keeps data within a 12-hour window; a visitor arriving shortly before cleanup gets less time. Visiting again does not extend the deadline. The local app uses persistent workspaces and has no scheduled deletion.

See [hosting architecture](../architecture/public-studio.md) for the operator limits and isolation model.

## Waiting for a workspace

![Studio waiting room](../../pics/studio-waiting-room.png)

The screenshot uses a controlled two-slot test service to exercise the same waiting-room UI; the production limit is 256.

The hosted service supports 256 open workspaces and a bounded waiting room of 1,024 tickets. If every slot is occupied or shared storage is low, **Start a temporary session** shows your queue position and automatically retries every five seconds. Keep the tab open. Refreshing the same tab keeps the ticket; after three minutes without a successful heartbeat it expires and a later request rejoins at the back. A service restart also clears the waiting room, and connected tabs rejoin automatically. No exact wait time is promised because other visitors can keep their workspaces until cleanup.

Waiting tickets are separate from the training queue and allocate no dataset directory or background supervisor. Once admitted, your experiments still share one compute slot and run in order. Increasing workspace capacity does not multiply GPU throughput.

On **Server load**, choose **End workspace**, then confirm after downloading anything you need. This stops your jobs, deletes your temporary data and frees a slot without waiting for the next scheduled cutoff. Closing a browser tab alone does not delete its workspace.
