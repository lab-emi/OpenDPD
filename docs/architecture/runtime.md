# Task runtime (S03)

One supervisor thread inside the Studio server owns a queue and spawns **one
worker subprocess per run**. The SQLite store (`workspace/metadata.sqlite`)
is the only authority for run status; files in the run directory are the
worker's own outputs and are re-read by the supervisor when the worker exits.

## State machine

```
queued -> running -> succeeded
                  -> failed
                  -> cancel_requested -> cancelled
                  -> interrupted
queued -> cancelled | interrupted
```

`cancel_requested` is shown until the worker has actually exited. Terminal
states never change: a completion observed after cancellation records
`cancelled` with the reason "worker finished after cancellation was
requested; artifacts and result were kept". A retry creates a *new* run with
`parent_run_id`; the failed record is kept.

## Worker protocol

| Channel | Direction | Content |
|---|---|---|
| `python -m opendpd.runtime.worker --workspace W --run-id R` | server → worker | started with `cwd=<run_dir>`, `stdin` closed, its own session/process group |
| `<run_dir>/events.jsonl` | worker → server | one JSON object per line: `{"ts": ISO-8601, "type": <RunEventType>, "payload": {...}}`; flushed per line; heartbeat every 5 s |
| `<run_dir>/logs/worker.log` | worker → file | stdout + stderr of the legacy trainer (never parsed) |
| `<run_dir>/CANCEL` | server → worker | cooperative stop request, polled at the end of every epoch |
| exit code | worker → server | 0 succeeded, 1 failed, 3 cancelled; anything else = died |

Event payloads:

| type | payload |
|---|---|
| `progress` | `{"epoch": i, "total_epochs": n, "phase": "epoch_end"}` |
| `metric` | `{"epoch": i, "split": "val"\|"test", "values": {"NMSE": …, "EVM": …, "ACLR_L": …, "ACLR_R": …, "ACLR_AVG": …}, "train_loss": …}` |
| `heartbeat` | `{"pid": …}` |
| `artifact` | `{"artifact_id": …, "kind": …, "path": …}` |
| `error` | `RunError` fields |
| `status` | written by the store, not the worker: `{"from": …, "to": …, "reason": …}` |

The store assigns `seq` (1, 2, 3 … per run) when it ingests a line. Clients
resume with `after=<seq>`; because nothing is pruned in this version,
`first_seq` is always the start and replay never needs a snapshot.

## Cancellation and cleanup

1. `cancel(run_id)` on a queued run → `cancelled` immediately.
2. On a running run → `cancel_requested`, `CANCEL` file written (returns in
   well under a second).
3. The worker stops at the next epoch boundary and exits with code 3; the
   partial checkpoint and logs stay registered.
4. If it is still alive after `cancel_grace` (default 30 s), the supervisor
   terminates the whole process tree (`psutil`), waits, then kills.

Worker identity is `pid` **plus process creation time**; a recycled PID is
never mistaken for a live worker.

## Restart and crash recovery

At start-up `Supervisor.recover()` marks every `running` /
`cancel_requested` / `queued` run as `interrupted` with a reason. An
orphaned worker that is still alive (matching pid and creation time) is
terminated first, and any events it wrote to `events.jsonl` are ingested.
Nothing is resumed automatically.

`Supervisor.stop(timeout)` (Ctrl-C path): stop accepting runs, write
`CANCEL` to every active worker, wait at most `timeout`, terminate the rest
and record them as `interrupted` ("service shut down while the run was in
progress").

## Limits

- Serial per device (`max_per_device=1`); other devices run in parallel.
- Workspace preflight (writable, ≥ 500 MB free) runs at submission, not at
  the end of training.
- Log lines stay in `worker.log`; only structured events enter SQLite.
