# OpenDPD 2.2.6

Studio adds server-load visibility and a single precise cleanup timestamp, while strengthening shared-service isolation and resource admission.

## Changes

- **Server load** shows estimated active users, running/queued experiments, workspace capacity, CPU, RAM, GPU utilization and GPU memory. The public API VM and compute host have separate panels. Local Studio reports its own machine.
- One background sample every five seconds serves every viewer. Stale or unavailable readings are clearly marked; leaving the page stops its polling. Only aggregate data is exposed.
- The top bar shows the exact scheduled workspace cleanup start in UTC. Repeated per-page temporary-workspace alerts are removed, with a compact mobile layout.
- Directory identifiers are validated before filesystem writes. The public API explicitly blocks local import-root discovery and About no longer initiates an outbound activity request.
- Workspace mutations serialize; expensive analysis/generation/export work has a global concurrency limit. Heavy writers reserve temporary storage before accepting data, and cleanup scans no longer block the event loop.
- Duplicate security headers and malformed loopback hosts/origins are rejected. Local sessions have bounded server-side lifetime and count. Private GPU telemetry is authenticated, validated and size-limited.
- Dependency floors include available security fixes. The GPU runtime uses a pinned PyTorch 2.14/CUDA 13.2 base, Ubuntu updates and fewer build/install tools. Frontend and resolved Python dependency audits run in CI.
- README, installation and hosting guides, load documentation and GUI screenshots are updated.

## Install

Follow the [uv installation steps](../install.md), then install or upgrade in your environment:

```sh
uv pip install --python .venv "opendpd==2.2.6" --torch-backend=auto
uv run --no-project --python .venv opendpd gui
```

[Hosted Studio](https://opendpd.com/studio/) · [Server load](../guides/server-load.md) · [Validation](../performance/studio-2.2.6.md) · [Security review](security-review-2.2.6.md)
