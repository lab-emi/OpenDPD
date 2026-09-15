# OpenDPD Studio 2.2.8

Studio starts with less JavaScript, sends fewer idle requests and releases
abandoned workspace resources without blocking other visitors' HTTP requests.

- **Lighter startup:** load each tool when its page opens. The browser's first
  Home visit loads about 0.82 MB of JavaScript/CSS instead of 1.45 MB in 2.2.7.
  Plotly and LaTeX remain available when their views need them.
- **Fewer idle requests:** hosted idle run lists and counts refresh every
  30 seconds. Active run lists retain ten-second updates and the existing event
  feed. Navigating away cancels obsolete reads and frees their request slots.
- **Cheaper shared status:** empty workspaces need no SQLite job-count reads.
  Session authentication uses a direct admission-ticket lookup. Cleanup moves
  worker shutdown, database close and directory deletion off the API event loop.
- **Trial inactivity rule:** **2 hours without user activity clears every
  temporary workspace created from that IP**. Foreground Studio interaction
  renews that group; background polling and running jobs do not. Workspaces
  retain independent credentials and files even on a shared network.
- **Clear deadlines:** the start prompt explains the rule. The top bar shows
  the earlier of inactivity expiry and the existing 11:55/23:55 UTC scheduled
  cutoff. A tab checks the server before signing out in case another workspace
  renewed its group. All data still has a maximum 12-hour retention window.

Install with `uv pip install "opendpd==2.2.8" --torch-backend=auto`, or use
[Studio on the web](https://opendpd.com/studio/).

The hosted service retains 256 workspace slots, 1,024 waiting tickets and one
shared compute slot. This release does not change scientific metrics or training
algorithms. Local Studio keeps persistent workspaces with no inactivity cleanup.
See [measurements and limits](../performance/studio-2.2.8.md),
[cleanup details](../guides/server-load.md), and
[deployment policy](../architecture/public-studio.md).
