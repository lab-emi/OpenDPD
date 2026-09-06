# OpenDPD Studio: one command to the workbench

```bash
pip install "opendpd[gui]"
opendpd gui
```

`opendpd gui` starts a local service on `127.0.0.1`, waits until it answers,
prints a one-time URL such as `http://127.0.0.1:8765/bootstrap?token=…` and
opens it in your default browser. No Node.js, no second terminal, no
copying of addresses. Press Ctrl+C to stop; running workers are terminated
and the workspace lock is released.

| Option | Effect |
|---|---|
| `--workspace DIR` | where datasets, runs and results live (default `$OPENDPD_WORKSPACE` or `~/opendpd-workspace`) |
| `--port N` | use exactly this port (fails if busy); without it the first free port from 8765 is used |
| `--no-browser` | print the URL only (SSH sessions, servers without a desktop) |

Starting `opendpd gui` again for the same workspace while one is running
opens the existing instance instead of a second server.

## What the browser shows

1. **Home**: the workspace path and the built-in example (measured DPA 200 MHz
   data). Register it with one click.
2. **Experiments → New**: pick a recipe (smoke recipes prove the pipeline in a
   minute and say so; research recipes are the paper settings), a dataset and
   a device. The server validates every change; errors are shown on the field.
3. **Run detail**: live progress, per-epoch metrics, logs, artifacts and the
   resolved configuration. Refreshing the page never resubmits anything.
4. **Results**: metrics with units and their evidence type (PA model, DPD on
   the surrogate, DPD measured). Mock data used for interface work is always
   badged **MOCK** and cannot be exported.

## When something is wrong

- `opendpd doctor` prints versions, whether the frontend assets are present,
  the workspace state and a free port, and lists every blocking problem.
- If the page says the frontend is not available, the installed wheel was
  built without the frontend (source checkout without `npm run build`);
  install a release wheel or run `cd frontend && npm ci && npm run build`.
- Lost session (server restarted, cookie cleared): open the URL printed by
  `opendpd gui`, or paste its token into the "Session required" page.
- The service only ever listens on loopback. For remote machines use an SSH
  tunnel (`ssh -L 8765:127.0.0.1:8765 host`) and `--no-browser` on the host.

## Platform status

See `docs/releases/support-matrix.md`. Linux is verified by the packaged
test (`tests/packaging/test_wheel_install.py`); macOS and Windows launches
must be verified by a person on a real desktop before they are listed as
supported.
