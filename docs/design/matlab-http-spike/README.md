# Feasibility spike for the MATLAB HTTP transport

The scripts behind section 3 of [`../matlab-http-transport.md`](../matlab-http-transport.md). They are a measurement aid, not
part of the toolbox or the product, and they change nothing outside a temporary directory.

| File | What it does |
| --- | --- |
| `run_spike.sh a` / `b` | Starts what the part needs, runs MATLAB in batch mode, stops everything on exit |
| `real_service.py` | Starts a scratch Studio service through the SDK and stops it when asked (part A) |
| `spike_http_a.m` | Part A: mints a bootstrap token, opens a session, checks the cookie, CSRF and Host rules with `matlab.net.http` only |
| `standin_server.py` | A throwaway Starlette server with `POST /infer` (`.npy` in, `.npy` out, `y = 0.5·x`), `GET /big` and `POST /sink` (part B) |
| `spike_http_b.m` | Part B: round trips of 32 and 128 MiB, raw uploads of 64 and 512 MiB, 30 small calls; prints time and extra peak memory per variant |

The stand-in is **not** OpenDPD's server: part B measures MATLAB's HTTP client and the loopback path only. Memory figures read
`/proc` and therefore need Linux. Results on MATLAB R2026a Update 5 (Linux, 32 cores) are in the design document; other
machines will differ, and Windows and macOS were not tried.
