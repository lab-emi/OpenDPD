# Performance report (S13, G2 targets of plan §8.2)

Measured on 2026-09-06T14:11:37+00:00 at commit `68953a1ea1db` (dirty tree). Every number below was produced by `scripts/perf_report.py` against a real `opendpd gui` server on this machine; browser numbers come from `frontend/e2e/live.spec.ts` (Playwright) against the same server. Nothing is inferred. Phases re-run with `--phases` are merged into the same JSON; the date above is that of the latest run.

## Environment

| Item | Value |
|---|---|
| OS | Linux-7.2.2-1-cachyos-x86_64-with-glibc2.44 |
| CPU | 13th Gen Intel(R) Core(TM) i9-13900HX (32 threads) |
| RAM | 101.0 GB |
| Python / torch / numpy | 3.13.14 / 2.13.0+cu132 / 2.4.4 |
| OpenDPD | 2.2.0.dev0 (editable checkout in the development venv (pip install -e .)) |
| Node / Playwright | v26.8.1 / Version 1.63.0 |
| Browsers | chromium-1243, chromium_headless_shell-1234, chromium_headless_shell-1243, ffmpeg-1011, firefox-1543, webkit-2359 (headless (Playwright); no window manager involved) |
| Data | built-in `DPA_200MHz` (measured, 38,400 paired samples); history fixture: 1,000 cloned finished runs + one 50,000-line worker log; stress: 100,000,000 synthetic float32 paired samples |
| Workspace | `/home/cgao/.cache/opendpd-perf/ws` (local SSD) |

## Results against the frozen targets

| Metric | Target | Measured | Verdict |
|---|---|---|---|
| Command to service available | p95 ≤ 10 s | 0.39 s | pass |
| Service ready to first page interactive | p95 ≤ 2 s | 168.34 ms | pass |
| Ordinary interface feedback (tab switch) | p95 ≤ 200 ms | 108.21 ms | pass |
| Status/metadata API while a run trains | p95 ≤ 250 ms | 19.83 ms | pass |
| Cached chart re-render (enlarge) | p95 ≤ 300 ms | 195.77 ms | pass |
| Training status visible latency | ≤ 2 s | 0.36 s | pass |
| GUI overhead on training (paired) | median ≤ 5 % | 1.73 % | pass |
| Control-plane memory growth (server RSS, last 20 min) | ≤ 100 MB | 0.19 MB | pass |
| Stress import extra RSS (parse/convert process) | ≤ 1 GB | 2.91 MB | pass |
| Cancel request feedback | ≤ 1 s | 0.00 s | pass |
| 50 000-line log and 1 000 runs browse, search, page | usable; DOM windowed | log viewer keeps 26 rows in the DOM for 50,000 loaded lines; experiments page shows 50 rows (916 DOM nodes) of 1,000 runs; server search/paging p95 in the API table | pass |
| Offline use | no host but loopback | no external host: CSP `connect-src 'self'`, `tests/unit/test_offline_assets.py`, and the Playwright offline journey (`e2e/a11y.spec.ts`) abort every non-loopback request and still complete | pass |

## Details

### Startup (20 launches)

- command to `/readyz` ready: p50 0.37 s, p95 0.39 s, max 0.41 s; first launch of the session 0.39 s
- Ctrl+C to process exit: p50 0.21 s, p95 0.26 s
- warm launches of the installed environment (Python byte-code and OS file cache warm); the very first launch of the session is reported separately; a cold-cache start needs root to drop caches and was not measured

### API latency (ms per request, sequential, same machine)

| Endpoint | idle p50 | idle p95 | idle max | training p50 | training p95 | training max |
|---|---:|---:|---:|---:|---:|---:|
| `/api/v1/runs?limit=50` | 10.52 | 11.62 | 18.34 | 11.63 | 19.83 | 20.76 |
| `/api/v1/runs/count` | 8.90 | 12.26 | 15.80 | 10.07 | 16.17 | 17.21 |
| `/api/v1/runs/{run}` | 0.72 | 1.11 | 1.37 | 0.54 | 1.01 | 1.88 |
| `/api/v1/system/capabilities` | 1.15 | 1.59 | 640.29 | 0.45 | 0.96 | 2.40 |
| `/api/v1/datasets` | 1.49 | 1.86 | 2.17 | 0.98 | 1.31 | 1.39 |
| `/api/v1/runs/{run}/events/list?after=0&limit=100` | 1.59 | 2.47 | 2.84 | 0.85 | 1.35 | 2.73 |
| `/api/v1/runs/{big}/logs?offset=0&limit=200` | 0.94 | 2.56 | 2.81 | 0.71 | 1.47 | 2.99 |
| `/api/v1/results/{run}` | 2.14 | 2.71 | 2.88 | 0.76 | 1.33 | 3.76 |

100 samples per cell; the training column was taken while a 80-epoch run was active in the worker.

### Training status visibility

- 245 events observed; latency p50 0.189 s, p95 0.314 s, max 0.359 s (server receipt (poll every 100 ms) minus the worker's own event timestamp, same machine clock; only events emitted after the poll loop opened are counted)
- slowest events: 0.359 s (metric, seq 236, the poll itself took 0.001 s); 0.335 s (progress, seq 191, the poll itself took 0.001 s); 0.335 s (metric, seq 192, the poll itself took 0.001 s); 0.335 s (metric, seq 193, the poll itself took 0.001 s); 0.326 s (progress, seq 146, the poll itself took 0.002 s)

### Cancel

- POST cancel returned in 2.2 ms; `cancel_requested` visible after 3.6 ms; final status `cancelled` 2.03 s after submission (cooperative stop at the epoch boundary)

### GUI overhead on training (3 pairs, 30 epochs each)

| pair | order | CLI train (s) | GUI train (s) | overhead | CLI process (s) | GUI submit → done (s) | GUI queue wait (s) |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 | cli first | 6.95 | 7.12 | 2.5 % | 7.63 | 9.03 | 0.15 |
| 1 | gui first | 7.23 | 7.22 | -0.1 % | 7.98 | 9.04 | 0.16 |
| 2 | cli first | 7.10 | 7.22 | 1.7 % | 7.80 | 9.05 | 0.17 |

Diagnostic: the same pairs with an explicit budget of 8 threads on both legs (median -0.5 %, min -1.3 %, max 2.3 %). This is not the target figure; it shows how much of the overhead is thread oversubscription.

| pair | order | CLI train (s) | GUI train (s) | overhead |
|---|---|---:|---:|---:|
| 0 | cli first | 6.52 | 6.49 | -0.5 % |
| 1 | gui first | 6.65 | 6.56 | -1.3 % |
| 2 | cli first | 6.50 | 6.66 | 2.3 % |

median 1.7 % (min -0.1 %, max 2.5 %); same config and seed, both legs use torch default (physical cores); order alternates per pair; the CLI trains in its own process, the GUI path trains in a worker subprocess while the server ingests its events and this script polls the run once per second; train = wall clock between started_at and finished_at, stamped by the one executor both paths share (data loading, training, evaluation); lifecycle = everything around it (interpreter and torch start-up, queueing, exit detection, this script's polling), reported separately and not part of the target.

### Control-plane memory (30.0 min of load)

- server RSS 809.9 MB at start, 817.6 MB after warm-up, 817.8 MB at the end: net 0.2 MB over the measured window
- load: 16,281 requests, 90 runs submitted, 0 request errors; server process RSS only (workers are separate processes); load: ~10 API requests/s including log pages and event replays, one 2-epoch run every 20 s
- samples (s, MB): 0:809.9, 40:817.5, 80:817.5, 120:817.5, 160:817.5, 200:817.5, 240:817.5, 280:817.5, 320:817.5, 360:817.5, 400:817.5, 440:817.5, 480:817.5, 520:817.5, 560:817.5, 600:817.6, 640:817.6, 680:817.6, 720:817.6, 760:817.6, 800:817.6, 840:817.6, 880:817.6, 920:817.6, 960:817.6, 1000:817.6, 1040:817.6, 1080:817.6, 1120:817.6, 1160:817.7, 1200:817.7, 1240:817.7, 1280:817.7, 1320:817.7, 1360:817.7, 1400:817.7, 1440:817.7, 1480:817.7, 1520:817.7, 1560:817.7, 1600:817.7, 1640:817.7, 1680:817.8, 1720:817.8, 1760:817.8, 1800:817.8, 1800:817.8

- browser heap (Chromium, CDP `JSHeapUsedSize`) over 120 page switches: samples 186.6, 219.0, 260.1, 307.0, 347.2, 395.3, 447.3, 479.2, 531.7, 571.4, 98.8, 145.7 MB; floor of the second half minus floor of the first half -87.8 MB, peak 571.4 MB (the heap oscillates with garbage collection; a leak shows as a rising floor)

### Stress import

- 100,000,000 paired float32 samples (1.60 GB raw, `(2, n, 2)` .npy) imported in 168.5 s (exit code 0); peak anonymous memory 29.0 MB vs 26.1 MB baseline: extra 2.9 MB; file-backed (memory-mapped) pages peaked at 2311.1 MB and total RSS at 2340.3 MB (2286.2 MB above the baseline when the file cache is counted); peak anonymous memory (RssAnon, sampled every 20 ms) of the import process minus an interpreter that only imported the service; the file-backed pages of the memory-mapped source and outputs are the OS file cache (excluded by plan §8.2) and are reported separately; CSV split files are written chunk by chunk

### Browser (Playwright, real server)

- chromium-1366: exit code 0

```
Running 3 tests using 1 worker

[1/3] [chromium-1366] › e2e/live.spec.ts:42:1 › bootstrap, train the smoke recipe, read the result and export a share package
[2/3] [chromium-1366] › e2e/live.spec.ts:76:1 › timings: page load, tab switching, chart re-render, DOM size, heap growth
[3/3] [chromium-1366] › e2e/live.spec.ts:141:1 › a very long worker log stays windowed and searchable
  3 passed (1.3m)
```
- firefox-1366: exit code 0

```

[1/3] [firefox-1366] › e2e/live.spec.ts:42:1 › bootstrap, train the smoke recipe, read the result and export a share package
[2/3] [firefox-1366] › e2e/live.spec.ts:76:1 › timings: page load, tab switching, chart re-render, DOM size, heap growth
[3/3] [firefox-1366] › e2e/live.spec.ts:141:1 › a very long worker log stays windowed and searchable
  1 skipped
  2 passed (10.0s)
```

- page load to experiments table (20 navigations): p50 126 ms, p95 168 ms
- tab switch (100 samples): p50 88 ms, p95 108 ms
- chart enlarge re-render (100 samples): p50 174 ms, p95 196 ms
- long log: 50,000 lines loaded in 0.3 s (2 000-line pages), 26 rows in the DOM, filter finds a line near the end

## Not measured here

- Cold-cache start (needs root to drop the page cache) — only warm launches and the first launch of the session are reported.
- macOS, Windows, Safari and the system default browser: no such machine here (support matrix marks them unverified).
- GPU: the targets above are control-plane targets; GPU training throughput is a benchmark-protocol matter (L4).
- CSV sources of Stress size: the CSV reader keeps the parsed arrays in RAM (Standard tier, 10^7 rows ≈ 160 MB, is fine); binary (.npy/.npz) sources stream. Preprocessing versions of Stress captures are not chunked (about 5× the raw size in RAM).

## Reproduce

```bash
python scripts/perf_report.py --workspace /tmp/perf-ws --out docs/releases/performance-report.md --minutes 30.0 --stress
```
