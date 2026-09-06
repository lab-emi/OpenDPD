# Studio API conventions (S04)

Base path `/api/v1`; the OpenAPI document is committed at
`docs/contracts/openapi.json` (CI fails when it is stale; regenerate with
`python scripts/export_openapi.py`). Frontend types are generated from that
file (S05).

## Sessions

1. The launcher prints `http://127.0.0.1:<port>/bootstrap?token=…`. Opening it
   exchanges the one-time token for an `HttpOnly; SameSite=Strict` cookie and
   redirects to `/` so the token never stays in the address bar.
2. `GET /api/v1/session` returns `{authenticated, csrf_token}`. Every
   state-changing request (POST) must send `X-OpenDPD-CSRF: <csrf_token>`.
3. Non-browser clients call `POST /api/v1/session/bootstrap {"token"}` and
   get the same cookie plus the CSRF token in the body.

Security decisions and their tests: `docs/architecture/threat-model.md`.

## Errors

Every non-2xx body has one shape:

```json
{"error": {"code": "invalid_config", "message": "…", "details": [{"field": "model.key", "message": "…", "hint": "…"}], "hint": "…"}}
```

| Status | Code | Meaning |
|---|---|---|
| 400 | `host_not_allowed` | Host header is not a loopback name |
| 401 | `unauthorized`, `bad_bootstrap_token` | no session / wrong token |
| 403 | `csrf_required`, `cross_origin_write`, `cors_not_supported` | write without CSRF header, foreign Origin, preflight |
| 404 | `run_not_found`, `result_not_available`, `artifact_not_found`, `artifact_missing`, `not_found` | |
| 409 | `run_not_finished`, `cursor_out_of_range`, `workspace_error` | state conflicts; the cursor code means "resynchronise from a snapshot" |
| 413 | `payload_too_large` | body above 2 MB |
| 422 | `invalid_request`, `invalid_config` | request shape / experiment configuration (field-level details) |
| 503 | `shutting_down` | server is stopping; nothing was started |

`POST /experiments/validate` never uses 4xx for configuration problems: it
returns `{ok, errors[], warnings[], resolved}` so the form can show them
inline.

## Runs and events

- `POST /runs` returns 201 with the queued record; with an `idempotency_key`
  that already exists it returns 200 and the existing record.
- `GET /runs/{id}` includes `heartbeat_stale` (running run silent > 60 s):
  the UI shows "log disconnected", never invents a terminal state.
- Events are numbered by `seq` (1-based, gap-free per run).
  `GET /runs/{id}/events/list?after=N` pages; `GET /runs/{id}/events?after=N`
  streams SSE with `id: <seq>`, `event: <type>` and a final `event: end`
  when the run is terminal; `Last-Event-ID` is honoured on reconnect.
  A cursor beyond the last seq is a 409; the client re-reads the run
  snapshot and starts from `after=0`. Events are never pruned.
- Logs are paged by byte offset (`GET /runs/{id}/logs?offset=&limit=`) so a
  multi-GB log never has to be loaded.
- Artifacts are downloaded only by registered id
  (`GET /artifacts/{run_id}/{artifact_id}`); clients never send paths.

## Plan §7.1 mapping

| Plan route | Status |
|---|---|
| `GET /healthz`, `GET /readyz` | done (`readyz` also reports whether the built frontend is present and matches the package version) |
| `GET /system/capabilities`, `GET /models` | done (plus `GET /recipes`) |
| `POST /datasets/import`, `POST /datasets/{id}/diagnostics` | done (S07): import roots (`GET /datasets/import-roots`, `…/{root}/files`), `POST /datasets/inspect`, `POST /datasets/import` (by root + relative path only), `POST /datasets/upload` (multipart, streamed, 2 GB cap), `GET/POST /datasets/{id}/diagnostics`, `POST /datasets/{id}/manifest`, `POST /datasets/{id}/preprocess[/preview]`; plus `import-builtin` |
| `POST /experiments/validate`, `POST /runs`, `GET /runs/{id}`, `GET /runs/{id}/events`, `POST /runs/{id}/cancel` | done (plus `retry`, `config`, `artifacts`, `logs`, `events/list`, and `GET /runs/{id}/lineage` (S10): PA surrogate / DPD model / retry links with the checkpoint hashes used, read from resolved configurations; S16 adds the `evaluate_measured` task: a `measurement` block names the played `run_dpd` run and capture files under `imports/` (uploaded through `POST /datasets/upload`), binding fills dataset, DPD weights and file hashes, and the lineage link `measured_playback` points at the played run) |
| `GET /results/{id}` | done (result id = run id; `?profile=` serves the result under another registered profile, `GET /results/{id}/profiles` lists what is stored). `run_dpd` runs have results too (S10): the exported `u` scored through the bound surrogate, with the x/u/y chain, baselines under the same reference, surrogate coverage and scaling (`docs/protocols/simulation-chain.md`). `evaluate_measured` runs (S16) carry `evidence_type: dpd_measured` and a `measurement` block: attestation, declared conditions, per-capture hash/delay/correlation/gain, level difference; the comparison key adds the declared operating point and never ranks measured against simulated results (`docs/protocols/measured-dpd.md`) |
| `GET /metrics/profiles`, `GET /metrics/profiles/{id}` | done (S08): the registry behind every score (`opendpd/core/metrics`); every registered profile is scored at run end from the best checkpoint |
| `GET /artifacts/{id}` | done as `/artifacts/{run_id}/{artifact_id}` (artifact ids are scoped to a run) |
| `POST /exports` | done (S11): `{run_id, kind: full|share}` writes `<workspace>/exports/<id>.zip` and returns the manifest + `download_url`; `GET /exports/{export_id}` serves it; `POST /imports` (multipart, 2 GB cap) verifies every hash before writing and returns the import report; conflicts and damaged packages are 422 with a specific code (`docs/protocols/experiment-packages.md`) |
| `GET /results/{id}/report?format=html|md` | done (S11): reports bound to the stored result and plot data; nothing recomputed |
| `GET /results/compare?runs=&profile=&format=json|csv` | done (S11): results side by side under one profile with the pairwise incompatibilities (dataset, data version, split, reference, profile version, evidence, execution semantics) stated explicitly; `GET /runs/{id}/history` serves the per-epoch curves; plot data (`plot-spectrum`, `plot-time`, `plot-amam`, plots-v1) are artifacts with fixed budgets |
| `GET /metrics/profiles` (`MetricProfile.validation`) | S15: every profile carries `validation` (`golden`, `analytic`, `pending_cross_validation`, `cross_validated`); the GUI offers only profiles past `pending_cross_validation`, the service computes and stores all of them; `SignalSpec.waveform` (`WaveformBinding`) records a dataset's binding to a reference waveform set by `datasets import --waveform` |

## Listing, search and paging (S13)

`GET /api/v1/runs?limit=&offset=&status=&q=` returns one page of runs, newest
first; `q` is a case-insensitive substring search over the run id and the
stored record (name, dataset, model key), with SQL wildcards treated as
literal characters. `GET /api/v1/runs/count?status=&q=` returns `{"count": n}`
for the same filters so a client can page a history of thousands of runs
without ever loading it whole. Both index the workspace first, so runs made by
the CLI or the Python API appear as soon as they finish.

## Response headers (S13)

Every response carries a Content-Security-Policy (`default-src 'self'`,
`script-src 'self'`, no inline scripts, no external hosts, `frame-ancestors
'none'`), `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY` and
`Referrer-Policy: same-origin`; API, bootstrap and health responses add
`Cache-Control: no-store`. Request bodies are capped whether or not they
announce a `Content-Length` (413 `payload_too_large`; uploads 413
`too_large`).
