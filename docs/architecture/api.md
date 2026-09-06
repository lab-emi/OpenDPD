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
| `POST /datasets/import`, `POST /datasets/{id}/diagnostics` | S07; today only `POST /datasets/import-builtin` |
| `POST /experiments/validate`, `POST /runs`, `GET /runs/{id}`, `GET /runs/{id}/events`, `POST /runs/{id}/cancel` | done (plus `retry`, `config`, `artifacts`, `logs`, `events/list`) |
| `GET /results/{id}` | done (result id = run id: one formal result per run) |
| `GET /artifacts/{id}` | done as `/artifacts/{run_id}/{artifact_id}` (artifact ids are scoped to a run) |
| `POST /exports` | S11 |
