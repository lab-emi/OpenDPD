# MATLAB over HTTP: a transport that keeps Python out of MATLAB's process

**Status: proposal for 2.5, not implemented.** Nothing in the server or the toolbox changes with this document. It answers
the roadmap's M3 item "HTTP-native transport" (decision 4: should HTTP become the default transport in 2.5?) with what the
code does today, what a feasibility spike measured (§3), and the smallest server surface that would make the transport
possible (§4). Statements marked *verified* were run on MATLAB R2026a Update 5, Linux, against a real local Studio service
or a stand-in server; everything else is design.

## 1. Why another transport

| Transport | How MATLAB reaches OpenDPD | What it costs |
| --- | --- | --- |
| In-process Python (`py.`, MATLINK, `opendpd.studio`) | Python runs inside MATLAB | Python and MATLAB versions must be compatible; a crash in an extension takes MATLAB down; `pyenv` is process-wide and cannot change once Python is loaded |
| Process (`opendpd.fit`, `Job.apply`, `Job.export`) | MATLAB starts `python -m opendpd.sdk._...` per call and exchanges files | No version matrix and no crash propagation, but every call starts Python and imports PyTorch again (at least 0.85 s on the spike machine, before any model is rebuilt), and nothing stays resident between calls |
| HTTP (this proposal) | MATLAB talks to the Studio service with `matlab.net.http` | One long-lived service per workspace; binary I/Q goes over the wire; the same client can later talk to a service on another machine |

The process transport is what 2.4 ships and it stays. HTTP adds what it cannot give: a resident model (iterative DPD,
lab loops, repeated `apply`), no per-call interpreter start, and a path to a remote GPU.

## 2. What exists (read from the code)

* The service listens on `127.0.0.1` on an OS-chosen port and writes `.studio.lock` (pid, creation time, port, URL,
  launcher secret) into the workspace. `POST /bootstrap/mint` with the launcher secret in `X-OpenDPD-Launcher` returns a
  single-use token (valid 120 s); `POST /api/v1/session/bootstrap` turns it into a session cookie (`opendpd_session`) and a
  CSRF token. Every non-GET request needs `X-OpenDPD-CSRF`; `Host` must be a loopback name; duplicate security headers are
  refused. (`opendpd/server/security.py`, `opendpd/sdk/client.py`)
* Request bodies are capped at 2 MiB except on a few listed upload paths (`/api/v1/imports` allows 2 GiB, the CSV uploads
  26 MiB, a few others less). Custom-data entry points (`/datasets/import`, `/datasets/upload`, `/imports`, …) are paused by `DatasetImportBoundary` wherever the service is
  created with `allow_custom_datasets=False` (covered by `tests/integration/test_custom_dataset_gate.py`).
* `services.datasets.import_arrays` is the strict array path: paired, finite, real N×2 values inside the float32 range,
  contiguous split with guard samples. The Python SDK reaches it by staging an `.npz` under the workspace's `imports` root and
  calling `POST /datasets/import`, which needs a shared file system: a remote client cannot do that.
* `Job.apply` runs `python -m opendpd.sdk._infer` for every call, which rebuilds the model from the checkpoint
  (`services.inference.apply_waveform`; at most 2^25 samples; `gru`, `tres_gru`, `gmp`, `mp_ls`, `gmp_ls`). `Job.export`
  does the same for `opendpd-model-v1` through `opendpd.sdk._export`; no HTTP route returns that package. The routes
  `POST /api/v1/exports`, `POST /api/v1/deploy/exports` and `GET /api/v1/exports/{id}` exist for experiment and
  `fixed-point-v1` packages.
* The Python client deliberately ignores system proxies for the loopback service (`ProxyHandler({})`) and refuses redirects.

## 3. Feasibility spike (verified)

The scripts are in [`matlab-http-spike/`](matlab-http-spike/README.md) (`run_spike.sh a` and `b`); the numbers below were
reproduced from that folder, part B within run-to-run noise. Part A used a real local service in a scratch workspace and no
Python inside MATLAB. Part B used a **stand-in** server (Starlette, loopback) that reads an `.npy` body, multiplies by 0.5
and answers with an `.npy`: it measures MATLAB's HTTP client and the loopback path, **not** OpenDPD's server, which will add
parsing, validation and model time. 32-core Linux workstation; no other load.

**A. The handshake works with `matlab.net.http` alone.**

| Step | Result |
| --- | --- |
| Mint a bootstrap token with the launcher secret from `.studio.lock` | 200, JSON token (first call 0.6 s) |
| Bootstrap a session | 200; one `Set-Cookie` (`opendpd_session`, `HttpOnly`, `SameSite=strict`) and a 43-character CSRF token |
| `GET /api/v1/system/capabilities` with the cookie, built by hand from `SetCookieField.convert()` | 200, service version and workspace as expected |
| Same without the cookie | 401 |
| Mutating POST with the cookie, no CSRF header | 403 `csrf_required` |
| Mutating POST with cookie and CSRF, invalid body | 422 `invalid_request` (the boundary let it through to validation) |
| `Host: evil.example` | 400 |

MATLAB did not carry the cookie by itself in this flow; the client must put it back on each request. An empty POST body (the
mint request) makes MATLAB emit warning `MATLAB:http:BodyExpectedFor`; the client must silence it locally.

**B. Binary bodies: which MATLAB classes to use.** Round trips of an N×2 single I/Q file, both directions equal in size.

| Upload → download | 32 MiB | 128 MiB | Extra memory (128 MiB case) |
| --- | --- | --- | --- |
| `FileProvider` → `FileConsumer` | 0.68 s | 0.39 s | 4 MiB |
| `MessageBody(uint8)` → `FileConsumer` | 0.24 s | 0.28 s | 381 MiB |
| `MessageBody(uint8)` → `BinaryConsumer` | 3.1 s | **76.6 s** | **4.2 GiB** |
| `FileProvider` → `BinaryConsumer` | 3.2 s | 80 s (first run) | about 4.2 GiB (first run) |

* `BinaryConsumer` grows its buffer quadratically; never use it for payloads. Download with `FileConsumer`, then read the file.
* `FileProvider` streams from disk: 512 MiB went up in 0.41 s (1.2 GiB/s) with 1 MiB of extra memory. A `uint8` `MessageBody`
  costs about three times the body in extra memory (1.5 GiB for 512 MiB).
* `MessageBody(uint8)` with `Content-Type: application/x-npy` is refused by the client (it only converts `uint8` for known
  binary types such as `application/octet-stream`); `FileProvider` accepts any type.
* Keep the `FileProvider` in a variable until `send` returns: a provider created inline inside the `RequestMessage`
  expression was deleted before sending in one run (error "object invalid or deleted").
* A 65,536-sample round trip (0.5 MiB each way, stand-in compute) took a median of 24 ms (range 23–32 ms, 30 calls). The
  process transport's fixed cost for the same call is at least the 0.85 s that `import torch` plus the inference service take
  to start here (5 runs, 0.84–0.86 s), before the model is rebuilt.
* Writing the 128 MiB `.npy` with the toolbox's `writeNpy` took 0.06 s; the answer was read back with `readNpy` and equals
  `0.5·x` exactly in every case.

**Not verified.** Windows and macOS; MATLAB releases other than R2026a; proxies, antivirus and firewalls (the spike set
`UseProxy=false`; whether MATLAB's default would route loopback traffic through a configured proxy was not tested);
interruption (Ctrl+C) during a transfer; automatic cookie handling by MATLAB; a real OpenDPD upload or inference route
(none exists yet); TLS; a server on another machine.

## 4. Proposal

### 4.1 Principles

1. The local boundary is unchanged: loopback only, session cookie plus CSRF, size caps per path, no cross-origin use.
2. No code is ever deserialised: arrays are parsed with `allow_pickle=False` after an explicit header check; the model
   packages stay data-only (`opendpd-model-v1`, `fixed-point-v1`).
3. New routes reuse the existing services (`import_arrays`, `apply_waveform`, the export writers), so results are identical
   to the process transport by construction and a parity test can say so.
4. Binary payloads are single-body requests so the MATLAB client can stream them from or to a file (§3 B).

### 4.2 Routes

| Route | Body → response | Reuses | Notes |
| --- | --- | --- | --- |
| `GET /readyz` (exists) | add `binary_transport_version: 1`; limits and devices in `GET /api/v1/system/capabilities`: `iq_profile`, `max_upload_bytes`, `max_samples`, `devices` | the `matlink_protocol_version` field the SDK already checks | A client refuses an older service with a clear message instead of a 404 |
| `POST /api/v1/uploads` | raw `.npy` (`Content-Type: application/x-npy`) → 201 `{upload_id, bytes, sha256, shape, dtype}` | quarantine storage of the CSV flow | Streams to disk, checks the header while receiving, computes SHA-256, stored for a short TTL |
| `POST /api/v1/datasets/arrays` | JSON `{dataset_id, display_name, input_upload, output_upload, signal, origin, guard_samples, notes}` → the dataset manifest | `import_arrays` | Same validation and errors as the in-process path; consumes the two uploads; added to `DatasetImportBoundary.PATHS` |
| `GET /api/v1/datasets/{id}/arrays?version=&which=input\|output` | → `.npy` stream, `ETag` = SHA-256 | `load_version_arrays` | Generated captures and exported datasets back to MATLAB |
| `POST /api/v1/runs/{run_id}/model-exports` | JSON `{format: "opendpd-model-v1"}` → 201 with `download_url` | the SDK's `_export` code path | Then `GET /api/v1/exports/{id}` (exists); deterministic bytes |
| `POST /api/v1/runs/{run_id}/infer?execution=&chunk_samples=&device=` | `.npy` (N×2) → `.npy` (N×2); metadata in `X-OpenDPD-Inference` (JSON) | `apply_waveform` | Resident model cache (§4.4); `device` defaults to `cpu` |
| `POST /api/v1/runs/{run_id}/streams`, `POST /api/v1/streams/{id}/chunks`, `DELETE /api/v1/streams/{id}` | chunk `.npy` → chunk `.npy` | the streaming variants of the registry | Hidden state stays on the server (and on the GPU); idle TTL; per-session cap. Chunks of thousands of samples, not single samples |

Existing routes cover the rest of a `fit`: `POST /runs` (submit), `GET /runs/{id}` (status), the run artifacts and exports.

**NPY profile `iq-npy-v1`.** Format version 1.0 or 2.0; `descr` `<f4` (and `<f8`, converted on arrival); shape `(N, 2)`;
C or Fortran order (MATLAB's column-major N×2 matrix needs no permute copy in Fortran order); header ≤ 4 KiB; data length
equal to shape × item size exactly; finite values; N ≤ `max_samples`. Object dtypes, structured dtypes, big-endian data and
any extra header key are refused with `npy_invalid`. The toolbox's `readNpy`/`writeNpy` already implement the same profile
on the MATLAB side.

**Errors** keep the existing envelope `{"error": {"code", "message", "details"}}`. New codes: `npy_invalid` and `iq_invalid`
(422), `upload_too_large` (413), `upload_not_found` (404), `quota_exceeded` (429), `device_unavailable` (422),
`stream_not_found` (404), `stream_expired` (410). `run_not_finished` and `payload_too_large` exist.

### 4.3 MATLAB client

* `opendpd.internal.http.*` on `matlab.net.http` only; `Transport="http"` next to `"process"` and `"python"`.
* The rules from §3: `HTTPOptions` with `UseProxy=false` for loopback and explicit connect, data and response timeouts;
  `FileProvider` for every upload and `FileConsumer` for every download, with temporary files removed by `onCleanup`; the
  session cookie and CSRF token held in the client object and attached to every request; errors mapped from the envelope to
  `opendpd:*` identifiers, as `fit` does for the process transport.
* Lifecycle: start `python -m opendpd.sdk._server --workspace W` as a child process with the same Java `ProcessBuilder` helper
  the process transport uses (no `pyenv`), wait for `.studio.lock` and `/healthz`, attach, and stop it with the existing
  stop-file protocol on `closeProject` (a service that was already running is attached to and never stopped).
* Python is never loaded into MATLAB; the Python version only has to satisfy OpenDPD, not MATLAB's compatibility table.

### 4.4 Server behaviour worth deciding explicitly

* **Limits.** Each new route gets its own body cap in `LocalBoundaryMiddleware` (today a path not in the lists is capped at
  2 MiB): 2^25 samples × 8 bytes = 256 MiB per array. Upload storage has a byte quota and a TTL sweep that also runs after a
  crash; an aborted upload (MATLAB interrupted) removes its partial file.
* **Resident models.** An LRU keyed by run id, checkpoint hash and device, bounded by count and bytes; one lock per model
  instance; compute off the event loop; evicted when the run is deleted. Whether inference queues behind training on a shared
  GPU is open (§6).
* **Gating.** `uploads`, `datasets/arrays` and the dataset download go in `DatasetImportBoundary` so custom data stays paused
  where it is paused today.
* **Generated clients.** `docs/contracts/openapi.json` and `frontend/src/api/schema.ts` are regenerated
  (`scripts/export_openapi.py`) and `test_openapi_is_exportable_and_committed` fails until they are.

## 5. Phasing and acceptance

| Step | Deliverable | Acceptance |
| --- | --- | --- |
| P0 | This document and the spike | Done; nothing shipped |
| P1 | Server routes behind `binary_transport_version` | NPY fuzz (truncated, object dtype, huge header, NaN, wrong shape) refused with the right code; body caps hold for chunked bodies; gate test with `allow_custom_datasets=False`; `infer` equals `apply_waveform` bit for bit on CPU |
| P2 | MATLAB client and `Transport="http"` for `importIQ`, `fit`, `apply`, `export` | Real-service tests in the style of `TestTransport`; fault injection against a stand-in (timeout, 401, 403, 413, 5xx, truncated download, interrupt); no leftover temp files or processes |
| P3 | Cross-transport parity and timing report | Same inputs through `process` and `http` give identical dataset hashes, identical inference output, and the timing table of §3 measured against the real routes |
| P4 | HTTP becomes the default in 2.5 (decision 4) | `process` stays as fallback; the quick-start works with no `pyenv` on a clean Windows machine |
| P5 | Remote service over an SSH tunnel | Needs a way to bootstrap a session without the lock file (for example pasting the single-use token the service prints); verified on a second machine |
| P6 | Hosted Studio | Out of scope until decision 5: non-browser authentication, quotas and a data-privacy statement need their own review |

## 6. Open questions for the maintainers

1. Two-step upload (`uploads` then `datasets/arrays`, proposed) or one multipart request? The spike measured single-body
   streaming only; multipart through `MultipartFormProvider` was not tried.
2. Accept `<f8` arrays, or require `<f4` and make the client convert (halves the bytes for double-precision MATLAB data)?
3. Should `infer` and `streams` exist on a service with custom data paused? They take user waveforms, so the proposal gates
   them the same way.
4. Inference and training on one GPU: queue inference behind running jobs, reserve a share, or CPU-only until measured?
5. Is a second public API surface acceptable before the contract tests and the generated clients cover it, or should P1 land
   with the OpenAPI additions in the same change?

## 7. Out of scope

Authentication for non-browser clients on a hosted service, TLS termination, multi-user quotas, and any change to the
registry, execution semantics or model causality. The sample-by-sample `step` of a System object over HTTP is not a goal:
frames of thousands of samples cost 24 ms round trip (§3), single samples would be dominated by it.
